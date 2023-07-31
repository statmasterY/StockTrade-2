# Copyright (c) 2022 MX
# 邢子文(XING-ZIWEN) <ziwen.xing@muxinasset.com>
# STAMP|380907

# from quantvale.bt_pilots.index_alpha.INDEX_ALPHA_BT_V5 import TaskProto

# from .strategy import Strategy

# class Task(TaskProto):
#     def execute(self):
#         self.delegate(Strategy)

import os

import pandas as pd
import numpy as np
import datetime
from datetime import date
from operator import itemgetter
from scipy.stats import spearmanr
import statsmodels.api as sm
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor ##
from pytorch_tabnet.metrics import Metric ##
import torch ##

import quantstats as qs
import pyfolio as pf
import warnings

from quantvale import utils
from quantvale import useful
from quantvale import backtest
from quantvale import access
from quantvale.access_app.factor_data import get_factors
from quantvale import preload


PRELOADID_FULL_DAY_BARS = '__PL__A__FULL_DAY_BARS'


class Task(backtest.BaseBacktestTask):
    def execute(self):
        try:
            exe_start_time = utils.now()
            # 保存参数
            # nums_xg=30 # 训练xgboost得到的因子数量 备选：10,20
            # nums_fin=10 # 回归筛选的因子数量 备选：10,20
            # factor_quantile= 20 # 股票分组数 备选：5,10
            # method_return_cal = 'long_short' # 因子单日收益率估计方法 备选：'long_short','long_only','short_only'
            # method_factor_weight = 'equal' # 股票权重分配方法  备选：'return','equal'
            # method_factor = 'long' # 因子收益率计算方法 备选：'long','long_only','short_only','long_short'
            # backdays = 252 # 因子收益率计算周期 备选：252,126,31,7
            # period = 5 # 调仓周期 备选：5,7,10,20
            # return_period = 5 # 计算最近因子收益的周期，用于回归筛选特征 备选：5,10,20
            # factor_period = 21 # 风格切换的周期 备选：10,21，42
            # amount_num = 5 # 股票账户数量 备选：5,7
            # fee_rate = 0.003 # 手续费率 备选：0.003,0.001
            # if_xgboost = False # 是否使用xgboost模型 备选：True,False

            # regression_period = 21 ##新增修改（用于回归的样本点个数） 备选：252,126,21,7
            # sample_weight = 'equal_weight'
            # if_indicators = False # 回归筛选因子时是否把自变量变换为±1 备选：True,Falses

            # index='date' # 日期列名
            # y_index='return_norm' # 因变量列名
            # n_splits=10   # 交叉验证折数
            # verbose=1 # 交叉验证输出信息等级
            # FactorData="//192.168.2.39/WuyihangStrategies/ALL_factorValues_2021-2023.csv" #待修改

            ####解包参数，写为args形式
            self.taskID, self.FactorData, self.args, self.path \
                = list(itemgetter('TASKID', 'FactorData', 'Arguments', 'Path')(self.params))

            start_date,end_date,backdays,period,return_period,factor_period,amount_num,nums_fin,factor_quantile,if_industry_market_cap_norm,\
            sample_weight,regression_period,method_return_cal,method_factor_weight,method_stock_weight,method_factor,\
            fee_rate,nums_xg,if_xgboost,if_indicators,n_splits,verbose,index,y_index = tuple(self.args.values())
            return_period_upper = 22##

            start_date = pd.to_datetime(start_date).date()
            end_date = pd.to_datetime(end_date).date()

            # self.trade_cal, self.allQvcodes = itemgetter('trade_cal', 'all_qvcodes')(kwargs) # 交易日列表，股票池
            # self.allQvcodes = itemgetter('all_qvcodes')(self.kwargs) # 交易日列表，股票池
            # self.data_dir = os.path.join('../qv_task_root', self.taskID) # 这个是输出文件夹的目录，方便进行debug
            self.data_dir = self.gen_data_dir() 
            self.log.record(self.data_dir)
            self.progress(10)

            clt = access.get_default_access_client()

            factordata_df = pd.read_csv(self.FactorData) #做了修改调整
            factordata_df['date']=pd.to_datetime(factordata_df['date']).dt.date
            factordata_df.set_index(['date', 'qscode'], inplace=True)

            self.progress(20)

            ####数据预处理
            # 行业所在列
            index_industry = factordata_df.columns.get_loc('中证一级行业')
            # 只要行业所在列，return列后面的列，并去掉vol_std_126这一列
            try:
                factordata_df = factordata_df.iloc[:, index_industry:].drop(columns=['vol_std_126','log_return'])
            except:
                pass
            # return列所在的位置
            index_return = factordata_df.columns.get_loc('return')
            factordata_df = factordata_df.drop(factordata_df.columns[1:index_return], axis=1)
            # 处理缺失值
            factordata_df = factordata_df.groupby(level='qscode',sort=False,group_keys=False).fillna(method= 'ffill')
            factordata_df = factordata_df.dropna()

            ####行业中性化
            # 行业哑变量
            factordata_df = pd.get_dummies(factordata_df, columns=['中证一级行业'])

            def industry_norm(df):
                # 准备数据
                returns = df["return"]  # 股票收益率数据
                industry_factors = df.iloc[:,df.columns.str.contains('^中证一级行业')]  # 行业因子数据
                # 拟合线性回归模型
                model = sm.OLS(returns, industry_factors)
                results = model.fit()
                r = returns - np.dot(industry_factors ,(results.pvalues <0.05)*results.params)          
                return r

            # 市值中性化
            def market_cap_norm(df):
                # 准备数据
                returns = df["return"]  # 股票收益率数据
                market_cap = df["lncap"]  # 市值数据
                # 因子数据
                # factor_value = df.iloc[:,~df.columns.str.contains('^中证一级行业')].drop(columns=['return','log_return','lncap']) # 因子数据
                #factors = pd.concat([market_cap, factor_value], axis=1)
                # 为因子数据添加常数项
                #factors = sm.add_constant(factors)
                # 拟合线性回归模型
                model = sm.OLS(returns, market_cap)
                results = model.fit()
                if results.pvalues[0] > 0.05:
                    return returns
                residuals = returns - market_cap*results.params[0]
                return residuals

            # 同时进行行业和市值中性化
            def industry_market_cap_norm(df):
                # 准备数据
                returns = df["return"]  # 股票收益率数据
                industry_factors = df.iloc[:,df.columns.str.contains('^中证一级行业')]  # 行业因子数据
                market_cap = df["lncap"]  # 市值数据
                # # 因子数据
                # factor_value = df.iloc[:,~df.columns.str.contains('^中证一级行业')].drop(columns=['return','log_return','lncap']) # 因子数据
                # factors = pd.concat([industry_factors, market_cap, factor_value], axis=1)
                # # 拟合线性回归模型
                # model = sm.OLS(returns, factors)
                factors = pd.concat([industry_factors, market_cap], axis=1)
                # # 拟合线性回归模型
                model = sm.OLS(returns, factors)
                results = model.fit()
                # 方程市值的p值大于0.1，说明市值不显著，
                if results.pvalues[industry_factors.shape[-1]]> 0.1:
                    return industry_norm(df)
                r = returns - np.dot(factors ,(results.pvalues <0.05)*results.params)
                return r

            # 市值和行业中性化
            if if_industry_market_cap_norm:
                factordata_df['return_norm'] = factordata_df.groupby(level='date',sort=False,group_keys=False).apply(industry_market_cap_norm)
            else:
                factordata_df['return_norm'] = factordata_df['return']

            # 筛选变量
            factordata_df = factordata_df.drop(columns=['lncap'])
            # 删除包含行业哑变量的列,使用正则表达式，行业哑变量的列都是以中证一级行业开头的
            factordata_df = factordata_df.loc[:, ~factordata_df.columns.str.contains('^中证一级行业')]
            
            # 加权一致性相关系数CCC损失函数
            def ccc_loss(y_true, y_pred):
                if isinstance(y_true, np.ndarray):
                    y_true = y_true.reshape(-1, 1)
                    y_pred = y_pred.reshape(-1, 1)
                elif torch.is_tensor(y_true):
                    y_true = y_true.detach().numpy().reshape(-1, 1)
                    y_pred = y_pred.detach().numpy().reshape(-1, 1)

                y_true_mean = np.mean(y_true)
                y_pred_mean = np.mean(y_pred)
                y_true_std = np.std(y_true)
                y_pred_std = np.std(y_pred)

                cov = np.mean((y_true - y_true_mean) * (y_pred - y_pred_mean))
                rho = cov / (y_true_std * y_pred_std)

                ccc = 2 * rho * y_true_std * y_pred_std / (
                    y_true_std ** 2 + y_pred_std ** 2 + (y_true_mean - y_pred_mean) ** 2
                )
                return ccc

            ccc_scorer = make_scorer(ccc_loss, greater_is_better=True)
            # 自定义评估指标
            class CCCMetric(Metric):
                def __init__(self):
                    self._name = "ccc"
                    self._maximize = True

                def __call__(self, y_true, y_pred):
                    y_true = y_true.reshape(-1, 1)
                    y_pred = y_pred.reshape(-1, 1)
                    return ccc_loss(y_true, y_pred)

            ccc_metric = CCCMetric()

            #####################################

            # 训练XGBoost模型 来筛选特征

            def rolling_train(df, index, y_index, nums_xg=20, n_splits=10, verbose=1, params=None):
                '''按日期滚动训练模型
                Args:
                    df: 数据集
                    nums_xg: 筛选变量的个数
                    index: 日期索引 = 'date'
                    n_splits: 折数
                    verbose: 是否打印训练信息
                    params: 模型参数（字典）
                    params = {
                        'max_depth': 3,
                        'n_estimators': 100,
                        'learning_rate': 0.1
                    }
                '''
                # 按照日期索引进行排序
                df_sorted = df.sort_index(level=index)
                # 获取日期索引的唯一值
                unique_dates = df_sorted.index.get_level_values(index).unique()

                oof_preds = []  # 保存每个折叠模型在验证集上的预测结果
                oof_targets = []  # 保存每个折叠模型的真实结果
                scores = []  # 保存每个折叠模型的得分
                # 初始化特征重要性，索引为整数，添加一列为特征名，后续每一列为一折的特征重要性
                feature_importance_df = pd.DataFrame()
                feature_importance_df['feature'] = df.columns[:-1]

                # 创建TimeSeriesSplit对象，并进行划分
                tscv = TimeSeriesSplit(n_splits)

                print("开始训练模型...")

                fold_nb = 1 # 折数,从1开始计数，一直到n_splits
                for train_index, test_index in tscv.split(unique_dates):
                    train_dates = unique_dates[train_index]
                    test_dates = unique_dates[test_index]

                    print(f"正在训练第{fold_nb}折")

                    if verbose:
                        print(f"训练集日期：{train_dates[0]} - {train_dates[-1]}")
                    
                    # 根据训练日期和测试日期选择数据
                    # train_data = df_sorted.loc[train_dates] 
                    train_data = df_sorted.loc[df_sorted.index.get_level_values(index).isin(train_dates)]
                    # test_data = df_sorted.loc[test_dates]
                    test_data = df_sorted.loc[df_sorted.index.get_level_values(index).isin(test_dates)]

                    # 获取特征和目标变量
                    X_train = train_data.drop(columns=[y_index])
                    y_train = train_data[y_index]
                    X_test = test_data.drop(columns=[y_index])
                    y_test = test_data[y_index]

                    # 创建XGBRegressor对象
                    model = XGBRegressor()
                    model.set_params(**params)

                    # 用ccc_metric，训练数据和目标数据作为评估指标进行训练
                    model.fit(X_train, y_train, verbose=verbose)

                    # 在测试集上进行预测
                    y_pred = model.predict(X_test)

                    # 计算评估指标（这里使用ccc_metric作为示例）
                    ccc_score = ccc_metric(y_test.values, y_pred)
                    if verbose:
                        print("CCC得分:", ccc_score)

                    # 保存每个折叠模型在验证集上的预测结果和真实标签
                    scores.append(ccc_score)
                    # print(len(y_pred), len(y_test))
                    oof_preds.append(y_pred)
                    oof_targets.append(y_test)

                    # 保存每个模型的特征重要性，保存到feature_importance_df
                    feature_importances = model.feature_importances_
                    feature_importance_df[f'fold_{fold_nb}'] = feature_importances

                    fold_nb += 1

                # 计算所有折叠模型的平均得分
                mean_score = np.mean(scores)
                print("平均得分：", mean_score)

                # 平均相关系数
                corr = []
                for task_id in range(len(oof_targets)):
                    corr.append(np.corrcoef(oof_targets[task_id],
                                            oof_preds[task_id])[0, 1])
                mean_corr = np.mean(corr)
                print(f"平均相关系数：{mean_corr }")

                # 计算特征重要性的均值,按照特征来求均值（每一列的均值）,作为行添加到feature_importance_df中
                feature_importance_df['mean'] = feature_importance_df.mean(axis=1)
                feature_importance_df = feature_importance_df.sort_values(by='mean', ascending=False)

                # 可视化特征重要性
                plt.figure(figsize=(16, 12))
                sns.barplot(x='mean', y='feature', data=feature_importance_df)

                # 保存最重要的nums_xg个特征
                top_20_features = feature_importance_df['feature'][:nums_xg].values

                print("训练结束！")

                return [mean_score, mean_corr , top_20_features]

            # params
            params = {
                'max_depth': 3,
                'n_estimators': 100,
                'learning_rate': 0.1,
                'eval_metric': ccc_metric
            }

            if if_xgboost:
                # 用rolling_train函数训练模型
                res = rolling_train(factordata_df.drop(columns='return'), index=index, y_index=y_index,nums_xg=nums_xg, n_splits=n_splits, verbose=verbose, params=params )
                # 用最优模型来筛选特征
                top_20_features = res[2]
            else:
                top_20_features = ['exp_wgt_momentum_1m', 'trix_15', 'bb_pos_20', 'williams_20', 'macd_histogram_12_26', 'macd_12_26', 'cci_20', 'wgt_momentum_1m', 'eom_20', 'vol_std_21', 'turnover_std_1m', 'vr_20', 'ma_20', 'bb_range_20', 'exp_wgt_momentum_3m', 'vol_std_63', 'cmo_20', 'qstick_20', 'pet_20', 'ep']
            
            self.progress(30)
     
            ####计算因子收益率矩阵（尝试引入手续费，做了修改调整）
            def factor_return(factordata_df, factor_name=top_20_features, factor_quantile=5, method_return_cal = 'long_short', backdays = 252, return_period = 5):
                factors_r = pd.DataFrame(columns=factor_name)
                df = factordata_df.copy()
                for factor in factor_name:
                    #df是因子值矩阵
                    # 加入rank of return 避免重复值
                    df['factor_quantile_'+factor] = df.groupby(level='date', sort = False, group_keys = False).\
                        apply(lambda x: pd.qcut(0.9999*x[factor].rank()+0.0001*x['return_norm'].rank(), factor_quantile, labels=False, duplicates='drop'))
                    df = df.dropna()
                    # 1月1号构建的组合，利用1月2号的收益率来计算，保存为1月1号的因子收益率
                    ##################################
                    ##################################
                    if method_return_cal == 'long_short': # 买顶卖底
                        factors_r[factor] = df.groupby('date', sort = False, group_keys = False).apply(lambda x: \
                            x[x['factor_quantile_'+factor]==factor_quantile-1]['forward_return_norm'].mean()-x[x['factor_quantile_'+factor]==0]['forward_return_norm'].mean())
                    elif method_return_cal == 'long':
                        factors_r[factor] = df.groupby('date', sort = False, group_keys = False).apply(lambda x: \
                            x[x['factor_quantile_'+factor]==factor_quantile-1]['forward_return_norm'].mean()-x[x['factor_quantile_'+factor]==0]['forward_return_norm'].mean())
                    elif method_return_cal == 'long_only': # 买顶
                        factors_r[factor] = df.groupby('date', sort = False, group_keys = False).apply(lambda x: \
                            x[x['factor_quantile_'+factor]==factor_quantile-1]['forward_return_norm'].mean())
                    elif method_return_cal == 'short_only': # 卖底
                        factors_r[factor] = df.groupby('date', sort = False, group_keys = False).apply(lambda x: \
                            -1*x[x['factor_quantile_'+factor]==0]['forward_return_norm'].mean())
                
                #############################################新增部分
                ##记录每个因子的股票权重的变化，并引入手续费
                ## 先记录每个因子股票的买卖情况
                stock_weight_on_factors = df.filter(regex='factor_quantile')
                stock_weight_on_factors.replace(0,-1*(method_return_cal != 'long_only'),inplace=True)
                stock_weight_on_factors.replace(range(1,factor_quantile-1),0,inplace=True)
                stock_weight_on_factors.replace(factor_quantile-1,1*(method_return_cal != 'short_only'),inplace=True)
                ## 每一天对权重进行归一化处理
                stock_weight_on_factors = stock_weight_on_factors.groupby('date', sort = False, group_keys = False).apply(lambda x: x / x.abs().sum())
                ## 计算每个因子每天的股票权重的变化
                weight_change = stock_weight_on_factors.groupby('qscode', sort = False, group_keys = False).diff()
                weight_change.loc[weight_change.index.get_level_values('date')==weight_change.index.get_level_values('date')[0]] = stock_weight_on_factors.loc[stock_weight_on_factors.index.get_level_values('date')==weight_change.index.get_level_values('date')[0]]
                weight_change = weight_change.abs().groupby('date', sort = False, group_keys = False).sum()
                weight_change.columns = [factor.replace('factor_quantile_','') for factor in weight_change.columns]
                fee = weight_change*fee_rate
                factors_r_long = factors_r - fee #考虑手续费后做多因子的收益率
                factors_r_short = -factors_r - fee #考虑手续费后做空因子的收益率
                fac_mom_long = (factors_r_long + 1).rolling(backdays).apply(lambda x: x.prod()) # day1用到了day2的return
                fac_mom_short = (factors_r_short + 1).rolling(backdays).apply(lambda x: x.prod())
                # fac_mom_ternary用于决策因子是否要进行交易，以及交易的方向，只有当过去因子收益率剔除手续费的影响后仍有利可图时，才进行交易
                fac_mom_ternary = fac_mom_long.copy() #先不用dropna
                # 把非nan的值都变成0
                fac_mom_ternary = fac_mom_ternary.where(fac_mom_ternary.isna(),0)
                fac_mom_ternary[fac_mom_long > 1] = 1
                fac_mom_ternary[fac_mom_short > 1] = -1

                fac_mom_l = (factors_r + 1).rolling(backdays).apply(lambda x: x.prod()) # day1用到了day2的return
                fac_mom_s = (factors_r + 1).rolling(return_period).apply(lambda x: x.prod())

                return factors_r, df, fac_mom_l, fac_mom_s, fac_mom_ternary

            def feature_collect(factors_r_all,fac_mom_l, fac_mom_s,factor_name = top_20_features, nums_fin = 10, start_date = datetime.date(2022,6,15), return_period = 7,sample_weight = 'integer_weight',if_indicators=False):
                coefficients = []
                coefficients2 = []
                const = []
                const2 = []
                t = []
                t2 = []
                std = []
                std2 = []
                r2 = []
                sign = []
                # df 是 fac_mom, 取start_date之前的数据 ,shift 1 day,避免使用未来数据
                # df = (fac_mom_l>1)....
                df = (2*(fac_mom_l>1)-1).shift(1+return_period).loc[:start_date] # 过去一年的收益率不应包含当前考察时间段内的收益率，所以shift 1+period day
                df2 = fac_mom_s.shift(1).loc[:start_date]
                df3 = factors_r_all.shift(1).loc[:start_date].iloc[-63:]
                df3['x'] = range(1, len(df3)+1)
                for factor in factor_name:##后续改为factor_name
                    X = sm.add_constant(df[factor]) #################################### 不能用数值来计算
                    y = df2[factor]
                    cleaned_data = pd.concat([X, y], axis=1).dropna().iloc[-factor_period:]
                    position_index = ([j // return_period * return_period for j in range(0, len(cleaned_data))])
                    ##这里可以优化，引入样本点的权重
                    if sample_weight == 'integer_weight':
                        position_index = list(range(0,len(cleaned_data),return_period))
                        position_index = [[i]*(position_index.index(i)+1) for i in position_index]
                        position_index = sum(position_index,[])
                    X = cleaned_data.iloc[position_index,:-1]
                    if if_indicators:
                        X = np.sign(X)
                    y = cleaned_data.iloc[position_index,-1]
                    model = sm.OLS(y, X).fit()

                    # X2从1到len(df3)
                    x = pd.DataFrame({factor:df3['x']})
                    X2 = sm.add_constant(x)
                    y2 = (df3[factor]+1).cumprod()
                    cleaned_data2 = pd.concat([X2, y2], axis=1).iloc[-151:].dropna()
                    X2 = cleaned_data2.iloc[:,:-1]
                    y2 = cleaned_data2.iloc[:,-1]
                    model2 = sm.OLS(y2, X2).fit()

                    #提取回归系数
                    coefficients.append(model.params[factor])
                    coefficients2.append(abs(model2.params[factor]))
                    #提取截距
                    const.append(model.params['const'])
                    const2.append(abs(model2.params['const']))
                    #提取t值
                    t.append(model.tvalues[factor])
                    t2.append(abs(model2.tvalues[factor]))
                    #提取残差标准差
                    std.append(np.sqrt(model.scale))
                    std2.append(np.sqrt(model2.scale))
                    #提取正负号
                    sign.append(np.sign(model2.params[factor]))
                    #提取R2
                    r2.append(model2.rsquared)
                coefficients_df = pd.DataFrame({'coefficients':coefficients,'coefficients2':coefficients2}, index=factor_name)
                const_df = pd.DataFrame({'const':const,'const2':const2}, index=factor_name)
                t_df = pd.DataFrame({'t':t,'t2':t2}, index=factor_name)
                std_df = pd.DataFrame({'std':std,'std2':std2}, index=factor_name)
                sign_df = pd.DataFrame({'sign2':sign}, index=factor_name)
                r2_df = pd.DataFrame({'r2':r2}, index=factor_name)
                coef_df = pd.concat([coefficients_df,const_df, t_df, std_df, sign_df,r2_df], axis=1)
                # coefficients的秩和coefficients2的秩加权求和 rank越小表示值越大
                coef_df['up'] = coef_df['coefficients2'] + 1.96*coef_df['std2']
                coef_df['down'] = coef_df['coefficients2'] - 1.96*coef_df['std2']
                # coef_df['rank'] = coef_df['up'].rank(ascending=False) + 1.2* coef_df['down'].rank(ascending=False)
                # coef_df['rank'] = coef_df['up'].rank(ascending=False) + 0.8* coef_df['down'].rank(ascending=False)
                coef_df['rank'] = coef_df['coefficients2'].rank(ascending=False) + 0.3*coef_df['r2'].rank(ascending=False) - 0.3* coef_df['std2'].rank(ascending=False)
                # coef_df['rank'] = coef_df['coefficients'].rank(ascending=False) - 0.1* coef_df['std2'].rank(ascending=False)
                top_features = coef_df.sort_values(by='rank', ascending= True).index[:nums_fin] #这里的10是参数

                return top_features , coef_df
            
            self.progress(40)

            # 归一化函数
            def trans(x, method_factor = 'long_short'):
                """
                method: long_short, long_only, long
                return: fac_mom_ternary
                输入：一天的因子收益率值
                输出：归一化的因子权重值
                """
                # 用以计算归一化后的因子权重
                # long_short: sum (weight<0) = -1/2, sum (weight>0) = 1/2 分方向归一化后，再一起归一化， w有正有负，不作为投资建议，因为无法实现，只能用来判断因子是否有效,正收益说明策略有效
                # long: sum abs(weight) = 1, w有正有负 做空的因子变成买底部的，做多的因子变成买顶部的
                # long_only : sum weight = 1, w全正,卖空的因子为0
                # short_only: sum weight = 1, w全正,做多的因子为0
                if method_factor == 'long_short':
                    negative_mask = x < 0
                    positive_mask = x > 0
                    negative_sum = x[negative_mask].sum()
                    positive_sum = x[positive_mask].sum()
                    if negative_sum != 0:
                        x[negative_mask] = -x[negative_mask] / negative_sum
                    if positive_sum != 0:
                        x[positive_mask] = x[positive_mask] / positive_sum
                    x = x / x.abs().sum()
                elif method_factor == 'long_only':
                    x[x < 0] = 0
                    if x[x > 0].sum() != 0:
                        x = x / x[x > 0].sum()
                elif method_factor == 'short_only':
                    x[x > 0] = 0
                    if x[x < 0].sum() != 0:
                        x =  x / x[x < 0].sum()
                elif method_factor == 'long':
                    x = x / x.abs().sum()
                else:
                    print('method error')
                return x

            # 计算每个因子的权重
            def get_weight(fac_r,fac_mom_l,fac_mom_s,fac_mom_63,fac_mom_ternary,coef_df,top_features,method_factor_weight = 'equal',method_factor = 'long'):  ###做了修改调整
                """
                fac_mom_ternary: 每个因子是否买入的矩阵，只包含选出来的特征
                method: 权重分配方法
                return: 每个因子的权重 days * factors ,shape = len(days-1)*len(factors)，有正有负
                """

                # 1月3号的持仓权重保存在1月2号（1月2号收盘时的数据作为1月3号的权重）
                # 但是基于1月1号的因子值（因为用了1月2号的收益率来计算1月1号的因子收益率），所以要用shift(1)来调整
                # 在backtest中用第二天的收益率来计算策略收益的。

                # 每个因子买卖的权重
                if method_factor_weight == 'equal':
                    weight = fac_mom_ternary[top_features].shift(1).apply(lambda x : trans(x,method_factor=method_factor), axis=1)
                elif method_factor_weight == 'return':
                    # 因子做多还是做空：用np.sign(0.8*(fac_mom_l-1)+0.19*(fac_mom_63-1)+0.01*(fac_mom_s-1))来判断
                    # 如果近期收益方向与过去收益方向相反，则降低最近的权重 修正后的值 = 0.87*(fac_mom_l-1)+0.12*(fac_mom_63-1)+0.01*(fac_mom_s-1)
                    fac_mom_l = fac_mom_l[top_features].reindex(index=fac_mom_s.index) # 保留和fac_mom_s相同的index对应的值
                    fac_mom_63 = fac_mom_63[top_features].reindex(index=fac_mom_s.index) # 保留和fac_mom_s相同的index对应的值

                    # 方法 1：
                    # weight = np.sign(0.8*(fac_mom_l-1)+0.19*(fac_mom_63-1)+0.01*(fac_mom_s-1))*abs(0.85*(fac_mom_l-1)+0.14*(fac_mom_63-1)+0.01*(fac_mom_s-1))\
                    #     .shift(1).apply(lambda x : trans(x,method_factor=method_factor), axis=1)
                    # weight = fac_mom_ternary.shift(1).apply(lambda x : trans(x,method_factor=method_factor), axis=1)
                    # weight = (fac_mom_l-1).shift(1).apply(lambda x : np.sign(x), axis=1).\
                    #     apply(lambda x : trans(x,method_factor=method_factor), axis=1)

                    # 方法 2：
                    # 类布林带策略
                    df2 = fac_r[top_features].shift(1).iloc[-(63+factor_period):,:].copy()
                    df3 = (df2+1).cumprod(axis=0)
                    weight = df3.copy()
                    weight['x'] = range(1,63+factor_period+1)
                    coef_df2 = coef_df[coef_df.index.isin(top_features)].copy()
                    # 加减仓的上界
                    up_1 = weight.apply(lambda x: coef_df2['coefficients2']*coef_df2['sign2']*x['x']+ coef_df2['const2'] + 1*coef_df2['std2'],axis = 1)
                    up_2 = weight.apply(lambda x: coef_df2['coefficients2']*coef_df2['sign2']*x['x']+ coef_df2['const2'] + 2*coef_df2['std2'],axis = 1)
                    up_3 = weight.apply(lambda x: coef_df2['coefficients2']*coef_df2['sign2']*x['x']+ coef_df2['const2'] + 3*coef_df2['std2'],axis = 1)
                    up_4 = weight.apply(lambda x: coef_df2['coefficients2']*coef_df2['sign2']*x['x']+ coef_df2['const2'] + 4*coef_df2['std2'],axis = 1)
                    # 加减仓的下界
                    down_1 = weight.apply(lambda x: coef_df2['coefficients2']*coef_df2['sign2']*x['x']+ coef_df2['const2'] - 1*coef_df2['std2'],axis = 1)
                    down_2 = weight.apply(lambda x: coef_df2['coefficients2']*coef_df2['sign2']*x['x']+ coef_df2['const2'] - 2*coef_df2['std2'],axis = 1)
                    down_3 = weight.apply(lambda x: coef_df2['coefficients2']*coef_df2['sign2']*x['x']+ coef_df2['const2'] - 3*coef_df2['std2'],axis = 1)
                    down_4 = weight.apply(lambda x: coef_df2['coefficients2']*coef_df2['sign2']*x['x']+ coef_df2['const2'] - 4*coef_df2['std2'],axis = 1)
                    # up_1 = coef_df['coefficients2']*coef_df['sign2']*range(63+1,63+21+1) + coef_df['const2'] + 1*coef_df['std2']
                    mean_0 = weight.apply(lambda x: coef_df2['coefficients2']*coef_df2['sign2']*x['x']+ coef_df2['const2'],axis = 1)
                    weight = weight.drop(columns=['x'])

                    # 把weight按照up_1的columns来排序
                    weight = weight[up_1.columns]
                    weight2 = weight.copy()
                    
                    # 达到上界，则做多的因子权重减少，做空的因子权重增加；达到下界，则做多的因子权重增加，做空的因子权重减少
                    # 达到上界或下界的触发信号，每当连续出现1时，保留第一次出现的1，后面重复这个
                    signal_up_0 = (weight2 > mean_0) & ~(weight2 > mean_0).shift(1).fillna(False)
                    signal_up_1 = (weight2 > up_1) & ~(weight2 > up_1).shift(1).fillna(False)
                    signal_up_2 = (weight2 > up_2) & ~(weight2 > up_2).shift(1).fillna(False)
                    signal_up_3 = (weight2 > up_3) & ~(weight2 > up_3).shift(1).fillna(False)
                    signal_up_4 = (weight2 > up_4) & ~(weight2 > up_4).shift(1).fillna(False)
                    signal_down_0 = (weight2 < mean_0) & ~(weight2 < mean_0).shift(1).fillna(False)
                    signal_down_1 = (weight2 < down_1) & ~(weight2 < down_1).shift(1).fillna(False)
                    signal_down_2 = (weight2 < down_2) & ~(weight2 < down_2).shift(1).fillna(False)
                    signal_down_3 = (weight2 < down_3) & ~(weight2 < down_3).shift(1).fillna(False)
                    signal_down_4 = (weight2 < down_4) & ~(weight2 < down_4).shift(1).fillna(False)

                    # 方法一：每个区间一个权重，中等收益中等回撤
                    position = pd.DataFrame(np.nan,index=weight2.index,columns=weight2.columns)
                    position[signal_up_0] = 0
                    position[signal_up_1] = -0.1 # 越过上界，做多的因子权重减少
                    position[signal_up_2] = -0.15
                    position[signal_up_3] = -0.2 
                    position[signal_up_4] = -0.1 # 因子动量反转了，所以做多转为做空，做空转为做多
                    position[signal_down_0] = 0
                    position[signal_down_1] = 0.1 # 越过下界，做多的因子权重增加
                    position[signal_down_2] = 0.15
                    position[signal_down_3] = 0.2 
                    position[signal_down_4] = 0.1 # 因子动量反转了，所以做多转为做空，做空转为做多
                    position = position.fillna(method='ffill').fillna(0)* coef_df2['sign2'] + 1
                    # 初始权重
                    weight2.iloc[-64:,] = (np.exp(120*coef_df2['coefficients2']) * coef_df2['sign2']).reindex(up_1.columns).\
                        transform(lambda x: x / sum(abs(x)))
                    # 假如day1的因子收益率很高，触发减仓信号，说明day2的实际收益率很高，day2当天收盘的时候减仓，即day3的持有仓位减少
                    # 而day3的持有仓位保存在day2（收盘调整后拿到第二天收盘的持仓权重），所以触发信号后第二天的权重调整

                    weight2 = weight2*(position)
                    weight = weight2.iloc[-factor_period:,].copy().apply(lambda x: x / sum(abs(x)),axis=1)

                    # 方法 3:
                    # t = pd.Series([0, 1, 2, 1, 0, 1, 0, 2, -1, 0, -1, -2])
                    # segments = (np.sign(t.replace(0, method='ffill')).diff().fillna(0) != 0).cumsum()
                    # target = t.abs().groupby(segments).cummax() * np.sign(t.replace(0, method='ffill'))

                weight.columns =  ['factor_quantile_'+ i for i in weight.columns]

                return weight

            # 每个股票的权重
            def get_stock_weight(df, method_stock_weight='equal'):
                """
                weight: 每个因子买卖的权重
                group_df: 每个股票的分组
                return: 每个股票的权重,大于0,sum(abs(w)) = 1 days * stocks
                """
                # 默认方法：按前一日收益率来分配当日的权重，因为是收盘买卖
                stock_weight = df['return_norm'].unstack(level='qscode').apply(lambda x: x / x.sum(), axis=1)
                # 每个股票的权重
                # 等权重，忽略NA
                if method_stock_weight == 'equal':
                    stock_weight = stock_weight.apply(lambda x: (1-pd.isna(x))/ x.count(), axis=1)
                elif method_stock_weight == 'return':
                    # 惩罚下跌过多的股票，权重变小，同时惩罚涨得过高比如接近涨停的股票，权重变小
                    # stock_weight = df['return_norm'].unstack(level='qscode')\
                    #     .apply(lambda x: np.exp(5*(x)-2*((0.0093>x-0.09) &(x-0.09 >0))-3*((x-0.1)>0)),axis=1)\
                    #         .apply(lambda x: x.abs() / x.abs().sum(), axis=1)
                    stock_weight = df['return'].unstack(level='qscode')\
                        .apply(lambda x: np.exp(7*(x)-2*((0.0093>x-0.09) &(x-0.09 >0))-5*((x-0.103)>0)-2*((-0.0699>x) &(x>-0.0996))),axis=1)\
                            .apply(lambda x: x.abs() / x.abs().sum(), axis=1)

                stock_weight = stock_weight.fillna(0)
                return stock_weight

            # 合并成每个组合中股票的权重
            def get_weight_stock_final(factor_df,stock_weight,factor_weight,method_factor='long',quantile=5):
                #数据预处理
                df = factor_df.copy()
                #当method == 'long_only'时，将尾部那组替换为0，否则应该替换为-1以卖空
                df[df.filter(regex='factor_quantile').columns] = df.filter(regex='factor_quantile').replace(0, -1 * (method_factor != 'long_only')) 
                df[df.filter(regex='factor_quantile').columns] = df.filter(regex='factor_quantile').replace(range(1, quantile-1), 0)
                #当method == 'short_only'时，将头部那组替换为0，否则应该替换为1以买入
                df[df.filter(regex='factor_quantile').columns] = df.filter(regex='factor_quantile').replace(quantile-1, 1 * (method_factor != 'short_only'))

                weight_s = stock_weight.stack()
                weight_s.name = 'weight'
                df = pd.concat([df.reindex(weight_s.index),weight_s],axis=1).dropna(how = 'all')

                # 每一天每只股票每个因子的权重 = 每一天每只股票的权重 点乘 每一天每个股票每个因子的买卖情况
                # 不应该在这里对不可交易的股票进行处理，应该在后面进行处理
                for i in df.filter(regex='factor_quantile').columns:
                    df[i] = df['weight'] * df[i]

                # 每一天每一只股票的权重
                weight_stock_day = stock_weight.copy()

                #如果涨停，则以1指示；如果跌停，则以-1指示；其余的用0指示(这里存在nan，可能会造成干扰)
                is_limit = df['return'].unstack(level='qscode').apply(lambda x: ((x.abs() > 0.0993) & (x.abs() < 0.1004)) | (x.abs() > 0.1991), axis=1) * np.sign(df['return'].unstack(level='qscode'))
                
                if method_factor == 'long_short':
                    for date in stock_weight.index:
                        # day1 的数据
                        f = factor_weight.loc[date]
                        d = df.loc[df.index.get_level_values(0)==date].filter(regex='factor_quantile')
                        # 保留所有weight，按因子加权求和
                        dsf = d.mul(np.sign(f),axis=1).fillna(0).apply(lambda x: x/x.abs().sum(), axis=0).fillna(0)
                        # 按因子加权求和
                        weight_stock_day.loc[date]= np.dot(dsf,f.abs().T)
                        # day1的数据不用如下处理
                        if date != stock_weight.index[0]:
                            delta_stock_weight = weight_stock_day.loc[date] - weight_stock_day.iloc[weight_stock_day.index.get_loc(date)-1]
                            # 向量对应元素相乘，如果大于0，则说明该操作不可执行
                            is_tradable = (delta_stock_weight * is_limit.loc[date]) <= 0 #这里的判断条件必须包含=0
                            isnot_tradable = (delta_stock_weight * is_limit.loc[date]) > 0 #通过分别记录可交易与不可交易的bool向量来排除nan的干扰
                            # 把不可交易的股票的增量权重设为0
                            delta_stock_weight[isnot_tradable] = 0
                            weight_stock_day.loc[date] = weight_stock_day.iloc[weight_stock_day.index.get_loc(date)-1] + delta_stock_weight
                            # 记录可交易股票的权重和及不可交易股票的权重和，这里的bool会排除掉nan的值
                            tradable_sum = weight_stock_day.loc[date][is_tradable].sum() 
                            untradable_sum = weight_stock_day.loc[date][isnot_tradable].sum()
                            ##再对可交易的股票权重进行“归一化”
                            weight_stock_day.loc[date][is_tradable] = weight_stock_day.loc[date][is_tradable] * (1-untradable_sum) / tradable_sum
                else: # method == 'long_only' or 'short_only','long'
                    for date in stock_weight.index:
                        f = factor_weight.loc[date]
                        d = df.loc[df.index.get_level_values(0)==date].filter(regex='factor_quantile')
                        # 乘以正负号判断是否保留，保留大于0的权重并在同一个因子内归一化
                        dsf = d.mul(np.sign(f),axis=1).apply(lambda x: x*(x>0), axis=0).fillna(0).apply(lambda x: x/x.sum(), axis=0).fillna(0)
                        # 按因子加权求和
                        weight_stock_day.loc[date]= np.dot(dsf,f.abs().T)
                        # day1的数据不用如下处理
                        if date != stock_weight.index[0]:
                            delta_stock_weight = weight_stock_day.loc[date] - weight_stock_day.iloc[weight_stock_day.index.get_loc(date)-1]
                            # 向量对应元素相乘，如果大于0，则说明该操作不可执行
                            is_tradable = (delta_stock_weight * is_limit.loc[date]) <= 0 #这里的判断条件必须包含=0
                            isnot_tradable = (delta_stock_weight * is_limit.loc[date]) > 0 #通过分别记录可交易与不可交易的bool向量来排除nan的干扰
                            # 把不可交易的股票的增量权重设为0
                            delta_stock_weight[isnot_tradable] = 0
                            weight_stock_day.loc[date] = weight_stock_day.iloc[weight_stock_day.index.get_loc(date)-1] + delta_stock_weight
                            # 记录可交易股票的权重和及不可交易股票的权重和，这里的bool会排除掉nan的值
                            tradable_sum = weight_stock_day.loc[date][is_tradable].sum() 
                            untradable_sum = weight_stock_day.loc[date][isnot_tradable].sum()
                            ##再对可交易的股票权重进行“归一化”
                            weight_stock_day.loc[date][is_tradable] = weight_stock_day.loc[date][is_tradable] * (1-untradable_sum) / tradable_sum

                return weight_stock_day
            
            def plot_cumprod(factors_r):
                df_cumprod = (factors_r+1).cumprod()
                plt.figure(figsize=(20, 6))
                # 遍历每个列并绘制折线图
                for column in df_cumprod.columns:
                    plt.plot(df_cumprod.index, df_cumprod[column], label=column)
                plt.xlabel('日期')  # 设置 x 轴标签
                plt.ylabel('累积乘积')  # 设置 y 轴标签
                plt.title('折线图')  # 设置图表标题
                plt.legend()  # 显示图例
                plt.xticks(rotation=45)  # 旋转 x 轴刻度标签，使其更易读

            # 回测函数
            def backtest(df, stock_weight_tol, period = 5, amount_num = 5, fee_rate = 0.003):
                col = ['Position' + str(i) for i in range(1, amount_num +1)] # 不同的股票帐号
                portfolio_r = pd.DataFrame(0, columns =col, index = stock_weight_tol.index) 

                weight_shift = pd.Series(np.zeros(len(stock_weight_tol)))
                weight_shift.index = stock_weight_tol.index
                    #######################################################
                #每天按顺序更新其中一个组合，并记录每个组合每天的成分股和该组合当天的收益率（目前求的是等权加总）
                #后续引入股票权重时，直接在merge_data中前20列乘以相应的股票权重
                for i in range(len(col)):
                    position_index = ([0]* i)+([j // period * period + i for j in range(0, len(stock_weight_tol))])
                    position_index = position_index[:len(stock_weight_tol)]
                    w = stock_weight_tol.iloc[position_index, :] # days*stocks
                    w.iloc[:i, :] = 0
                    # calculate turnover rate
                    w_change = w.diff().abs().sum(axis=1).fillna(0)
                    w_change.index = stock_weight_tol.index
                    weight_shift = weight_shift.add(w_change, level = 'date')
                    # 用下一天的收益率计算当天的持仓收益率，因为是收盘后才能知道当天买入的持仓
                    # 实际29号收盘买入，30号的持仓收益率保存到了29号
                    return_ = df.loc[(stock_weight_tol.index, slice(None)), :]['forward_return_raw'].unstack() # days*stocks
                    return_ = return_.fillna(0)
                    dfnew=pd.concat([w,return_],keys=['weight','return'],axis=0)
                    # 串成序列
                    res =pd.Series(np.diag(np.dot(dfnew.loc['weight'],dfnew.loc['return'].T)),index=stock_weight_tol.index)
                    # 把res添加到portfolio_r中，按照index对齐
                    portfolio_r.iloc[:,i] = portfolio_r.iloc[:,i].add(res,fill_value=0)
                    # 计算换手率
                    # turnover = abs(w - w.shift(1)).sum(axis = 1).sum() / 2

                portfolio_r['total_return'] = portfolio_r.mean(axis = 1)
                portfolio_r['turnover'] = weight_shift / amount_num
                portfolio_r['total_return'] = portfolio_r['total_return'] - portfolio_r['turnover'] * fee_rate
                portfolio_r['compound_return'] = (portfolio_r.total_return+1).cumprod() - 1
                portfolio_r['benchmark'] = df.loc[(portfolio_r.index, slice(None)), :]['forward_return_raw'].groupby(level='date').mean()
                return portfolio_r

            self.progress(50)

            factordata_df['forward_return_norm'] = factordata_df.groupby(level='qscode')['return_norm'].shift(-1)
            factordata_df['forward_return_raw'] = factordata_df.groupby(level='qscode')['return'].shift(-1)
            factordata_df = factordata_df.dropna() #做了修改调整

            # 所有因子的特征名称,去掉return,return_norm和forward_return_raw列
            all_features = factordata_df.columns.drop(['return','return_norm','forward_return_raw','forward_return_norm'])
            factors_r_all, factordata_df, fac_mom_l, fac_mom_s, fac_mom_ternary = factor_return(factordata_df,factor_name=all_features,\
                factor_quantile=factor_quantile,method_return_cal=method_return_cal, \
                    backdays=backdays,return_period = return_period)

            fac_mom_63 = (factors_r_all + 1).rolling(63).apply(lambda x: x.prod())
            self.log.record(f'所有的因子名称为：{all_features}')
        
            self.progress(60)

            datelist = factordata_df.index.get_level_values(0).unique()
            start_index = 0
            end_index = backdays+ factor_period+ 1
            # 用于存储每个时间段的因子权重
            weight_factor = pd.DataFrame()
            # 用于存储每个时间段的最终股票权重
            weight_stock_final = pd.DataFrame()
            warnings.simplefilter(action='ignore', category=FutureWarning)
            factor_df = factordata_df.loc[(datelist[start_index]<factordata_df.index.get_level_values(0)) & (factordata_df.index.get_level_values(0)<=datelist[end_index])]
            fac_mom_l2 = fac_mom_l.loc[(datelist[start_index]<fac_mom_l.index.get_level_values(0)) & (fac_mom_l.index.get_level_values(0)<=datelist[end_index])]
            fac_mom_s2 = fac_mom_s.loc[(datelist[start_index]<fac_mom_s.index.get_level_values(0)) & (fac_mom_s.index.get_level_values(0)<=datelist[end_index])]
            fac_mom_632 = fac_mom_63.loc[(datelist[start_index]<fac_mom_63.index.get_level_values(0)) & (fac_mom_63.index.get_level_values(0)<=datelist[end_index])]
            fac_r = factors_r_all.loc[(datelist[start_index]<factors_r_all.index.get_level_values(0)) & (factors_r_all.index.get_level_values(0)<=datelist[end_index])]

            for i in range(end_index, len(datelist), factor_period):

                if if_xgboost:
                    # 用xgboost来筛选特征
                    top_features, coef_df = feature_collect(factors_r_all,fac_mom_l2, fac_mom_s2, factor_name=top_20_features, nums_fin=nums_fin, start_date = datelist[end_index],sample_weight=sample_weight, if_indicators=if_indicators)
                else:
                    # 用回归系数来筛选特征
                    top_features, coef_df = feature_collect(factors_r_all,fac_mom_l2, fac_mom_s2, factor_name=all_features, nums_fin=nums_fin, start_date = datelist[end_index],sample_weight=sample_weight,if_indicators=if_indicators)

                # 名称包含top_features的变量
                top_features_qt = ['factor_quantile_'+factor for factor in list(top_features)]

                # 往后取这些特征的数据
                end_index = min(end_index+ factor_period, len(datelist)-1)
                start_index = end_index - backdays - factor_period - 1
                
                factor_df = factordata_df.loc[(datelist[start_index]<factordata_df.index.get_level_values(0)) & (factordata_df.index.get_level_values(0)<=datelist[end_index])]
                fac_mom_l2 =fac_mom_l.loc[(datelist[start_index]<fac_mom_l.index.get_level_values(0)) & (fac_mom_l.index.get_level_values(0)<=datelist[end_index])]
                fac_mom_s2 = fac_mom_s.loc[(datelist[start_index]<fac_mom_s.index.get_level_values(0)) & (fac_mom_s.index.get_level_values(0)<=datelist[end_index])]
                fac_mom_632 = fac_mom_63.loc[(datelist[start_index]<fac_mom_63.index.get_level_values(0)) & (fac_mom_63.index.get_level_values(0)<=datelist[end_index])]
                fac_mom_ternary2 = fac_mom_ternary.loc[(datelist[start_index]<fac_mom_ternary.index.get_level_values(0)) & (fac_mom_ternary.index.get_level_values(0)<=datelist[end_index])]
                fac_r = factors_r_all.loc[(datelist[start_index]<factors_r_all.index.get_level_values(0)) & (factors_r_all.index.get_level_values(0)<=datelist[end_index])]

                # 需要的分组数据
                factor_df2 = factor_df[[*top_features_qt ,'return_norm','return','forward_return_norm','forward_return_raw']]

                # 都是当日收盘时的权重
                factor_weight = get_weight(fac_r,fac_mom_l2[[*top_features]],fac_mom_s2[[*top_features]],fac_mom_632[[*top_features]],fac_mom_ternary2[[*top_features]],coef_df,top_features,method_factor_weight = method_factor_weight,method_factor = method_factor).loc\
                    [datelist[end_index-factor_period+1]:datelist[end_index]]
                stock_weight = get_stock_weight(factor_df2, method_stock_weight=method_stock_weight).loc\
                    [datelist[end_index-factor_period+1]:datelist[end_index]]
                stock_weight_tol = get_weight_stock_final(factor_df2,stock_weight,factor_weight,method_factor=method_factor,quantile=factor_quantile)
                stock_weight_tol = stock_weight_tol.dropna(how='all')
                stock_weight_tol = stock_weight_tol.fillna(0)

                # 每个时间段的因子权重
                weight_factor = weight_factor.append(factor_weight).dropna(how='all')

                # 每个时间段的最终股票权重
                weight_stock_final = weight_stock_final.append(stock_weight_tol)

            # weight_factor = weight_factor.fillna(0)
            weight_stock_final = weight_stock_final.fillna(0)

            # weight_factor去掉重复的index
            weight_factor = weight_factor[~weight_factor.index.duplicated(keep='first')]

            # weight_stock_final去掉重复的index
            weight_stock_final = weight_stock_final[~weight_stock_final.index.duplicated(keep='first')]

            # 恢复警告
            warnings.resetwarnings()

            # 保存数据
            weight_factor.to_csv(os.path.join(self.data_dir, 'weight_factor.csv'), encoding = 'gbk')
            weight_stock_final.to_csv(os.path.join(self.data_dir, 'weight_stock_final.csv'), encoding = 'gbk')

            # 绘制因子权重
            weight_factor.loc[datetime.date(2022,4,30):,].plot(figsize=(20,10))

            self.progress(70)
            
            portfolio_r = backtest(factordata_df, weight_stock_final, period = period, amount_num = amount_num, fee_rate = fee_rate)

            # 绘制换手率图
            fig = plt.figure(figsize=(16, 9))
            ax1 = fig.add_subplot(111)
            ax1.plot(portfolio_r['turnover'], color='red', label='turnover')
            ax1.legend(loc='upper left')

            # 画图
            portfolio_r.index = pd.to_datetime(portfolio_r.index)

            # 回测的开始和结束时间
            end_date = weight_stock_final.dropna().index[-1]

            # 获取系统时间，只需要日期、小时和分钟
            now = datetime.datetime.now()
            now = now.strftime('%m%d%H%M')

            # 回测区间内的数据
            bt_data = portfolio_r.loc[start_date:end_date, :]

            path = self.path
            para = 'period'+str(period)+'_groupnums:'+str(factor_quantile)+'_method:'+method_factor+'_'+method_factor_weight+'_backdays:'+str(backdays)

            # 把total_return和benchmark用qs.reports.html画图
            qs.reports.html(bt_data['total_return'],bt_data['benchmark'], output=path+'Res_2_'+now+'.html', title='Backtest Result_'+para)

            stock_weight_tol.to_csv(os.path.join(self.data_dir, now+'.csv'), encoding = 'gbk')
            # portfolio_r.to_csv('F:\data\portfolio_r'+now+'.csv',encoding='gbk')

            self.progress(100)
            self.chrono.stop('End')
            # Output report to server
            self.end(dict(
                RunTime=dict(
                    start=utils.strdatetime(exe_start_time),
                    stop=utils.now_str(),
                    total=self.chrono.total(),
                    chrono=self.chrono.recall(),
                ),
            ))

        except Exception as e:
            from quantvale import error
            self.log.record(error.get_traceback())
            self.end_with_exception()