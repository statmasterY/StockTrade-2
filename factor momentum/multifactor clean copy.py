#%%
import pandas as pd
import numpy as np
import datetime
from standardize import mad5_truncate
from datetime import date

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
from scipy.stats import pearsonr, linregress
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing as pp
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import joblib
from pytorch_tabnet.metrics import Metric
import torch
import torch.nn as nn

import statsmodels.api as sm

from Roll import roll_np

from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from matplotlib import pyplot as plt
import seaborn as sns


# 读取CSV文件
df2 = pd.read_csv('D:\杨钦\计算机语言与笔记\quant\牧鑫\ALL_factorValues_2021-2023.csv')

df2['date']=pd.to_datetime(df2['date']).dt.date
df2.set_index(['date', 'qscode'], inplace=True)

# 行业所在列
index_industry = df2.columns.get_loc('中证一级行业')

# 只要行业所在列，return列后面的列，并去掉vol_std_126这一列
df2 = df2.iloc[:, index_industry:].drop(columns=['vol_std_126'])

# return列所在的位置
index_return = df2.columns.get_loc('return')

df2 = df2.drop(df2.columns[1:index_return], axis=1)

# 处理缺失值
df2 = df2.groupby(level='qscode',sort=False,group_keys=False).fillna(method= 'ffill')
df2 = df2.dropna()

# 行业中性化
# 行业哑变量
df2 = pd.get_dummies(df2, columns=['中证一级行业'])
# 行业中性化,用行业哑变量的回归系数乘以行业哑变量的值，然后再用原来的值减去这个值

def industry_norm(df):
    # 准备数据
    returns = df["return"]  # 股票收益率数据
    industry_factors = df.iloc[:,df.columns.str.contains('^中证一级行业')]  # 行业因子数据

    # 拟合线性回归模型
    model = sm.OLS(returns, industry_factors)
    results = model.fit()

    r = returns - np.dot(industry_factors ,(results.pvalues <0.05)*results.params)
    
    # residuals = results.resid
    
    return r

# 市值中性化
def market_cap_norm(df):
    # 准备数据
    returns = df["return"]  # 股票收益率数据
    market_cap = df["lncap"]  # 市值数据

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
df2['return_norm'] = df2.groupby(level='date',sort=False,group_keys=False).apply(market_cap_norm)

# 筛选变量
df2 = df2.drop(columns=['log_return','lncap'])

# 删除包含行业哑变量的列,使用正则表达式，行业哑变量的列都是以中证一级行业开头的
df2 = df2.loc[:, ~df2.columns.str.contains('^中证一级行业')]

top_20_features = ['exp_wgt_momentum_1m', 'trix_15', 'bb_pos_20', 'williams_20', 'macd_histogram_12_26', 'macd_12_26', 'cci_20', 'wgt_momentum_1m', 'eom_20', 'vol_std_21', 'turnover_std_1m', 'vr_20', 'ma_20', 'bb_range_20', 'exp_wgt_momentum_3m', 'vol_std_63', 'cmo_20', 'qstick_20', 'pet_20', 'ep']

# 所有因子的特征名称,去掉return和return_norm列
all_features = df2.columns.drop(['return','return_norm'])

# 选取最优特征和目标变量
#df3 = df2[[*top_20_features,'return_norm','return']]
df3 = df2[[*all_features,'return_norm','return']]
df3['forward_return_norm'] = df3.groupby(level='qscode')['return_norm'].shift(-1)
df3 = df3.dropna()

#%%
####################

def factor_return(df, factor_name=top_20_features, factor_quantile=5, method = 'long_short', backdays = 252):
    factors_r = pd.DataFrame(columns=factor_name)
    for factor in factor_name:
        #df是因子值矩阵
        # 加入rank of return 避免重复值
        df['factor_quantile_'+factor] = df.groupby(level='date', sort = False, group_keys = False).\
            apply(lambda x: pd.qcut(0.9999*x[factor].rank()+0.0001*x['return_norm'].rank(), factor_quantile, labels=False, duplicates='drop'))
        df = df.dropna()
        # 1月1号构建的组合，利用1月2号的收益率来计算，保存为1月1号的因子收益率
        if method == 'long_short':
            factors_r[factor] = df.groupby('date', sort = False, group_keys = False).apply(lambda x: \
                x[x['factor_quantile_'+factor]==factor_quantile-1]['forward_return_norm'].mean()-x[x['factor_quantile_'+factor]==0]['forward_return_norm'].mean())
        elif method == 'long':
            factors_r[factor] = df.groupby('date', sort = False, group_keys = False).apply(lambda x: \
                x[x['factor_quantile_'+factor]==factor_quantile-1]['forward_return_norm'].mean())
    fac_mom = (factors_r + 1).rolling(backdays).apply(lambda x: x.prod())
    fac_mom_ternary = fac_mom.dropna()
    fac_mom_ternary[fac_mom_ternary <= 1] = -1
    fac_mom_ternary[fac_mom_ternary > 1] = 1
    return factors_r, df, fac_mom, fac_mom_ternary

# factors_r 是当日的因子收益率矩阵，df是因子值矩阵, fac_mom是因子累积收益率矩阵，fac_mom_ternary是三态因子动量bool矩阵
# factors_r_all, df3, fac_mom, fac_mom_ternary = factor_return(df3,factor_name=all_features,factor_quantile=10)

# def feature_collect(df, factor_name = top_20_features, nums = 10):
#     coefficients = []
#     for factor in factor_name:##后续改为factor_name
#         X = sm.add_constant(df[factor])
#         y = factors_r_all[factor]
#         cleaned_data = pd.concat([X, y], axis=1).dropna()
#         X = cleaned_data.iloc[:,:-1]
#         y = cleaned_data.iloc[:,-1]
#         model = sm.OLS(y, X).fit()
#         #提取回归系数
#         coefficients.append(model.params[factor])
#     coefficients = pd.DataFrame(coefficients, index=factor_name, columns=['coefficient'])
#     top_features = coefficients.sort_values(by='coefficient', ascending= False).index[:nums] #这里的10是参数

#     return top_features, coefficients

# top_features, coefficients = feature_collect(fac_mom, factor_name=all_features, nums=10)

top_features = ['wgt_momentum_3m',
 'oi',
 'np',
 'tte_ttm',
 'exp_wgt_momentum_3m',
 'turnover_1m',
 'abn_inv',
 'ocfr_ttm',
 'macd_histogram_12_26',
 'macd_12_26']


# 名称包含top_features的变量
df3 = df2[[*top_features,'return_norm','return']]
df3['forward_return_norm'] = df3.groupby(level='qscode')['return_norm'].shift(-1)
df3['forward_return_raw'] = df3.groupby(level='qscode')['return'].shift(-1)
df3 = df3.dropna()

factors_r, df3, fac_mom, fac_mom_ternary = factor_return(df3,factor_name=top_features,factor_quantile=10)

########################## 回测
#%%

# 每个因子买卖的权重

def get_weight(factors_r,method = 'equal_long',backdays= 252):

    # 过去backdays的累计收益率
    fac_mom = (factors_r + 1).rolling(backdays).apply(lambda x: x.prod())
    fac_mom_bool = fac_mom > 1
    fac_mom_bool = fac_mom_bool.replace(True, 1)
    fac_mom_bool = fac_mom_bool.replace(False, -1)

    # 每个因子买卖的权重
    if method == 'equal_long':
        # 只做多，做空的因子通过买因子值小的股票来代替，所以sum abs(weight) = 1，正负表示买顶还是买底
        # >0: 1/5  <0: -1/15 再归一化
        weight = fac_mom_bool.shift(1).apply(lambda x: (x>0)/sum(x>0)*(x >0) - (x<0)/sum(x<0)*(1-x>0), axis=1)\
            .apply(lambda x: x/sum(abs(x)), axis=1)
    elif method == 'equal_long_only':
        # 只做多，忽略做空的因子 
        # >0: 1/5, <0:0
        weight = fac_mom_bool.shift(1).apply(lambda x: (x>0)/sum(x>0)*(x >0) , axis=1)
    elif method == 'equal_short_only':
        # 只做空，忽略做多的因子
        # >0: 0, <0: -1/15
        weight = fac_mom_bool.shift(1).apply(lambda x: -1*(x<0)/sum(x<0)*(1-x>0), axis=1)
    elif method == 'equal_ls':
        # 可以做多和做空 sum abs(weight) = 1
        # >0: 1/5   <0: -1/15
        weight = fac_mom_bool.shift(1).apply(lambda x: (x>0)/sum(x>0)*(x >0) - (x<0)/sum(x<0)*(1-x>0), axis=1).apply(lambda x: x/sum(abs(x)), axis=1)
    elif method == 'return_long':
        weight = (fac_mom-1).shift(1).apply(lambda x: x/sum(abs(x)), axis=1)
    elif method == 'return_long_only':
        weight = (fac_mom-1).shift(1).apply(lambda x: x*(x>0), axis=1).apply(lambda x: x/sum(abs(x)), axis=1)
    elif method == 'return_short_only':
        weight = (fac_mom-1).shift(1).apply(lambda x: x*(x<0), axis=1).apply(lambda x: x/sum(abs(x)), axis=1)
    elif method == 'return_ls':
        weight = (fac_mom-1).shift(1).apply(lambda x: (x>0)/sum(x>0)*(x >0) - (x<0)/sum(x<0)*(1-x>0), axis=1)

    weight.columns =  ['factor_quantile_'+ i for i in weight.columns]

    return weight

# 每个股票的权重

def get_stock_weight(df, method='equal'):

    stock_weight = df['return_norm'].unstack(level='qscode').apply(lambda x: x / x.sum(), axis=1)

    # 每个股票的权重
    # 等权重，忽略NA
    if method == 'equal':
        stock_weight = stock_weight.apply(lambda x: (1-pd.isna(x))/ x.count(), axis=1)
    # 按一个月内的收益率分配权重，实质是波动性大的权重多一点
    elif method == 'return':
        stock_weight = stock_weight.apply(lambda x: x / x.sum(), axis=1)

    ############################## 缓和一点的方法

    stock_weight = stock_weight.fillna(0)

    return stock_weight

# 都是当日收盘时的权重
factor_weight = get_weight(factors_r,method = 'equal_long')
stock_weight = get_stock_weight(df3, method='equal')

#%%
# 把股票权重加入到df中
def get_weight_stock_final(df, stock_weight,factor_weight,method='long'):
    weight_s = stock_weight.stack()
    weight_s.name = 'weight'
    df = pd.concat([df,weight_s],axis=1).dropna(how = 'all')

    # 每一天每只股票每个因子的权重 = 每一天每只股票的权重 点乘 每一天每个股票每个因子的买卖情况
    for i in df.filter(regex='factor_quantile').columns:
        df[i] = df['weight'] * df[i]

    # 归一化 ，每一天，每一个因子，股票之间的权重要归一化

    # 每一天每一只股票的权重
    weight_stock_day = stock_weight.copy()
    
    if method == 'long':
        for date in stock_weight.index:
            # day1 的数据
            f = factor_weight.loc[date]
            d = df.loc[df.index.get_level_values(0)==date].filter(regex='factor_quantile')
            # 乘以正负号判断是否保留，保留大于0的权重并在同一个因子内归一化
            dsf = d.mul(np.sign(f),axis=1).apply(lambda x: x*(x>0), axis=0).fillna(0).apply(lambda x: x/x.sum(), axis=0)
            # 按因子加权求和
            weight_stock_day.loc[date]= np.dot(dsf,f.abs().T)

    return weight_stock_day

def backtest(df, stock_weight, factor_weight, factor_quantile =5 ,method = 'long_short'):
    #生成记录投资组合收益率的dataframe
    #回测需要注意不同日期有数据的股票并不完全相同，在引入权重时须尤其注意
    col = ['Position' + str(i) for i in range(1, 6)] # 不同的股票帐号
    portfolio_r = pd.DataFrame(0, columns =col, index = factor_weight.index) 

    #计算每一天的股票权重
    if method == 'long_short':
        df[df.filter(regex='factor_quantile').columns] = df.filter(regex='factor_quantile').replace(0, -1) # 卖底 0是最小的
        df[df.filter(regex='factor_quantile').columns] = df.filter(regex='factor_quantile').replace(range(1, 4), 0) # 不买卖
        df[df.filter(regex='factor_quantile').columns] = df.filter(regex='factor_quantile').replace(4, 1) # 买顶
        stock_weight_tol = get_weight_stock_final(df, stock_weight,factor_weight)
        stock_weight_tol = stock_weight_tol.dropna(how='all')
        stock_weight_tol = stock_weight_tol.fillna(0)
        # 归一化
        stock_weight_tol = stock_weight_tol.apply(lambda x: (x>0)/sum(x>0) - (x<0)/sum(x<0), axis=1)
    elif method == 'long':
        df[df.filter(regex='factor_quantile').columns] = df.filter(regex='factor_quantile').replace(0, -1) # 卖底 0是最小的
        df[df.filter(regex='factor_quantile').columns] = df.filter(regex='factor_quantile').replace(range(1, factor_quantile-1), 0) # 不买卖
        df[df.filter(regex='factor_quantile').columns] = df.filter(regex='factor_quantile').replace(factor_quantile-1, 1) # 买顶
        stock_weight_tol = get_weight_stock_final(df, stock_weight,factor_weight,method='long')
        #######################################################

    #每天按顺序更新其中一个组合，并记录每个组合每天的成分股和该组合当天的收益率（目前求的是等权加总）
    #后续引入股票权重时，直接在merge_data中前20列乘以相应的股票权重
    if method == 'long':
        for i in range(len(col)):
            position_index = ([0]* i)+([j // 5 * 5 + i for j in range(0, len(stock_weight_tol))])
            position_index = position_index[:len(stock_weight_tol)]
            w = stock_weight_tol.iloc[position_index, :] # days*stocks
            w.iloc[:i, :] = 0
            # 用下一天的收益率计算当天的持仓收益率，因为是收盘后才能知道当天买入的持仓
            # 实际29号收盘买入，30号的持仓收益率保存到了29号
            return_ = df.loc[(stock_weight_tol.index, slice(None)), :]['forward_return_raw'].unstack() # days*stocks
            return_ = return_.fillna(0)
            dfnew=pd.concat([w,return_],keys=['weight','return'],axis=0)
            # 串成序列
            res =pd.Series(np.diag(np.dot(dfnew.loc['weight'],dfnew.loc['return'].T)),index=stock_weight_tol.index)
            # 把res添加到portfolio_r中，按照index对齐
            portfolio_r.iloc[:,i] = portfolio_r.iloc[:,i].add(res,fill_value=0)
    else: #method == 'long_short' or 'long_only'
        for i in range(len(col)):
            position_index = ([0]* i)+([j // 5 * 5 + i for j in range(0, len(stock_weight_tol))])
            position_index = position_index[:len(stock_weight_tol)]
            w = stock_weight_tol.iloc[position_index, :] # days*stocks
            w.iloc[:i, :] = 0
            # 用下一天的收益率计算当天的持仓收益率，因为是收盘后才能知道当天买入的持仓
            # 实际29号收盘买入，30号的持仓收益率保存到了29号
            return_ = df.loc[(stock_weight_tol.index, slice(None)), :]['forward_return_raw'].unstack() # days*stocks
            return_ = return_.fillna(0)
            dfnew=pd.concat([w,return_],keys=['weight','return'],axis=0)
            # 串成序列
            res =pd.Series(np.diag(np.dot(dfnew.loc['weight'],dfnew.loc['return'].T)),index=stock_weight_tol.index)
            # 把res添加到portfolio_r中，按照index对齐
            portfolio_r.iloc[:,i] = portfolio_r.iloc[:,i].add(res,fill_value=0)

    portfolio_r['total_return'] = portfolio_r.mean(axis = 1)
    portfolio_r['compound_return'] = (portfolio_r.total_return+1).cumprod() - 1
    portfolio_r['benchmark'] = df.loc[(portfolio_r.index, slice(None)), :]['forward_return_raw'].groupby(level='date').mean()

     
    return stock_weight_tol,portfolio_r

df4 = df3.loc[:, [*df3.filter(regex='factor_quantile').columns,'forward_return_norm','return','forward_return_raw']].copy()

stock_weight_tol,portfolio_r = backtest(df4, stock_weight, factor_weight, factor_quantile=10, method = 'long')

#%% 
# 画图
import quantstats as qs

portfolio_r.index = pd.to_datetime(portfolio_r.index)

# 回测的开始和结束时间
start_date = factor_weight.dropna().index[0]
end_date = factor_weight.dropna().index[-1]

# 回测区间内的数据
bt_data = portfolio_r.loc[start_date:end_date, :]

qs.reports.html(bt_data.total_return, output='stats.html', title='Backtest Result')

# 超额收益
qs.reports.html(bt_data.total_return - bt_data.benchmark, output='extra return.html', title='Extra return Result')

# 把total_return和benchmark用qs.reports.html画图
qs.reports.html(bt_data[['total_return', 'benchmark']], output='stats2.html', title='Backtest Result')

# %%
