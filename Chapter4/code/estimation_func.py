# 機械学習モデル作成_6week(全代入 評価)

import time
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,  mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy import signal
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def linear_model(feature,target,storage_week):
    '''
    storage_week : 何週間貯蔵まで加えるか
    返り値は実測値と予測値が混ざったdataframe
    ※標準化は既に行った特徴量を使うべし
    '''
    feature = feature.iloc[:20*(storage_week+1),:]
    target = target.iloc[:20*(storage_week+1),:]
    
    model = LinearRegression()
    model.fit(feature,target)
    model_coef = model.coef_
    y_pred = pd.DataFrame(model.predict(feature))
    y_pred.index = target.index
    y_act_pred = pd.concat([target,y_pred],axis=1)
    y_act_pred.columns = ['Actual', 'Predict']
    
    return y_act_pred



def linearregression_6(target):
    '''
    重回帰分析
    全サンプル代入バージョン
    実測値vs予測値の散布図を返す
    '''
    
#####各週間(6week)ごとにプロットの色を変えて描写する
    y_pred_0 = []
    y_real_0 = []
    y_pred_1 = []
    y_real_1 = []
    y_pred_2 = []
    y_real_2 = []
    y_pred_3 = []
    y_real_3 = []
    y_pred_4 = []
    y_real_4 = []
    y_pred_5 = []
    y_real_5 = []
    y_pred_6 = []
    y_real_6 = []

    for i in range(target.shape[0]):
        if target.index[i][0] == '0':
            y_pred_0.append(target['Predict'][i])
            y_real_0.append(target['Actual'][i])
        elif target.index[i][0] == '1':
            y_pred_1.append(target['Predict'][i])
            y_real_1.append(target['Actual'][i])
        elif target.index[i][0] == '2':
            y_pred_2.append(target['Predict'][i])
            y_real_2.append(target['Actual'][i])
        elif target.index[i][0] == '3':
            y_pred_3.append(target['Predict'][i])
            y_real_3.append(target['Actual'][i])
        elif target.index[i][0] == '4':
            y_pred_4.append(target['Predict'][i])
            y_real_4.append(target['Actual'][i])
        elif target.index[i][0] == '5':
            y_pred_5.append(target['Predict'][i])
            y_real_5.append(target['Actual'][i])
        else:
            y_pred_6.append(target['Predict'][i])
            y_real_6.append(target['Actual'][i])
            
    for i in range(7):
        y_real_temp = [y_real_0,y_real_1,y_real_2,y_real_3,y_real_4,y_real_5,y_real_6][i]
        y_pred_temp = [y_pred_0,y_pred_1,y_pred_2,y_pred_3,y_pred_4,y_pred_5,y_pred_6][i]
        print(f'RMSE_{i}week : {np.sqrt(mean_squared_error(y_real_temp,y_pred_temp))}')
            
#####別々に分けるの終了
    y_real = target['Actual']
    y_pred = target['Predict']
    fig = plt.figure(figsize=(10,10))
    ax = plt.subplot(1,1,1)
    ax.text(0.01,0.85,f'R: {np.corrcoef(y_real, y_pred)[1,0]:.3f}\nRMSE : {np.sqrt(mean_squared_error(y_real,y_pred)):.3f}',
            transform=ax.transAxes,size=28)
    ax.text(0.6,0.05,f'n: {target.shape[0]}',transform=ax.transAxes,size=20)
    #####6週間分を色を変えてプロット
    
    plt.scatter(y_real_0,y_pred_0,marker="D",linewidths=4,c='r',label='0week')
    plt.scatter(y_real_1,y_pred_1,marker="D",linewidths=4,c='orange',label='1week')
    plt.scatter(y_real_2,y_pred_2,marker="D",linewidths=4,c='yellow',label='2week')
    plt.scatter(y_real_3,y_pred_3,marker="D",linewidths=4,c='greenyellow',label='3week')
    plt.scatter(y_real_4,y_pred_4,marker="D",linewidths=4,c='turquoise',label='4week')
    plt.scatter(y_real_5,y_pred_5,marker="D",linewidths=4,c='dodgerblue',label='5week')
    plt.scatter(y_real_6,y_pred_6,marker="D",linewidths=4,c='m',label='6week')
    
    plt.plot([0,28],[0,28],c='k',label='y=x')
    plt.plot([0,28],[3.3,31.3],linestyle='dashed',c='gray')
    plt.plot([0,28],[-3.3,24.7],linestyle='dashed',c='gray')
    plt.ylabel('Predicted Firmness (N)',fontsize=24)
    plt.xlabel('Measured  Firmness (N)',fontsize=24)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.xlim(18,28)
    plt.ylim(18,28)
    plt.grid(True)
    plt.legend(loc ='lower right',fontsize=14,frameon=False)
    plt.close()

#     print('_______________')
#     for i in range(feature_scale_delay_df.shape[1]):
#         print(f'{feature_scale_delay_df.columns[i]} : {model.coef_[i]:.3f}')
#     print(f'Intercept : {model.intercept_}')
#     print('_______________')
    
    return fig


def linearregression_7(target):
    '''
    重回帰分析
    全サンプル代入バージョン
    実測値vs予測値の散布図を返す
    '''
    
#####各週間(6week)ごとにプロットの色を変えて描写する
    y_pred_0 = []
    y_real_0 = []
    y_pred_1 = []
    y_real_1 = []
    y_pred_2 = []
    y_real_2 = []
    y_pred_3 = []
    y_real_3 = []
    y_pred_4 = []
    y_real_4 = []
    y_pred_5 = []
    y_real_5 = []
    y_pred_6 = []
    y_real_6 = []
    y_pred_7 = []
    y_real_7 = []

    for i in range(target.shape[0]):
        if target.index[i][0] == '0':
            y_pred_0.append(target['Predict'][i])
            y_real_0.append(target['Actual'][i])
        elif target.index[i][0] == '1':
            y_pred_1.append(target['Predict'][i])
            y_real_1.append(target['Actual'][i])
        elif target.index[i][0] == '2':
            y_pred_2.append(target['Predict'][i])
            y_real_2.append(target['Actual'][i])
        elif target.index[i][0] == '3':
            y_pred_3.append(target['Predict'][i])
            y_real_3.append(target['Actual'][i])
        elif target.index[i][0] == '4':
            y_pred_4.append(target['Predict'][i])
            y_real_4.append(target['Actual'][i])
        elif target.index[i][0] == '5':
            y_pred_5.append(target['Predict'][i])
            y_real_5.append(target['Actual'][i])
        elif target.index[i][0] == '6':
            y_pred_6.append(target['Predict'][i])
            y_real_6.append(target['Actual'][i])
        else:
            y_pred_7.append(target['Predict'][i])
            y_real_7.append(target['Actual'][i])
            
    for i in range(8):
        y_real_temp = [y_real_0,y_real_1,y_real_2,y_real_3,y_real_4,y_real_5,y_real_6,y_real_7][i]
        y_pred_temp = [y_pred_0,y_pred_1,y_pred_2,y_pred_3,y_pred_4,y_pred_5,y_pred_6,y_pred_7][i]
        print(f'RMSE_{i}week : {np.sqrt(mean_squared_error(y_real_temp,y_pred_temp))}')
        
        print(f'Mean of Measured  Firmness {i}week : {np.mean(y_real_temp)}')
        print()
        print(f'Mean of Predicted Firmness {i}week : {np.mean(y_pred_temp)}')
        print()
        print(f'RMSE_{i}week : {np.sqrt(mean_squared_error(y_real_temp,y_pred_temp))}')
        print('----------')
            
#####別々に分けるの終了
    y_real = target['Actual']
    y_pred = target['Predict']
    fig = plt.figure(figsize=(10,10))
    ax = plt.subplot(1,1,1)
    ax.text(0.01,0.85,f'R: {np.corrcoef(y_real, y_pred)[1,0]:.3f}\nRMSE : {np.sqrt(mean_squared_error(y_real,y_pred)):.3f}',
            transform=ax.transAxes,size=28)
    ax.text(0.6,0.05,f'n: {target.shape[0]}',transform=ax.transAxes,size=20)
    #####6週間分を色を変えてプロット
    
    plt.scatter(y_real_0,y_pred_0,marker="D",linewidths=4,c='r',label='0week')
    plt.scatter(y_real_1,y_pred_1,marker="D",linewidths=4,c='orange',label='1week')
    plt.scatter(y_real_2,y_pred_2,marker="D",linewidths=4,c='yellow',label='2week')
    plt.scatter(y_real_3,y_pred_3,marker="D",linewidths=4,c='greenyellow',label='3week')
    plt.scatter(y_real_4,y_pred_4,marker="D",linewidths=4,c='turquoise',label='4week')
    plt.scatter(y_real_5,y_pred_5,marker="D",linewidths=4,c='dodgerblue',label='5week')
    plt.scatter(y_real_6,y_pred_6,marker="D",linewidths=4,c='m',label='6week')
    plt.scatter(y_real_7,y_pred_7,marker="D",linewidths=4,c='pink',label='7week')
    
    plt.plot([0,28],[0,28],c='k',label='y=x')
    plt.plot([0,28],[3.3,31.3],linestyle='dashed',c='gray')
    plt.plot([0,28],[-3.3,24.7],linestyle='dashed',c='gray')
    plt.ylabel('Predicted Firmness (N)',fontsize=24)
    plt.xlabel('Measured  Firmness (N)',fontsize=24)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.xlim(18,28)
    plt.ylim(18,28)
    plt.grid(True)
    plt.legend(loc ='lower right',fontsize=14,frameon=False)
    plt.close()

#     print('_______________')
#     for i in range(feature_scale_delay_df.shape[1]):
#         print(f'{feature_scale_delay_df.columns[i]} : {model.coef_[i]:.3f}')
#     print(f'Intercept : {model.intercept_}')
#     print('_______________')
    
    return fig


def holdout(feature,target):
    '''
    feature: 特徴量
    sample : '_1_'などで定義
    random : 0はある番号を一括、1はrandomで選択、2は自分でサンプル指定
    '''
    # 全indexを抽出
    index_temp = []
    for period in range(7):
        for num in [str(num+1) for num in range(5)]:
            index_temp.append(f'{period}_{num}')
            
    # 配列に変換
    temp = np.array(index_temp)
    index_temp = temp.reshape(7,5)
    
    # test用のindexを抽出
    feature = feature.iloc[:140,:]
    test_index  = ['0_1_1','0_1_2','0_1_3','0_1_4','1_1_1','1_1_2','1_1_3','1_1_4','2_4_1','2_4_2','2_4_3','2_4_4','3_1_1','3_1_2','3_1_3','3_1_4','4_3_1','4_3_2','4_3_3','4_3_4','5_2_1','5_2_2','5_2_3','5_2_4','6_5_1','6_5_2','6_5_3','6_5_4']

    train_index = feature.index.drop(test_index)
    
    # 各貯蔵期間ごとに5つのリンゴを使うが、4つをtrain, 1つをtestのhold-outを行う
    X_train = feature.loc[train_index,:]
    X_test = feature.loc[test_index,:]

    target = target.iloc[:140,:]
    y_train = target.loc[train_index,:]
    y_test = target.loc[test_index,:]

    linear = LinearRegression()
    model = linear.fit(X_train,y_train)
    y_pred = pd.DataFrame(model.predict(X_test),index=y_test.index,columns=y_test.columns)
    y_pred_real = pd.DataFrame(model.predict(X_train),index=y_train.index,columns=y_train.columns)

    rscore_list_kf = []
    rmse_list_kf = []
    rscore_list_kf.append(pd.concat([y_test,y_pred],axis=1).corr().iloc[0,1])
    rmse_list_kf.append(np.sqrt(mean_squared_error(y_test,y_pred)))
    
    # 描写
    fig = plt.figure(figsize=(12,12))
    ax1 = fig.add_subplot(1, 1, 1)
    y_real = y_test.values.flatten()
    y_pred = y_pred.values.flatten()
    
    ax1.scatter(y_train, model.predict(X_train).ravel(),c='b',label='training',s=100)
    ax1.scatter(y_test,   y_pred,c='r',label='test',s=100)
    ax1.text(0.01,0.85,f'R_test: {np.corrcoef(y_real, y_pred)[1,0]:.3f}\nRMSE_test : {np.sqrt(mean_squared_error(y_real,y_pred)):.3f}',
             transform=ax1.transAxes,size=28)

    ax1.plot([0,28],[0,28],c='k',label='y=x')
    ax1.plot([0,28],[3.3,31.3],linestyle='dashed',c='gray')
    ax1.plot([0,28],[-3.3,24.7],linestyle='dashed',c='gray')
    ax1.set_ylabel('Predicted Firmness (N)',fontsize=32)
    ax1.set_xlabel('Measured  Firmness (N)',fontsize=32)
    ax1.set_xticklabels([18,20,22,24,26,28],fontsize=28)
    ax1.set_yticklabels([18,20,22,24,26,28],fontsize=28)
    ax1.set_xlim([18, 28])
    ax1.set_ylim([18, 28])
    ax1.grid(True)
    ax1.legend(loc ='lower right',fontsize=28,frameon=False)
    
#     if feature.columns[-1][5:] == '15mm':
#         ax1.set_title('6-week-storage-15mm model', fontsize=28)
#     else:
#         ax1.set_title('6-week-storage model', fontsize=28)

    if feature.columns[-1][5:] == '15mm':
        ax1.text(0.01,0.95,'a)',  transform=ax1.transAxes, fontsize=28)
    else:
        ax1.text(0.01,0.95,'b)',  transform=ax1.transAxes, fontsize=28)
    
    # fig.savefig(f'result_compare/fig_{feature.columns[-1][5:]}.jpg')
    
    return fig