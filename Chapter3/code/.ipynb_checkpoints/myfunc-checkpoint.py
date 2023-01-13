import os
import time
import warnings
warnings.simplefilter('ignore')
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize_scalar
from scipy import signal
from sklearn.metrics import r2_score, mean_squared_error
import cv2


def delete_distance(distance, apple_df):
    '''
    最大30 mmまでのdataframeを作成
    '''
    
    # 1 mm未満を削除する処理
    apple_df = apple_df.iloc[distance.index,:]
    distance.index = [i for i in range(distance.shape[0])]
    apple_df.index = [i for i in range(apple_df.shape[0])]

    # 最大30 mmとした処理
    distance_30mm = distance[distance <= 30.1]
    apple_df_30mm = apple_df.iloc[distance_30mm.index,:]
    
    return distance_30mm, apple_df_30mm


def AVE_SD_CV(apple):
    '''
    同一サンプルを複数回計測した際のプロファイルの平均と標準偏差，変動係数を返す関数
    引数は副スカイ計測した際のプロファイル
    '''
    
    ave = apple.mean(axis=1)
    sd = apple.std(axis=1)
    CV = 100*(sd/ave)
    
    return ave,sd, CV


def plot_profile(distance, data1, data2):
    '''
    平均値±標準偏差のプロット，　変動係数のプロット
    '''
    
    fig, ax = plt.subplots(2, 2, figsize=(12,12))

    AVE,SD, CV = AVE_SD_CV(data1)
    
    ax[0,0].plot(distance, AVE,c='k',alpha=1)
    ax[0,0].fill_between(distance,(AVE-SD),(AVE+SD),color='k',alpha=0.3)
    ax[0,0].set_ylabel('Intensity',fontsize=16)
    ax[0,0].set_xlabel('Distance from incident point [mm]',fontsize=16)
    ax[0,0].set_xlim(0,30)
    ax[0,0].set_ylim(2,8)
    ax[0,0].text(1,7.7,'(a)',fontsize=20)



    ax[1,0].plot(distance, CV,c='k',alpha=0.5)
    ax[1,0].hlines(5,xmin=0,xmax=30,linestyles='dashed',color='k')
    ax[1,0].set_ylim(0,8)
    ax[1,0].set_xlim(0,30)
    ax[1,0].set_ylabel('Coefficient of Variance (%)',fontsize=16)
    ax[1,0].set_xlabel('Distance from incident point [mm]',fontsize=16)
    ax[1,0].text(1,7.6,'(b)',fontsize=20)




    AVE,SD, CV = AVE_SD_CV(data2)
    
    ax[0,1].plot(distance, AVE,c='k',alpha=1)
    ax[0,1].fill_between(distance,(AVE-SD),(AVE+SD),color='k',alpha=0.3)
    ax[0,1].set_ylabel('Intensity',fontsize=16)
    ax[0,1].set_xlabel('Distance from incident point [mm]',fontsize=16)
    ax[0,1].set_xlim(0,30)
    ax[0,1].set_ylim(2,8)
    ax[0,1].text(1,7.7,'(c)',fontsize=20)

    ax[1,1].plot(distance, CV,c='k',alpha=0.5)
    ax[1,1].hlines(5,xmin=0,xmax=30,linestyles='dashed',color='k')
    ax[1,1].set_ylim(0,8)
    ax[1,1].set_xlim(0,30)
    ax[1,1].set_ylabel('Coefficient of Variance (%)',fontsize=16)
    ax[1,1].set_xlabel('Distance from incident point [mm]',fontsize=16)
    ax[1,1].text(1,7.6,'(d)',fontsize=20)

    plt.tight_layout()
    
    return fig
    
    