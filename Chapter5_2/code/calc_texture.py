def compare_texture_line(column):
    '''
    特徴量を指定して可視化
    
    特徴量は以下
    'firmness_skin'
    'firmness_flesh'
    'AUC_skin'
    'AUC_flesh'
    'CI_flesh'
    'firmness_flesh_only'
    'AUC_flesh_only'
    'CI_flesh_only'
    
    '''
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # 特徴量を抽出
    df = pd.read_csv('./../data/texture_all.csv',header=0,index_col=0)

    # 品種のカラムを作成
    func = lambda x:x[:-7]
    temp = pd.DataFrame(df.index, 
                        columns = ['cultivar'],
                        index = df.index).applymap(func)

    df['cultivar'] = temp

    # 貯蔵期間のカラムを作成
    func = lambda x:x[-6:][0:2]
    temp = pd.DataFrame(df.index, 
                        columns = ['period'],
                        index = df.index).applymap(func)


    df['period'] = temp

    # ヒストグラムを作成

    # 各品種・貯蔵期間ごとの平均値・標準偏差をdfに格納
    df_mean = df.groupby(['cultivar','period']).mean()
    df_std = df.groupby(['cultivar','period']).std()

    # 品種ごとのCIの平均値・標準偏差を算出
    df_mean_fuji = df_mean.loc[[('Fuji')],:][column].values
    df_mean_kougyoku = df_mean.loc[[('Kougyoku')],:][column].values
    df_mean_ourin = df_mean.loc[[('Ourin')],:][column].values

    df_std_fuji = df_std.loc[[('Fuji')],:][column].values
    df_std_kougyoku = df_std.loc[[('Kougyoku')],:][column].values
    df_std_ourin = df_std.loc[[('Ourin')],:][column].values

    labels = [i+1 for i in range(int(df['period'][-1]))]
    # 5週目はないので、特殊処理
    labels.pop(4)

    # マージンを設定
    margin = 0.2  #0 <margin< 1
    totoal_width = 1 - margin

    # 棒グラフをプロット
    fig = plt.figure(figsize=(9,6))
    for i, label in enumerate(labels):
        
        plt.errorbar(label, df_mean_fuji[i], yerr =df_std_fuji[i], fmt='o', capsize=5, markersize=10, ecolor='r', markeredgecolor = "r", color='r')
        plt.errorbar(label, df_mean_kougyoku[i], yerr =df_std_kougyoku[i], capsize=5, fmt='o', markersize=10, ecolor='g', markeredgecolor = "g", color='g')
        plt.errorbar(label, df_mean_ourin[i], yerr =df_std_ourin[i], capsize=5, fmt='o', markersize=10, ecolor='b', markeredgecolor = "b", color='b')
    
    plt.plot(labels, df_mean_fuji,c='r',label='ふじ')
    plt.plot(labels, df_mean_kougyoku,c='g',label='紅玉')
    plt.plot(labels, df_mean_ourin,c='b',label='王林')

    # ラベルの設定
    plt.xticks(labels,fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Storage period (week)',fontsize=14)
    
    if column in ['firmness_skin', 'firmness_flesh','firmness_flesh_only']:
        plt.ylabel('Firmness [N]',fontsize=14)
        if column in ['firmness_flesh','firmness_flesh_only']:
            plt.ylim(0,10)
        else:
            plt.ylim(0,30)
        
    elif column in ['AUC_skin', 'AUC_flesh','AUC_flesh_only']:
        plt.ylabel('AUC [J]',fontsize=14)
        if column in ['AUC_flesh','AUC_flesh_only']:
            plt.ylim(0,450)
        else:
            plt.ylim(0,250)
    
    else:
        plt.ylabel('Crispness Index (CI) [N]',fontsize=14)
        plt.ylim(0,35)

    plt.legend(loc='upper center',
               fontsize=14,
               frameon=False)
    plt.show()
    plt.close()
    
    return fig