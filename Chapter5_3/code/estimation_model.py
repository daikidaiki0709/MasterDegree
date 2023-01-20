import time
# import warnings
# warnings.simplefilter('ignore')
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


def arrange_feature(METHOD):
    '''
    METHOD: 0は従来法，1はレーザー散乱法
    '''
    if METHOD == 0:
        condition = 'data_laser_1condition'
    else:
        condition = 'data_laser'
    
    ###### 633 nm ######
    wavelen = 633

    # 補正後のprofile特徴量
    feature_First_df = pd.read_csv(
        f'./../data/data_FirstStorage/{condition}/feature/feature_CB_HDR/feature_shake/{wavelen}nm/feature_all.csv',
        index_col=0
    )
    
    feature_Second_df = pd.read_csv(
        f'./../data/data_SecondStorage/{condition}/feature/feature_CB_HDR/feature_shake/{wavelen}nm/feature_all.csv',
        index_col=0
    )
    
    feature_df_633 = pd.concat([feature_First_df, feature_Second_df],axis=0)
    feature_df_633.columns = [f'{col}_{wavelen}' for col in feature_df_633.columns]

    ###### 850 nm ######
    wavelen = 850

    # 補正後のprofile特徴量
    feature_First_df = pd.read_csv(
        f'./../data/data_FirstStorage/{condition}/feature/feature_CB_HDR/feature_shake/{wavelen}nm/feature_all.csv',
        index_col=0
    )
    
    feature_Second_df = pd.read_csv(
        f'./../data/data_SecondStorage/{condition}/feature/feature_CB_HDR/feature_shake/{wavelen}nm/feature_all.csv',
        index_col=0
    )
    
    feature_df_850 = pd.concat([feature_First_df, feature_Second_df],axis=0)
    feature_df_850.columns = [f'{col}_{wavelen}' for col in feature_df_850.columns]

    ##### 目的変数 #####
    target_First_df = pd.read_csv(f'./../data/data_FirstStorage/data_mealiness_shake/Mealiness.csv',index_col=0)
    target_First_df = target_First_df.iloc[:feature_First_df.shape[0],0].astype(float) #初期状態が文字列だったので，小数に変換
    target_Second_df = pd.read_csv(f'./../data/data_SecondStorage/data_mealiness_shake/Mealiness.csv',index_col=0)
    target_Second_df = target_Second_df.iloc[:feature_Second_df.shape[0],0].astype(float) #初期状態が文字列だったので，小数に変換

    target_df = pd.concat([target_First_df, target_Second_df],axis=0)
    target_df.index = feature_df_633.index


    ###### 合成と削除 ######
    feature_df = pd.concat([feature_df_633, feature_df_850],axis=1)

    # 実験に失敗したサンプル2(Firststorage, 04_03_2)を削除
    # 実験に失敗したサンプル2(Secondstorage, 04_08_4)を削除
    target_df = target_df.dropna(axis=0)
    feature_df = feature_df.reset_index()
    target_df = target_df.reset_index()

    target_df = target_df.drop(target_df.index[84],axis=0)
    feature_df = feature_df.drop([feature_df.index[84],feature_df.index[195]],axis=0)

    feature_df = feature_df.drop(['index'],axis=1)
    target_df.index = target_df['index']
    target_df = target_df.drop(['index'],axis=1)
    feature_df.index = target_df.index
    
    return feature_df, target_df



def scale(feature_df, target_df):
    '''
    feature: 特徴量
    '''
    X_test = feature_df[feature_df.index.str.contains('9|10')]
    X_train = feature_df.drop(X_test.index, axis=0)
    y_test = target_df[target_df.index.str.contains('9|10')]
    y_train = target_df.drop(y_test.index, axis=0)

    # 標準化
    scaler = StandardScaler()

    X_train_scale = scaler.fit_transform(X_train)
    X_train_scale = pd.DataFrame(X_train_scale,
                                 columns = X_train.columns,
                                 index = X_train.index)


    X_test_scale = scaler.transform(X_test)
    X_test_scale = pd.DataFrame(X_test_scale,
                                 columns = X_test.columns,
                                 index = X_test.index)
    
    return X_train_scale, X_test_scale, y_train, y_test


def select_feature(X_train_scale, X_test_scale, y_train, THRESH=0.1):
    '''
    相関係数が閾値以下の変数を削除
    '''
    temp = abs(pd.concat([X_train_scale,y_train],axis=1).corr())['Mealiness (%)'][abs(pd.concat([X_train_scale, y_train],axis=1).corr())['Mealiness (%)'].sort_values(ascending=False)>THRESH].index
    temp = temp.drop(['Mealiness (%)'])


    X_train_scale = X_train_scale.loc[:,temp]
    X_test_scale = X_test_scale.loc[:,temp]
    
    return X_train_scale, X_test_scale

def divide_mealiness(y, THRESH=15):
    '''
    THRESH: 閾値以上を粉質化（1），それ以下を非粉質化（0）とする
    '''
    tmp = y.copy()
    
    tmp.loc[tmp['Mealiness (%)']<=THRESH,'Mealiness (%)'] = int(0)
    tmp.loc[tmp['Mealiness (%)']>THRESH,'Mealiness (%)'] = int(1)
    
    return tmp


def optimize_hyperparameter(param_pls,param_svm,param_ann, X_train_scale, y_train):
    '''
    dict: 各モデルにおけるハイパーパラメータの設定
    '''
    # CVの設定
    SEED = 100
    kf_inner = KFold(n_splits=10, shuffle=True, random_state=SEED)    

    ##### GridSearch #####
    # PLSのGridSearch 
    CV_pls = GridSearchCV(
        PLSRegression(),
        param_pls,
        scoring='neg_mean_squared_error',
        cv=kf_inner,
        n_jobs = -1
    )
    # SVMのGridSearch
    CV_svm = GridSearchCV(
        SVC(),
        param_svm,
        scoring='f1',
        cv=kf_inner,
        n_jobs=-1
    )
    # ANNのGridSearch
    CV_ann = GridSearchCV(
        MLPClassifier(early_stopping=True,random_state=SEED),
        param_ann,
        scoring='f1',
        cv=kf_inner,
        n_jobs=-1
    )


    # GridSearchの結果表示
    results_pls = CV_pls.fit(X_train_scale, y_train)
    results_svm = CV_svm.fit(X_train_scale,y_train.values.ravel())
    results_ann = CV_ann.fit(X_train_scale.values,y_train.values.ravel())
    
    return results_pls, results_svm, results_ann


def calc_metrics(feature_df,target_df,THRESH_RANGE):
    
    SEED = 100
    N = 3
    
    acc_list = []
    f1_list = []
    for MEALINESS in THRESH_RANGE:
        
        ##### 標準化
        X_train_scale, X_test_scale, y_train, y_test = scale(feature_df=feature_df, target_df=target_df)


        ##### 変数選択
        X_train_scale, X_test_scale = select_feature(X_train_scale=X_train_scale, X_test_scale=X_test_scale, y_train=y_train)


        ##### 目的変数のバイナリー化
        y_train = divide_mealiness(y_train,THRESH=MEALINESS).astype('int')
        y_test = divide_mealiness(y_test,THRESH=MEALINESS).astype('int')

        ##### 目的変数のオーバーサンプリング
        smote = SMOTE(sampling_strategy='minority', random_state=SEED,k_neighbors=N,n_jobs=-1)
        X_train_scale, y_train = smote.fit_resample(X_train_scale, y_train)
        y_train.astype('int')


        ##### ハイパーパラメータの最適化
        param_pls = {'n_components':[i for i in range(1,100)]}
        param_svm = {
            # 'C':[i/10 for i in range(1,201)],
            # 'kernel':['rbf'],
            # 'gamma':[1e-1,1e-2,5e-2,1e-3,5e-3,1e-4,5e-4,1e-5,5e-5],
            'kernel':['poly'],
            'degree':[i for i in range(1,13)]
        }
        param_ann = {'hidden_layer_sizes':[(55,),(15,),(6,)],#(88,),(44,), 
                     'activation':['relu'],
                     'solver':['adam'],
                     'alpha':[1000,100,10,1,1e-2,1e-3,1e-4],
                     'batch_size':[25,50,75,100]
                    }

        results_pls, results_svm, results_ann = optimize_hyperparameter(
            param_pls = param_pls, 
            param_svm = param_svm,
            param_ann = param_ann,
            X_train_scale = X_train_scale,
            y_train = y_train
        )

        ##### 最適なハイパーパラメータでモデル構築

        # PLS
        pls = PLSRegression(n_components=results_pls.best_params_['n_components'])
        model_PLS = pls.fit(X_train_scale,y_train)

        # SVM
        svm = SVC(
        #     kernel='rbf',
        #     C=results_svm.best_params_['C'], 
            # gamma=results_svm.best_params_['gamma'],
            kernel='poly',
            degree=results_svm.best_params_['degree'],
        )
        model_SVM = svm.fit(X_train_scale, y_train)

        # ANN
        ann = MLPClassifier(hidden_layer_sizes=results_ann.best_params_['hidden_layer_sizes'],
                           activation=results_ann.best_params_['activation'],
                           solver=results_ann.best_params_['solver'],
                           alpha=results_ann.best_params_['alpha'],
                           batch_size=results_ann.best_params_['batch_size'],
                           random_state=SEED
                          )

        model_ANN = ann.fit(X_train_scale, y_train)

        ##### モデルの予測 #####

        # trainの予測
        y_pred_train_ANN = model_ANN.predict(X_train_scale).flatten()
        y_pred_train_SVM = model_SVM.predict(X_train_scale).flatten()
        y_pred_train_PLS = (model_PLS.predict(X_train_scale).flatten() > 0.5).astype('uint8')

        # testの予測
        y_pred_SVM = model_SVM.predict(X_test_scale).flatten()
        y_pred_ANN = model_ANN.predict(X_test_scale).flatten()
        y_pred_PLS = (model_PLS.predict(X_test_scale).flatten() > 0.5).astype('uint8')


        tmp = pd.DataFrame(
            [
                [
            # accuracy_score(y_train,y_pred_train_PLS),
            accuracy_score(y_test,y_pred_PLS),
            # accuracy_score(y_train,y_pred_train_SVM),
            accuracy_score(y_test,y_pred_SVM),
            # accuracy_score(y_train,y_pred_train_ANN),
            accuracy_score(y_test,y_pred_ANN),
                ],

                [
            # f1_score(y_train,y_pred_train_PLS),
            f1_score(y_test,y_pred_PLS),
            # f1_score(y_train,y_pred_train_SVM),
            f1_score(y_test,y_pred_SVM),
            # f1_score(y_train,y_pred_train_ANN),
            f1_score(y_test,y_pred_ANN),
                ]
            ]
        )

        tmp.index = ['Accuracy','F1 score']
        tmp.columns = ['PLS_test','SVM_test','ANN_test']
        acc = tmp.iloc[0,:].max()
        f1 = tmp.iloc[1,:].max()
        
        acc_list.append(acc)
        f1_list.append(f1)
        
    return acc_list, f1_list
    
    

    
    # tmp
