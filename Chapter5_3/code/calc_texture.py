import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def calc_texture(DIRECT):
    
    '''
    引数：テクスチャーアナライザー（G101 or 2023年以降はG208？）で得られるcsvが格納されているディレクトリ
    返り値：テクスチャー特徴量
    
    ##### テクスチャー特徴量を作成 #####
    以下の順で格納していく
        1. Fs：硬さ
        2. Ff：7mmまでの硬さ
        3. D：Fsまでの距離
        4. Ws：Dまでの面積
        5. Wf：7mmまでのAUC
        6. Grad：0からFsまでの傾き
        7. FLC：x=Dと、Ff付近で得られる直線の交点における荷重
        8. CI：果肉プロファイルの総和
        
    なお上記の指標は以下の論文を参考に作成
    - Crispness Index(CI)と最大荷重によるカキ果肉部の物性評価, 2017
    - RELATIONSHIP BETWEEN APPLE SENSORY ATTRIBUTES AND INSTRUMENTAL PARAMETERS OF TEXTURE, 2006
    '''
    
    #####　データの読み込み＆データの整形　#####
    # csvファイルの読み込み
    df = pd.read_csv(DIRECT,encoding='Shift-JIS')
    
    # このままだとcolumnが気持ち悪いので、名無しカラムをNoneに変更する
    replace_columns = [column for column in df.columns if 'Unnamed' in column] # 気持ち悪いカラム名を取得
    none_columns = ['None' for i in range(len(replace_columns))] #上記のカラムと同数のNoneカラムを用意
    
    # カラム名を変更
    df = df.rename(columns=dict(zip(replace_columns,none_columns)))
    
    
    #####　データからテクスチャー特徴量を作成　#####
    all_texture_list = []# 全サンプルの指標を格納する空リストを用意
    
    sample_num = 10
    # csvから1サンプルのプロファイルを取得できる
    for i in range(sample_num):
        for j in range(2):
            
            one_texture_list = []# 各サンプルごとの指標を格納

            sample_name = f'{i+1} _ {j+1}'

            # sample名に対応した列番号を取得
            sample_column_number = df.columns.get_loc(sample_name)

            # sample名に対応した、「時間、試験力、ストローク」を取得する
            df_temp = df.iloc[:,sample_column_number:sample_column_number+3]

            # 0番目の行をカラム名にしたいので、それに対応する処理
            df_temp.columns = df_temp.iloc[0,:] # カラム名変更
            df_temp = df_temp.drop([0],axis=0) # 0番目の行がカラム名と重複するので削除

            # 必要なデータのみ取得
            if df_temp.columns[0] == '時間':
                time_df = df_temp['時間'][1:]
            else:
                time_df = df_temp['Time'][1:]
            
            distance = df_temp['ストローク'][1:]
            load = df_temp['試験力'][1:]

            # 実はdf内はstr型なので、floatになおす
            time_df = time_df.astype('float')
            distance = distance.astype('float')
            load = load.astype('float')

            ##### テクスチャー特徴量を作成 #####
            # 以下の順で格納していく
            # 1. Fs：硬さ
            # 2. Ff：7mmまでの硬さ
            # 3. D：Fsまでの距離
            # 4. Ws：Dまでの面積
            # 5. Wf：7mmまでのAUC
            # 6. Grad：0からFsまでの傾き
            # 7. FLC：x=Dと、Ff付近で得られる直線の交点における荷重
            # 8. CI：果肉プロファイルの総和

            # 1. Fs
            Fs = np.max(load) # 念の為果皮範囲に絞った
            one_texture_list.append(Fs)

            # 2. Ff
            Ff = load[distance>= 7.00].iloc[0]
            one_texture_list.append(Ff)

            # 3. D
            D = distance[load == Fs].to_numpy()
            # 配列ではなく、数値として保存
            D = D[0]
            one_texture_list.append(D)

            # 4. Ws：Dまでの面積
            distance_D = distance[distance <= D].values
            load_D = load[distance <= D].values
            # ここから計算
            import scipy.integrate as integrate
            Ws = integrate.trapz(load_D,distance_D)
            one_texture_list.append(Ws)

            # 5. Wf：7mmまでのAUC
            distance_7mm = distance[distance<=7].values
            load_7mm = load[distance<=7].values
            # AUCの取得
            import scipy.integrate as integrate
            Wf = integrate.trapz(load_7mm, distance_7mm)
            one_texture_list.append(Wf)

            # 6. Grad：0からFsまでの傾き
            # distance=0のloadを取得
            distance_0 = distance[distance >=0].values[0]
            load_0 = load[distance == distance_0].values[0]
            # FsのdistanceはDなので計算なし
            distance_Fs = D
            load_Fs = Fs
            # Gradの算出
            Grad = (load_Fs - load_0) / (distance_Fs - distance_0)
            one_texture_list.append(Grad)

            # 7. x=Dと、Ff付近で得られる直線の交点における荷重
            # Ff近辺ののデータを取得
            distance_Ffnear = distance[(distance >= 6.5)&(distance<= 7.5)]
            load_Ffnear = load[(distance >= 6.5)&(distance<= 7.5)]
            # 近辺データから誤差が最小となる直線を作成
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            result = model.fit(distance_Ffnear.values.reshape(-1,1), load_Ffnear.values.reshape(-1,1))
            # x=Dの予測値を算出
            FLC = result.intercept_ + result.coef_ * D
            FLC = FLC[0][0]
            one_texture_list.append(FLC)

            # 8. CI：果肉プロファイルの総和
            # 0.05秒間隔の時間データを取得
            time_df_5intervals = time_df[::5]
            time_df_CI = time_df_5intervals[(time_df_5intervals>=6) & (time_df_5intervals<=10)]
            # CIの算出
            CI = 0
            for t in time_df_CI:
                F_t2 = load[time_df == round(t + 0.05,3)].values[0] # 丸め誤差が発生するので、小数点第二位に強制変更
                F_t1 = load[time_df == t].values[0]
                F_t0 = load[time_df == round(t - 0.05,3)].values[0]

                CI += abs((F_t2 + F_t0) - 2*F_t1)

            one_texture_list.append(CI)

            # 全部を一つのリストにまとめる
            all_texture_list.append(one_texture_list)

    # 全データを成型
    texture_df = pd.DataFrame(all_texture_list)
    texture_df.columns = ['Fs[N]','Ff[N]','D[mm]','Ws[mJ]','Wf[mJ]','Grad[kN/m]','FLC[N]','CI[N]']
    texture_df.index = [f'{i+1} _ {j+1}' for i in range(sample_num) for j in range(2)]
    
    return texture_df

#==============================================================================================================================#

def combine_csv(DIRECT):
    '''
    同じディレクトリ内にあるテクスチャー特徴量のcsvを結合する
    '''
    
    allfile = os.listdir(DIRECT)
    csv_list = [csvfile for csvfile in allfile if 'Feature' in csvfile]
    csv_list = [csvfile for csvfile in csv_list if 'all' not in csvfile]
    csv_list.sort()

    feature_all = pd.DataFrame()
    for csv_path in csv_list:
        df_temp_2 = pd.read_csv(f'{DIRECT}/{csv_path}', header=0, index_col=0)
        # indexの番号に+1して、index改名 (レーザーに合わせるため)
        # df_temp.index = temp_list
        feature_all = pd.concat([feature_all, df_temp_2],axis=0)

    return feature_all
    