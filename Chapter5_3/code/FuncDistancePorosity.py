import numpy as np
import pandas as pd
import cv2
import os
import pydicom

# 連続で読み込み、distance-porosity分布データを得る

def calc_porosity(condition, storage_period):
    '''
    各画像におけるporosityを取得する（画像
    '''
    voxel = 10
    porosity_list = []
    for num in range(5):
        
        # path関連の操作
        path = f'./../data/data_{condition}/data_microstructure/voxel{voxel}um/{storage_period}_{num+1}'
        
        # 画像全てを読み込む
        CNT = len(os.listdir(path))
        porosity_list_temp = []
        for i in range(CNT):
            # ゼロパディング
            i = str(i).zfill(4)
            
            #　dicomを読み込み
            ds = pydicom.dcmread(f'{path}/{i}.dcm',force=True)
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            
            # uint16に変更
            img = ds.pixel_array
            img = img[100:400,100:400]
            img = img.astype('uint16')


            # 平滑化
            if voxel == 5:
                img_gauss = cv2.GaussianBlur(img,(15,15),3)
            else:
                img_gauss = cv2.GaussianBlur(img,(5,5),5)

            # 2値化
            thresh, img_binary = cv2.threshold(img_gauss, 0, img.max(), cv2.THRESH_OTSU)
            img_binary = np.where(img_binary>0,2**16,0)
            
            # 画像の保存
            # 元画像
            cv2.imwrite(f'./../data/data_{condition}/data_microstructure/voxel{voxel}um_png/{storage_period}_{num+1}/{i}.png',
                        img)
            # 2値画像
            cv2.imwrite(f'./../../data_{condition}/data_microstructure/voxel{voxel}um_binary/{storage_period}_{num+1}/{i}.png',
                                img_binary)
            
            # 空隙率を算出
            boid_area = len(img_binary[img_binary == 0])
            porosity = 100 * (boid_area / len(img_binary.flatten()))
            
            porosity_list_temp.append(porosity)
          
        # sampleごとに深度ごとのporosityを保存
        porosity_list.append(porosity_list_temp)
        
    results = pd.DataFrame(porosity_list).T
    distance_df = pd.DataFrame([voxel*i for i in range(CNT)], 
                               columns=['distance (μm)'],
                               index=results.index)
    
    results = pd.concat([distance_df, results],axis=1)
    
    return results