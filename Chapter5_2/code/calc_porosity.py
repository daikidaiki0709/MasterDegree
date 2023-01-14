import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import pydicom

# 連続で読み込み、distance-porosity分布データを得る

def calc_porosity(cultivar):
    '''
    cultivar: 'Fuji','Kougyoku','Ourin'
    '''
    cultivar = cultivar
    df = pd.DataFrame()
    
    for i_stor in ['01','02','03','04','06','07','08','09','10','11','12']:
        porosity_list = []
        
        for i_num in range(1,4):

            path = f'./../data/Binarydata/{cultivar}/{cultivar}_{i_stor}_{i_num}_350'
            
            # 画像全てを読み込む
            files = os.listdir(path)
            files.sort()
            
            temp = []
            
            for i_file in files:
                
                #　dicomを読み込み
                ds = pydicom.dcmread(f'{path}/{i_file}',force=True)
                ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
                
                 # uint16に変更
                img = ds.pixel_array

                # 空隙率を算出
                pore_area = len(img[img == 0])
                porosity = 100 * (pore_area / len(img.flatten()))
                temp.append(porosity)
            
            porosity = np.mean(temp)
            porosity_list.append(porosity)
            
        df = pd.concat([df,pd.DataFrame(porosity_list,columns=[i_stor])],axis=1)
    
        
    return df