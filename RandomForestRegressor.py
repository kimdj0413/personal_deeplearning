###########################################
#####   랜덤 포레스트 중고차 가격 예측  #####
###########################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_url = "https://media.githubusercontent.com/media/musthave-ML10/data_source/main/car.csv"
data = pd.read_csv(file_url)

# print(data.head())
# print(data.describe())
# print(data.info())
##  선형 모델에서는 아웃라이어 처리가 필요하지만 트리 모델에서는 필요하지 않다.

##############################
### engine 변수 전처리 하기 ###
##############################

data[['engine', 'engine_unit']]= data['engine'].str.split(expand=True) #공백 기준 문자 분할 및 별도의 변수로 저장
# print(data['engine']) #데이터 타입을 확인해보면 object형
data['engine'] = data['engine'].astype('float32') #자료형 변환
# print(data['engine_unit'].unique()) #engine_unit의 고윳값 확인
data.drop('engine_unit', axis=1, inplace=True) #'CC'와 NaN만 저장되어 있는 칼럼은 제거

##################################
### max_power 변수 전처리 하기  ###
##################################

data[['max_power','max_power_unit']] = data['max_power'].str.split(expand=True)
# print(data['max_power'])
# data['max_power'] = data['max_power'].astype('float32') #string to float error
# print(data[data['max_power']=='bhp']) #error가 난 부분 확인하기

def isFloat(value): #에러 처리 함수 정의하기
    try:
        num = float(value) #값을 숫자로 반환
        return num
    except ValueError:
        return np.NaN #에러면 난수 처리
    
data['max_power']=data['max_power'].apply(isFloat)
# print(data['max_power_unit'].unique())
data.drop('max_power_unit', axis=1, inplace=True) #칼럼제거

##############################
### mileage 변수 전처리하기 ###
##############################

data[['mileage','mileage_unit']]=data['mileage'].str.split(expand=True)
data['mileage']=data['mileage'].astype('float32')
# print(data['mileage_unit'].unique()) #출력해보면 단위가 전부다름
# print(data['fuel']) #fuel 칼럼과 비교
##  리터당 가격으로 mileage를 통일

def mile(x):
    if x['fuel']=='Petrol':
        return x['mileage'] / 80.43
    elif x['fuel']=='Diesel':
        return x['mileage'] / 73.56
    elif x['fuel']=='LPG':
        return x['mileage'] / 40.85
    else:
        return x['mileage'] / 44.23
    
data['mileage'] = data.apply(mile, axis=1) #axxis=1을 하는 이유는 그렇지 않으면 컬럼명을 컬럼에서 찾지않고 인덱스에서 찾음.
print(data['mileage'])
data.drop('mileage_unit', axis=1, inplace=True)