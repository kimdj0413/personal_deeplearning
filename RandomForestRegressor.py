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
data.drop('mileage_unit', axis=1, inplace=True)

###############################
### torque 변수 처리하기    ###
##############################

data['torque'] = data['torque'].str.upper()

def toruque_unit(x):
    if 'NM' in str(x):
        return 'Nm'
    elif 'KGM' in str(x):
        return 'kgm'
    
data['torque_unit'] = data['torque'].apply(toruque_unit) #null 값도 존재하니 확인
# print(data[data['torque_unit'].isna()]['torque'].unique()) #torque_unit이 결측치인 라인의 torque 변수 고윳값 확인
data['torque_unit'].fillna('Nm', inplace=True) #결측치를 Nm으로 채움

#     #enumerate 함수 활용
# string_example = '12,7@ 2,700(KGM@ RPM)'
# for i, j in enumerate(string_example):
#     print(i,'번째 텍스트: ',j)
#     #반환값을 받는 변수가 두개(i,j)여서 인덱스 값과 그에 해당하는 값 리턴

def split_num(x):
    x = str(x) #여러 자료형이 있을 수 있으니 확실히 문자열로 형변환
    for i,j in enumerate(x):
        if j not in '0123456789.':
            cut = i
            break
    return x[:cut]

data['torque'] = data['torque'].apply(split_num)
data['torque'] = data['torque'].replace('',np.NaN) #''를 결측치로 대체(형변환을 위해)
data['torque'] = data['torque'].astype('float64') #형변환

def torque_trans(x): #단위 통일 함수
    if x['torque_unit'] == 'kgm':
        return x['torque']*9.8066
    else:
        return x['torque']
    
data['torque'] = data.apply(torque_trans, axis=1)
data.drop('torque_unit', axis=1, inplace=True)

###########################
### name 변수 처리하기  ###
##########################

data['name'] = data['name'].str.split(expand=True)[0] #띄어쓰기 기준으로 맨 처음 값만 저장
# print(data['name'].unique()) #Lnad Rover가 Land만 나옴
data['name'] = data['name'].replace('Land','Land Rover')

######################################
### 결측치 처리 및 더미 변수 변환   ###
######################################

# print(len(data)) #길이 : 8128
# print(data.isna().mean()) #결측치 비율이 2% 밖에 안되므로 행삭제.
data.dropna(inplace=True)
data = pd.get_dummies(data, columns=['name', 'fuel', 'seller_type', 'transmission', 'owner'], drop_first=True) #빈 칼럼을 더미 변수로 변환.
# print(len(data)) #길이 : 7906

#######################
### 모델링 및 평가  ###
######################

    #RMSE 평가
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.drop('selling_price', axis=1), data['selling_price'], test_size=0.2, random_state=100)
from sklearn.ensemble import RandomForestRegressor

# model = RandomForestRegressor(random_state=100) #랜덤 포레스트는 매번 다르게 나무를 만드므로 random state를 지정하면 동일하게 가능
# model.fit(X_train, y_train)
# train_pred = model.predict(X_train)
# test_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error #RMSE를 사용한 평가
# print(mean_squared_error(y_train, train_pred)**0.5, mean_squared_error(y_test, test_pred)**0.5)

    #K-폴드 교차검증
##  데이터를 K개로 쪼개 그 중 하나씩 선택해서 시험셋으로 사용 후 K번 반복해 평가.
from sklearn.model_selection import KFold
# print(data) #인덱스 값과 줄의 길이가 맞지 않음
data.reset_index(drop=True, inplace=True) #인덱스를 다시 맞춤, drop=True를 안하면 기존 인덱스 값을 새로운 칼럼으로 가져옴.
# print(data)

kf = KFold(n_splits=5) #5개 분할 KFold 객체 생성
X = data.drop('selling_price', axis=1)
y = data['selling_price']
# for i,j in kf.split(X): #5개로 나뉘어진 분할 확인(훈렷셋, 시험셋으로 나뉘므로 두개인 인덱스 필요)
#     print(i,j)

# train_rmse_total = []
# test_rmse_total = []

# for train_index, test_index in kf.split(X):
#     X_train, X_test = X.loc[train_index], X.loc[test_index]
#     y_train, y_test = y[train_index], y[test_index]

#     model = RandomForestRegressor(random_state=100)
#     model.fit(X_train,y_train)
#     train_pred = model.predict(X_train)
#     test_pred = model.predict(X_test)

#     train_rmse = mean_squared_error(y_train, train_pred)**0.5
#     test_rmse = mean_squared_error(y_test, test_pred)**0.5
#     train_rmse_total.append(train_rmse)
#     test_rmse_total.append(test_rmse)

# print("train_rmse: ", sum(train_rmse_total)/5, " test_rmse: ",sum(test_rmse_total)/5)

##  랜덤 포레스트는 오버피팅을 막는데 효율적.
##  전체 트리를 사용하는게 아니라 일부만 사용해서 단순 예측력을 떨어짐.

##############################
### 하이퍼 파라미터 튜닝    ###
##############################

##  n_estimators : 결정트리 개수(기본값 100개)
##  max_depth : 트리의 최대 깊이 제한
##  min_samples_split : 노드를 나눌것인지 말지.
##  min_samples_leaf : 분리된 노드의 데이터에 최소 몇 개의 데이터가 있어야 할지 결정.
##  n_jobs : 병렬 처리에 사용되는 CPU 코어 수. -1은 모든 코어 사용.
train_rmse_total = []
test_rmse_total = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = RandomForestRegressor(n_estimators=300,max_depth=50,min_samples_split=5,min_samples_leaf=1,n_jobs=-1,random_state=100)
    model.fit(X_train,y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_rmse = mean_squared_error(y_train, train_pred)**0.5
    test_rmse = mean_squared_error(y_test, test_pred)**0.5
    train_rmse_total.append(train_rmse)
    test_rmse_total.append(test_rmse)

print("train_rmse: ", sum(train_rmse_total)/5, " test_rmse: ",sum(test_rmse_total)/5)
##  하이퍼파라미터 튜닝으로 오버피팅이 줄어들었으니 그 전보다 좋은 모델.