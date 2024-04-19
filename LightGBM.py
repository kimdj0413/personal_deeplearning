########################################
#####   LightGBM 이상거래 예측하기  #####
########################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_url = 'C:\\personal_coding\\personal_machineLearning\\fraud.csv'
data = pd.read_csv(file_url)

# print(data.info(show_counts=True)) #데이터가 많으면 Non-Null Count가 안나오므로 옵션
# print(round(data.describe(), 2))

##############################
### 전처리 : 데이터클리닝   ###
##############################

data.drop(['first','last','street','city','state','zip','trans_num','unix_time','job','merchant'],axis=1,inplace=True) #필요없는 데이터 삭제
data['trans_date_trans_time']=pd.to_datetime(data['trans_date_trans_time']) #Object->datetime64 날짜형으로 변환

##############################
### 전처리 : 피처엔지니어링 ###
##############################

    #결제금액
##  Z 점수(Z-Score, 표준점수) : 평균과 표준편자를 이용해 특정값이 정규분포 범위에서 어느 수준에 위치 하는지를 나타냄.
amt_info = data.groupby('cc_num').agg(['mean','std'])['amt'].reset_index() #cc_num별 amt 평균과 표준편차 계산
# print(amt_info.head())
data = data.merge(amt_info, on='cc_num', how='left') #데이터합치기
data['amt_z_score']=(data['amt']-data['mean'])/data['std'] #z-score 계산

data.drop(['mean','std'],axis=1,inplace=True) #남은 변수 제거
    #범주
category_info = data.groupby(['cc_num','category']).agg('mean','std')['amt'].reset_index() #cc_num과 category 기준으로 amt의 평균, 표준편차 계산
data=data.merge(category_info, ono=['cc_num','category'], how='left') #데이터 합치기
data['cat_z_score'] = (data['amt'] - data['mean'])/data['std'] #z-score 계산
data.drop(['mean','std'], axis=1, inplace=True) #변수제거

    #거리 계산
import geopy.distance #거리계산용 라이브러리

data['merch_coord'] = pd.Series(zip(data['merch_lat'], data['merch_long'])) #상점 위도, 경도 한 변수로 합치기
data['cust_coord'] = pd.Series(zip(data['lat'], data['long'])) #고객 주소 위도, 경도 한 변수로 합치기

data['distance'] = data.apply(lambda x: geopy.distance(x['merch_coord'], x['cust_coord']).km, axis=1) #거리계산

##  거리 z-score 계산
distance_info = data.groupby('cc_num').agg(['mean','std'])['distance'].reset_index()
data = data.merge(distance_info, on='cc_num', how='left')
data['distance_z_score']=(data['distance']- data['mean'])/data['std']
data.drop(['mean','std'],axis=1,inplace=True)

    #나이 구하기
data['age'] = 2024 - pd.to_datetime(data['dob']).dt.year #dt함수를 이용한 나이 계산

data.drop(['cc_num','lat','long','merch_long','dob','merch_coord','cust_coord'], axis=1, inplace=True) #계산하고 필요없는 변수들 제거

    #더미 변수 변환
data = pd.get_dummies(data, columns=['category', 'gender'], drop_first=True) #카테고리와 성별은 더미변수로 변환

data.set_index('trans_date_trans_time', inplace=True) #trans_date_trans_time 인덱스로 활용

##########################
### 모델링 및 평가하기  ###
##########################

##  train_test_split()을 사용하지 않고 특정 날짜 기준으로 훈렷셋과 시험셋 나눔
train = data[data.index < '2020-07-01']
test = data[data.index >= '2020-07-01']

print(len(test)/ len(data)) #시험셋 비율 확인

X_train = train.drop('is_fraud', axis=1)
X_test = test.drop('is_fraud', axis=1)
y_train = train['is_fraud']
y_test = test['is_fraud']

import lightgbm as lgb

model_1 = lgb.LGBMClassifier(random_state=100)
model_1.fit(X_train, y_train)
pred_1 = model_1.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
print(accuracy_score(y_test, pred_1))
##  정확도가 99.6%이지만 이미 is_fraud 가 0인 경우가 99%이므로 큰 의미가 없음.
print(confusion_matrix(y_test, pred_1)) #혼동행렬 확인
print(classification_report(y_test, pred_1)) #분류 리포트 확인. 민감하게 반응하는 지수인 재현율(reacall)이 중요

proba_1 = model_1.predict_proba(X_test) #그동안의 predict는 0과 1의 결과로 반올림 되어서 나옴. proba는 0과 1이 아닌 소수점으로 나옴.
##  proba : 2열 x 데이터 수 만큼 값 나옴.[0에 대한 예측 값, 1에 대한 예측 값]....
print(proba_1[:,1]) #1에 대한 예측 결과만 출력(거래 이상만)
proba_1 = proba_1[:,1] #예측 결과 재설정

proba_int1 = (proba_1 > 0.2).astype('int') #0.2 기준으로 분류
proba_int2 = (proba_1 > 0.8).astype('int') #0.8 기준으로 분류

print(confusion_matrix(y_test, proba_int1))
print(classification_report(y_test, proba_int1)) #정밀도는 낮아졌으나 재현율은 높아짐.
print(confusion_matrix(y_test, proba_int2))
print(classification_report(y_test, proba_int2)) #정밀도는 높아졌으나 재현율은 낮아짐.

roc_auc_score(y_test, proba_1) #정확도 확인

########################
### 이진분류 평가지표 ###
########################

##  정확도 : 전체 예측값 중 몇 %나 맞췄는지
##  오차행렬 : 실제 참/거짓, 예측 참/거짓을 2X2 테이블로 표현
##  정밀도 : 양성으로 예측된 것 중 참 양성의 비율
##  재현율 : 실제 양성 중 참 양성으로 예측 한 비율
##  F1-점수 : 정밀도와 재현도의 조화 평균
##  민감도 : 참 양성과 거짓 음성으로 예측된 것 중 참 양성의 비율
##  특이도 : 거짓 양성과 참 음 음성으로 예측된 것 중 참 음성의 비율
##  AUC : 모델이 데이터를 얼마나 명료하게 분류하는지를 나타내는 지표

##############################################
### 하이퍼 파라미터 튜닝 : 랜덤 그리드 서치 ###
##############################################

from sklearn.model_selection import RandomizedSearchCV

params = {
    'n_estimators':[100,500,1000],
    'learning_rate':[0.01,0.05,0.1,0.3],
    'lambda_l1':[0,10,20,30,50], #L1 정규화
    'lambda_l2':[0,10,20,30,50], #L2 정규화
    'max_depth':[5,10,15,20],
    'subsample':[0.6,0.8,1]
}
##  L1&L2 정규화 : XGBoost에서도 사용이 가능하며 선형 회귀 모델을 만들면 각 변수에 대한 계수(기울기)가 구해지는데
##                계수에 페널티를 부여해 너무 큰 계수가 나오지 않도록 강제하는 방법.(오버피팅 방지)
model_2 = lgb.LGBMClassifier(random_state=100)
rs = RandomizedSearchCV(model_2, param_distributions=params, n_iter=30, scoring='roc_auc', random_state=100, n_jobs=-1)

rs.fit(X_train, y_train)
print(rs.best_params_)

rs_proba = rs.predict_proba(X_test)
print(roc_auc_score(y_test, rs_proba[:,1]))

rs_proba_int = (rs_proba[:,1]>0.2).astype('int')
print(confusion_matrix(y_test, rs_proba_int))
print(classification_report(y_test, rs_proba_int))