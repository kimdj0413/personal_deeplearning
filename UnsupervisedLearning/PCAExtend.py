####################################
#####   PCA 성능 향상 테스트    #####
####################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_url = 'C:\\personal_coding\\personal_machineLearning\\UnsupervisedLearning\\anonymous.csv'
anonymous = pd.read_csv(file_url)
# print(anonymous.head())
# print(anonymous['class'].unique())
# print(anonymous['class'].mean())
# print(anonymous.isna().sum().sum()) #결측치의 총 합을 구해서 결측치가 있는지 확인

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(anonymous.drop('class', axis=1), anonymous['class'], test_size=0.2, random_state=100)

###################
### 스케일링    ###
##################

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

###################
### 성능 비교   ###
###################

##  깡성능
from sklearn.ensemble import RandomForestClassifier
# model_1 = RandomForestClassifier(random_state=100)

import time
# start_time = time.time()
# model_1.fit(X_train_scaled, y_train)
# print(time.time()-start_time) #소요사건

from sklearn.metrics import accuracy_score, roc_auc_score

# pred_1 = model_1.predict(X_test_scaled)
# print(accuracy_score(y_test, pred_1)) #정확도 예측

# proba_1 = model_1.predict_proba(X_test_scaled) #proba를 사용해야 소수점 형태(AUC)로 예측 가능
# print(roc_auc_score(y_test, proba_1[:, 1]))

##  차원 축소
from sklearn.decomposition import PCA

pca = PCA(n_components = 2) #4천개 중 주성분 2개로
pca.fit(X_train_scaled)
# print(pca.explained_variance_ratio_) #데이터 반영 비율 확인
##  데이터 반영 비율이 너무 낮으니 엘보우 기법으로 최적 개수 구함.

# var_ratio = []
# for i in range(100,550,50):
#     pca = PCA(n_components=i)
#     pca.fit_transform(X_train_scaled)
#     ratio = pca.explained_variance_ratio_.sum()
#     var_ratio.append(ratio)
# sns.lineplot(x=range(100,550,50), y=var_ratio)
# plt.show() #그래프로 확인 #주성분 100~550 사이에서 얻을 수 있는 반영 비율은 62%~82%


##  PCA 성능
pca = PCA(n_components=400, random_state=100) #주성분 400개로 지정
pca.fit(X_test_scaled)
X_train_scaled_pca = pca.transform(X_train_scaled)
X_test_scaled_pca = pca.transform(X_test_scaled)

model_2 = RandomForestClassifier(random_state = 100)
start_time = time.time()
model_2.fit(X_train_scaled_pca, y_train)
# print(time.time() - start_time)

pred_2 = model_2.predict(X_test_scaled_pca)
print(accuracy_score(y_test,pred_2))

proba_2 = model_2.predict_proba(X_test_scaled_pca)
print(roc_auc_score(y_test, proba_2[:,1]))