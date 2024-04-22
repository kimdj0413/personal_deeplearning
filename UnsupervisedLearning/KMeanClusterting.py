########################################
#####   K-평균 군집화 쇼핑몰 추천   #####
########################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

##################################
### 학습용 데이터로 학습하기    ###
##################################

file_url = 'https://raw.githubusercontent.com/musthave-ML10/data_source/main/example_cluster.csv'
data = pd.read_csv(file_url)

# print(data)
# sns.scatterplot(x='var_1', y='var_2', data=data)
# plt.show() #산점도 그래프로 한눈에 보기. 3그룹으로 나뉘어져 있으니 K-평균 군집화로 3개로 나눔.

from sklearn.cluster import KMeans

kmeans_model = KMeans(n_clusters=3, random_state=100) #그룹을 3개로 지정
kmeans_model.fit(data) #학습
# print(kmeans_model.predict(data)) #예측. 3개의 레이블이 배열에 저장.
data['label'] = kmeans_model.predict(data) #예측값을 label로 저장.
# # sns.scatterplot(x='var_1', y='var_2', data=data, hue='label', palette='rainbow') #hue와 paletter는 색상 부여.
# plt.show()

##  엘보우 기법 : 최적의 클러스터 개수(K)를 확인하는 방법.
##  이너셔(관성) : 각 그룹에서 중심과 각 그룹에 해당하는 데이터 간의 거리에 대한 합.
## 둘은 반비례 관계.

### 엘보우 기법으로 최적의 K값 구하기

# print(kmeans_model.inertia_) #이너셔는 학습 할때 자동으로 계산

distance = []
for k in range(2,10):
    k_model = KMeans(n_clusters=k)
    k_model.fit(data)
    distance.append(k_model.inertia_) #이니셔를 리스트에 저장

# print(distance)
# sns.lineplot(x=range(2,10), y=distance)
# plt.show() #distance 값이 급격히 작아지는 지점(엘보우)의 K값.

######################
### 고객 데이터셋   ###
######################

file_url = 'https://raw.githubusercontent.com/musthave-ML10/data_source/main/customer.csv'
customer = pd.read_csv(file_url)

# print(customer) #cc_num(카드번호)를 고객 ID로 사용
# print(customer['cc_num'].nunique()) #고객 100명
# print(customer['category'].nunique()) #범주 11개

##############################
### 전처리 : 피처엔지니어링 ###
##############################

customer_dummy = pd.get_dummies(customer, columns=['category']) #더미 변수로 변환
# print(customer_dummy)
cat_list = customer_dummy.columns[2:] #변수 이름 리스트 생성
for i in cat_list:
    customer_dummy[i] = customer_dummy[i] * customer_dummy['amt'] #금액으로 변수 업데이트
# print(customer_dummy)
customer_agg = customer_dummy.groupby('cc_num').sum() #거래 건으로 정리된 데이터를 고객 레벨로 취합.
# print(customer_agg)

    #데이터 스케일링
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(customer_agg),
            columns = customer_agg.columns,
            index=customer_agg.index)
# print(scaled_df) #0의 근삿값으로 스케일링 완료

#########################################
### 고객 데이터 모델링 및 실루엣 계수   ###
#########################################

    #엘보우 기법으로 k 값 구하기
distance = []
for k in range(2,10):
    k_model = KMeans(n_clusters=k)
    k_model.fit(scaled_df)
    labels = k_model.predict(scaled_df)
    distance.append(k_model.inertia_)

# sns.lineplot(x=range(2,10), y=distance)
# plt.show()
##  그래프가 완만하게 떨어짐 -> 더욱 복잡하고 오래걸리는 실루엣 계수 사용

from sklearn.metrics import silhouette_score

silhouette = []
for k in range(2,10):
    k_model = KMeans(n_clusters=k)
    k_model.fit(scaled_df)
    labels = k_model.predict(scaled_df)
    silhouette.append(silhouette_score(scaled_df, labels))

# sns.lineplot(x=range(2,10), y=silhouette)
# plt.show()
##  실루엣 계수가 클수록 좋은 분류 K=4

##################################
### 최종 예측 모델 및 결과 해석 ###
##################################

k_model = KMeans(n_clusters=4)
k_model.fit(scaled_df)
labels = k_model.predict(scaled_df)
scaled_df['label'] = labels

scaled_df_mean = scaled_df.groupby('label').mean()
scaled_df_count = scaled_df.groupby('label').count()['category_travel']

scaled_df_count = scaled_df_count.rename('count')
scaled_df_all = scaled_df_mean.join(scaled_df_count)
print(scaled_df_all)