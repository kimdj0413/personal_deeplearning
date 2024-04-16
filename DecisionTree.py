################################
#####   결정트리 연봉예측   #####
################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_url = 'https://media.githubusercontent.com/media/musthave-ML10/data_source/main/salary.csv'
data = pd.read_csv(file_url, skipinitialspace=True) #skipinitialspace는 데이터 첫자리 공란 제거

# print(data.head())
# print(data['class'].unique())
# print(data.info())
# print(data.describe(include = 'all')) #include = all -> object형 포함해서 출력

##############################
### 전처리 : 범주형 데이터  ###
##############################

data['class'] = data['class'].map({'<=50K':0, '>50K':1}) #숫자로 변환(공란 제거 안하면 오류)

    #데이터 타입이 object인 변수 리스트에 저장
obj_list = []
for i in data.columns:
    if data[i].dtype == 'object':
        obj_list.append(i)
        
    #object 리스트에서 고윳값이 10개 이상인거 출력
# for i in obj_list:
#     if data[i].nunique() >= 10:
#         print(i, data[i].nunique())

    #education 변수 처리
# print(data['education'].value_counts()) #고윳값 출현 빈도 확인
        #education과 edaucation-num 상관관계 확인 & 매칭 가능한지 확인
# print(np.sort(data['education-num'].unique()))
# print(data['education'].nunique())
# print(data[data['education-num']==1]) #education-num이 1인 값만 출력해서 education 동일 한지 살펴보기
# print(data[data['education-num']==1]['education'].unique()) #실제로 education-num == 1일때 education 값 출력
# for i in np.sort(data['education-num'].unique()): #모든값이 1대1 매칭 되는지 확인
    # print(i, data[data['education-num']==i]['education'].unique())
data.drop('education', axis=1, inplace=True) #education을 삭제

    #native-country 변수 처리
# print(data['native-country'].value_counts()) #고윳값 출현 빈도 확인
# print(data.groupby('native-country').mean(numeric_only=True).sort_values('class')) #class 열과 상관관계 확인(불일치)
country_group = data.groupby('native-country').mean(numeric_only=True)['class'] #native-country와 class열을 저장(native-country가 인덱스, class가 변수)
country_group = country_group.reset_index() #인덱스는 따로 만들고 native-countyu와 class를 변수로.(인덱스 값으로 native-country 대체)
data = data.merge(country_group, on='native-country',how='left') #data와 country_group 합치기
data.drop('native-country', axis=1, inplace=True) #기존 native-country는 삭제
data = data.rename(columns={'class_x':'class','class_y':'native-country'}) #이름 다시 설정 해주기

#############################################
### 전처리 : 결측치 처리 및 더미 변수 변환  ###
#############################################

# print(data.isna().mean()) #결측치 비율 확인
data['native-country'] = data['native-country'].fillna(-99) #결측치 -99로 채우기. 트리기반에서 유용. 선형 모델에선 X.
# print(data['workclass'].value_counts()) #고윳값별 출현 빈도 확인
data['workclass'] = data['workclass'].fillna('Private') #출현 빈도가 압도적으로 높은 'Private'로 결측치 채움
# print(data['occupation'].value_counts()) #고윳값별 출현 빈도 확인
data['occupation'] = data['occupation'].fillna('Unknown') #압도적인게 없으므로 'Unknown'으로 결측치 채움
data = pd.get_dummies(data, drop_first=True) #범주형 데이터를 더미 변수로 변환

######################
### 모델링 및 평가  ###
######################

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.drop('class', axis=1), data['class'], test_size=0.4, random_state=100)
from sklearn.tree import DecisionTreeClassifier

# model = DecisionTreeClassifier() #모델 생성
# model.fit(X_train, y_train) #학습
# train_pred = model.predict(X_train) #예측
# test_pred = model.predict(X_test) #예측

# from sklearn.metrics import accuracy_score
# print(accuracy_score(y_train, train_pred),accuracy_score(y_test, test_pred)) #훈련셋 : 98%, 테스트셋 : 82% -> 오버피팅

##  오버피팅(과적합) 문제 : 예측 모델이 훈련셋을 지나치게 잘 예측한다면 새로운 데이터를 예측할 때 큰 오차 유발.
##  언더피팅(과소적합) 문제 : 모델이 충분히 학습하지 않아 훈련셋에 대해서도 좋은 예측을 하지 못함.
##  오버피팅 해결 : 트리의 깊이를 낮춘다.

model = DecisionTreeClassifier(max_depth=7) #오버피팅 문제 해결 위해 트리 깊이를 설정
model.fit(X_train, y_train)
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_train, train_pred),accuracy_score(y_test, test_pred))

from sklearn.tree import plot_tree
plt.figure(figsize=(30,15))
# plot_tree(model) #트리가 너무커서 확인하기 힘듬
plot_tree(model, max_depth=3, fontsize=15, feature_names=X_train.columns) #depth 3까지만 확인.
plt.show() #지니(gini) 인덱스가 낮을 수록 노드의 순도가 높음. 순도가 높은쪽으로 가지를 뻗어나감.