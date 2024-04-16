#####결정트리 연봉예측#####

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

#전처리 : 범주형 데이터
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