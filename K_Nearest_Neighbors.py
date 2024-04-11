#####K 최근접 이웃 와인 등급 예측#####

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_url = 'https://media.githubusercontent.com/media/musthave-ML10/data_source/main/wine.csv'
data = pd.read_csv(file_url)

#데이터 살펴보기
print(data.head()) #상위 5행 출력
print(data.info()) #178개가 되지 않는 결측치가 2개 있음.
print(data.describe()) #통계 정보 출력

#목표치 고윳값 확인하기
data['class'].unique() #목표변수 고윳값 출력
data['class'].nunique() #고윳값 가짓수 출력
data['class'].value_counts() #각 고윳값에 해당하는 개수 출력
# sns.barplot(x = data['class'].value_counts().index, y = data['class'].value_counts()) #막대그래프로 확인
sns.countplot(data['class']) #더 쉽게 막대그래프 그리기
plt.show()
