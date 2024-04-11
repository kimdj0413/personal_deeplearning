#####K 최근접 이웃 와인 등급 예측#####

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_url = 'https://media.githubusercontent.com/media/musthave-ML10/data_source/main/wine.csv'
data = pd.read_csv(file_url)
"""
#데이터 살펴보기
print(data.head()) #상위 5행 출력
print(data.info()) #178개가 되지 않는 결측치가 2개 있음.
print(data.describe()) #통계 정보 출력
"""
#목표치 고윳값 확인하기
data['class'].unique() #목표변수 고윳값 출력
data['class'].nunique() #고윳값 가짓수 출력
data['class'].value_counts() #각 고윳값에 해당하는 개수 출력
# sns.barplot(x = data['class'].value_counts().index, y = data['class'].value_counts()) #막대그래프로 확인
#sns.countplot(data['class']) #더 쉽게 막대그래프 그리기
# plt.show()
"""
#결측치 처리하기
print(data.isna()) #결측치가 True로 표시됨
print(data.isna().sum()) #False=0, True=1 이므로 합을 구하면 결측치를 한눈에 볼 수 있음.
print(data.isna().mean()) #평균치 출력으로 % 알 수 있음.

    #첫번째 방법
data.dropna() #결측치 행 제거
data.dropna(subset=['alcohol']) #지정된 변수의 결측치 행만 제거

    #두번째 방법
data.drop(['alcohol','nonflavored_phenols'], axis=1) #결측치 변수를 통으로 제거
"""
    #세번째 방법
data.fillna(data.mean(), inplace=True) #결측치를 채워넣기(평균적으로 중윗값이나 평균값 채워넣음.)

#스케일링(데이터 스케일 맞추기)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
"""
    #첫번째 : 표준화 스케일링(평균을 0으로 표준편차를 1으로. 데이터 고르게 분포)
st_scaler = StandardScaler() #스케일러 지정
st_scaler.fit(data) #학습
st_scaled = st_scaler.transform(data) #계산(스케일링)
st_scaled = pd.DataFrame(st_scaled, columns = data.columns) #넘파이 배열을 판다스 데이터프레임으로 변경
print(st_scaled)
print(round(st_scaled.describe(), 2)) #평균과 표준편차 확인

    #두번째 : 로버스트 스케일링(데이터의 아웃라이어 영향력이 크고 이를 피함.)
rb_scaler = RobustScaler() #스케일러 지정
rb_scaled = rb_scaler.fit_transform(data) #학습 및 계산(스케일링)
rb_scaled = pd.DataFrame(rb_scaled, columns = data.columns) #넘파이 배열을 판다스 데이터프레임으로 변경
print(round(rb_scaled.describe(), 2))

    #세번째 : 최대최소 스케일링(최댓값 1, 최솟값 0. 데이터의 특성을 최대한 유지)
mm_scaler = MinMaxScaler() #스케일러 지정
mm_scaled = mm_scaler.fit_transform(data) #학습 및 계산(스케일링)
mm_scaled = pd.DataFrame(mm_scaled, columns=data.columns) #넘파이 배열을 판다스 데이터프레임으로 변경
print(round(mm_scaled.describe(),2))
"""
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data.drop('class', axis=1),data['class'], test_size=0.2, random_state=100)

mm_scaler = MinMaxScaler()
X_train_scaled = mm_scaler.fit_transform(X_train) #훈련셋 학습 및 스케일링
X_test_scaled = mm_scaler.transform(X_test) #시험셋 학습 및 스케일링
#목표값(종속변수)는 스케일링에서 제외

#모델링 및 평가
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

scores = []
for i in range(1,21):
    knn = KNeighborsClassifier(n_neighbors=i) #모델 생성(예측에 참고할 이웃 수 = i(기본은 5))
    #for문 돌려서 range(1,21)에서 정확도가 가장 높은 것으로 이웃 수 책정(i=13)
    knn.fit(X_train_scaled,y_train) #학습
    pred = knn.predict(X_test_scaled) #예측
    acc = accuracy_score(y_test, pred) #평가
    scores.append(acc) #평가 저장

print(scores)
sns.lineplot(x=range(1,21), y=scores)
plt.show()
    
