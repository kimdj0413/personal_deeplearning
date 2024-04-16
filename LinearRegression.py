####################################
#####   선형회귀 보험 데이터 셋 #####
####################################

import pandas as pd
from sklearn.model_selection import train_test_split
#선형회귀 라이브러리
from sklearn.linear_model import LinearRegression 

#모델생성
model = LinearRegression()

file_url= 'https://media.githubusercontent.com/media/musthave-ML10/data_source/main/insurance.csv'
data = pd.read_csv(file_url)

#독립변수
X = data[['age','sex','bmi','children','smoker']]
#종속변수
y = data['charges'] 

#독립/종속변수, 학습/시험셋, 학습셋 80% 시험셋 20%, 랜덤 샘플링 번호 100번(100번이라는 고유한 랜덤 생성)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=100)

#학습(독립변수, 종속변수)
model.fit(X_train, y_train) 
#모델을 사용해서 예측(X_test 값에서 y_test값 예측 후 실제값과 비교)
pred = model.predict(X_test) 


comparison = pd.DataFrame({'actual':y_test, 'pred':pred})
""" 테이블로 평가하기
print(comparison)
"""

""" 그래프로 평가하기
import matplotlib.pyplot as plt
import seaborn as sns #산점도 그래프 그리기 라이브러리
plt.figure(figsize=(10,10))
sns.scatterplot(x='actual', y='pred',data=comparison)

plt.show()
"""

#통계적 방법으로 평가하기(RMSE) -> 가장 많이 사용
from sklearn.metrics import mean_squared_error #MSE 라이브러리
print(mean_squared_error(y_test, pred, squared = False))
#R squred 값을 구하는 함수 score
#R squred = SSR(모델~종속변수 y의 평균값 사이의 거리) / SST(모델~실제 데이터까지의 거리)
print(model.score(X_train, y_train))

#모델의 계수(기울기) 구하기
#계수의 절대값이 클수록 영향도가 큰 변수
model.coef_ #pd.Series(model.coef_, index = X.columns) -> 판다스로 보기쉽게
#모델의 y절편 구하기
model.intercept_

#최종 수식
#계수 a, b, c, d, e(해당 값이 1 증가하는 만큼 기울기 만큼 charges 증가)
#y절편 i
#charges(종속변수) = a x age + b x sex + c x bmi + d x children + e x smoker + i