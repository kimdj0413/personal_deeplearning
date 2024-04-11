#로지스틱 회귀 타이타직 생존자

import pandas as pd

file_url = 'https://media.githubusercontent.com/media/musthave-ML10/data_source/main/titanic.csv'
data = pd.read_csv(file_url)
"""
data.head() #상위 5행 출력
data.describe() #통계 정보 출력
data.corr(numeric_only=True) #상관관계 출력(1에 가까울수록 상관관계 up), 옵션은 숫자만 계산하기 위함

#상관관계를 히트맵으로 보기
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(20, 20))
sns.heatmap(data.corr(numeric_only=True), cmap='coolwarm', vmin=-1, vmax=1, annot=True)
plt.show()
"""
#더미변수와 원-핫인코딩(피처 엔지니어링)
data['Name'].nunique(),data['Ticket'].nunique() #해당 열의 변수 갯수 보기
data = data.drop(['Name','Ticket'], axis=1) #큰 관련없는 문자 변수 삭제
data = pd.get_dummies(data, columns=['Sex','Embarked'], drop_first=True) #더미변수를 만들고 원핫인코딩 후 열 삭제

from sklearn.model_selection import train_test_split

#모델링 및 예측하기
X = data.drop('Survived', axis=1) #독립변수에서 종속변수 제거
y = data['Survived'] #종속변수
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

#평가하기
#이진분류는 RMSE 사용은 적합하지 않음.
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pred))

#각 변수별 영향력 확인
model.coef_
print(pd.Series(model.coef_[0], index = X.columns)) #2차원 배열이므로 [0]길이로 길이 재정비
#로지스틱 회귀 분석은 선형 회귀처럼 수식 표현불가. 연산을 더 거침.

#피처 엔지니어링(기존 데이터를 손보아 더 나은 변수를 만드는 기법)
#다중 공신성 문제 해결(상관관계가 높은 변수를 합칩)
data['family'] = data('SibSp')+data('Parch') #부모/자식, 형제/자매 데이터 합쳐서 가족 만들기
data.drop(['SibSp','Parch'], axis=1, inplace=True) #삭제
#이걸로 학습하면 정확도 조금 높아짐.