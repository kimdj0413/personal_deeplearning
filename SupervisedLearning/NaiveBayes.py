############################################
#####   나이브 베이즈 스팸 문자 처리    #####
############################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_url = 'https://media.githubusercontent.com/media/musthave-ML10/data_source/main/spam.csv'
data = pd.read_csv(file_url)

#데이터 확인
# print(data.head())
# print(data['target'].unique())

##############################
### 전처리 : 특수기호 제거  ###
##############################

import string

# print(string.punctuation) #특수기호출력

def remove_punc(x):
    new_string = []
    for i in x:
        if i not in string.punctuation:
            new_string.append(i)
    new_string = ''.join(new_string) #리스트를 문자열 형태로 저장
    return new_string

data['text'] = data['text'].apply(remove_punc) #한 행에 하나씩 함수 적용(이거 안하면 통째로 적용함.)

##############################
### 전처리 : 불용어 처리    ###
##############################

import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords

# print(stopwords.words('english')) #영어 불용어 출력

def stop_words(x):
    new_string=[]
    for i in x.split(): #띄어쓰기 단위로 나눠주기
        if i.lower() not in stopwords.words('english'): #소문자화 후 불용어에 없으면 추가
            new_string.append(i.lower())
    new_string = ' '.join(new_string) #공백 단위로 묶기
    return new_string

data['text'] = data['text'].apply(stop_words) #함수적용
# print(data['text'])

##################################################
### 전처리 : 목표 컬럼 형태 변환(문자 -> 숫자)  ###
##################################################
data['target'] = data['target'].map({'spam':1, 'ham':0}) #map 함수는 딕셔너리 형태로 값을 매핑
# print(data['target'])

####################################################################################
### 전처리 : 카운트 기반 벡터화(모든 단어를 인덱스화 시켜 각 문장마다 카운트 값)    ###
####################################################################################

X = data['text'] #독립변수
y = data['target'] #종속변수

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer() #객체 생성
cv.fit(X) #학습
# print(cv.vocabulary_) #각 단어별 인덱스 값 출력
# print(cv.vocabulary_['go']) #go의 인덱스 값 출력
X = cv.transform(X) #트랜스폼
# print(X) #결과물 -> (행 번호, 단어 인덱스) 츨현횟수

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=100)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB() #객체 생성
model.fit(X_train, y_train) #학습
pred = model.predict(X_test) #예측

from sklearn.metrics import accuracy_score, confusion_matrix
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred)) #혼동(오차)행렬로 평가
##  음성을 음성으로 판단    음성을 양성으로 판단
##  양성을 음성으로 판단    양성을 양성으로 판단
sns.heatmap(confusion_matrix(y_test,pred), annot=True, fmt='.0f') #시각화해서 보여주기
plt.show()