########################################
#####   XGBoost 커플성사여부 예측   #####
########################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_url = 'https://media.githubusercontent.com/media/musthave-ML10/data_source/main/dating.csv'
data = pd.read_csv(file_url)
# pd.options.display.max_columns = 40 #40열을 모두 살펴보기.
# print(data.info())
# print(round(data.describe(), 2))

##############################
### 전처리 : 결측치 처리    ###
##############################

# print(data.isna().mean()) #결측치 비율 출력
data = data.dropna(subset=['pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence', 'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests', 'attractive_important', 'sincere_important', 'intellicence_important', 'funny_important', 'ambtition_important', 'shared_interests_important'])
# 중요도 결측치 행 제거
data = data.fillna(-99) #나머지는 -99로 채움

##################################
### 전처리 : 피처 엔지니어링    ###
##################################

    #나이 전처리
def age_gap(x):
    if x['age'] == -99:
        return -99
    elif x['age_o'] == -99:
        return -99
    elif x['gender'] == 'female':
        return x['age_o'] - x['age']
    else:
        return x['age'] - x['age_o']

data['age_gap'] = data.apply(age_gap, axis=1)
data['age_gap_abs'] = abs(data['age_gap'])
# print(data['age_gap'])

    #인종 전처리
def same_race(x):
    if x['race'] == -99:
        return -99
    elif x['race_o'] == -99:
        return -99
    elif x['race'] == x['race_o']:
        return 1
    else:
        return -1
    
data['same_race'] = data.apply(same_race, axis=1)

def same_race_point(x):
    if x['same_race'] == -99:
        return -99
    else:
        return x['same_race'] * x['importance_same_race']
    
data['same_race_point'] = data.apply(same_race_point, axis=1)

    #평가/중요도 전처리
def rating(data, importance, score):
    if data[importance] == -99:
        return -99
    elif data[score] == -99:
        return -99
    else:
        return data[importance] * data[score]
    
partner_imp = data.columns[8:14] #상대방 중요도
partner_rate_me = data.columns[14:20] #본인에 대한 상대방의 평가
my_imp = data.columns[20:26] #본인의 중요도
my_rate_partner = data.columns[26:32] #상대방에 대한 본인의 평가

new_label_partner = ['attractive_p', 'sincere_partner_p', 'intelligence_p', 'funny_p', 'ambition_p', 'shared_interests_p)']
new_label_me = ['attractive_m', 'sincere_partner_m', 'intelligence_m', 'funny_m', 'ambition_m', 'shared_interests_m']

for i,j,k in zip(new_label_partner, partner_imp, partner_rate_me):
    data[i] = data.apply(lambda x: rating(x,j,k), axis=1)

#     ##  람다 함수를 적용하지 않으면
# def apply_rating(data, importance_col, score_col, new_col):
#     for i in range(len(data)):
#         data.loc[i, new_col] = rating(data.loc[i], importance_col, score_col)
#         #loc함수는 loc[해당 행, 해당 열]
# for i,j,k in zip(new_label_partner, partner_imp, partner_rate_me):
#     apply_rating(data, j, k, i)
# # 이렇게 한줄 한줄 돌리는 함수가 하나 더 필요함.

for i,j,k in zip(new_label_me, my_imp, my_rate_partner):
    data[i] = data.apply(lambda x: rating(x,j,k), axis=1)

    #더미 변수 변환(원-핫 인코딩)
data = pd.get_dummies(data, columns=['gender','race','race_o'], drop_first=True)

#######################
### 모델링 및 평가  ###
######################

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.drop('match',axis=1),data['match'], test_size=0.2, random_state=100)
import xgboost as xgb
# model = xgb.XGBClassifier(n_estimators = 500, max_depth=5,random_state=100)
# model.fit(X_train, y_train)

# pred = model.predict(X_test)
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# # print(accuracy_score(y_test, pred))
# # print(confusion_matrix(y_test, pred)) #편향된 데이터임을 판별
# # print(classification_report(y_test, pred))

# ##########################################
# ### 하이퍼파라미터튜닝 : 그리드 서치    ###
# ##########################################

# ##  파라미터 값들을 설정 해주면 조합해서 최적의 파라미터를 찾아줌
# from sklearn.model_selection import GridSearchCV

# parameters = {
#     'learning_rate' : [0.01, 0.1, 0.3],
#     'max_depth' : [5, 7, 10],
#     'subsample' : [0.5, 0,7, 1],
#     'n_estimators' : [300, 500, 1000]
# }
# ##  learning_rate : 경사하강법에서 '매개변수'를 얼만큼씩 이동할지. XG부스트는 강화된 경사 부스팅.
# ##  max_depth : 트리의 깊이 제한
# ##  subsample : 모델을 학습할 때 일부 데이터만 사용하여 트리를 만드는 비율. 오버피팅 방지용.
# ##  n_estimators : 전체 나무의 개수

# model = xgb.XGBClassifier()
# gs_model = GridSearchCV(model, parameters, n_jobs=-1, scoring='f1', cv=5)

# ##  n_jobs : 사용할 코어 수
# ##  scoring : 모델링 판단 기준
# ##  cv : K-폴드 값

# gs_model.fit(X_train, y_train)
# # print(gs_model.best_params_) #최고의 파라미터를 출력
# pred = gs_model.predict(X_test)
# print(accuracy_score(y_test, pred))
# print(classification_report(y_test, pred))

######################
### 중요 변수 확인  ###
######################

model = xgb.XGBClassifier(learning_rate=0.3, max_depth=5,n_estimators=1000,subsample=0.5,random_state=100) #최적의 파라미터로 모델 생성
model.fit(X_train, y_train)
# print(model.feture_importances_)
feature_imp = pd.DataFrame({'features':X_train.columns, 'values':model.feature_importances_})
plt.figure(figsize=(20,10))
sns.barplot(x='values', y='features', data=feature_imp.sort_values(by='values',ascending=False).head(10))
plt.show()