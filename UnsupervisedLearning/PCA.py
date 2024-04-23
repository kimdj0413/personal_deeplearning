###########################
#####	PCA 차원 축소	#####
##########################

##	PCA = 주성분 분석 = 차원 축소(변수가 3개면 3차원 -> 2개 2차원으로)
##	어떤 것을 예측하거나 분석하지는 않음.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_url = 'https://raw.githubusercontent.com/musthave-ML10/data_source/main/customer_pca.csv'
customer = pd.read_csv(file_url)
# print(customer.head())

customer_X = customer.drop('label', axis=1) #독립변수
customer_y = customer['label'] #종속변수
##	종속변수 값은 유지돼야 해서 PCA 적용 대상에서 제외

###################
### 차원축소    ###
##################

from sklearn.decomposition import PCA

pca = PCA(n_components=2) #주성분 개수는 사용자가 지정(2개)
pca.fit(customer_X)
customer_pca = pca.transform(customer_X)
# print(customer_pca)

customer_pca = pd.DataFrame(customer_pca, columns=['PC1','PC2']) #데이터프레임으로 변환
customer_pca = customer_pca.join(customer_y) #label 값을 추가
# print(customer_pca)
# sns.scatterplot(x='PC1',y='PC2', data=customer_pca, hue='label',palette='rainbow')
# plt.show() #산점도 그리기

# print(pca.components_) #주성분과 변수의 상관 관계 확인
df_comp = pd.DataFrame(pca.components_, columns=customer_X.columns) #데이터프레임으로 확인
# print(df_comp)
sns.heatmap(df_comp,cmap='coolwarm')
plt.show()