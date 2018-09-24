import pandas as pd
from pandas import Series
from pandas import DataFrame
import numpy as np
df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                 'python-machine-learning-book-2nd-edition'
                 '/master/code/ch10/housing.data.txt',
                 header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

#Part 1: Exploratory Data Analysis
#Head, tail and statistical summary of the data
print('\n', "Head, tail and statistical summary of the data:")
print(df.head())
df.head().to_excel("head.xls")
print(df.tail())
df.tail().to_excel("tail.xls")
summary=df.describe()
summary.to_excel("statistical summary.xls")
print(summary) 

import matplotlib.pyplot as plt
import seaborn as sns

#generate scatterplot matrix
print('\n', "Scatterplot Matrix:")
cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
sns.pairplot(df[cols], size=2.5)
plt.title('Scatterplot Matrix')
plt.tight_layout()
plt.savefig("scatterplot matrix")
plt.show()


#generate the correlation matrix array as a heat map
print('\n', "Correlation matrix array as a heat map:")
import numpy as np
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
plt.rcParams['figure.figsize']=(13.0, 13.0)
plt.title('Correlation Heat Map')
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
plt.savefig("correlation heat map")
plt.show()


#Generate the box plot
array=df.values
plt.rcParams['figure.figsize']=(6.0, 6.0)
plt.title('Box Plot')
plt.boxplot(array)
plt.xlabel("Attribute Index")
plt.ylabel("Quartile Ranges")
plt.savefig("box plot")
plt.show()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
np_std = sc.fit_transform(df)
df_std = DataFrame(np_std)
array=df_std.values
plt.rcParams['figure.figsize']=(6.0, 6.0)
plt.title('Standardized Box Plot')
plt.boxplot(array)
plt.xlabel("Attribute Index")
plt.ylabel("Quartile Ranges")
plt.savefig("standarized box plot")
plt.show()


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X = df.iloc[:, :-1].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Part 2: Linear regression
slr = LinearRegression()
slr.fit(X_train, y_train)
np_ci=np.append(slr.intercept_,slr.coef_)
df_ci=DataFrame(np_ci,index=['intercept','coef1','coef2','coef3','coef4','coef5','coef6','coef7','coef8','coef9','coef10','coef11','coef12','coef13'],columns=['value'])
df_ci=df_ci.T
df_ci.to_excel('df_ls_ci.xls')
print(df_ci)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

# plot the residual plot
plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.title('LR Residual Plot')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.savefig("slr residual plot")
plt.show()

#compute the MSE of our training and test predictions
from sklearn.metrics import mean_squared_error
print('MSE train: %.3f, MSE test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
#MSE train: 19.958, test: 27.196

from sklearn.metrics import r2_score
print('R^2 train: %.3f, R^2 test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)),'\n')
print("---------------------------------------------------------------------------")
#R^2 train: 0.765, test: 0.673



#Part 3.1: Ridge regression
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math
list_i_l2=[]
list_R2tr_l2=[]
list_R2te_l2=[]
list_msetr_l2=[]
list_msete_l2=[]
for i in range (-3,3):
    print("alpha = 10^(",i, "), the result of Ridge regression is as belows:",'\n')
    ridge = Ridge(alpha=math.pow( 10, i ))
    ridge.fit(X_train, y_train)
    
    np_ci=np.append(ridge.intercept_,ridge.coef_)
    df_ci=DataFrame(np_ci,index=['intercept','coef1','coef2','coef3','coef4','coef5','coef6','coef7','coef8','coef9','coef10','coef11','coef12','coef13'],columns=['value'])
    df_ci=df_ci.T
    print("Model description:")
    print(df_ci)
    
    y_train_pred = ridge.predict(X_train)
    y_test_pred = ridge.predict(X_test)
    
    # plot the residual plot
    plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
    plt.title('L2 Residual Plot')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
    plt.xlim([-10, 50])
    plt.show()

    #define MSE arrays
    alpha=math.pow( 10, i)
    list_i_l2.append(alpha)
    
    R2tr= r2_score(y_train, y_train_pred)
    list_R2tr_l2.append(R2tr)
    
    R2te= r2_score(y_test, y_test_pred)
    list_R2te_l2.append(R2te)
    
    mse_train= mean_squared_error(y_train, y_train_pred)
    list_msetr_l2.append(mse_train)
    
    mse_test= mean_squared_error(y_test, y_test_pred)
    list_msete_l2.append(mse_test)
    
    #compute the MSE of our training and test predictions
    print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
 
    #compute the R^2 of our training and test predictions
    print('R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)),'\n')
    print("---------------------------------------------------------------------------")
array_l2=np.vstack((list_i_l2,  list_R2tr_l2,  list_R2te_l2, list_msetr_l2, list_msete_l2))   
df_l2=pd.DataFrame(array_l2,columns=['','','','','',''])
df_l2.index=Series(['alpha','R^2 train','R^2 test','MSE train','MSE test'])
df_l2.to_excel('Ridge_MSE.xls')
print(df_l2)


#Part 3.2: LASSO regression
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math
list_i_l1=[]
list_R2tr_l1=[]
list_R2te_l1=[]
list_msetr_l1=[]
list_msete_l1=[]
for i in range (-3,3):
    print('\n','\n',"alpha = 10^(",i, "), the result of LASSO regression is as belows:")
    lasso = Lasso(alpha=math.pow( 10, i ))
    lasso.fit(X_train, y_train)
    
    np_ci=np.append(lasso.intercept_,lasso.coef_)
    df_ci=DataFrame(np_ci,index=['intercept','coef1','coef2','coef3','coef4','coef5','coef6','coef7','coef8','coef9','coef10','coef11','coef12','coef13'],columns=['value'])
    df_ci=df_ci.T
    print("Model description:")
    print(df_ci)
    
    y_train_pred = lasso.predict(X_train)
    y_test_pred = lasso.predict(X_test)

    # plot the residual plot
    plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
    plt.title('L1 Residual Plot')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
    plt.xlim([-10, 50])
    plt.show()
    
    #define MSE arrays
    alpha=math.pow( 10, i)
    list_i_l1.append(alpha)
    
    R2tr= r2_score(y_train, y_train_pred)
    list_R2tr_l1.append(R2tr)
    
    R2te= r2_score(y_test, y_test_pred)
    list_R2te_l1.append(R2te)
    
    mse_train= mean_squared_error(y_train, y_train_pred)
    list_msetr_l1.append(mse_train)
    
    mse_test= mean_squared_error(y_test, y_test_pred)
    list_msete_l1.append(mse_test)
    
    #compute the MSE of our training and test predictions
    print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
    
    #compute the R^2 of our training and test predictions
    print('R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))
   
array_l1=np.vstack((list_i_l1,  list_R2tr_l1,  list_R2te_l1, list_msetr_l1, list_msete_l1))   
df_l1=pd.DataFrame(array_l1,columns=['','','','','',''])
df_l1.index=Series(['alpha','R^2 train','R^2 test','MSE train','MSE test'])
df_l1.to_excel('LASSO_MSE.xls')
print(df_l1)


#Part 3.3: Elastic Net regression
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math
list_i_EN=[]
list_R2tr_EN=[]
list_R2te_EN=[]
list_msetr_EN=[]
list_msete_EN=[]
for i in range (-3,3):
    print('\n','\n',"l1 ratio = 10^(",i, "), the result of ElasticNet regression is as belows:")
    elanet = ElasticNet(alpha=1.0,l1_ratio=math.pow( 10, i ))
    elanet.fit(X_train, y_train)

    np_ci=np.append(elanet.intercept_,elanet.coef_)
    df_ci=DataFrame(np_ci,index=['intercept','coef1','coef2','coef3','coef4','coef5','coef6','coef7','coef8','coef9','coef10','coef11','coef12','coef13'],columns=['value'])
    df_ci=df_ci.T
    print("Model description:")
    print(df_ci)
    
    y_train_pred = elanet.predict(X_train)
    y_test_pred = elanet.predict(X_test)

    # plot the residual plot
    plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
    plt.title('ElasticNet Residual Plot')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
    plt.xlim([-10, 50])
    plt.show()
    
    #define arrays
    alpha=math.pow( 10, i)
    list_i_EN.append(alpha)
    
    R2tr= r2_score(y_train, y_train_pred)
    list_R2tr_EN.append(R2tr)
    
    R2te= r2_score(y_test, y_test_pred)
    list_R2te_EN.append(R2te)
    
    mse_train= mean_squared_error(y_train, y_train_pred)
    list_msetr_EN.append(mse_train)
    
    mse_test= mean_squared_error(y_test, y_test_pred)
    list_msete_EN.append(mse_test)
  
    #compute the MSE of our training and test predictions
    print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
    
    
    #compute the R^2 of our training and test predictions
    print('R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))    

array_ElasticNet=np.vstack((list_i_EN, list_R2tr_EN, list_R2te_EN, list_msetr_EN, list_msete_EN))   
df_EN=pd.DataFrame(array_ElasticNet,columns=['','','','','',''])
df_EN.index=Series(['l1 ratio','R^2 train','R^2 test','MSE train','MSE test'])
df_EN.to_excel('ElasticNet_MSE.xls')
print(df_EN)
print('\n')

print("My name is QI GUO")
print("My NetID is: qiguo3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
######STOP HERE######################
