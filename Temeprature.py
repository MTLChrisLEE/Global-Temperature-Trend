import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("GLB.Ts+dSST.csv",na_values="***")
X = dataset.iloc[90:138,0:1].values-1970
avgTemperature = dataset.iloc[90:138,13:14]

janTemperature = dataset.iloc[90:138,1:2]
febTemperature = dataset.iloc[90:138,2:3]
marchTemperature = dataset.iloc[90:138,3:4]
aprilTemperature = dataset.iloc[90:138,4:5]
mayTemperature = dataset.iloc[90:138,5:6]
juneTemperature = dataset.iloc[90:138,6:7]
julyTemperature = dataset.iloc[90:138,7:8]
augTemperature = dataset.iloc[90:138,8:9]
sepTemperature = dataset.iloc[90:138,9:10]
octTemperature = dataset.iloc[90:138,10:11]
novTemperature = dataset.iloc[90:138,11:12]
decTemperature = dataset.iloc[90:138,12:13]

# Fitting Simple Linear Regression and Polynomial Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, avgTemperature)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree =9)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, avgTemperature)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, avgTemperature)


import statsmodels.api as sm
print("====================================GLOBAL TEMP LINEAR=====================================")
estGlobalTemperatureLinear=sm.OLS(avgTemperature, X)
print(estGlobalTemperatureLinear.fit().summary())

print("====================================GLOBAL TEMP POLY====================================")
estGlobalTestPolynomial=sm.OLS(avgTemperature, X_poly)
print(estGlobalTestPolynomial.fit().summary())


# Visualising the Training set results with Linear Regression
plt.figure("Global Temperature")
plt.scatter(X, avgTemperature, color = 'red')
plt.plot(X, regressor.predict(X), color = 'black', label="Linear")
plt.title('Global Temperature')
plt.xlabel('Year')
plt.ylabel('Temperature Anomalies')
plt.show()


#Visualising the Testing set results with Linear Regression
plt.figure("Global Temperature")
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'green',label="Polynomial")
plt.title('Global Temperature')
plt.xlim(xmin=0)
plt.xlabel('Year')
plt.ylabel('Temperature Anomalies')
plt.legend()
plt.savefig('GlobalTemperature.png')
plt.show()



from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
def MonthRegression(arrayoftime, monthTemperature,titleofsummary,label,color):
    X_train, X_test, y_train, y_test = train_test_split(arrayoftime, monthTemperature, test_size = 0.3, random_state = 0)

    regressor=LinearRegression()
    regressor.fit(X_train,y_train)
    print("===================================="+titleofsummary+"=====================================")

    estMonthTemperature=sm.OLS(y_train, X_train)
    print(estMonthTemperature.fit().summary())
    
    plt.figure("Monthly Temperature with real values(Training Set)")
    plt.scatter(X_train, y_train, color = color)
    plt.plot(X, regressor.predict(X), color = color, label=label)
    plt.xlim(xmin=0)
    plt.title('Monthly Temperature with real values(Training Set)')
    plt.xlabel('Year')
    plt.ylabel('Temperature Anomalies')
    plt.legend(loc=2, ncol=2)
    plt.show()
    plt.savefig('Month\'s Global Temperature Trend with Real values, Training Set.png')
    
    plt.figure("Monthly Temperature with real values(Test Set)")
    plt.scatter(X_test, y_test, color = color)
    plt.plot(X, regressor.predict(X), color = color, label=label)
    plt.xlim(xmin=0)
    plt.title('Monthly Temperature with real values(Test Set)')
    plt.xlabel('Year')
    plt.ylabel('Temperature Anomalies')
    plt.legend(loc=2, ncol=2)
    plt.show()
    plt.savefig('Month\'s Global Temperature Trend without Real values, Test Set.png')
    
    print("===========================")
    print("The temperature in" + label + " will be " + str(regressor.predict(48)))
    

MonthRegression(X,janTemperature,"JAN TEMP LINEAR","JAN",'#5ebfff')
MonthRegression(X,febTemperature,"FEB TEMP LINEAR","FEB",'#3b769e') 
MonthRegression(X,marchTemperature,"MARCH TEMP LINEAR","MAR",'#2b5673') 
MonthRegression(X,aprilTemperature ,"APRIL TEMP LINEAR","APRIL",'#244961') 
MonthRegression(X,mayTemperature ,"MAY TEMP LINEAR", "MAY",'#92c97b')
MonthRegression(X,juneTemperature ,"JUNE TEMP LINEAR","JUNE",'#79a364') 

MonthRegression(X,julyTemperature ,"JULY TEMP LINEAR","JULY",'#79a364')
MonthRegression(X,augTemperature ,"AUG TEMP LINEAR","AUG",'#445c39') 
MonthRegression(X,sepTemperature ,"SEPT TEMP LINEAR","SEPT",'#ff4046')
MonthRegression(X,octTemperature ,"OCT TEMP LINEAR","OCT",'#a12729') 
MonthRegression(X,novTemperature ,"NOV TEMP LINEAR","NOV",'#781d1e')
MonthRegression(X,decTemperature,"DEC TEMP LINEAR","DEC",'#472b2b') 






