import pandas as pd #visualization
from sklearn.model_selection import train_test_split #Train/Test/Split
from sklearn.linear_model import LinearRegression #Actual Algorithm
import joblib
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt

df = pd.read_csv('Test.csv')
df.head()

x = df[["TotalCases", "NewCases", "LocalTransmission", "ReturningResidents", "ROFW", "Deaths", "APOR", "LDOrder", "Increaseincases", "IncreaseinPositivityRate"]]
y = df["Positivity"]

lr = LinearRegression()
LinearRegression()
x_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 0) #80% to be trained
lr.fit(x_train, y_train)

y_pred_test = lr.predict(X_test)
print(y_pred_test)


predtest = lr.predict(X_test)
msetest = mean_squared_error(y_test, predtest)
rmsetest = sqrt(msetest)
print(msetest)



joblib.dump(lr, "clf.pkl")