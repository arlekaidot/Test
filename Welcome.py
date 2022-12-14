import pandas as pd #visualization
from sklearn.model_selection import train_test_split #Train/Test/Split
from sklearn.linear_model import LinearRegression #Actual Algorithm
import matplotlib.pyplot as plt #To plot data
from sklearn.metrics import r2_score, mean_squared_error
import streamlit as st
import datetime
import seaborn as sn
from math import sqrt





lr = LinearRegression()
df = pd.read_csv('Covid.csv')
df.head()


df['Month'] = df['Month'].apply(lambda x : datetime.datetime.strptime(x, "%Y-%m-%d"))
df = df.drop('Month', axis=1)



st.title("Development of Multiple Linear Regression with Covid-19 data")
st.markdown("""
	The data set gathered contains important data gathered from the official Department of Health
	Website as well as other variables used from Iloilo City Covid-19 Emergency Operations Center Page.
	The data we have gathered here contains 45 weeks of data due to the fact many of these datasets were
	omitted over time. The data sets gathered here are as follows:
	### Data Description
	-   Positivity (Retrieved from DOH)
	-   Total Cases
	-   New Cases
	-   Local Transmissions
	-   Returning Residents
	-   ROFW (Returning Overseas Filipino Workers)
	-   Deaths
	-   APOR (Authorized Personnel Outside Residence)
	-   Lockdown Order
	-   Notes (Types of Lockdowns)
	-   Increase in Cases (Compared to previous day's data)
	-   Increase in Positivity Rates (Compared to previous day's data)
""")
st.sidebar.title("Dataset Options for Viewing")


q1 = st.sidebar.checkbox("Show Raw Data", False)
q2 = st.sidebar.checkbox("Show Positivity Rates Only", False)
q3 = st.sidebar.checkbox("Show Intercept", False)
q4 = st.sidebar.checkbox("Show Coefficients", False)
q5 = st.sidebar.checkbox("Show Trained Data Prediction", False)
q6 = st.sidebar.checkbox("Show Trained Data r2 score", False)
q7 = st.sidebar.checkbox("Show Test Data Prediction", False)
q8 = st.sidebar.checkbox("Show Test Data r2 score", False)
q9 = st.sidebar.checkbox("Show Heatmap of Correlating Variables", False)
q10 = st.sidebar.checkbox("Show RMSE and MSE Scores", False)




DATA_URL = ('Covid.csv')
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data
data = load_data(10000)


if q1:
    st.subheader('Raw data')
    st.write(data)

x = df[["TotalCases", "NewCases", "LocalTransmission", "ReturningResidents", "ROFW", "Deaths", "APOR", "LDOrder", "Increaseincases", "IncreaseinPositivityRate"]]
y = df["Positivity"]

if q2:
    st.subheader("Positivity Rates Only")
    st.write(y)



x_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0) #80% to be trained
lr.fit(x_train, y_train)
LinearRegression()


c = lr.intercept_

if q3:
    st.subheader("Intercept")
    st.write(c)



m =  lr.coef_
print(m)

if q4:
    st.subheader("Coefficients")
    st.write(m)

y_pred_train = lr.predict(x_train)


if q5:
    st.subheader('Trained Data')
    plt.scatter(y_train, y_pred_train)
    plt.xlabel('Actual Positivity Rate')
    plt.ylabel('Predicted Positivity Rate')
    st.pyplot(plt)

if q6:
    st.subheader('Trained Data r2 score')
    score = r2_score(y_train, y_pred_train)
    st.write(score)


y_pred_test = lr.predict(X_test)

if q7:
    st.subheader('Test Data')
    plt.scatter(y_test, y_pred_test)
    plt.xlabel('Actual Positivity Rate')
    plt.ylabel('Predicted Positivity Rate')
    st.pyplot(plt)

if q8:
    st.subheader('Test Data r2 score')
    testscore = r2_score(y_test, y_pred_test)
    st.write(testscore)

if q9:
    st.subheader('Heatmap of Correlating Variables')
    fig, ax = plt.subplots()
    sn.heatmap(df.corr()[['Positivity']].sort_values(by='Positivity', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
    st.write(fig)


if q10:
    st.subheader("MSE and RMSE score of Trained Data")
    predtrain = lr.predict(x_train)
    msetrain = mean_squared_error(y_train, predtrain)
    rmsetrain = sqrt(msetrain)
    st.write("MSE Score")
    msetrain
    st.write("RMSE Score")
    rmsetrain


    st.subheader("MSE and RMSE score of Test Data")
    predtest = lr.predict(X_test)
    msetest = mean_squared_error(y_test, predtest)
    rmsetest = sqrt(msetest)
    st.write("MSE Score")
    msetest
    st.write("RMSE Score")
    rmsetest

