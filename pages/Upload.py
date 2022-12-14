import streamlit as st
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from math import sqrt
import pandas_profiling

if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar:
    st.title("Upload Own Files for Prediction")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling"])
    st.info("Please make sure categorical data is changed to numerical data beforehand")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)


if choice == "Modelling":
    chosen_target = st.selectbox('Choose Your Independent Variable', df.columns)
    x = df.drop([chosen_target],axis = 1).values
    chosen_size = st.slider('Select Test Size', 0.0, 1.0)
    if st.button('Run Modelling'):
        y = df[chosen_target]


        #Initializing Algorithm
        lr = LinearRegression()
        x_train, X_test, y_train, y_test = train_test_split(x, y, test_size=chosen_size, random_state=0)
        lr.fit(x_train, y_train)
        y_pred = lr.predict(X_test)
        y_pred_train = lr.predict(x_train)
        y_pred_test = lr.predict(X_test)



        #TRAINING DATA DETAILS
        st.subheader('Trained Data')
        plt.scatter(y_train, y_pred_train)
        plt.xlabel('Actual Positivity Rate')
        plt.ylabel('Predicted Positivity Rate')
        st.pyplot(plt)
        st.subheader('Details:')
        score = r2_score(y_train, y_pred_train)
        st.write("r2 score:", score)
        predtrain = lr.predict(x_train)
        msetrain = mean_squared_error(y_train, predtrain)
        rmsetrain = sqrt(msetrain)
        st.write("MSE score:", msetrain)
        st.write("RMSE score:", rmsetrain)


        #TEST DATA DETAILS
        st.subheader('Test Data')
        plt.scatter(y_test, y_pred_test)
        plt.xlabel('Actual Positivity Rate')
        plt.ylabel('Predicted Positivity Rate')
        st.pyplot(plt)
        st.subheader('Details:')
        testscore = r2_score(y_test, y_pred_test)
        st.write("r2 score:", testscore)
        predtest = lr.predict(X_test)
        msetest = mean_squared_error(y_test, predtest)
        rmsetest = sqrt(msetest)
        st.write("MSE score:", msetest)
        st.write("RMSE score:", rmsetest)



        #PREDICTIONS
        st.subheader('Predictions:')
        comparison = pd.DataFrame({"Actual value": y_test, "Predicted Value": y_pred, "Difference": y_test - y_pred})
        st.write(comparison)

