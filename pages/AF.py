import streamlit as st
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os
import matplotlib.pyplot as plt
import datetime
import pandas_profiling
import pyaf.ForecastEngine as autof
import numpy as np

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
    convert_date = st.selectbox('Choose Column with Date Configuration',df.columns)
    df[convert_date] = df[convert_date].apply(lambda x : datetime.datetime.strptime(x, "%Y-%m-%d"))

    if st.button('Run Modelling'):
        lEngine = autof.cForecastEngine()
        lEngine.train(df, convert_date, chosen_target, 12);
        lEngine.getModelInfo()
        df = lEngine.forecast(df, 12);
        df2 = df.fillna(0) #Changing NAN values to 0
        st.write(df2)

        forecast = df[[convert_date , chosen_target , 'Positivity_Forecast', "Positivity_Forecast_Upper_Bound", "Positivity_Forecast_Lower_Bound"]]
        df3 = forecast.fillna(0) #Changing NAN values to 0
        st.write(df3)

        st.line_chart(forecast, x =convert_date, y=[chosen_target, "Positivity_Forecast", "Positivity_Forecast_Upper_Bound", "Positivity_Forecast_Lower_Bound"])

