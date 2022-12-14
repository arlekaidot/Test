import streamlit as st
import pandas as pd
import joblib

st.header("Covid-19 Positivity Rate Prediction Tool")

#"TotalCases", "NewCases", "LocalTransmission", "ReturningResidents", "ROFW", "Deaths", "APOR", "LDOrder", "Increaseincases", "IncreaseinPositivityRate"

# Input bar 1
TotalCases = st.number_input("Total Cases",)

# Input bar 2
NewCases = st.number_input("New Cases")

# Input bar 2
LocalTransmission = st.number_input("Local Transmissions")

# Input bar 2
ReturningResidents = st.number_input("Returning Residents")

# Input bar 2
ROFW = st.number_input("ROFWs")

# Input bar 2
Deaths = st.number_input("Deaths")

# Input bar 2
APOR = st.number_input("Authorized Personnel Outside Residence")

# Input bar 2
LDOrder = st.selectbox("Select Lockdown Order", ("Travel Protocol", "General", "Modified", "Enhanced"))

# Input bar 2
Increaseincases = st.number_input("Enter Increase in Cases")

# Input bar 2
IncreaseinPositivityRate = st.number_input("Enter Increase in Positivity Rate")



# If button is pressed
if st.button("Submit"):
    # Unpickle classifier
    clf = joblib.load("clf.pkl")

    # Store inputs into dataframe
    x = pd.DataFrame([[TotalCases, NewCases, LocalTransmission, ReturningResidents, ROFW, Deaths, APOR, LDOrder, Increaseincases, IncreaseinPositivityRate]],
                     columns=["TotalCases", "NewCases", "LocalTransmission", "ReturningResidents", "ROFW", "Deaths", "APOR", "LDOrder", "Increaseincases", "IncreaseinPositivityRate"])
    x = x.replace(["Travel Protocol", "General", "Modified", "Enhanced"], [0, 1, 2, 3])


    # Get prediction
    prediction = clf.predict(x)[0]

    # Output prediction
    st.text(f"The predicted Positivity Rate is: {prediction}")