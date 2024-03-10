#Importing dependencies 
import random
import csv
import streamlit as st
import plost
import pandas as pd
import numpy as np
import altair as alt

from datetime import datetime
from faker import Faker
from faker.providers import BaseProvider


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


#________Uploading data_________

# Define column names
columns = ["Systolic_BP", "Diastolic_BP", "Glucose(mg/dL)", "Insulin (mmol–¹)"]

# Generate random numbers for DataFrame
biochem_data = np.random.randint(40, 250, size=(60, len(columns)))
dka_risk = np.random.randint(0, 2, size=(60, 1))


# Map 0 to 'No Risk' and 1 to 'At Rísk'
dka_risk = np.where(dka_risk == 0, 'No Risk', 'At Risk')

# Combine the biochem_data with the dka_risk
data_combined = np.concatenate((biochem_data, dka_risk), axis=1)

# Create DataFrame without indexes
df = pd.DataFrame(data_combined, columns=columns + ['DKA Risk'], index=None)




#Convert Systolic_BP, Diastolic_BP, Glucose, Insulin to numerical data
data = df[["Systolic_BP", "Diastolic_BP","Glucose(mg/dL)", "Insulin (mmol–¹)"]].astype('int64')
data = pd.DataFrame(data)


#Convert Diastolic_BP to 2/3 Systolic_BP
data["Diastole"] = data["Systolic_BP"]*2/3
data["Diastolic_BP"] = data["Diastole"]
data["Diastolic_BP"] = np.ceil(data["Diastolic_BP"])
data.drop(columns=["Diastole"], inplace=True)

#Add "DKA Risk" column to dataset
data = data.join(df["DKA Risk"])



#Glucose level
glucose_value = data.iloc[59:60, 2].values[0]
glucose_str = str(glucose_value)+ "mg/dL"


#Blood pressure (systolic blood pressure, diastolic blood pressure)
sbp_value = data.iloc[59:60, 0].values[0]
dbp_value = data.iloc[59:60, 1].values[0]
sbp_str = str(sbp_value)
dbp_str = str(dbp_value)

#Blood pressure value
BP = sbp_str + "/" + dbp_str+ " mmHg"



#____Logistic Regression Model To Predict Risk Of DKA Occuring_____
# Split data into features(X) and target(y)
X = data.iloc[:, 0:4].values
y = data.iloc[:, 4].values


# Create training and test sets from data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize model
model = LogisticRegression()

# Fit the model
model.fit(X_train, y_train)

# Make predictions using test sets
y_pred = model.predict(X_test)


#Evaluate LogisticRegression Model
accuracy = accuracy_score(y_test, y_pred)

#Model accuracy 
#print("Model Accuracy:", accuracy)

#Predict risk of DKA
risk = y_pred[-1]






# Function to create donut chart for countdown timer
def create_donut_chart(seconds_remaining, total_seconds):
    data = pd.DataFrame({
        'time': [seconds_remaining, total_seconds - seconds_remaining],
        'category': ['Remaining', 'Elapsed']
    })

    chart = alt.Chart(data).mark_arc().encode(
        theta='time:Q',
        color='category:N',
        tooltip=['category:N', 'time:Q']
    ).properties(
        width=300,
        height=300
    )

    return chart

#Daika User Interface 

#Main function
def main():
    #Blood glucose metrics
    st.title("Daika")
    st.markdown("### Your Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric(label = "Glucose levels", value = glucose_str , delta = "1.2")
    col2.metric(label = "DKA", value = risk , delta = "-0.002")
    col3.metric(label = "B.P", value = BP)



    #Blood pressure metrics
    st.markdown("### Blood Pressure Chart")
    bp_chart = pd.DataFrame(data, columns = ["Systolic_BP","Diastolic_BP"])
    st.line_chart(bp_chart)

    st.markdown("Inject Insulin In: ")

    st.sidebar.title("Daika")
    total_seconds = st.sidebar.number_input("Set total duration (seconds):", min_value=1, value=21600)

    start_time = st.session_state.get('start_time', datetime.now())
    st.session_state.start_time = start_time

    elapsed_time = (datetime.now() - start_time).total_seconds()
    seconds_remaining = max(total_seconds - elapsed_time, 0)

    st.write(f"Time remaining: {int(seconds_remaining)} seconds")

    # Create countdown timer in donut shape
    st.altair_chart(create_donut_chart(seconds_remaining, total_seconds), use_container_width=True)

    # Update countdown timer every second
    if seconds_remaining > 0:
        st.experimental_rerun()


if __name__ == "__main__":
    main()
