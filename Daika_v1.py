import streamlit as st
import plost
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder




#Daika Dashboard

#__________Uploading Data__________

# Define column names
columns = ["Systolic_BP", "Diastolic_BP", "Glucose(mg/dL)"]

# Generate random numbers for DataFrame
biochem_data = np.random.randint(40, 250, size=(10000, len(columns)))

#Convert biochem_data to a Dataframe
data = pd.DataFrame(biochem_data, columns = columns)



#Refine Systolic_BP and Diastolic_BP data

df = data[["Systolic_BP", "Diastolic_BP","Glucose(mg/dL)"]].astype('int64')
df = pd.DataFrame(df)

#Convert Diastolic_BP to 2/3 Systolic_BP
df["Diastole"] = df["Systolic_BP"]*2/3
df["Diastolic_BP"] = df["Diastole"]
df["Diastolic_BP"] = np.ceil(df["Diastolic_BP"])

df.drop(columns=["Diastole"], inplace=True)




#Data for Ketone values

ketone_data = np.random.randint(0.02,30.8, size = (10000))
#ketone


#_______Convert ketone_data to a dataframe_____
ketone_pd = pd.DataFrame(ketone_data, columns=['ketone_value'])

# Initialize an empty list to store messages
messages = []

for k in ketone_pd['ketone_value']:
    if k < 12.4:
        messages.append("No DKA Risk")
    elif 12.4 <= k < 15.5:
        messages.append("Moderate Risk Of DKA")
    elif 15.5 <= k <=20:
        messages.append("DKA")
    elif k > 20:
        messages.append("DKA...High Ketone Levels Detected")
    else:
        messages.append(" ")

# Add the messages list as a new column to the DataFrame
#ketone_pd["message"] = messages
ketone_pd["Risk"] = messages


# Display dataFrame of ketone values
#ketone_pd



#___Join biochem data & ketone data to form a single data frame___
df = df.join(ketone_pd)


#________PARAMETERS TO MONITOR__________


# Blood Pressure
sbp_value = df.iloc[59:60, 0].values[0]
dbp_value = df.iloc[59:60, 1].values[0]
sbp_str = str(sbp_value)
dbp_str = str(dbp_value)
BP = sbp_str + "/" + dbp_str+ "mmHg"


# Glucose Levels
glucose_value = df.iloc[59:60, 2].values[0]
glucose_str = str(glucose_value)+ "mg/dL"







# Ketone Levels
ketone_level = df.iloc[59:60, 3].values[0]
ketone_str = str(ketone_level) + "mmol/L"

# ____Differences in ketone values at 60sec and 59sec _____

ketone_val = []  # Initialize an empty list
# Extracting the ketone values at 59secs and 60secs
ketone59 = int(df["ketone_value"].iloc[59])
ketone60 = int(df["ketone_value"].iloc[60])

delta = ketone60 - ketone59
delta_str = str(delta)

if ketone60  > ketone59:
    message = "+" + delta_str + "units"

elif ketone60 < ketone59:
    message = delta_str + "units"

else:
    print("")




message = message









#____Logistic Regression Model To Predict Risk Of DKA Occuring_____

# Split data into features(X) and target(y)
X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values


# Create training and test sets from data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
model = LogisticRegression()

# Fit the model
model.fit(X_train, y_train)

# Make predictions using test sets
y_pred = model.predict(X_test)


#Evaluate LogisticRegression Model

accuracy = accuracy_score(y_test, y_pred)
#print("Model Accuracy:", accuracy)
risk = y_pred[-1]






#_______DAIKA USER INTERFACE_______



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



#Main function
def main():
    #Blood glucose metrics
    st.markdown("### Your Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label = "Glucose levels", value = glucose_str , delta = "1.2")
    col2.metric(label = "DKA", value = risk , delta = "-0.002")
    col3.metric(label = "B.P", value = BP)
    col4.metric(label = "Ketones", value = ketone_str, delta = message)


    #Blood pressure metrics
    st.markdown("### Blood Pressure Chart")
    bp_chart = pd.DataFrame(data, columns = ["Systolic_BP","Diastolic_BP"])
    st.line_chart(bp_chart)

    st.markdown("Inject In ")

    total_seconds = st.sidebar.number_input("Set total duration (seconds):", min_value=1, value=60)

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
