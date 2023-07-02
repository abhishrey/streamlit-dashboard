import pandas as pd
import streamlit as st

# Load the CSV data

st.title("Student Assignment Submission Data for students last submission on same hour")
data = pd.read_csv("c:/OVGU\DBSC/final_data.csv")  # Replace "your_data.csv" with the actual file name

# Convert eq_ts column to datetime format
data["eq_ts"] = pd.to_datetime(data["eq_ts"])

# Create the initial selectbox for egrp_instructor
selected_instructor = st.selectbox("Select Instructor", data["egrp_instructor"].unique())

# Filter the data based on the selected instructor
filtered_data = data[data["egrp_instructor"] == selected_instructor]

# Create the selectbox for eq_taskid
selected_taskid = st.selectbox("Select Task ID", filtered_data["eq_taskid"].unique())

# Filter the data based on the selected task ID
filtered_data = filtered_data[filtered_data["eq_taskid"] == selected_taskid]

# Find eq_UserID, SQL_category, count(eq_ts), first_submission_time, and last_submission_time
filtered_data = filtered_data.groupby(["eq_UserID", "SQL_category"]).agg(
    Total_Attempt=("eq_ts", "count"),
    first_submission_time=("eq_ts", "min"),
    last_submission_time=("eq_ts", "max")
)

# Filter the data where last_submission_time of minimum 2 eq_UserID lies in the same hour of the same date
filtered_data = filtered_data.groupby(pd.Grouper(freq="H", key="last_submission_time")).filter(lambda x: x.shape[0] >= 2)

# Show the resulting table
st.table(filtered_data)
