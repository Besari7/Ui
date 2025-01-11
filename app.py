import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# Load the dataset
data = pd.read_csv("student_performance_dataset_fixed.csv")

# Convert Final_Grade to numerical value
grade_map = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
data["Final_Grade_Num"] = data["Final_Grade"].map(grade_map)

# Remove unnecessary columns
data = data.drop(["Student_ID", "Final_Grade"], axis=1)

# Features and target variable
X = data.drop("Final_Grade_Num", axis=1)
y = data["Final_Grade_Num"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Numerical and categorical features
numeric_features = [
    "Age",
    "GPA",
    "Attendance_Rate",
    "Participation_Rate",
    "Homework_Completion_Rate",
    "Midterm_Score",
    "Final_Score",
    "Study_Hours_Per_Week",
    "Extra_Curricular_Activities",
]
categorical_features = ["Gender", "Major", "Part_Time_Job"]

# Preprocessing steps
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
categorical_transformer = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Model (Random Forest)
model = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier())]
)

# Train the model
model.fit(X_train, y_train)

# Streamlit application
st.title("Student Performance Prediction")

# Input fields
gender = st.selectbox("Gender", X["Gender"].unique())
age = st.number_input("Age", min_value=18, max_value=30, value=20)
major = st.selectbox("Major", X["Major"].unique())
gpa = st.number_input("GPA", min_value=0.0, max_value=4.0, value=2.5, step=0.1)
attendance_rate = st.number_input(
    "Attendance Rate (%)", min_value=0, max_value=100, value=80, step=1
)
participation_rate = st.number_input(
    "Participation Rate (%)", min_value=0, max_value=100, value=50, step=1
)
homework_completion_rate = st.number_input(
    "Homework Completion Rate (%)", min_value=0, max_value=100, value=70, step=1
)
midterm_score = st.number_input(
    "Midterm Score", min_value=0, max_value=100, value=60, step=1
)
final_score = st.number_input(
    "Final Score", min_value=0, max_value=100, value=70, step=1
)
study_hours_per_week = st.number_input(
    "Study Hours Per Week", min_value=0, max_value=50, value=10, step=1
)
extra_curricular_activities = st.number_input(
    "Extra Curricular Activities", min_value=0, max_value=5, value=2, step=1
)
part_time_job = st.selectbox("Part-Time Job", X["Part_Time_Job"].unique())

# Prediction button
if st.button("Predict"):
    # Convert user inputs to a DataFrame
    input_data = pd.DataFrame(
        {
            "Gender": [gender],
            "Age": [age],
            "Major": [major],
            "GPA": [gpa],
            "Attendance_Rate": [attendance_rate],
            "Participation_Rate": [participation_rate],
            "Homework_Completion_Rate": [homework_completion_rate],
            "Midterm_Score": [midterm_score],
            "Final_Score": [final_score],
            "Study_Hours_Per_Week": [study_hours_per_week],
            "Extra_Curricular_Activities": [extra_curricular_activities],
            "Part_Time_Job": [part_time_job],
        }
    )

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display the result
    st.write("Predicted Grade:", {4: "A", 3: "B", 2: "C", 1: "D", 0: "F"}[prediction])
