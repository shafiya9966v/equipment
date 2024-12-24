import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
pd1 = pd.read_csv("./equipment_anomaly_data.csv")
pd1
import joblib # Edit
# model = joblib.load("svm_classifier.pkl")  # Provide the correct path
st.title("Predictive Maintenance: Alerts and Graphs")
st.write("""
This application displays performance trends and failure alerts from sensor data. 
It helps monitor equipment health and predict maintenance needs.
""")
st.sidebar.header("Upload Sensor Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

# Visualization settings
st.sidebar.header("Graph Settings")
time_column = st.sidebar.text_input("Time Column (e.g., 'faulty')", "faulty")

# Main section
if uploaded_file:
    # Load data
    data = pd.read_csv(uploaded_file)
    st.header("Uploaded Data")
    st.write(data.head())

    # Ensure the time column exists
    if time_column not in data.columns:
        st.error(f"Time column '{time_column}' not found in the data.")
    else:
        # Convert time column to datetime
        data[time_column] = pd.to_datetime(data[time_column])

        # Display performance trends
        st.subheader("Performance Trends")
        numeric_columns = data.select_dtypes(
            include=np.number).columns.tolist()

        if numeric_columns:
            selected_metric = st.selectbox(
                "Select a Metric to View Trends", numeric_columns)
            fig = px.line(data, x=time_column, y=selected_metric,
                          title=f"Trend: {selected_metric}")
            st.plotly_chart(fig)
        else:
            st.warning("No numeric columns found for trend visualization.")

        # Alerts for potential failures
        alert_threshold = 0.7 # Edit
        st.subheader("Failure Alerts")
        if "failure_probability" in data.columns:
            alerts = data[data["failure_probability"] >= alert_threshold]
            if not alerts.empty:
                st.warning(
                    "⚠ Potential issues detected in the following records:")
                st.write(alerts)

                # Plot failure probabilities
                fig = px.scatter(alerts, x=time_column, y="failure_probability", color="failure_probability",
                                 title="Failure Probability Over Time", labels={"failure_probability": "Failure Probability"})
                st.plotly_chart(fig)
            else:
                st.success("No critical issues detected.")
        else:
            st.error(
                "Column 'failure_probability' not found in the data. Cannot generate alerts.")

else:
    st.info("Upload a CSV file to view alerts and graphs.")
# Prediction for single input

st.title("Faulty Equipment Detection")
st.write("Detect whether equipment is faulty based on sensor data.")

# Option for input: Single input or upload dataset
st.sidebar.header("Input Method")
input_method = st.sidebar.radio(
    "Select Input Method", ("Manual Input", "Upload Dataset"))

# Single Equipment Input
if input_method == "Manual Input":
    st.subheader("Enter Sensor Readings")
    vibration = st.number_input(
        "Vibration Level", min_value=0.0, max_value=100.0, value=50.0)
    temperature = st.number_input(
        "Temperature (°C)", min_value=-10.0, max_value=150.0, value=30.0)
    humidity = st.number_input(
        "Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
    pressure = st.number_input(
        "Pressure (kPa)", min_value=0.0, max_value=200.0, value=100.0)
    if st.button("Check Equipment Status"):
        input_data = pd.DataFrame([[vibration, temperature, humidity, pressure]],
                                  columns=["vibration", "temperature", "humidity", "pressure"])
        # prediction = svm_classifier.predict(input_data)[0]
        # status = "Faulty" if prediction == 1 else "Healthy"
        # st.write(f"The equipment status is: *{status}*")
        if input_method == "manual Input":
            st.warning("This equipment needs maintenance.")
        else:
            st.success("This equipment is functioning well.")

# Batch Input via Dataset
elif input_method == "Upload Dataset":
    st.subheader("Upload a Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Load dataset
        input_df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(input_df)

        # Ensure columns match the model's input features
        required_columns = ["vibration", "temperature", "humidity", "pressure"]
        if all(col in input_df.columns for col in required_columns):
            # Predict for batch input
            predictions = model.predict(input_df[required_columns])
            input_df["Status"] = ["Faulty" if pred ==
                                  1 else "Healthy" for pred in predictions]

            # Display predictions
            st.subheader("Predicted Equipment Status")
            st.dataframe(
                input_df[["vibration", "temperature", "humidity", "pressure", "Status"]])

            # Count faulty equipment
            faulty_count = sum(predictions)
            total_count = len(predictions)
            st.write(f"Total Equipment: {total_count}")
            st.write(f"Faulty Equipment: {faulty_count}")

            # Highlight maintenance insights
            if faulty_count > 0:
                st.warning("Some equipment needs maintenance.")
            else:
                st.success("All equipment is functioning well.")
        else:
            st.error(f"Uploaded file is missing required columns: {required_columns}")


def load_data():
    np.random.seed(42)
    data = {
        "Humidity": np.random.uniform(20, 100, 100),
        "Pressure": np.random.uniform(50, 150, 100),
        "Temperature": np.random.uniform(20, 120, 100),
        "Vibration": np.random.uniform(5, 100, 100),
    }


data = {
    'equipment': ['turbine', 'pump', 'compressor', 'turbine', 'pump', 'compressor', 'turbine', 'pump'],
    'vibration': [2.1, 3.2, 1.5, 2.3, 3.0, 1.8, 2.1, 3.2]
}

pd2 = pd.DataFrame(data)

# Count plot to visualize equipment types and vibration
plt.figure(figsize=(8, 6))
pd2['vibration_level'] = pd.cut(pd2['vibration'], bins=3, labels=["Low", "Medium", "High"]) # Edit
sns.countplot(x='equipment', hue='vibration', data=pd2, palette='viridis')

plt.title('Count Plot of Equipment Types with Vibration')
plt.xlabel('Equipment Type')
plt.ylabel('Count')
plt.legend(title='Vibration')
plt.show()
# Sidebar options for the user to select features
st.sidebar.header("Graph Options")
x_axis = st.sidebar.selectbox("Select X-axis:", pd1.columns)
y_axis = st.sidebar.selectbox("Select Y-axis:", pd1.columns)

# Filter options for the graph
st.sidebar.header("Filter Options")
humidity_filter = st.sidebar.slider("Filter Humidity:", min_value=float(pd1["humidity"].min()), max_value=float(
    pd1["humidity"].max()), value=(float(pd1["humidity"].min()), float(pd1["humidity"].max())))
pressure_filter = st.sidebar.slider("Filter Pressure:", min_value=float(pd1["pressure"].min()), max_value=float(
    pd1["pressure"].max()), value=(float(pd1["pressure"].min()), float(pd1["pressure"].max())))

# Apply filters
filtered_df = pd1[
    (pd1["humidity"] >= humidity_filter[0]) &
    (pd1["humidity"] <= humidity_filter[1]) &
    (pd1["pressure"] >= pressure_filter[0]) &
    (pd1["pressure"] <= pressure_filter[1])
]

# Show filtered data
st.subheader("Filtered Data")
st.write(filtered_df)
# Plot using Matplotlib
st.subheader("Graph Using Matplotlib")
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(filtered_df[x_axis], filtered_df[y_axis], color="blue", alpha=0.7)
ax.set_xlabel(x_axis)
ax.set_ylabel(y_axis)
ax.set_title(f"{y_axis} vs {x_axis}")
st.pyplot(fig)

# Plot using Plotly
st.subheader("Interactive Graph Using Plotly")
fig_plotly = px.scatter(filtered_df, x=x_axis, y=y_axis, color="humidity",
                        size="pressure", title=f"{y_axis} vs {x_axis} (Interactive)")
st.plotly_chart(fig_plotly)
