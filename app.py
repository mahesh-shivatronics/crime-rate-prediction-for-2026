import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Title
st.title("Crime Rate Prediction System")

# Load dataset
df = pd.read_csv("crime_data.csv")

# Training data
X = df[['Year', 'Population']]
y = df['CrimeRate']

# Train model
model = LinearRegression()
model.fit(X, y)

# User input
st.header("Enter Details")

year = st.number_input("Enter Year", min_value=2018, max_value=2050, value=2026)
population = st.number_input("Enter Population", min_value=1000, value=750000)

# Predict button
if st.button("Predict Crime Rate"):

    prediction = model.predict([[year, population]])

    st.success(f"Predicted Crime Rate: {int(prediction[0])}")

# Show dataset
st.subheader("Crime Dataset")
st.write(df)

# Graph
st.subheader("Crime Rate Graph")

fig, ax = plt.subplots()

ax.plot(df['Year'], df['CrimeRate'], marker='o')
ax.set_xlabel("Year")
ax.set_ylabel("Crime Rate")
ax.set_title("Crime Rate Over Years")

st.pyplot(fig)