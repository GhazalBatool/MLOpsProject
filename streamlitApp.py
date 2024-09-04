import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st

model = joblib.load("liveModelV1.pkl")
model = pd.read_csv('mobile_price_rating.csv')
X = data.iloc[:,:-1]
Y = data.iloc[:,-1]
#test train split
X_train,X_test, Y_train, Y_test = train_test_split(X,Y, testsize= 0.2, random_state= 42)

#make preeiction
y_pred = model.predict(X_test)
#calculate accuarcy
accuracy = accuracy_score(Y_test, y_pred)
#page title
st.title("Model Accuracy and Real-Time Prediction")

#display Accuracy
st.write(f"Model {accuracy}")

#real time prediction based on user inputs
st.header("Real-Time Prediction")
input_data = []
for col in X_test.columns:
    input_value = st.number_input(f'Input for feature {col}', value=0)
    input_data.append(input_value)
#convert input data to dataframe
input_df = pd.DataFrame([input_data], columns= X_test.columns)

#Make Predictions
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.writes(f'Prediction: {prediction[0]}')
