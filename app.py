import streamlit as st
import numpy as np
import pickle
loaded_model = pickle.load(open("trained_model.sav",'rb'))

def diabetes_pred(input_data):
    
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'
    
def main():
    st.title("Diabetes Prediction")
    age = st.text_input('Age')
    pregnancies = st.text_input('Number of pregnancies')
    glucose = st.text_input('Glucose Levels')
    bloodPressure = st.text_input('Blood Pressure')
    skinThickness = st.text_input("Skin Thickness")
    insulin = st.text_input('Insulin Levels')
    bmi = st.text_input("BMI")
    diabetesPedigreeFunction = st.text_input("Diabetes Pedigree")

    diagnosis = ''
    if st.button("Predict"):
        diagnosis = diabetes_pred([age,pregnancies,glucose,bloodPressure,skinThickness,insulin,
                                   bmi,diabetesPedigreeFunction])
    st.success(diagnosis)

if __name__ == '__main__':
    main()
