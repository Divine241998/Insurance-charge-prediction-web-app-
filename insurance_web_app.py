import streamlit as st
import numpy as np
import pandas as pd
import pickle

#creating a function for prediction
def insurance_prediction(input_data):
    loaded = pickle.load(open("insurance_prediction.pkl","rb")) #for the model
    loaded2 = pickle.load(open("preprocess.pkl","rb")) #for scaling
    new_x = loaded2.transform(input_data)
    predicted_value = loaded.predict(new_x)[0]
    return predicted_value



def main():

    st.title("Insurance Charges Web App")
    age_input = st.text_input("Age","please type age in numbers ...")
    gender_input = st.selectbox("Gender",["male","female"])
    bmi_input = st.text_input("BMI","please input digits..." )
    children = st.text_input("Children","what is the number of children you have?")
    smoker = st.radio("smoker",("yes","no"))
    region = st.selectbox("Region",['southwest', 'southeast', 'northwest', 'northeast'])


    insurance = ""
    if st.button("Estimate insurance charges"):
        insurance = insurance_prediction(pd.DataFrame([{"age":float(age_input),"sex":gender_input,"bmi":float(bmi_input),"children": float(children),"smoker":smoker,"region":region}]))
        
    
    return st.success(insurance)

if __name__ == '__main__':
    main()
