import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
import pybase64

st.set_page_config(page_title="Customer Churn Prediction - Broadband", page_icon="",layout="wide", initial_sidebar_state="expanded")
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return pybase64.b64encode(data).decode()
img = get_img_as_base64("bg.jpg")
st.markdown(
    """
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .sidebar .sidebar-content {
        width: 300px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{img}");
background-size: cover;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}

</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
st.title("Customer Churn Prediction - Broadband")


menu = option_menu(menu_title='',options=['Home','Contact'],orientation='horizontal',
        default_index=0, icons=['house','envelope'])
if menu == 'Home':
    st.markdown('__<p style="font-family: verdana; text-align:left; font-size: 15px; color: #FAA026">'
                'Our web application provides an interactive platform to predict broadband customer '
                'churn.</P>__', unsafe_allow_html=True)
    col1,col2,col3 = st.columns([0.25,1,0.75])
    with col2:
        age = st.slider(label='Age',min_value=18,max_value=70,value=18)
        bill = st.number_input(label='Avg Monthly Bill', min_value=30, max_value=100)
        #with col2:
        gen = st.radio(label='Gender',options=['Male', 'Female'])
        gender = 1 if gen == 'Male' else 0
        data = st.number_input(label='Avg Data Usage', min_value=50, max_value=500)

        #with col3:
        slm = st.number_input(label='Subscription_Length_Months', min_value=1, max_value=24)
        location = st.selectbox(label='Customer Location', options=['Houston', 'Los Angeles', 'Miami', 'Chicago', 'New York'])
        city = {
            'Houston': 0,
            'Los Angeles': 0,
            'Miami': 0,
            'Chicago': 0,
            'New York': 0
        }
        city[location] = 1
        predict = st.button(label='Predict')

        new_datapoint = [age,gender,slm,bill,data,city['Chicago'],city['Houston'],city['Los Angeles'],city['Miami'],city['New York']]
        #st.write(*new_datapoint)

        #loading pickled datapoints
        with open('scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)

        with open('xgboost_model.pkl', 'rb') as model_file:
            xgb_model = pickle.load(model_file)
        if predict:
            new_data = np.array(new_datapoint)
            # Reshape the input data to be 2D
            new_data = new_data.reshape(1, -1)
            new_data_scaled = scaler.transform(new_data)
            #st.write(new_data_scaled)

            predictions = xgb_model.predict(new_data_scaled)
            # 'predictions' will contain the predicted class (0 or 1) based on your XGBoost model
            print(predictions)
            if predictions[0] == 1:
                st.error("The likelihood of customer churn is high.")

            else:
                st.success("The likelihood of customer churn is less.")
elif menu == 'Contact':
    st.markdown('__<p style="text-align:left; font-size: 20px; color: #FAA026">Applications and Packages Used:</P>__',
                unsafe_allow_html=True)
    st.write("  * Python")
    st.write("  * Scikit-Learn")
    st.write("  * Pandas")
    st.write("  * Numpy")
    st.write("  * Pickle")
    st.write("  * Streamlit")
    st.write("  * Github")
    st.markdown(
        '__<p style="text-align:left; font-size: 20px; color: #FAA026">For feedback/suggestion, connect with me on</P>__',
        unsafe_allow_html=True)
    st.subheader("LinkedIn")
    st.write("https://www.linkedin.com/in/selvamani-a-795580266/")
    st.subheader("Email ID")
    st.write("selvamani.ind@gmail.com")
    st.subheader("Github")
    st.write("https://github.com/selvamani1992")