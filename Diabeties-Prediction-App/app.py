import pandas as pd
import pickle
import time
import plotly.express as px
import streamlit as st
from PIL import Image

st.sidebar.header("User Input Parameters : ")
def user_input_features():
    Pregnancy = st.sidebar.slider('Pregnancy Period', 0, 12, 1)
    Glucose = st.sidebar.slider('Glucose',40, 220, 1)
    BloodPressure = st.sidebar.slider('BloodPressure',30, 200, 1)
    SkinThickness = st.sidebar.slider('SkinThickness',0, 120, 1)
    Insulin = st.sidebar.slider('Insulin',0, 120, 1)
    BMI = st.sidebar.slider('BMI',17, 50, 1)
    DiabetesPedigreeFunction = st.sidebar.slider('DiabetesPedigreeFunction',0.0, 3.0, 0.1)
    Age = st.sidebar.slider('Age',0.0, 100.0, 0.1)
    data={'Pregnancies': Pregnancy,
            'Glucose':Glucose,
            'BloodPressure': BloodPressure,
            'SkinThickness': SkinThickness,
            'Insulin': Insulin,
            'BMI': BMI,
            'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
            'Age': Age}
    data = pd.DataFrame(data, index=[0])
    return data

limit={'Pregnancies': 140,
            'Glucose':70,
            'BloodPressure': 160,
            'SkinThickness': 200,
            'Insulin': 370,
            'BMI': 100,
            'DiabetesPedigreeFunction': 100,
            'Age': 110}

st.write("## Diabeties Prediction App")

#opening the image
image = Image.open('app/data-science/Diabeties-Prediction-App/IMG.png')
image = image.resize((500,300))
#displaying the image on streamlit app

st.image(image, caption='Diabeties')

data = user_input_features()
st.write("---")
st.header("Your Parameters : ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
  # Update the progress bar with each iteration
  bar.progress(i + 1)
  time.sleep(0.01)
st.write(data)


df=pd.read_csv("app/data-science/Diabeties-Prediction-App/diabetes.csv")
def add_label(row):
    if row['Outcome']==1:
        return "Diabetic"
    else:
        return "Non-Diabetic"
df['label']=df.apply(add_label,axis=1)





import plotly.graph_objs as go


# Add marker at a specific value


## PREDICTION
st.write("---")
st.header("Prediction : ")
loaded_model = pickle.load(open("Diabeties.sav", 'rb'))
prediction=loaded_model.predict(data)

if(prediction[0]==0):
    prediction="Not Diabeties"
else:
    prediction="Diabeties"
st.write(prediction)

st.write("---")
st.header("General Discussion ")
st.write("Your Input Paameters positions are sh")

for i in df.columns[0:-2]:
    fig = px.histogram(df, x=i,color='label')

    fig.add_trace(go.Scatter(
        x=[data[i][0]],
        y=[limit[i]],
        text=[
              "Input Parameter value Position"],
        mode="text",
    ))

    fig.add_shape(type="line",
        x0=data[i][0], y0=0, x1=data[i][0], y1=limit[i],
        line=dict(color="White",width=3)
    )

    st.plotly_chart(fig, use_container_width=True)
