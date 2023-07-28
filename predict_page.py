import streamlit as st
import  pickle
import numpy as np
from streamlit_folium import st_folium, folium_static
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import folium
from geopy.geocoders import ArcGIS
from sklearn import tree
from sklearn.model_selection import GridSearchCV,\
RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor



def load_model():
    with open('random.pkl','rb') as file:
        data =  pickle.load(file)
    return data

data = load_model()


regression_model = data["model"]
le_house_type = data["le_house_type"]
le_town = data["le_town"]
le_state = data["le_state"]

def show_predict_page():
    global P
    st.title("Nigerian house price prediction")

    st.write("""### Please fill the data with information""")

    house_types =( 'Detached Bungalow','Semi Detached Duplex','Terraced Duplexes','Detached Duplex','Block of Flats','Semi Detached Bungalow','Terraced Bungalow' )

    Abuja_town =('Lokogoma District','Gwarinpa','Katampe','Jahi',
                        'Guzape District',
                        'Life Camp',
                        'Gaduwa',
                        'Utako',
                        'Lugbe District',
                        'Kubwa',
                        'Galadimawa',
                        'Durumi',
                        'Mabushi',
                        'Wuye',
                        'Karmo',
                        'Mbora (Nbora)',
                        'Dakwo',
                        'Jabi',
                        'Kaura'
                        'Apo'
    )
    Anambara_town = ('Ibeju Lekki',
                        'Port Harcourt',
                        'Alimosho',
                        'Ibadan',
                        'Ajah',
                        'Ikorodu',
                        'Lokogoma District',
                        'Guzape District',
                        'Magboro'
    )
    Delta_town = ['Asaba']
    Edo_town = ['Oredo']
    Enugu_town = ['Enugu']
    Imo_town = ['Owerri Municipal']
    Lagos_town = ('Lekki',
                    'Victoria Island (VI)',
                    'Magodo',
                    'Ajah',
                    'Agege',
                    'Isheri North',
                    'Ojodu',
                    'Ikeja',
                    'Isolo'
    )
    Ogun_town = ('Sango Ota',
                    'Mowe Ofada',
                    'Mowe Town',
                    'Magboro',
                    'Isheri North',
                    'Arepo',
                    'Ifo'
    )
    Oyo_town = ['Ibadan']
    Rivers_town =['Port Harcourt']


    bedroom = st.slider("number of bedrooms",0,6,1)
    bathroom = st.slider("number of bathrooms",0,6,1)
    house_type = st.selectbox("house_type",house_types)
    state = st.selectbox("state",options=['Abuja', 'Anambara', 'Delta', 'Edo', 'Enugu', 'Imo', 'Lagos',
       'Ogun', 'Oyo', 'Rivers'])

    if state == 'Abuja':
        town = st.selectbox("town", Abuja_town)
    if state == 'Anambara':
        town = st.selectbox("town", Anambara_town)
    if state == 'Delta':
        town = st.selectbox("town", Delta_town)
    if state == 'Edo':
        town = st.selectbox("town", Edo_town)
    if state =='Enugu':
        town = st.selectbox("town", Enugu_town)
    if state == 'Imo':
        town = st.selectbox("town", Imo_town)
    if state == 'Lagos':
        town = st.selectbox("town", Lagos_town)
    if state == 'Ogun':
        town = st.selectbox("town", Ogun_town)
    if state == 'Oyo':
        town = st.selectbox("town", Oyo_town)
    if state == 'Rivers':
        town = st.selectbox("town", Rivers_town)




    ok = st.button("calculate house price")
    if ok:
        x = np.array([[bedroom, bathroom, house_type, town, state]])
        x[:, 2] = le_house_type.transform(x[:, 2])
        x[:, 3] = le_town.transform(x[:, 3])
        x[:, 4] = le_state.transform(x[:, 4])
        P = x.astype(float)

        House_price = regression_model.predict(P)
        st.subheader(f"The estimated house price ₦{House_price[0]/1e6:.1f} million Naira")

    if st.button('get location'):
        # center on Liberty Bell, add marker
        nom = ArcGIS()
        s = nom.geocode(str(town) + ',' + str(state) + ',' + 'Nigeria')
        m = folium.Map(location=[s.latitude, s.longitude], zoom_start=12)
        folium.Marker(
            [s.latitude, s.longitude], popup=f'{house_type,town, state}', ).add_to(m)
        House_price = regression_model.predict(P)
        st.subheader(f"The estimated house price ₦{House_price[0] / 1e6:.1f} million Naira")
        # call to render Folium map in Streamlit
        folium_static(m, width=800, height=500)