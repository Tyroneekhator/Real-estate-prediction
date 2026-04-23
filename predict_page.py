<<<<<<< HEAD
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
=======
import joblib
import numpy as np
import streamlit as st
import folium
from geopy.geocoders import ArcGIS
from streamlit_folium import folium_static


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


def build_feature_array(bedroom, bathroom, house_type, town, state, encoders):
    le_house_type = encoders["le_house_type"]
    le_town = encoders["le_town"]
    le_state = encoders["le_state"]

    x = np.array([[bedroom, bathroom, house_type, town, state]], dtype=object)
    x[:, 2] = le_house_type.transform(x[:, 2])
    x[:, 3] = le_town.transform(x[:, 3])
    x[:, 4] = le_state.transform(x[:, 4])
    return x.astype(float)


def show_predict_page():
    data = load_model()

    regression_model = data["model"]
    encoders = {
        "le_house_type": data["le_house_type"],
        "le_town": data["le_town"],
        "le_state": data["le_state"],
    }

    house_types = data["house_types"]
    states = data["states"]
    towns_by_state = data["towns_by_state"]

    st.markdown(
        """
        <div class="app-hero">
            <h1 style="margin:0 0 8px 0;">Estimate Property Value</h1>
            <p class="small-muted" style="margin:0;">
                Enter the property details below to generate a house price estimate
                based on learned patterns in the dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Property Details</div>', unsafe_allow_html=True)

    bedroom = st.slider("Number of bedrooms", 0, 10, 3)
    bathroom = st.slider("Number of bathrooms", 0, 10, 3)
    house_type = st.selectbox("House type", house_types)
    state = st.selectbox("State", states)

    available_towns = towns_by_state.get(state, [])
    town = st.selectbox("Town", available_towns)

    predict_clicked = st.button("Estimate Price", use_container_width=True)

    if predict_clicked:
        try:
            features = build_feature_array(
                bedroom, bathroom, house_type, town, state, encoders
            )
            predicted_price = float(regression_model.predict(features)[0])

            st.session_state["predicted_price"] = predicted_price
            st.session_state["selected_inputs"] = {
                "bedroom": bedroom,
                "bathroom": bathroom,
                "house_type": house_type,
                "town": town,
                "state": state,
            }
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    if "predicted_price" in st.session_state:
        selected = st.session_state["selected_inputs"]

        st.markdown(
            f"""
            <div class="result-card" style="margin-top:16px;">
                <p style="margin:0 0 10px 0; opacity:0.9;">Estimated Property Value</p>
                <h2 style="margin:0;">₦{st.session_state["predicted_price"]:,.0f}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="app-card" style="margin-top:16px;">
                <div class="section-title">Location</div>
                <p><strong>Town:</strong> {selected["town"]}</p>
                <p><strong>State:</strong> {selected["state"]}</p>
                <p><strong>House Type:</strong> {selected["house_type"]}</p>
                <p><strong>Layout:</strong> {selected["bedroom"]} bedroom(s), {selected["bathroom"]} bathroom(s)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        try:
            geolocator = ArcGIS()
            location_query = f'{selected["town"]}, {selected["state"]}, Nigeria'
            location = geolocator.geocode(location_query)

            st.markdown(
                """
                <div class="app-card" style="margin-top:16px;">
                    <div class="section-title">Map View</div>
                """,
                unsafe_allow_html=True,
            )

            if location is None:
                st.warning("The selected location could not be found on the map.")
            else:
                m = folium.Map(
                    location=[location.latitude, location.longitude],
                    zoom_start=12,
                )
                
                marker_text = (
                    f'House Type: {selected["house_type"]} | '
                    f'Town: {selected["town"]} | '
                    f'State: {selected["state"]}'
                )
                
                folium.Marker(
                        [location.latitude, location.longitude],
                        popup=marker_text,
                        tooltip=marker_text,
                ).add_to(m)

                folium_static(m, width=1100, height=450)

            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Could not load the map: {e}")

    else:
        st.markdown(
            """
            <p class="small-muted" style="margin-top:12px;">
                Click <strong>Estimate Price</strong> to see the estimated property value,
                location details, and map below.
            </p>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <p class="small-muted" style="margin-top:12px;">
            The estimate is based on historical housing data and should be treated
            as a guide, not a final market valuation.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)
>>>>>>> b8b30a8 (Initial commit)
