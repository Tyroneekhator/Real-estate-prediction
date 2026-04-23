<<<<<<< HEAD
# To run the application

- Download and open on Python
- NOTE:  DO NOT TOUCH THE  JUYPTER NOTEBOOK FILE
- Streamlit run app.py

---

# Results
## Home screen
![home_screen](/homescreen.png)  
---
## estimate price and location
![estimate_price_with_location](/resultandlocation.png)  
---
## EDA house prices
![EDA_house_prices](/EDAhouseprices.png)  
=======
# Nigerian House Price Prediction App

This project is a Streamlit web application for predicting house prices in Nigeria and exploring housing price trends through simple visual analysis.

---

## To Run the Application

1. Download or clone the project folder.
2. Open the project in Python or your code editor of choice.
3. Create and activate a virtual environment.
4. Install the required dependencies.
5. Run the Streamlit app.

> **Note:** Do **not** modify the Jupyter Notebook file unless you specifically want to retrain or rework the model development process.

## Run Command

```bash
streamlit run app.py
```

---

## Project Overview

This application has two main parts:

- **Predict Page**  
  Allows the user to estimate house prices based on selected property features.

- **Explore Page**  
  Displays charts and visual analysis of house prices in Nigeria.

---

## Features

## House Price Prediction

The prediction page allows the user to provide information such as:

- Number of bedrooms
- Number of bathrooms
- House type
- Town
- State

The application then estimates the house price.

## Location Display

The app can also show the selected property location on a map.

## Exploratory Data Analysis (EDA)

The app includes charts that help users understand:

- The most expensive towns
- Average house prices by state
- Average house prices by house type
- Price patterns in the dataset

---

## Results

## Home Screen

![home_screen](/homescreen.png)

---

## Estimate Price and Location

![estimate_price_with_location](/resultandlocation.png)

---

## EDA House Prices

![EDA_house_prices](/EDAhouseprices.png)

---

## Important Note

- Do **not** touch the Jupyter Notebook file if your goal is only to run the app.
- To run the application successfully, make sure all required Python packages are installed.
- Run the app from the project root folder.

---

## Suggested Setup

## Create a Virtual Environment

```bash
python -m venv venv
```

## Activate the Virtual Environment

### On Windows PowerShell

```powershell
venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available yet, install the needed packages manually:

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn geopy folium streamlit-folium joblib
```

---

## Start the Application

```bash
python -m streamlit run app.py
```

---

## Files Used in the Project

- `app.py` — main Streamlit app entry point
- `predict_page.py` — price prediction page
- `explore_page.py` — exploratory analysis page
- `nigeria_houses_data.csv` — dataset used in the app
- `README.md` — project documentation
- `Nigeria (1).ipynb` — notebook used during model development

---

## Output

After running the command, Streamlit will open the application in your browser, usually at:

```text
http://localhost:8501
```

---

## Summary

This project is a simple real estate prediction app built with Streamlit. It allows users to estimate Nigerian house prices, view property locations, and explore housing data visually.
>>>>>>> b8b30a8 (Initial commit)
