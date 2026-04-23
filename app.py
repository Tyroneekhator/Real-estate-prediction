import streamlit as st
<<<<<<< HEAD
from predict_page import  show_predict_page
from explore_page import show_explore_page

page = st.sidebar.selectbox("Explore or Predict",("Predict","Explore"))

if page =="Predict":
    show_predict_page()
elif page =="Explore":
    show_explore_page()



=======
from predict_page import show_predict_page
from explore_page import show_explore_page


st.set_page_config(
    page_title="Nigerian House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_global_css():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #F8FAFC;
            color: #0F172A;
        }

        section[data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid #E5E7EB;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1350px;
        }

        .sidebar-brand {
            background: linear-gradient(135deg, #166534 0%, #0F766E 100%);
            color: white;
            padding: 18px 16px;
            border-radius: 16px;
            margin-bottom: 20px;
        }

        .sidebar-brand h2 {
            margin: 0 0 6px 0;
            font-size: 1.15rem;
        }

        .sidebar-brand p {
            margin: 0;
            font-size: 0.9rem;
            opacity: 0.92;
            line-height: 1.4;
        }

        .app-hero {
            background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
            border: 1px solid #E5E7EB;
            border-radius: 20px;
            padding: 26px 24px;
            margin-bottom: 20px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
        }

        .app-card {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 18px;
            padding: 20px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
            margin-bottom: 20px;
        }

        .metric-card {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 16px;
            padding: 18px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.03);
            min-height: 110px;
        }

        .metric-label {
            color: #64748B;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 8px;
        }

        .metric-value {
            color: #0F172A;
            font-weight: 700;
            font-size: 1.35rem;
            margin-bottom: 6px;
        }

        .metric-subtext {
            color: #475569;
            font-size: 0.92rem;
        }

        .result-card {
            background: linear-gradient(135deg, #166534 0%, #0F766E 100%);
            color: white;
            border-radius: 18px;
            padding: 24px;
            box-shadow: 0 12px 26px rgba(22, 101, 52, 0.18);
            margin-bottom: 20px;
        }

        .result-card h3, .result-card h2, .result-card p {
            color: white !important;
        }

        .small-muted {
            color: #64748B;
            font-size: 0.95rem;
        }

        .section-title {
            font-size: 1.08rem;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 0.6rem;
        }

        div[data-testid="stButton"] > button {
            background: #166534;
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.6rem 1rem;
            font-weight: 600;
        }

        div[data-testid="stButton"] > button:hover {
            background: #14532D;
            color: white;
        }

        div[data-testid="stSelectbox"] label,
        div[data-testid="stSlider"] label {
            font-weight: 600;
            color: #0F172A;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_global_css()

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <h2>House Price App</h2>
            <p>Predict property prices and explore market trends across Nigeria.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.selectbox("Navigate", ("Price Prediction", "Market Insights"))

if page == "Price Prediction":
    show_predict_page()
else:
    show_explore_page()
>>>>>>> b8b30a8 (Initial commit)
