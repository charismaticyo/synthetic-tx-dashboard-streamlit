%%writefile app.py

import time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from faker.providers.address.en_GB import Provider as EnGbAddressProvider
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from urllib.request import urlopen
import json

from data import *

st.set_page_config(layout="wide")
st.title("COMP 1831 - Transaction Generator with Anomaly Detection")

# Initialize session state
if "df" not in st.session_state:
    columns = ["timestamp", "amount", "purpose", "country", "is_outlier"]
    st.session_state['df'] = pd.DataFrame(columns=columns)

# Isolation Forest Detection Function
def detect_outliers(df):
    if df.empty:
        return df

    df_copy = df.copy()

    # One-hot encode categorical features
    encoded = pd.get_dummies(df_copy[['purpose', 'country']])
    features = pd.concat([df_copy[['amount']], encoded], axis=1)

    # Train Isolation Forest
    model = IsolationForest(contamination=0.05, random_state=42)
    df_copy["is_outlier"] = model.fit_predict(features)
    df_copy["is_outlier"] = df_copy["is_outlier"].apply(lambda x: 1 if x == -1 else 0)

    return df_copy

# Data generation
def generate_clicked():
    data = generate_timeseries_data(
        num_rows=st.session_state.dp_rows,
        start_timestamp=st.session_state.get("last_tx_timestamp", None),
        amount_min=st.session_state.dp_amount_range[0],
        amount_max=st.session_state.dp_amount_range[1],
        purposes=st.session_state.dp_purposes,
        countries=st.session_state.dp_countries,
    )
    st.session_state.last_tx_timestamp = data.iloc[-1]["timestamp"]
    combined_df = pd.concat([st.session_state.df.drop(columns="is_outlier", errors='ignore'), data], ignore_index=True)
    st.session_state.df = detect_outliers(combined_df)

# Sidebar - Data Params
def data_params_widget():
    st.subheader("Data Parameters")
    st.info("Use the form below to configure new data parameters.")
    st.slider('Transaction range', 0.0, 100_000.0, (5_000.0, 80_000.0), key="dp_amount_range")
    st.multiselect('Purposes', 
        ('Entertainment', 'Holiday', 'Transportation', 'Bills', 'Medical', 'Misc'), 
        key="dp_purposes")
    st.multiselect('Countries', EnGbAddressProvider.alpha_3_country_codes, key="dp_countries")
    st.number_input('Rows', min_value=1, step=1, key="dp_rows")
    st.button('Generate', on_click=generate_clicked)

# Choropleth Map
def choropleth_widget():
    if st.session_state.df.empty:
        return

    with urlopen('https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json') as response:
        countries = json.load(response)

    grouped = st.session_state.df.groupby("country").size().reset_index(name="count")
    fig = px.choropleth(
        grouped,
        geojson=countries,
        locations='country',
        color='count',
        color_continuous_scale="hot_r",
        range_color=(0, grouped["count"].max()),
    )
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=1,
        mapbox_center={"lat": 30, "lon": 0},
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )
    st.plotly_chart(fig, use_container_width=True)

# Bar and Time Series Charts
def charts_widget():
    if st.session_state.df.empty:
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        data = st.session_state.df.groupby("purpose").size().reset_index(name="count")
        fig = px.bar(data, x="purpose", y="count", color="purpose", text="count")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.line(
            st.session_state.df,
            x="timestamp",
            y="amount",
            color=st.session_state.df["is_outlier"].map({0: "Normal", 1: "Suspicious"}),
            labels={"color": "Transaction Type"},
        )
        st.plotly_chart(fig, use_container_width=True)

# Metric Display
def metrics_widget():
    st.subheader("Transaction Metrics")
    col1, col2 = st.columns(2)
    df = st.session_state.df
    with col1:
        st.metric(
            "Total Transactions",
            len(df),
            help="Number of all transactions"
        )
    with col2:
        suspicious_count = df["is_outlier"].sum() if "is_outlier" in df.columns else 0
        st.metric(
            "Suspicious Transactions",
            int(suspicious_count),
            help="Number of detected outliers"
        )

# General controls
def general_widget():
    st.subheader("General")
    st.button('Clear state', on_click=lambda: st.session_state.clear())

# Data Table View
def data_table_widget():
    st.markdown('# Generated Data:')
    st.markdown('The dataframe below includes detected suspicious transactions (is_outlier=1).')
    st.dataframe(st.session_state.df)

# Layout
with st.sidebar:
    st.title("Configuration")
    data_params_widget()
    general_widget()

metrics_widget()
choropleth_widget()
charts_widget()
data_table_widget()
