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

# Isolation Forest Outlier Detection
def detect_outliers(df):
    if df.empty:
        return df

    df_copy = df.copy()

    # One-hot encode 'purpose' and 'country'
    encoded = pd.get_dummies(df_copy[['purpose', 'country']])
    features = pd.concat([df_copy[['amount']], encoded], axis=1)

    # Fit Isolation Forest
    model = IsolationForest(contamination=0.05, random_state=42)
    df_copy["is_outlier"] = model.fit_predict(features)
    df_copy["is_outlier"] = df_copy["is_outlier"].apply(lambda x: 1 if x == -1 else 0)

    return df_copy

# Generate data and detect outliers
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
    
    # Combine and detect outliers
    combined = pd.concat([st.session_state.df.drop(columns="is_outlier", errors='ignore'), data], ignore_index=True)
    st.session_state.df = detect_outliers(combined)

# Sidebar widgets
def data_params_widget():
    st.subheader("Data Parameters")
    st.info("Configure data generation below.")
    st.slider('Transaction Amount Range', 0.0, 100_000.0, (5_000.0, 80_000.0), key="dp_amount_range")
    st.multiselect('Purposes', 
        ('Entertainment', 'Holiday', 'Transportation', 'Bills', 'Medical', 'Misc'), 
        key="dp_purposes")
    st.multiselect('Countries', EnGbAddressProvider.alpha_3_country_codes, key="dp_countries")
    st.number_input('Number of Rows to Generate', min_value=1, step=1, key="dp_rows")
    st.button('Generate', on_click=generate_clicked)

def general_widget():
    st.subheader("General")
    st.button('Clear State', on_click=lambda: st.session_state.clear())

# Metric Display
def metrics_widget():
    st.subheader("Transaction Metrics")
    col1, col2 = st.columns(2)
    df = st.session_state.df
    with col1:
        st.metric("Total Transactions", len(df), help="All transactions so far")
    with col2:
        suspicious_count = df["is_outlier"].sum() if "is_outlier" in df.columns else 0
        st.metric("Suspicious Transactions", int(suspicious_count), help="Anomalies detected by Isolation Forest")

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

# Bar & Line Chart
def charts_widget():
    if st.session_state.df.empty:
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        grouped = st.session_state.df.groupby("purpose").size().reset_index(name="count")
        fig = px.bar(grouped, x="purpose", y="count", color="purpose", text="count")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        df = st.session_state.df
        fig = px.line(
            df,
            x="timestamp",
            y="amount",
            color=df["is_outlier"].map({0: "Normal", 1: "Suspicious"}),
            labels={"color": "Transaction Type"},
        )
        st.plotly_chart(fig, use_container_width=True)

# Data Table
def data_table_widget():
    st.markdown('# Generated Transaction Data')
    st.markdown('Outliers are flagged in the `is_outlier` column (1 = Suspicious).')
    st.dataframe(st.session_state.df)

# Sidebar Layout
with st.sidebar:
    st.title("Configuration")
    data_params_widget()
    general_widget()

# Main layout
metrics_widget()
choropleth_widget()
charts_widget()
data_table_widget()
