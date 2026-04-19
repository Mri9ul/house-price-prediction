import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


@st.cache_data
def load_data():
    column_names = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
        "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "PRICE"
    ]

    data = pd.read_csv(
        "data/housing.csv",
        sep=r"\s+",
        header=None,
        names=column_names
    )
    return data


@st.cache_resource
def train_model(dataframe):
    X = dataframe.drop("PRICE", axis=1)
    y = dataframe["PRICE"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return model, mae, mse, r2, X


data = load_data()
model, mae, mse, r2, X = train_model(data)

st.title("🏠 House Price Prediction")
st.markdown(
    """
    Predict house prices using an **XGBoost Regressor** trained on housing data.
    Enter feature values manually or autofill a sample row.
    """
)

with st.expander("Model Information"):
    st.write("**Algorithm:** XGBoost Regressor")
    st.write(f"**Mean Absolute Error:** {mae:.3f}")
    st.write(f"**Mean Squared Error:** {mse:.3f}")
    st.write(f"**R² Score:** {r2:.3f}")
    st.write("**Input Features:** 13")

st.divider()

st.subheader("Feature Input")

sample_index = st.selectbox(
    "Select a sample row to autofill",
    ["None"] + [str(i) for i in range(min(10, len(X)))]
)

if sample_index != "None":
    default_values = X.iloc[int(sample_index)].to_dict()
else:
    default_values = {
        "CRIM": 0.1,
        "ZN": 0.0,
        "INDUS": 8.0,
        "CHAS": 0,
        "NOX": 0.5,
        "RM": 6.0,
        "AGE": 65.0,
        "DIS": 4.0,
        "RAD": 4,
        "TAX": 300.0,
        "PTRATIO": 18.0,
        "B": 390.0,
        "LSTAT": 12.0
    }

col1, col2, col3 = st.columns(3)

with col1:
    crim = st.number_input("CRIM", value=float(default_values["CRIM"]))
    zn = st.number_input("ZN", value=float(default_values["ZN"]))
    indus = st.number_input("INDUS", value=float(default_values["INDUS"]))
    chas = st.selectbox("CHAS", [0, 1], index=int(default_values["CHAS"]))
    nox = st.number_input("NOX", value=float(default_values["NOX"]))

with col2:
    rm = st.number_input("RM", value=float(default_values["RM"]))
    age = st.number_input("AGE", value=float(default_values["AGE"]))
    dis = st.number_input("DIS", value=float(default_values["DIS"]))
    rad = st.number_input("RAD", value=int(default_values["RAD"]), step=1)
    tax = st.number_input("TAX", value=float(default_values["TAX"]))

with col3:
    ptratio = st.number_input("PTRATIO", value=float(default_values["PTRATIO"]))
    b = st.number_input("B", value=float(default_values["B"]))
    lstat = st.number_input("LSTAT", value=float(default_values["LSTAT"]))

st.divider()

if st.button("Predict House Price"):
    input_data = pd.DataFrame([{
        "CRIM": crim,
        "ZN": zn,
        "INDUS": indus,
        "CHAS": chas,
        "NOX": nox,
        "RM": rm,
        "AGE": age,
        "DIS": dis,
        "RAD": rad,
        "TAX": tax,
        "PTRATIO": ptratio,
        "B": b,
        "LSTAT": lstat
    }])

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")
    st.success(f"Predicted House Price: {prediction:.2f}")

st.divider()
st.caption("Built with Streamlit, Pandas, NumPy, and XGBoost.")