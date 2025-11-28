import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from urllib.request import urlopen
import pickle
import datetime

from copy import deepcopy
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector as selector

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


st.title("Prediction Page")
st.write("This is the prediction page. Here you can load a trained model and make predictions on new data.")

# ===================== LOAD DATAFRAME =====================
try:
    with open("data/dataset.pkl", "rb") as f:
        df = pickle.load(f)

    st.write("DataFrame loaded from pickle:")
    st.dataframe(df)
except FileNotFoundError:
    st.error("The file dataset.pkl was not found. Please go back to the previous page.")
    st.stop()

# ===================== MODEL UPLOADER =====================
model_file = st.file_uploader(
    label="Upload your trained model (pickle file)", type=["pkl"]
)

# ===================== DEFINE ORIGINAL FEATURES =====================
target_col = "Churn"
feature_cols = [c for c in df.columns if c != target_col]

st.write("### Features used for prediction:", feature_cols)

# ===================== BUILD INPUT UI FROM DATAFRAME =====================
st.write("## Enter values for prediction")

input_data = {}

for col in feature_cols:
    col_series = df[col].dropna()
    unique_vals = col_series.unique()

    # 1) Treat columns with few unique values as categorical (dropdown),
    #    even if they are numeric (e.g. SeniorCitizen = {0, 1})
    if 2 <= len(unique_vals) <= 10:
        st.write(f"**{col}** (categorical dropdown)")
        options = sorted(unique_vals.tolist())
        val = st.selectbox(f"{col}:", options, key=f"{col}_dropdown")
        input_data[col] = [val]

    # 2) Truly numeric continuous columns
    elif pd.api.types.is_numeric_dtype(col_series):
        st.write(f"**{col}** (numeric input)")
        default_value = float(col_series.median())
        val = st.number_input(f"{col}:", value=default_value, key=f"{col}_num")
        input_data[col] = [val]

    # 3) Text categorical columns
    else:
        st.write(f"**{col}** (text categorical)")
        options = sorted(unique_vals.tolist())
        val = st.selectbox(f"{col}:", options, key=f"{col}_text")
        input_data[col] = [val]

# ===================== MAKE PREDICTION =====================
if model_file is not None:
    model = pickle.load(model_file)
    st.success("Model loaded successfully!")

    # Find any ColumnTransformer inside the pipeline (optional, just for info)
    column_transformer = None
    if hasattr(model, "named_steps"):
        for name, step in model.named_steps.items():
            if "ColumnTransformer" in str(type(step)):
                column_transformer = step
                break

    if column_transformer is None:
        st.warning("No ColumnTransformer found inside the model pipeline (or model is not a Pipeline).")
    else:
        feature_names = column_transformer.get_feature_names_out()
        st.write("### Feature columns expected by the model (after transformation):")
        st.write(feature_names)

    if st.button("Predict"):
        input_df = pd.DataFrame(input_data)

        # Ensure dtypes in input_df match the original df dtypes
        original_dtypes = df[feature_cols].dtypes.to_dict()
        for c, dt in original_dtypes.items():
            if c in input_df.columns:
                try:
                    input_df[c] = input_df[c].astype(dt)
                except Exception:
                    # If casting fails, we just leave the value as is
                    pass

        st.write("### Raw input to model (same schema as dataset):")
        st.dataframe(input_df)

        prediction = model.predict(input_df)
        st.success(f"Prediction: {'Yes' if prediction[0] == 1 else 'No'}")

else:
    st.info("Please upload a trained model to make predictions.")
