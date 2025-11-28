import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from urllib.request import urlopen
import pickle

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import cross_val_score

from sidebar import render_sidebar

render_sidebar()

def load_data(path):
    df = pd.read_csv(path)
    return df

path = "data/telecom_users.csv"

st.title("Telecom Customer Churn Analysis")
st.write("""
This application analyzes customer churn in a telecom dataset.
""")

uploaded_file = st.file_uploader(
    label="Upload your CSV dataset",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_data(path)
    
if "Unnamed: 0" in df.columns and "customerID" in df.columns:
    df.drop(columns=["Unnamed: 0","customerID"], inplace=True)
    
columns = [col for col in df.columns if col != "Churn"]
   
cols_to_drop = st.multiselect(
    "Select columns to remove:",
    options=columns

)
# st.button("Drop Selected Columns")

df.drop(columns=cols_to_drop, inplace=True)
columns = [col for col in df.columns if col != "Churn"]

if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)
    
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
########### guardar el data set para otra pagina ###########
# Guardar DataFrame en pickle cuando se presione el botón
if st.button("Save Dataset"):
    with open("data/dataset.pkl", "wb") as f:
        pickle.dump(df, f)
    
    st.success("Dataset saved successfully.")
    st.write("Now you must go to Models page...")
    # Cambio de 'pantalla' para Streamlit 1.24.1
    # st.session_state.pages = "Models"
    # st.experimental_rerun()


######### Dataset Overview #########
unique = {}
type_counts = {}
nan_counts = {}

for col in df.columns:
    unique[col] = df[col].nunique()
    type_counts[col] = df[col].dtype
    nan_counts[col] = df[col].isna().sum()

df_unique = pd.DataFrame({
    "n_unique" : unique,
    "dtype" : type_counts, 
    "n_nan": nan_counts}
)

st.subheader("Dataset Overview")
st.write("The dataset contains information about telecom customers and whether they have churned or not.")
st.dataframe(df)
st.write("Dataset shape:", df.shape)
st.write(df_unique)

######### EDA #########
st.subheader("Exploratory Data Analysis (EDA)")
left_col,  right_col = st.columns(2)

with left_col:
    
    st.write("Churn distribution:")
     # Contar churn
     
    # churn_df = (
    #     df["Churn"]
    #     .value_counts()
    #     .rename_axis("Churn")         # nombre de la columna de categorías
    #     .reset_index(name="count_value")    # nombre de la columna de conteos
    # )
    # churn_df["Churn"] = churn_df["Churn"].map({0: "No Churn", 1: "Churn"})
    # st.write(churn_df)
    # st.write(churn_df.columns)

    # fig1 = px.pie(
    #     churn_df,
    #     names="Churn",     # columna con las etiquetas ("No Churn", "Churn")
    #     values="count_value",    # columna con los valores (4399, 1587, etc.)
    #     title="Churn Distribution",
    # )
    # fig1.update_traces(textposition="inside", textinfo="percent")
    # st.plotly_chart(fig1, use_container_width=True)

    
    churn_counts = df['Churn'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.legend(['No Churn', 'Churn'])

    ax1.axis('equal')
    st.pyplot(fig1)
    
with right_col:
    contract_churn = df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack()
    fig2 = px.bar(contract_churn, barmode='group', title='Churn Rate by Contract Type')
    st.plotly_chart(fig2)

cat_Columns = df.select_dtypes(include=["object"]).columns.tolist()
num_Columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

st.write("### Categorical Features vs Churn")
st.write(f"The categorical features in the dataset are: {', '.join(cat_Columns)}")
st.write("These stacked bar charts show how each categorical feature is distributed across Churn and Non-Churn customers.")
st.write("Each bar represents the percentage breakdown within each churn group, adding up to 100%.")
cols_per_row = 3

for i, col in enumerate(cat_Columns):
    # cada 3 gráficos creo una nueva fila de columnas
    if i % cols_per_row == 0:
        st_cols = st.columns(cols_per_row)

    table = pd.crosstab(df["Churn"], df[col], normalize='index') * 100

    # figura individual
    fig, ax = plt.subplots(figsize=(4, 4))
    table.plot(kind="bar", stacked=True, ax=ax)

    ax.set_ylabel("Percent (%)")
    ax.set_title(f"Percentual Distribution '{col}' by Churn")
    ax.legend(title=col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    # mandamos el gráfico a la columna que toca (0,1,2)
    with st_cols[i % cols_per_row]:
        st.pyplot(fig)
    # mandamos el gráfico a la columna que toca (0,1,2)
num_Columns_only = [col for col in num_Columns if col != "Churn"]    
st.write("### Numerical Features vs Churn")
st.write(f"The numercial features in the dataset are: {', '.join(num_Columns_only)}")
st.write("Box plots showing the distribution of numerical features based on churn status.")

for i, col in enumerate(num_Columns_only):
    # cada 3 gráficos creo una nueva fila de columnas
    if i % cols_per_row == 0:
        st_cols = st.columns(cols_per_row)

    # figura individual
    fig, ax = plt.subplots(figsize=(4, 4))
    df.boxplot(column=col, by='Churn', ax=ax)

    ax.set_title(f"Box Plot of '{col}' by Churn")
    ax.set_xlabel("Churn")
    ax.set_ylabel(col)
    plt.suptitle("")  # eliminar el título automático de pandas
    plt.tight_layout()

    # mandamos el gráfico a la columna que toca (0,1,2)
    with st_cols[i % cols_per_row]:
        st.pyplot(fig)


