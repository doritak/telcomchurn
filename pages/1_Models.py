
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
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector as selector

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import cross_val_score

from sidebar import render_sidebar

render_sidebar()

if "scores" not in st.session_state:
    # Try to load previous scores from pickle (optional)
    try:
        with open("scores.pkl", "rb") as f:
            st.session_state.scores = pickle.load(f)
    except FileNotFoundError:
        st.session_state.scores = []  # start empty list


st.title("Models Page")
st.write("This is the models page : Gradient Boosting.")

try:
    with open("data/dataset.pkl", "rb") as f:
        df = pickle.load(f)

    st.write("DataFrame loaded from pickle:")
    st.dataframe(df)

except FileNotFoundError:
    st.error("The file dataset.pkl was not found. Please go back to the previous page.")
    
######### Data Preprocessing #########
X = df.drop("Churn", axis=1)
y = df["Churn"]

num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(sparse_output=False,handle_unknown="ignore"
                             ).set_output(transform="pandas"))
                                          ])
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler().set_output(transform="pandas"))
                                      ])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, num_features),
    ("cat", categorical_transformer, cat_features)
    ]).set_output(transform="pandas")

preprocessor_new = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_features),
        ("cat", categorical_transformer, cat_features),
    ]
)

# scores = {}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)

st.write("Data Preprocessing completed. Ready for model training.")
st.write(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

### Select the model
models_options = {
    "Gradient Boosting": GradientBoostingClassifier(),
    "Logistic Regression CV": LogisticRegressionCV(),
    "AdaBoost": AdaBoostClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Bernoulli Naive Bayes": BernoulliNB(),
    
}

st.subheader("📌 Select a model to train")

model_name  = st.selectbox(
    "Available models:",
    list(models_options.keys())
)
treeModels = ["Random Forest", "Decision Tree", "Gradient Boosting", "XGBoost", "AdaBoost"]

if model_name in treeModels:
    st.write("You have selected a tree-based model.")
    rand_st = 42
    model = models_options[model_name].set_params(random_state=rand_st)
    pipeline_model = Pipeline(steps=[
        ("preprocessor", preprocessor_new),
        ("model", model)
    ])
    pipeline_model.fit(X_train, y_train)
    y_pred = pipeline_model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    st.write(f"F1 Score for {model_name}: {f1:.4f}")
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    # scores[model_name] = round(f1_score(y_test, y_pred, average='weighted'), 3)
    
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap=plt.cm.Blues)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=("Predicted No Churn", "Predicted Churn"))
    ax.yaxis.set(ticks=(0, 1), ticklabels=("Actual No Churn", "Actual Churn"))
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="red")  
    ax.set_title(f'Confusion Matrix for {model_name}')
    st.pyplot(fig)
    # Save the F1 score and timestamp in session state
    run_info = {
    "model": model_name,
    "f1": round(f1_score(y_test, y_pred, average='weighted'), 3),
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.scores.append(run_info)

else:
    if model_name == "Logistic Regression CV":  
        st.write("Selected model is Logistic Regression with Cross-Validation.")
        model = models_options[model_name].set_params(cv=5, class_weight='balanced')
    else:
        st.write("Selected model is not tree-based. ")
        model = models_options[model_name]
    
    pipeline_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    pipeline_model.fit(X_train, y_train)
    y_pred = pipeline_model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    st.write(f"F1 Score for {model_name}: {f1:.4f}")
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap=plt.cm.Blues)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=("Predicted No Churn", "Predicted Churn"))
    ax.yaxis.set(ticks=(0, 1), ticklabels=("Actual No Churn", "Actual Churn"))
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="red")
    ax.set_title(f'Confusion Matrix for {model_name}')
    st.pyplot(fig)
    
    # Save the F1 score and timestamp in session state
    run_info = {
    "model": model_name,
    "f1": round(f1_score(y_test, y_pred, average='weighted'), 3),
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.scores.append(run_info)
    
# Save scores to pickle
with open("scores.pkl", "wb") as f:
    pickle.dump(st.session_state.scores, f)
        
st.write("Model training and evaluation completed.")


st.write("F1 Scores of all trained models:")
scores_df = pd.DataFrame(st.session_state.scores)
scores_df = scores_df.iloc[::-1]   # reverse index order
st.dataframe(scores_df)
# st.write(st.session_state.scores)

#### guardar el modelo entrenado ####
model_filename = f"data/{model_name.replace(' ', '_').lower()}_model.pkl"
with open(model_filename, "wb") as f:
    pickle.dump(pipeline_model, f)
    
# ==== Allow user to download model to their PC ====
with open(model_filename, "rb") as f:
    st.download_button(
        label="Download trained model",
        data=f,
        file_name=f"{model_name.replace(' ', '_').lower()}_model.pkl",
        mime="application/octet-stream"
    )
st.success(f"Trained model saved as {model_filename}")

st.write("You can now proceed to the 'Model Evaluation' page to compare model performances.")
st.write("### End of Models Page")