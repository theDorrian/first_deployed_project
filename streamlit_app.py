# streamlit_app.py

import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# --- Константы ---
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
MODEL_PATH = "best_model.pkl"
MODEL_FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked_Q",
    "Embarked_S",
]

st.set_page_config(page_title="🚢 Titanic Survival Predictor", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)
    for col in ["Embarked_Q", "Embarked_S"]:
        if col not in df.columns:
            df[col] = 0
    return df[MODEL_FEATURES]

@st.cache_data
def get_preview(df: pd.DataFrame) -> pd.DataFrame:
    # возвращаем фиксированный сэмпл из 10 строк
    return df.sample(10, random_state=42)

# --- Основной код ---
df = load_data()
model = load_model()

st.title("🚢 Titanic Survival Predictor")
st.write(
    """
    Приложение загружает заранее обученную модель (`best_model.pkl`) и делает предсказания выживания пассажиров Titanic.
    """
)

# === Блок 1: Метрики на всем датасете ===
st.subheader("📊 Качество модели на всем датасете")
X_all = preprocess(df)
y_all = df["Survived"]
y_pred_all = model.predict(X_all)
y_proba_all = model.predict_proba(X_all)[:, 1]

col1, col2 = st.columns(2)
col1.metric("Accuracy", f"{accuracy_score(y_all, y_pred_all):.3f}")
col2.metric("ROC AUC", f"{roc_auc_score(y_all, y_proba_all):.3f}")

st.write("**Confusion Matrix**")
st.write(confusion_matrix(y_all, y_pred_all))

st.write("**Classification Report**")
st.text(classification_report(y_all, y_pred_all))

st.write("---")

# === Блок 2: Превью данных ===
st.subheader("🗂 Превью исходного датасета")
preview = get_preview(df)
st.dataframe(preview, use_container_width=True)

st.write("---")

# === Блок 3: Интерактивное предсказание ===
st.sidebar.header("🧑‍✈️ Ввод параметров пассажира")

pclass = st.sidebar.selectbox("Pclass", sorted(df["Pclass"].unique()))
sex = st.sidebar.selectbox("Sex", ["male", "female"])
sex = 0 if sex == "male" else 1
age = st.sidebar.slider("Age", float(df["Age"].min()), float(df["Age"].max()), float(df["Age"].median()))
sibsp = st.sidebar.number_input("SibSp", min_value=int(df["SibSp"].min()), max_value=int(df["SibSp"].max()), value=0)
parch = st.sidebar.number_input("Parch", min_value=int(df["Parch"].min()), max_value=int(df["Parch"].max()), value=0)
fare = st.sidebar.number_input("Fare", float(df["Fare"].min()), float(df["Fare"].max()), float(df["Fare"].median()))
embarked = st.sidebar.selectbox("Embarked", ["C", "Q", "S"])

new_passenger = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": embarked,
}])

X_new = preprocess(new_passenger)
pred = model.predict(X_new)[0]
proba = model.predict_proba(X_new)[0, 1]

st.sidebar.subheader("📋 Результат предсказания")
st.sidebar.markdown(f"**Выживет:** {'Да' if pred == 1 else 'Нет'}")
st.sidebar.markdown(f"**Вероятность выживания:** {proba:.2f}")
