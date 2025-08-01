# streamlit_app.py

import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

# --- Константы ---
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
MODEL_PATH = "best_model.pkl"
MODEL_FEATURES = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked_Q", "Embarked_S"
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

# --- Основной код ---
df = load_data()
model = load_model()

# Сэмплируем одного случайного пассажира при первом заходе
if "sample_passenger" not in st.session_state:
    st.session_state.sample_passenger = df.sample(1).iloc[0]

sample = st.session_state.sample_passenger

st.title("🚢 Titanic Survival Predictor")
st.write(
    """
    Приложение загружает заранее обученную модель (`best_model.pkl`) и делает предсказания выживания пассажиров Titanic.
    """
)

# === Блок 1: Метрики на всём датасете ===
st.subheader("📊 Качество модели на всём датасете")
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
st.subheader("🗂 Превью исходного датасета (фиксированная выборка)")
@st.cache_data
def get_preview(data):
    return data.sample(10, random_state=42)

preview = get_preview(df)
st.dataframe(preview, use_container_width=True)
st.write("---")

# === Блок 3: Интерактивное предсказание ===
st.sidebar.header("🧑‍✈️ Ввод параметров пассажира")

# Подготовим списки опций
pclass_opts = sorted(df["Pclass"].unique())
sex_opts    = ["male", "female"]
embark_opts = ["C", "Q", "S"]

# Значения по умолчанию из sample
pclass_def  = int(sample["Pclass"])
sex_def     = sample["Sex"]
age_def     = float(sample["Age"])
sibsp_def   = int(sample["SibSp"])
parch_def   = int(sample["Parch"])
fare_def    = float(sample["Fare"])
embark_def  = sample["Embarked"]

# Виджеты с дефолтами
pclass = st.sidebar.selectbox("Pclass", pclass_opts, index=pclass_opts.index(pclass_def))
sex    = st.sidebar.selectbox("Sex", sex_opts, index=sex_opts.index(sex_def))
age    = st.sidebar.slider("Age", float(df["Age"].min()), float(df["Age"].max()), age_def)
sibsp  = st.sidebar.number_input("SibSp", min_value=int(df["SibSp"].min()), max_value=int(df["SibSp"].max()), value=sibsp_def)
parch  = st.sidebar.number_input("Parch", min_value=int(df["Parch"].min()), max_value=int(df["Parch"].max()), value=parch_def)
fare   = st.sidebar.number_input("Fare", float(df["Fare"].min()), float(df["Fare"].max()), value=fare_def)
embarked = st.sidebar.selectbox("Embarked", embark_opts, index=embark_opts.index(embark_def))

# Составляем DataFrame для предсказания
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
st.sidebar.markdown(f"**Вероятность выживания:** {proba:.3f}")
