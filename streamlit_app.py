import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

DATA_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)
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
    df = pd.read_csv(DATA_URL)
    return df

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 1) Заполняем пропуски
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    # 2) Кодируем пол
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    # 3) One-hot для порта посадки
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)
    # 4) Гарантируем, что обе колонки есть
    for col in ["Embarked_Q", "Embarked_S"]:
        if col not in df.columns:
            df[col] = 0
    # 5) Отбираем нужные фичи в нужном порядке
    return df[MODEL_FEATURES]

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

acc = accuracy_score(y_all, y_pred_all)
rocauc = roc_auc_score(y_all, y_proba_all)

col1, col2 = st.columns(2)
col1.metric("Accuracy", f"{acc:.3f}")
col2.metric("ROC AUC", f"{rocauc:.3f}")

st.write("**Confusion Matrix**")
st.write(confusion_matrix(y_all, y_pred_all))

st.write("**Classification Report**")
st.text(classification_report(y_all, y_pred_all))

st.write("---")

# === Блок 2: Превью данных ===
st.subheader("🗂 Превью исходного датасета")
st.dataframe(df.sample(10), use_container_width=True)

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

# Собираем DataFrame для одного пассажира
new_passenger = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": embarked,
}])

# Предобработка и предсказание
X_new = preprocess(new_passenger)
pred = model.predict(X_new)[0]
proba = model.predict_proba(X_new)[0, 1]

st.sidebar.subheader("📋 Результат предсказания")
st.sidebar.markdown(f"**Выживет:** {'Да' if pred == 1 else 'Нет'}")
st.sidebar.markdown(f"**Вероятность выживания:** {proba:.2f}")
