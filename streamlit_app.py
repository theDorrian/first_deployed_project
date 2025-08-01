import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    confusion_matrix, classification_report
)

# --- Константы ---
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
MODEL_PATH = "best_model.pkl"
MODEL_FEATURES = [
    "Pclass","Sex","Age","SibSp","Parch","Fare","Embarked_Q","Embarked_S"
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
    df["Sex"] = df["Sex"].map({"male":0,"female":1})
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)
    for c in ["Embarked_Q","Embarked_S"]:
        if c not in df.columns:
            df[c] = 0
    return df[MODEL_FEATURES]

@st.cache_data
def get_preview(df: pd.DataFrame) -> pd.DataFrame:
    # отфильтровываем только строки, где нет NaN в Age, Fare, Embarked
    df_clean = df.dropna(subset=["Age","Fare","Embarked"])
    # возвращаем фиксированный sample
    return df_clean.sample(10, random_state=42)

# --- Основной код ---
df = load_data()
model = load_model()

# фиксированный «рандом» для дефолтов
if "sample_passenger" not in st.session_state:
    df_nonull = df.dropna(subset=["Age","Fare","Embarked"])
    st.session_state.sample_passenger = df_nonull.sample(1, random_state=42).iloc[0]
sample = st.session_state.sample_passenger

st.title("🚢 Titanic Survival Predictor")

# — Метрики —
st.subheader("📊 Качество модели")
X_all = preprocess(df)
y_all = df["Survived"]
y_pred = model.predict(X_all)
y_proba = model.predict_proba(X_all)[:,1]

c1, c2 = st.columns(2)
c1.metric("Accuracy", f"{accuracy_score(y_all,y_pred):.3f}")
c2.metric("ROC AUC", f"{roc_auc_score(y_all,y_proba):.3f}")

st.write("**Confusion Matrix**")
st.write(confusion_matrix(y_all, y_pred))
st.write("**Classification Report**")
st.text(classification_report(y_all, y_pred))
st.write("---")

# — Превью исходного датасета —
st.subheader("🗂 Превью исходного датасета")
preview = get_preview(df)
st.dataframe(preview, use_container_width=True)
st.write("---")

# — Интерактивное предсказание —
st.sidebar.header("🧑‍✈️ Ввод параметров пассажира")

opts_pclass = sorted(df["Pclass"].unique())
opts_sex    = ["male","female"]
opts_emb    = ["C","Q","S"]

pclass_def = int(sample["Pclass"])
sex_def    = sample["Sex"]
age_def    = float(sample["Age"])
sibsp_def  = int(sample["SibSp"])
parch_def  = int(sample["Parch"])
fare_def   = float(sample["Fare"])
emb_def    = sample["Embarked"]

pclass = st.sidebar.selectbox("Pclass", opts_pclass, index=opts_pclass.index(pclass_def))
sex    = st.sidebar.selectbox("Sex", opts_sex,    index=opts_sex.index(sex_def))
age    = st.sidebar.slider("Age", float(df["Age"].min()), float(df["Age"].max()), age_def)
sibsp  = st.sidebar.number_input("SibSp", int(df["SibSp"].min()), int(df["SibSp"].max()), value=sibsp_def)
parch  = st.sidebar.number_input("Parch", int(df["Parch"].min()), int(df["Parch"].max()), value=parch_def)
fare   = st.sidebar.number_input("Fare", float(df["Fare"].min()), float(df["Fare"].max()), value=fare_def)
emb    = st.sidebar.selectbox("Embarked", opts_emb, index=opts_emb.index(emb_def))

new_pass = pd.DataFrame([{
    "Pclass":pclass, "Sex":sex, "Age":age,
    "SibSp":sibsp, "Parch":parch, "Fare":fare,
    "Embarked":emb
}])
X_new = preprocess(new_pass)
pred  = model.predict(X_new)[0]
prob  = model.predict_proba(X_new)[0,1]

st.sidebar.subheader("📋 Результат предсказания")
st.sidebar.markdown(f"**Выживет:** {'Да' if pred==1 else 'Нет'}")
st.sidebar.markdown(f"**Вероятность:** {prob:.3f}")
