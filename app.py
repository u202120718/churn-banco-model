import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

# -------------------------------------------------------------------
# CONFIGURACIÓN DE LA APP
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Churn de Clientes - Banco ABC",
    layout="wide"
)

st.title("🧠 Predicción de Churn de Clientes - Banco ABC")
st.caption("App construida del Grupo4")

DATA_PATH = "Bank Customer Churn Prediction.csv"  # nombre EXACTO del archivo


# -------------------------------------------------------------------
# 1. CARGA DE DATOS
# -------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


# -------------------------------------------------------------------
# 2. FEATURE ENGINEERING (MISMO CRITERIO QUE EN EL NOTEBOOK)
# -------------------------------------------------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df_fe = df.copy()

    # Saldo por producto
    df_fe["balance_per_product"] = df_fe["balance"] / (
        df_fe["products_number"].replace(0, np.nan)
    )
    df_fe["balance_per_product"].fillna(0, inplace=True)

    # Relación salario / saldo
    df_fe["salary_balance_ratio"] = df_fe["estimated_salary"] / (
        df_fe["balance"].replace(0, np.nan)
    )
    df_fe["salary_balance_ratio"].replace([np.inf, -np.inf], np.nan, inplace=True)
    df_fe["salary_balance_ratio"].fillna(df_fe["salary_balance_ratio"].median(), inplace=True)

    # Grupos de edad
    bins_age = [0, 25, 35, 45, 55, 65, 100]
    labels_age = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
    df_fe["age_group"] = pd.cut(df_fe["age"], bins=bins_age, labels=labels_age)

    # Buckets de tenure
    df_fe["tenure_bucket"] = pd.cut(
        df_fe["tenure"],
        bins=[-1, 0, 2, 5, 10, 100],
        labels=["0", "1-2", "3-5", "6-10", "10+"],
    )

    # Flag de alto saldo
    df_fe["high_balance"] = (df_fe["balance"] >
                             df_fe["balance"].quantile(0.75)).astype(int)

    return df_fe


# -------------------------------------------------------------------
# 3. ENTRENAMIENTO DEL MODELO Y CÁLCULO DE MÉTRICAS
# -------------------------------------------------------------------
@st.cache_resource
def train_model():
    df = load_data(DATA_PATH)
    df_fe = feature_engineering(df)

    target = "churn"
    drop_cols = ["customer_id"]
    features = [c for c in df_fe.columns if c not in [target] + drop_cols]

    numeric_features = [
        "credit_score",
        "age",
        "tenure",
        "balance",
        "products_number",
        "estimated_salary",
        "balance_per_product",
        "salary_balance_ratio",
    ]

    categorical_features = [
        "country",
        "gender",
        "credit_card",
        "active_member",
        "age_group",
        "tenure_bucket",
        "high_balance",
    ]

    df_fe[categorical_features] = df_fe[categorical_features].astype("object")

    X = df_fe[features]
    y = df_fe[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
      steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
      ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    modelo_rf = RandomForestClassifier(
        random_state=42,
        n_estimators=200,
        max_depth=None
    )

    pipeline_rf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", modelo_rf),
        ]
    )

    pipeline_rf.fit(X_train, y_train)

    # Predicciones y métricas
    y_pred = pipeline_rf.predict(X_test)
    y_proba = pipeline_rf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba),
        "conf_matrix": confusion_matrix(y_test, y_pred),
        "y_test": y_test,
        "y_proba": y_proba,
    }

    # IMPORTANCIA DE VARIABLES (TOP 10)
    feature_names = pipeline_rf.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline_rf.named_steps["classifier"].feature_importances_
    fi_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    metrics["feature_importances"] = fi_df

    return df, pipeline_rf, metrics


# Intentamos entrenar el modelo. Si falta el CSV, mostramos error bonito.
try:
    df, pipeline_rf, metrics = train_model()
except FileNotFoundError:
    st.error(
        f"No se encontró el archivo **{DATA_PATH}**.\n\n"
        "➡ Colócalo en la MISMA carpeta que `app.py` "
        "o cambia la variable `DATA_PATH` en el código."
    )
    st.stop()


# -------------------------------------------------------------------
# 4. SIDEBAR DE NAVEGACIÓN
# -------------------------------------------------------------------
st.sidebar.header("Navegación")
page = st.sidebar.radio(
    "Ir a:",
    [
        "📊 Exploración de Datos",
        "📈 Métricas del Modelo",
        "🔮 Predicción Individual",
    ],
)


# -------------------------------------------------------------------
# 5. PÁGINA: EXPLORACIÓN DE DATOS
# -------------------------------------------------------------------
if page == "📊 Exploración de Datos":
    st.subheader("📊 Vista general del dataset")

    st.write(f"**Filas:** {df.shape[0]}   |   **Columnas:** {df.shape[1]}")
    st.dataframe(df.head())

    st.markdown("### Distribución de la variable objetivo (Churn)")
    churn_counts = df["churn"].value_counts().sort_index()
    fig, ax = plt.subplots()
    sns.barplot(x=churn_counts.index, y=churn_counts.values, ax=ax)
    ax.set_xlabel("Churn (0 = No abandona, 1 = Abandona)")
    ax.set_ylabel("Cantidad de clientes")
    st.pyplot(fig)

    st.markdown("### Estadísticos descriptivos de variables numéricas")
    st.dataframe(df.describe())


# -------------------------------------------------------------------
# 6. PÁGINA: MÉTRICAS DEL MODELO
# -------------------------------------------------------------------
elif page == "📈 Métricas del Modelo":
    st.subheader("📈 Resultados finales del modelo Random Forest")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    col2.metric("Precision", f"{metrics['precision']:.3f}")
    col3.metric("Recall", f"{metrics['recall']:.3f}")

    col4, col5 = st.columns(2)
    col4.metric("F1-Score", f"{metrics['f1']:.3f}")
    col5.metric("AUC ROC", f"{metrics['auc']:.3f}")

    st.markdown("### Matriz de confusión")
    conf_matrix = metrics["conf_matrix"]
    fig, ax = plt.subplots()
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax,
    )
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    st.pyplot(fig)

    st.markdown("### Curva ROC")
    fpr, tpr, _ = roc_curve(metrics["y_test"], metrics["y_proba"])
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {metrics['auc']:.3f}")
    ax2.plot([0, 1], [0, 1], "--", label="Azar")
    ax2.set_xlabel("Tasa de Falsos Positivos (FPR)")
    ax2.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
    ax2.set_title("Curva ROC")
    ax2.legend()
    st.pyplot(fig2)

    st.markdown("### Importancia de variables (Top 10)")
    fi_df = metrics["feature_importances"].head(10)
    st.dataframe(fi_df)

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=fi_df,
        x="importance",
        y="feature",
        ax=ax3,
    )
    ax3.set_title("Top 10 variables más importantes")
    ax3.set_xlabel("Importancia")
    ax3.set_ylabel("Variable")
    st.pyplot(fig3)


# -------------------------------------------------------------------
# 7. PÁGINA: PREDICCIÓN INDIVIDUAL
# -------------------------------------------------------------------
elif page == "🔮 Predicción Individual":
    st.subheader("🔮 Predicción de churn para un cliente específico")

    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score", 300, 900, 650)
        age = st.number_input("Edad", 18, 95, 35)
        tenure = st.number_input("Tenure (años como cliente)", 0, 20, 5)
        balance = st.number_input(
            "Balance (saldo en cuenta)",
            min_value=0.0,
            max_value=300000.0,
            value=50000.0,
            step=1000.0,
        )
        products_number = st.number_input("Número de productos", 1, 5, 1)

    with col2:
        estimated_salary = st.number_input(
            "Salario estimado",
            min_value=0.0,
            max_value=300000.0,
            value=60000.0,
            step=1000.0,
        )

        countries = sorted(df["country"].dropna().unique())
        genders = sorted(df["gender"].dropna().unique())
        bool_opts = [0, 1]

        country = st.selectbox("País de residencia", countries)
        gender = st.selectbox("Género", genders)
        credit_card = st.selectbox("¿Tiene tarjeta de crédito? (1=Sí, 0=No)", bool_opts)
        active_member = st.selectbox("¿Es cliente activo? (1=Sí, 0=No)", bool_opts)

    if st.button("Calcular probabilidad de Churn"):
        # Creamos un registro de ejemplo
        input_dict = {
            "customer_id": [0],  # valor dummy
            "credit_score": [credit_score],
            "country": [country],
            "gender": [gender],
            "age": [age],
            "tenure": [tenure],
            "balance": [balance],
            "products_number": [products_number],
            "credit_card": [credit_card],
            "active_member": [active_member],
            "estimated_salary": [estimated_salary],
            "churn": [0],  # dummy, no se usa en la predicción
        }

        df_input = pd.DataFrame(input_dict)
        df_input_fe = feature_engineering(df_input)

        drop_cols = ["customer_id"]
        target = "churn"
        features = [c for c in df_input_fe.columns if c not in [target] + drop_cols]

        X_new = df_input_fe[features]

        proba_churn = pipeline_rf.predict_proba(X_new)[:, 1][0]
        pred_label = pipeline_rf.predict(X_new)[0]

        st.write("---")
        st.write(f"**Probabilidad estimada de churn:** `{proba_churn:.3f}`")

        if pred_label == 1:
            st.error(
                "El modelo predice que **este cliente tiene ALTO riesgo de abandono (Churn = 1)**."
            )
        else:
            st.success(
                "El modelo predice que **este cliente probablemente permanecerá en el banco (Churn = 0)**."
            )
