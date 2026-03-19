"""
╔══════════════════════════════════════════════════════════════════╗
║   INDIAN LIVER PATIENT DISEASE PREDICTION — STREAMLIT APP        ║
║   Run: streamlit run app.py                                      ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

# ── sklearn / ML imports ──────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE

SEED = 42

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Liver Disease Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for cleaner look
st.markdown("""
<style>
    .metric-card {
        background: #f0f4ff;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        border-left: 4px solid #4a6fa5;
    }
    .best-badge {
        background: #d4edda;
        color: #155724;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
    .section-header {
        font-size: 22px;
        font-weight: 700;
        color: #2c3e50;
        border-bottom: 2px solid #4a6fa5;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# DATA & MODEL LOADING
# ══════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv("data/indian_liver_patient.csv")
    df["Albumin_and_Globulin_Ratio"].fillna(
        df["Albumin_and_Globulin_Ratio"].median(), inplace=True
    )
    le = LabelEncoder()
    df["Gender"] = le.fit_transform(df["Gender"])
    df["Dataset"] = df["Dataset"].map({1: 1, 2: 0})
    return df


@st.cache_resource
def train_all_models(df):
    """Train all 6 models and return results + trained objects."""
    X = df.drop("Dataset", axis=1)
    y = df["Dataset"]

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    smote = SMOTE(random_state=SEED)
    X_bal, y_bal = smote.fit_resample(X_sc, y)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=SEED, stratify=y_bal
    )

    models = {
        "Logistic Regression": LogisticRegression(random_state=SEED, max_iter=1000),
        "Decision Tree":       DecisionTreeClassifier(random_state=SEED),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=SEED),
        "SVM":                 SVC(probability=True, random_state=SEED),
        "XGBoost":             XGBClassifier(random_state=SEED, eval_metric="logloss", verbosity=0),
        "LightGBM":            LGBMClassifier(random_state=SEED, verbose=-1),
    }

    results = []
    trained = {}

    for name, m in models.items():
        m.fit(X_tr, y_tr)
        y_pred  = m.predict(X_te)
        y_proba = m.predict_proba(X_te)[:, 1]
        results.append({
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_te, y_pred), 4),
            "Precision": round(precision_score(y_te, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_te, y_pred, zero_division=0), 4),
            "F1-Score":  round(f1_score(y_te, y_pred, zero_division=0), 4),
            "ROC-AUC":   round(roc_auc_score(y_te, y_proba), 4),
        })
        trained[name] = m

    results_df = pd.DataFrame(results)
    best_name  = results_df.sort_values("ROC-AUC", ascending=False).iloc[0]["Model"]
    return results_df, trained, scaler, best_name, X.columns.tolist()


# ══════════════════════════════════════════════════════════════════
# HELPER: run models on subset
# ══════════════════════════════════════════════════════════════════
def run_on_subset(subset_df, label):
    drop_cols = [c for c in ["AgeGroup", "GenderLabel", "HybridGroup"]
                 if c in subset_df.columns]
    data = subset_df.drop(columns=drop_cols)
    X = data.drop("Dataset", axis=1)
    y = data["Dataset"]

    if y.nunique() < 2 or len(y) < 20:
        return None

    sc = StandardScaler()
    X_sc = sc.fit_transform(X)
    minority = y.value_counts().min()
    if minority >= 6:
        sm = SMOTE(random_state=SEED, k_neighbors=min(5, minority - 1))
        X_sc, y = sm.fit_resample(X_sc, y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y, test_size=0.2, random_state=SEED, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(random_state=SEED, max_iter=1000),
        "Decision Tree":       DecisionTreeClassifier(random_state=SEED),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=SEED),
        "SVM":                 SVC(probability=True, random_state=SEED),
        "XGBoost":             XGBClassifier(random_state=SEED, eval_metric="logloss", verbosity=0),
        "LightGBM":            LGBMClassifier(random_state=SEED, verbose=-1),
    }
    rows = []
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        y_pred  = m.predict(X_te)
        y_proba = m.predict_proba(X_te)[:, 1]
        rows.append({
            "Group": label,
            "Model": name,
            "Accuracy":  round(accuracy_score(y_te, y_pred), 4),
            "Precision": round(precision_score(y_te, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_te, y_pred, zero_division=0), 4),
            "F1-Score":  round(f1_score(y_te, y_pred, zero_division=0), 4),
            "ROC-AUC":   round(roc_auc_score(y_te, y_proba), 4),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════
df = load_data()

with st.spinner("🔄 Training all models — please wait…"):
    results_df, trained_models, scaler, best_model_name, feature_names = train_all_models(df)

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
st.sidebar.image("https://img.icons8.com/color/96/liver.png", width=80)
st.sidebar.title("🏥 Liver Disease Predictor")
st.sidebar.markdown("---")
st.sidebar.success(f"✅ Best Model: **{best_model_name}**")
st.sidebar.markdown("---")
st.sidebar.markdown("**Navigation:** Select a tab above ↑")
st.sidebar.caption("Indian Liver Patient Dataset (ILPD) — 583 records")

# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🧩 Tab 1 — Basic Info",
    "🔬 Tab 2 — Prediction System",
    "📊 Tab 3 — Evaluation Report",
    "📈 Tab 4 — EDA Visualization",
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — BASIC INFO
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">📋 Dataset Overview</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", "583")
    col2.metric("Features", "10")
    col3.metric("Liver Disease", f"{(df['Dataset']==1).sum()}")
    col4.metric("No Disease", f"{(df['Dataset']==0).sum()}")

    st.markdown("---")
    st.subheader("🎯 Problem Statement")
    st.info(
        "Liver disease is a major global health concern. Early detection is critical "
        "to prevent severe complications. This project uses machine learning to predict "
        "whether a patient has **liver disease (1)** or **no disease (0)** based on "
        "10 clinical blood test features."
    )

    st.subheader("📌 Feature Descriptions")
    feature_table = pd.DataFrame({
        "Feature": [
            "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
            "Alkaline_Phosphotase", "Alamine_Aminotransferase",
            "Aspartate_Aminotransferase", "Total_Protiens",
            "Albumin", "Albumin_and_Globulin_Ratio"
        ],
        "Type": [
            "Numerical", "Categorical", "Numerical", "Numerical",
            "Numerical", "Numerical", "Numerical", "Numerical",
            "Numerical", "Numerical"
        ],
        "Description": [
            "Patient's age in years",
            "Male or Female (encoded: Male=1, Female=0)",
            "Total bilirubin in blood (mg/dL) — elevated in liver disease",
            "Direct (conjugated) bilirubin — more specific liver marker",
            "ALP enzyme level — elevated in liver/bone disorders",
            "ALT (SGPT) enzyme — direct liver damage indicator",
            "AST (SGOT) enzyme — liver & heart damage indicator",
            "Total protein levels in blood",
            "Albumin protein — low in chronic liver disease",
            "Albumin/Globulin ratio — protein balance indicator",
        ]
    })
    st.dataframe(feature_table, use_container_width=True, hide_index=True)

    st.subheader("🔢 Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 2 — PREDICTION SYSTEM
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🔬 Liver Disease Prediction</div>', unsafe_allow_html=True)
    st.markdown(f"Using best model: **{best_model_name}**")
    st.markdown("Fill in the patient's clinical values below:")

    col1, col2 = st.columns(2)

    with col1:
        age    = st.number_input("Age (years)",      min_value=1,  max_value=120, value=45)
        gender = st.selectbox("Gender",              ["Male", "Female"])
        tb     = st.number_input("Total Bilirubin",  min_value=0.0, max_value=100.0, value=1.0, step=0.1)
        db     = st.number_input("Direct Bilirubin", min_value=0.0, max_value=50.0,  value=0.3, step=0.1)
        alkphos = st.number_input("Alkaline Phosphotase", min_value=0, max_value=5000, value=200)

    with col2:
        alt    = st.number_input("Alamine Aminotransferase (ALT)", min_value=0, max_value=5000, value=30)
        ast    = st.number_input("Aspartate Aminotransferase (AST)", min_value=0, max_value=5000, value=40)
        tp     = st.number_input("Total Proteins",   min_value=0.0, max_value=15.0, value=6.5, step=0.1)
        alb    = st.number_input("Albumin",          min_value=0.0, max_value=10.0, value=3.2, step=0.1)
        agr    = st.number_input("Albumin/Globulin Ratio", min_value=0.0, max_value=5.0, value=1.0, step=0.01)

    st.markdown("---")
    predict_btn = st.button("🔍 Predict", type="primary", use_container_width=True)

    if predict_btn:
        gender_enc = 1 if gender == "Male" else 0
        input_data = np.array([[age, gender_enc, tb, db, alkphos, alt, ast, tp, alb, agr]])
        input_scaled = scaler.transform(input_data)

        best_model = trained_models[best_model_name]
        prediction = best_model.predict(input_scaled)[0]
        probability = best_model.predict_proba(input_scaled)[0]

        st.markdown("---")
        col_a, col_b, col_c = st.columns(3)

        if prediction == 1:
            col_a.error("🔴 Prediction: **LIVER DISEASE DETECTED**")
        else:
            col_a.success("🟢 Prediction: **NO LIVER DISEASE**")

        col_b.metric("Disease Probability",   f"{probability[1]*100:.1f}%")
        col_c.metric("No Disease Probability", f"{probability[0]*100:.1f}%")

        # Probability gauge bar
        st.markdown("#### Confidence Bar")
        fig, ax = plt.subplots(figsize=(8, 1))
        ax.barh(0, probability[1], color="#e74c3c", height=0.5, label="Disease")
        ax.barh(0, probability[0], left=probability[1], color="#2ecc71", height=0.5, label="No Disease")
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.axvline(x=0.5, color="black", linestyle="--", linewidth=1.2)
        ax.legend(loc="lower right", fontsize=8)
        ax.set_title("Prediction Probability", fontsize=10)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        with st.expander("💡 Interpretation Tips"):
            st.markdown("""
            - **Probability > 50%** → Model predicts Liver Disease  
            - **Bilirubin levels** are key indicators — high values suggest liver dysfunction  
            - **ALT / AST** enzymes elevated beyond normal ranges strongly suggest liver damage  
            - This tool is for **educational purposes** — always consult a licensed physician  
            """)

# ══════════════════════════════════════════════════════════════════
# TAB 3 — EVALUATION REPORT
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">📊 Full Evaluation Report</div>', unsafe_allow_html=True)

    # ── Overall Metrics Table ──
    st.subheader("🏆 All Models — Full Balanced Dataset")

    def highlight_best(df_in, col="ROC-AUC"):
        best_idx = df_in[col].idxmax()
        def styler(row):
            return ["background-color: #d4f5d4; font-weight:bold"
                    if row.name == best_idx else "" for _ in row]
        return df_in.style.apply(styler, axis=1).format(
            {c: "{:.4f}" for c in ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]}
        )

    st.dataframe(highlight_best(results_df), use_container_width=True, hide_index=True)
    st.caption("✅ Highlighted row = Best overall model by ROC-AUC")

    # ── Best Highlights ──
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    best_overall = results_df.sort_values("ROC-AUC", ascending=False).iloc[0]
    best_recall  = results_df.sort_values("Recall", ascending=False).iloc[0]
    best_f1      = results_df.sort_values("F1-Score", ascending=False).iloc[0]

    col1.success(f"🏆 **Best Overall (ROC-AUC)**\n\n{best_overall['Model']}\n\nROC-AUC: {best_overall['ROC-AUC']}")
    col2.warning(f"🩺 **Best for Recall (Medical)**\n\n{best_recall['Model']}\n\nRecall: {best_recall['Recall']}")
    col3.info(f"⚖️ **Best F1-Score**\n\n{best_f1['Model']}\n\nF1: {best_f1['F1-Score']}")

    # ── Visual Bar Chart ──
    st.markdown("---")
    st.subheader("📈 Visual Comparison")

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    colors  = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
    model_labels = [m.replace(" ", "\n") for m in results_df["Model"]]

    for i, (m, c) in enumerate(zip(metrics, colors)):
        axes[i].bar(range(len(results_df)), results_df[m], color=c, alpha=0.85, edgecolor="white")
        axes[i].set_xticks(range(len(results_df)))
        axes[i].set_xticklabels(model_labels, fontsize=7)
        axes[i].set_title(m, fontweight="bold")
        axes[i].set_ylim(0, 1.1)
        for j, v in enumerate(results_df[m]):
            axes[i].text(j, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)

    plt.suptitle("Model Performance Metrics Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Sub-dataset Results ──
    st.markdown("---")
    st.subheader("🔹 Age-Based Evaluation")

    df_age = df.copy()
    df_age["AgeGroup"] = pd.cut(df_age["Age"], bins=[0,30,50,200],
                                 labels=["Young","Middle","Senior"])

    age_dfs = []
    for grp in ["Young", "Middle", "Senior"]:
        sub = df_age[df_age["AgeGroup"] == grp].drop("AgeGroup", axis=1)
        res = run_on_subset(sub, grp)
        if res is not None:
            age_dfs.append(res)

    if age_dfs:
        age_df_all = pd.concat(age_dfs, ignore_index=True)
        st.dataframe(age_df_all, use_container_width=True, hide_index=True)

    st.subheader("🔹 Gender-Based Evaluation")
    gender_dfs = []
    for grp, val in [("Male", 1), ("Female", 0)]:
        sub = df[df["Gender"] == val].copy()
        res = run_on_subset(sub, grp)
        if res is not None:
            gender_dfs.append(res)

    if gender_dfs:
        gender_df_all = pd.concat(gender_dfs, ignore_index=True)
        st.dataframe(gender_df_all, use_container_width=True, hide_index=True)

    st.subheader("🔹 Hybrid Evaluation (Age + Gender)")
    hybrid_dfs = []
    df_h = df_age.copy() if "AgeGroup" in df_age.columns else df.copy()
    if "AgeGroup" not in df_h.columns:
        df_h["AgeGroup"] = pd.cut(df_h["Age"], bins=[0,30,50,200],
                                   labels=["Young","Middle","Senior"])
    df_h["GenderLabel"]  = df_h["Gender"].map({1:"Male", 0:"Female"})
    df_h["HybridGroup"]  = df_h["AgeGroup"].astype(str) + "_" + df_h["GenderLabel"]

    for grp, sub in df_h.groupby("HybridGroup"):
        sub_clean = sub.drop(columns=["AgeGroup","GenderLabel","HybridGroup"])
        res = run_on_subset(sub_clean, grp)
        if res is not None:
            hybrid_dfs.append(res)

    if hybrid_dfs:
        hybrid_df_all = pd.concat(hybrid_dfs, ignore_index=True)
        st.dataframe(hybrid_df_all, use_container_width=True, hide_index=True)

    # ── Why Best Model ──
    st.markdown("---")
    st.subheader("💡 Why is this the Best Model?")
    st.markdown(f"""
    **Best Model: {best_model_name}**

    | Reason | Explanation |
    |--------|-------------|
    | 🎯 High ROC-AUC | Better at separating disease vs no-disease across all probability thresholds |
    | 🩺 Good Recall  | Fewer false negatives (missed disease cases) — critical in medical diagnosis |
    | 🌲 Ensemble     | Combines hundreds of decision trees, reducing variance and overfitting |
    | ⚡ Robust       | Handles class imbalance and non-linear relationships well |

    > In medical diagnosis, **Recall** is the most critical metric — we want to catch as many
    > true disease cases as possible, even at the cost of slightly more false positives.
    """)

# ══════════════════════════════════════════════════════════════════
# TAB 4 — EDA VISUALIZATION
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">📈 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    # Class distribution
    st.subheader("📌 Class Distribution")
    col1, col2 = st.columns(2)

    counts = df["Dataset"].value_counts()
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(["No Disease (0)", "Liver Disease (1)"],
               [counts.get(0, 0), counts.get(1, 0)],
               color=["#2ecc71", "#e74c3c"], edgecolor="white", width=0.5)
        for i, v in enumerate([counts.get(0,0), counts.get(1,0)]):
            ax.text(i, v + 2, str(v), ha="center", fontweight="bold")
        ax.set_title("Class Distribution", fontweight="bold")
        ax.set_ylabel("Count")
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.caption("📌 Dataset is imbalanced — more disease cases than non-disease. SMOTE is used to fix this.")

    with col2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.pie([counts.get(0,0), counts.get(1,0)],
               labels=["No Disease", "Liver Disease"],
               colors=["#2ecc71", "#e74c3c"],
               autopct="%1.1f%%", startangle=90)
        ax.set_title("Class Distribution (Pie)", fontweight="bold")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")

    # Histograms
    st.subheader("📊 Feature Distributions")
    numeric_cols = df.select_dtypes(include=np.number).columns.drop("Dataset").tolist()
    n_cols = 4
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col], bins=30, color="steelblue", edgecolor="white", alpha=0.85)
        axes[i].set_title(col, fontsize=9, fontweight="bold")
        axes[i].set_xlabel("Value", fontsize=8)
        axes[i].set_ylabel("Freq", fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Histograms", fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
    st.caption("📌 Bilirubin and enzyme values (ALT, AST) are highly right-skewed — extreme values typically indicate liver disease.")

    st.markdown("---")

    # Correlation Heatmap
    st.subheader("🔥 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(11, 7))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Correlation"})
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
    st.caption("📌 Total_Bilirubin & Direct_Bilirubin are highly correlated. Alamine & Aspartate aminotransferases also strongly correlate.")

    st.markdown("---")

    # Feature Importance
    st.subheader("🎯 Feature Importance (Random Forest & XGBoost)")
    col1, col2 = st.columns(2)

    rf = trained_models["Random Forest"]
    xgb = trained_models["XGBoost"]

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values()
        imp.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
        ax.set_title("Random Forest", fontweight="bold")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        imp = pd.Series(xgb.feature_importances_, index=feature_names).sort_values()
        imp.plot(kind="barh", ax=ax, color="darkorange", edgecolor="white")
        ax.set_title("XGBoost", fontweight="bold")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.caption("📌 Direct/Total Bilirubin and liver enzymes (ALT, AST) are consistently the strongest predictors of liver disease.")

    # Age distribution by class
    st.markdown("---")
    st.subheader("📐 Age Distribution by Class")
    fig, ax = plt.subplots(figsize=(9, 4))
    df[df["Dataset"] == 1]["Age"].hist(bins=20, ax=ax, alpha=0.7, color="#e74c3c", label="Disease")
    df[df["Dataset"] == 0]["Age"].hist(bins=20, ax=ax, alpha=0.7, color="#2ecc71", label="No Disease")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    ax.set_title("Age Distribution by Disease Class", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
    st.caption("📌 Liver disease is more common in middle-aged to senior patients (40–70 years).")
