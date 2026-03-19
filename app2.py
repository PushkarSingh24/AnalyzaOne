"""
╔══════════════════════════════════════════════════════════════════════╗
║  INDIAN LIVER PATIENT DISEASE PREDICTOR — PROJECT 2 (app2.py)       ║
║  Dark-themed medical dashboard · 9 Tabs · Stacking Classifier        ║
║  Run: streamlit run app2.py                                          ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from imblearn.over_sampling import SMOTE

SEED = 42
IMPORTANT_FEATURES = [
    "Total_Bilirubin", "Direct_Bilirubin",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Alkaline_Phosphotase",
]
IMP_LABELS = {
    "Total_Bilirubin": "Total Bilirubin",
    "Direct_Bilirubin": "Direct Bilirubin",
    "Alamine_Aminotransferase": "SGPT (ALT)",
    "Aspartate_Aminotransferase": "SGOT (AST)",
    "Alkaline_Phosphotase": "Alkaline Phosphatase",
}

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Liver Disease Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Light theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── global light background ── */
html, body, [class*="css"] {
    background-color: #f5f7fa !important;
    color: #1a1a2e !important;
    font-family: 'Segoe UI', sans-serif;
}
/* ── sidebar ── */
[data-testid="stSidebar"] {
    background-color: #eef2fb !important;
}
/* ── tab bar ── */
.stTabs [data-baseweb="tab-list"] {
    background-color: #e2e8f7;
    border-radius: 8px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #4a5568 !important;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background-color: #2563eb !important;
    color: #fff !important;
    border-radius: 6px;
}
/* ── metric boxes ── */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #d1daea;
    border-radius: 10px;
    padding: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
}
/* ── dataframe ── */
.stDataFrame { background: #ffffff !important; }
/* ── input widgets ── */
.stNumberInput input, .stSelectbox select, .stSlider {
    background-color: #ffffff !important;
    color: #1a1a2e !important;
}
/* ── buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white !important;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    padding: 10px 20px;
}
/* ── slogan banner ── */
.slogan-banner {
    background: linear-gradient(135deg, #dbeafe, #ede9fe);
    border-left: 4px solid #2563eb;
    padding: 14px 20px;
    border-radius: 8px;
    margin: 10px 0 20px 0;
}
.slogan-en { font-size: 18px; font-weight: 700; color: #1d4ed8; }
.slogan-hi { font-size: 16px; color: #5b21b6; margin-top: 4px; }
/* ── section header ── */
.sec-head {
    font-size: 20px; font-weight: 800; color: #1d4ed8;
    border-bottom: 2px solid #2563eb; padding-bottom: 6px; margin-bottom: 16px;
}
/* ── card ── */
.info-card {
    background: #ffffff; border-radius: 10px;
    padding: 16px; border: 1px solid #d1daea; margin-bottom: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
/* ── success / error override ── */
.stAlert { background: #ffffff !important; }
</style>
""", unsafe_allow_html=True)
 
# ─── matplotlib light style ────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#f5f7fa",
    "axes.facecolor":   "#ffffff",
    "axes.edgecolor":   "#cccccc",
    "axes.labelcolor":  "#1a1a2e",
    "xtick.color":      "#444444",
    "ytick.color":      "#444444",
    "text.color":       "#1a1a2e",
    "grid.color":       "#e2e8f0",
    "legend.facecolor": "#ffffff",
    "legend.edgecolor": "#cccccc",
})
 
SLOGAN_HTML = """
<div class="slogan-banner">
  <div class="slogan-en">💧 Be Salt Smart, Not Salt Scared.</div>
  <div class="slogan-hi">🌿 नमक स्वाद अनुसार नहीं, सेहत अनुसार!</div>
</div>
"""

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING (cached so it runs only once)
# ──────────────────────────────────────────────────────────────────────────────
def _get_models():
    return {
        "Logistic Regression": LogisticRegression(random_state=SEED, max_iter=1000),
        "Decision Tree":       DecisionTreeClassifier(random_state=SEED),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=SEED),
        "SVM":                 SVC(probability=True, random_state=SEED),
        "XGBoost":             XGBClassifier(random_state=SEED, eval_metric="logloss", verbosity=0),
        "LightGBM":            LGBMClassifier(random_state=SEED, verbose=-1),
    }

def _eval(model, Xtr, Xte, ytr, yte):
    model.fit(Xtr, ytr)
    yp = model.predict(Xte)
    yb = model.predict_proba(Xte)[:, 1]
    return {
        "Accuracy":  round(accuracy_score(yte, yp), 4),
        "Precision": round(precision_score(yte, yp, zero_division=0), 4),
        "Recall":    round(recall_score(yte, yp, zero_division=0), 4),
        "F1-Score":  round(f1_score(yte, yp, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(yte, yb), 4),
    }

def _prepare(subset_df, use_imp=True):
    drop = [c for c in ["Age","Gender","AgeGroup","GenderLabel","Hybrid"]
            if c in subset_df.columns]
    data = subset_df.drop(columns=drop)
    X = data[IMPORTANT_FEATURES] if use_imp else data.drop("Dataset", axis=1)
    y = data["Dataset"]
    if y.nunique() < 2 or len(y) < 20:
        return None
    sc = StandardScaler()
    Xsc = sc.fit_transform(X)
    mn = y.value_counts().min()
    if mn >= 6:
        sm = SMOTE(random_state=SEED, k_neighbors=min(5, mn-1))
        Xsc, y = sm.fit_resample(Xsc, y)
    return train_test_split(Xsc, y, test_size=0.2, random_state=SEED, stratify=y), sc

def _run_subset(sub_df, label):
    res = _prepare(sub_df)
    if res is None:
        return None, None, None
    (Xtr, Xte, ytr, yte), sc = res
    rows, trained = [], {}
    for n, m in _get_models().items():
        met = _eval(m, Xtr, Xte, ytr, yte)
        met.update({"Model": n, "Group": label})
        rows.append(met)
        trained[n] = m
    return pd.DataFrame(rows)[["Group","Model","Accuracy","Precision","Recall","F1-Score","ROC-AUC"]], trained, (Xtr, Xte, ytr, yte)

@st.cache_resource
def build_everything():
    df = load_data()
    X = df[IMPORTANT_FEATURES]
    y = df["Dataset"]
    Xall = df.drop("Dataset", axis=1)

    sc_imp = StandardScaler(); Ximp = sc_imp.fit_transform(X)
    sc_all = StandardScaler(); Xall_sc = sc_all.fit_transform(Xall)

    sm = SMOTE(random_state=SEED)
    Xb, yb = sm.fit_resample(Ximp, y)
    Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.2, random_state=SEED, stratify=yb)

    # full dataset results
    full_rows, full_trained = [], {}
    for n, m in _get_models().items():
        met = _eval(m, Xtr, Xte, ytr, yte)
        met["Model"] = n
        full_rows.append(met)
        full_trained[n] = m
    full_df = pd.DataFrame(full_rows)[["Model","Accuracy","Precision","Recall","F1-Score","ROC-AUC"]]

    # subsets
    dfw = df[IMPORTANT_FEATURES + ["Age","Gender","Dataset"]].copy()
    dfw["AgeGroup"] = pd.cut(dfw["Age"], bins=[0,30,50,200], labels=["Young","Middle","Senior"])
    dfw["GenderLabel"] = dfw["Gender"].map({1:"Male",0:"Female"})
    dfw["Hybrid"] = dfw["AgeGroup"].astype(str) + "_" + dfw["GenderLabel"]

    age_rows,  age_tr,  age_sp  = [], {}, {}
    for g, s in [("Young", dfw[dfw["AgeGroup"]=="Young"]),
                  ("Middle", dfw[dfw["AgeGroup"]=="Middle"]),
                  ("Senior", dfw[dfw["AgeGroup"]=="Senior"])]:
        r, t, sp = _run_subset(s.copy(), g)
        if r is not None:
            age_rows.append(r); age_tr[g] = t; age_sp[g] = sp

    gender_rows, gen_tr, gen_sp = [], {}, {}
    for g, v in [("Male",1),("Female",0)]:
        s = dfw[dfw["Gender"]==v].copy()
        r, t, sp = _run_subset(s, g)
        if r is not None:
            gender_rows.append(r); gen_tr[g] = t; gen_sp[g] = sp

    hybrid_rows, hyb_tr, hyb_sp = [], {}, {}
    for g, s in dfw.groupby("Hybrid"):
        r, t, sp = _run_subset(s.copy(), g)
        if r is not None:
            hybrid_rows.append(r); hyb_tr[g] = t; hyb_sp[g] = sp

    age_df    = pd.concat(age_rows,    ignore_index=True) if age_rows    else pd.DataFrame()
    gender_df = pd.concat(gender_rows, ignore_index=True) if gender_rows else pd.DataFrame()
    hybrid_df = pd.concat(hybrid_rows, ignore_index=True) if hybrid_rows else pd.DataFrame()

    # ── Stacking on full hybrid set ──
    dh = dfw.drop(columns=["Age","Gender","AgeGroup","GenderLabel","Hybrid"])
    Xh = dh[IMPORTANT_FEATURES]; yh = dh["Dataset"]
    sc_h = StandardScaler(); Xh_sc = sc_h.fit_transform(Xh)
    sm2 = SMOTE(random_state=SEED)
    Xhb, yhb = sm2.fit_resample(Xh_sc, yh)
    Xhtr, Xhte, yhtr, yhte = train_test_split(Xhb, yhb, test_size=0.2, random_state=SEED, stratify=yhb)

    stacking = StackingClassifier(
        estimators=[
            ("rf",  RandomForestClassifier(n_estimators=100, random_state=SEED)),
            ("svm", SVC(probability=True, random_state=SEED)),
            ("xgb", XGBClassifier(random_state=SEED, eval_metric="logloss", verbosity=0)),
        ],
        final_estimator=LogisticRegression(random_state=SEED, max_iter=1000),
        cv=5, stack_method="predict_proba"
    )
    stacking.fit(Xhtr, yhtr)

    # hybrid final comparison
    hyb_full_rows, hyb_full_trained = [], {}
    for n, m in _get_models().items():
        met = _eval(m, Xhtr, Xhte, yhtr, yhte)
        met["Model"] = n
        hyb_full_rows.append(met)
        hyb_full_trained[n] = m

    ysp = stacking.predict(Xhte)
    yspb = stacking.predict_proba(Xhte)[:,1]
    hyb_full_rows.append({
        "Model": "⭐ Stacking Classifier",
        "Accuracy":  round(accuracy_score(yhte, ysp), 4),
        "Precision": round(precision_score(yhte, ysp, zero_division=0), 4),
        "Recall":    round(recall_score(yhte, ysp, zero_division=0), 4),
        "F1-Score":  round(f1_score(yhte, ysp, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(yhte, yspb), 4),
    })
    hyb_final_df = pd.DataFrame(hyb_full_rows)[["Model","Accuracy","Precision","Recall","F1-Score","ROC-AUC"]]

    best_full_name = full_df.sort_values("ROC-AUC", ascending=False).iloc[0]["Model"]

    return {
        "df": df, "dfw": dfw,
        "sc_imp": sc_imp, "sc_all": sc_all, "sc_h": sc_h,
        "full_df": full_df, "full_trained": full_trained,
        "Xtr": Xtr, "Xte": Xte, "ytr": ytr, "yte": yte,
        "age_df": age_df, "age_tr": age_tr,
        "gender_df": gender_df, "gen_tr": gen_tr,
        "hybrid_df": hybrid_df, "hyb_tr": hyb_tr,
        "hyb_final_df": hyb_final_df, "hyb_full_trained": hyb_full_trained,
        "stacking": stacking,
        "Xhtr": Xhtr, "Xhte": Xhte, "yhtr": yhtr, "yhte": yhte,
        "best_full_name": best_full_name,
    }

# ──────────────────────────────────────────────────────────────────────────────
# LOAD
# ──────────────────────────────────────────────────────────────────────────────
with st.spinner("🔄 Training all models (including Stacking)…"):
    D = build_everything()

df          = D["df"]
full_df     = D["full_df"]
age_df      = D["age_df"]
gender_df   = D["gender_df"]
hybrid_df   = D["hybrid_df"]
hyb_final   = D["hyb_final_df"]
best_name   = D["best_full_name"]
stacking    = D["stacking"]

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("# 🏥 Liver Disease\nDashboard")
st.sidebar.markdown("---")
st.sidebar.success(f"Best Model: **{best_name}**")
st.sidebar.info("Features: Total/Direct Bilirubin, SGPT, SGOT, Alk. Phosphatase")
st.sidebar.markdown("---")
st.sidebar.markdown(SLOGAN_HTML, unsafe_allow_html=True)
st.sidebar.caption("ILPD · 583 patients · Project 2")

# ── TABS ───────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🧩 Basic Info",
    "🎯 Overall Prediction",
    "🔬 Advanced Prediction",
    "📊 Evaluation Report",
    "📈 EDA Visualization",
    "🔗 Feature Relationships",
    "⚗️ Lifestyle Simulator",
    "📉 Future Progression",
    "🥗 Diet & Prevention",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — BASIC INFO
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown(SLOGAN_HTML, unsafe_allow_html=True)
    st.markdown('<div class="sec-head">📋 Dataset Overview</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Patients", "583")
    c2.metric("Features Used", "5 Key")
    c3.metric("Liver Disease", str((df["Dataset"]==1).sum()))
    c4.metric("No Disease",    str((df["Dataset"]==0).sum()))

    st.markdown("---")
    st.subheader("🎯 Problem Statement")
    st.info(
        "Liver disease is one of the leading causes of mortality worldwide. "
        "Early and accurate prediction using blood test markers allows timely treatment. "
        "This project builds a complete ML pipeline using **5 key liver biomarkers** "
        "to classify patients as **Liver Disease (1)** or **No Disease (0)**."
    )

    st.subheader("🔬 Selected Features (SGPT/SGOT Based)")
    feat_info = pd.DataFrame({
        "Feature": list(IMP_LABELS.keys()),
        "Label": list(IMP_LABELS.values()),
        "Normal Range": ["0.2–1.2 mg/dL", "0–0.3 mg/dL", "7–56 U/L", "10–40 U/L", "44–147 U/L"],
        "What it Indicates": [
            "Total yellow pigment in blood — elevated in jaundice/liver disease",
            "More specific liver marker for bile duct obstruction",
            "ALT — primary liver damage enzyme (elevated in hepatitis)",
            "AST — liver + heart damage indicator",
            "ALP — elevated in liver and bone disorders",
        ]
    })
    st.dataframe(feat_info, use_container_width=True, hide_index=True)

    st.subheader("📊 Raw Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — OVERALL PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown(SLOGAN_HTML, unsafe_allow_html=True)
    st.markdown('<div class="sec-head">🎯 Overall Liver Disease Prediction</div>', unsafe_allow_html=True)
    st.markdown(f"Uses best model: **{best_name}** (highest ROC-AUC on full balanced dataset)")

    col1, col2 = st.columns([1,1])
    with col1:
        tb   = st.number_input("Total Bilirubin (mg/dL)",      0.0, 100.0, 1.0, 0.1, key="ov_tb")
        db   = st.number_input("Direct Bilirubin (mg/dL)",     0.0, 50.0,  0.3, 0.1, key="ov_db")
        sgpt = st.number_input("SGPT / ALT (U/L)",             0,   5000,  30,  1,   key="ov_sgpt")
        sgot = st.number_input("SGOT / AST (U/L)",             0,   5000,  40,  1,   key="ov_sgot")
        alkp = st.number_input("Alkaline Phosphatase (U/L)",   0,   5000,  200, 1,   key="ov_alkp")

    with col2:
        st.markdown("#### Reference Ranges")
        st.markdown("""
| Marker | Normal | Concern |
|--------|--------|---------|
| Total Bilirubin | 0.2–1.2 | > 2.0 |
| Direct Bilirubin | 0–0.3 | > 0.5 |
| SGPT (ALT) | 7–56 | > 100 |
| SGOT (AST) | 10–40 | > 80 |
| Alk. Phosphatase | 44–147 | > 300 |
""")

    st.markdown("---")
    if st.button("🔍 Predict Now", key="ov_pred", use_container_width=True):
        inp = np.array([[tb, db, sgpt, sgot, alkp]])
        inp_sc = D["sc_imp"].transform(inp)
        best_m = D["full_trained"][best_name]
        pred   = best_m.predict(inp_sc)[0]
        proba  = best_m.predict_proba(inp_sc)[0]

        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        if pred == 1:
            r1.error("🔴 LIVER DISEASE DETECTED")
        else:
            r1.success("🟢 NO LIVER DISEASE")
        r2.metric("Disease Probability",    f"{proba[1]*100:.1f}%")
        r3.metric("No Disease Probability", f"{proba[0]*100:.1f}%")

        fig, ax = plt.subplots(figsize=(8, 1.2))
        ax.barh(0, proba[1], color="#ef4444", height=0.5, label="Disease")
        ax.barh(0, proba[0], left=proba[1], color="#22c55e", height=0.5, label="No Disease")
        ax.axvline(0.5, color="#fff", linestyle="--", lw=1.5)
        ax.set_xlim(0,1); ax.set_yticks([])
        ax.legend(loc="lower right", fontsize=9)
        ax.set_title("Prediction Confidence", fontsize=10)
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ADVANCED PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown(SLOGAN_HTML, unsafe_allow_html=True)
    st.markdown('<div class="sec-head">🔬 Advanced Prediction System</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        pred_type = st.selectbox("Prediction Type", ["Age-Based","Gender-Based","Hybrid"])
    with c2:
        feat_mode = st.selectbox("Input Mode", ["Important Features Only","All Features"])
    with c3:
        model_opts = list(_get_models().keys())
        if pred_type == "Hybrid":
            model_opts = model_opts + ["⭐ Stacking Classifier"]
        sel_model = st.selectbox("Select Model", model_opts)

    st.markdown("---")
    st.subheader("📝 Enter Patient Values")

    if feat_mode == "Important Features Only":
        cf1, cf2 = st.columns(2)
        with cf1:
            atb   = st.number_input("Total Bilirubin",   0.0, 100.0, 1.0, 0.1, key="adv_tb")
            adb   = st.number_input("Direct Bilirubin",  0.0, 50.0,  0.3, 0.1, key="adv_db")
            asgpt = st.number_input("SGPT (ALT)",        0,   5000,  30,  1,   key="adv_sgpt")
        with cf2:
            asgot = st.number_input("SGOT (AST)",        0,   5000,  40,  1,   key="adv_sgot")
            aalkp = st.number_input("Alk. Phosphatase",  0,   5000,  200, 1,   key="adv_alkp")
        inp_arr = np.array([[atb, adb, asgpt, asgot, aalkp]])
        use_sc  = D["sc_imp"]
    else:
        # All features
        all_cols = [c for c in df.columns if c != "Dataset"]
        vals = {}
        col_pairs = [all_cols[i:i+3] for i in range(0, len(all_cols), 3)]
        for group in col_pairs:
            cs = st.columns(len(group))
            for ci, col in enumerate(group):
                if col == "Gender":
                    vals[col] = 1 if cs[ci].selectbox("Gender", ["Male","Female"], key=f"adv_{col}") == "Male" else 0
                else:
                    mn = float(df[col].min()); mx = float(df[col].max()); md_ = float(df[col].median())
                    vals[col] = cs[ci].number_input(col, mn, mx, md_, key=f"adv_{col}")
        inp_arr = np.array([[vals[c] for c in all_cols]])
        use_sc  = D["sc_all"]

    st.markdown("---")
    if st.button("🔍 Run Advanced Prediction", key="adv_pred", use_container_width=True):
        inp_sc = use_sc.transform(inp_arr[:, :len(IMPORTANT_FEATURES)] if feat_mode == "Important Features Only" else inp_arr)

        if sel_model == "⭐ Stacking Classifier":
            # Stacking always uses important features
            inp_sc2 = D["sc_h"].transform(inp_arr[:, :len(IMPORTANT_FEATURES)] if feat_mode=="Important Features Only" else inp_arr[:, :len(IMPORTANT_FEATURES)])
            pred  = stacking.predict(inp_sc2)[0]
            proba = stacking.predict_proba(inp_sc2)[0]
        else:
            # pick from appropriate trained dict
            if pred_type == "Age-Based":
                src = list(D["age_tr"].values())[0] if D["age_tr"] else D["full_trained"]
            elif pred_type == "Gender-Based":
                src = list(D["gen_tr"].values())[0] if D["gen_tr"] else D["full_trained"]
            else:
                src = list(D["hyb_tr"].values())[0] if D["hyb_tr"] else D["full_trained"]
            m = src.get(sel_model, D["full_trained"][sel_model])
            pred  = m.predict(inp_sc[:, :len(IMPORTANT_FEATURES)])[0]
            proba = m.predict_proba(inp_sc[:, :len(IMPORTANT_FEATURES)])[0]

        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        if pred == 1:
            r1.error(f"🔴 LIVER DISEASE DETECTED\n\nModel: {sel_model}")
        else:
            r1.success(f"🟢 NO LIVER DISEASE\n\nModel: {sel_model}")
        r2.metric("Disease Prob",    f"{proba[1]*100:.1f}%")
        r3.metric("No Disease Prob", f"{proba[0]*100:.1f}%")

        fig, ax = plt.subplots(figsize=(8, 1.2))
        ax.barh(0, proba[1], color="#ef4444", height=0.5, label="Disease")
        ax.barh(0, proba[0], left=proba[1], color="#22c55e", height=0.5, label="No Disease")
        ax.axvline(0.5, color="white", ls="--", lw=1.5)
        ax.set_xlim(0,1); ax.set_yticks([])
        ax.legend(fontsize=9, loc="lower right")
        ax.set_title(f"Confidence — {sel_model}", fontsize=10)
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EVALUATION REPORT
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown(SLOGAN_HTML, unsafe_allow_html=True)
    st.markdown('<div class="sec-head">📊 Final Evaluation Report</div>', unsafe_allow_html=True)

    def style_table(df_in):
        best_idx = df_in["ROC-AUC"].idxmax()
        def hl(row):
            return ["background-color:#1e3a5f;font-weight:bold" if row.name==best_idx else "" for _ in row]
        return df_in.style.apply(hl, axis=1).format(
            {c:"{:.4f}" for c in ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"] if c in df_in.columns}
        )

    st.subheader("🏆 All Models — Full Balanced Dataset")
    st.dataframe(style_table(full_df), use_container_width=True, hide_index=True)

    bo = full_df.sort_values("ROC-AUC",ascending=False).iloc[0]
    br = full_df.sort_values("Recall", ascending=False).iloc[0]
    c1,c2 = st.columns(2)
    c1.success(f"🏆 Best Overall (ROC-AUC): **{bo['Model']}** — AUC {bo['ROC-AUC']}")
    c2.warning(f"🩺 Best Recall (Medical) : **{br['Model']}** — Recall {br['Recall']}")

    # Visual comparison
    fig, axes = plt.subplots(1, 5, figsize=(18,4))
    metrics_v = ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]
    cols_v    = ["#3b82f6","#ef4444","#22c55e","#f59e0b","#a855f7"]
    for i, (m, c) in enumerate(zip(metrics_v, cols_v)):
        axes[i].bar(range(len(full_df)), full_df[m], color=c, alpha=0.85, edgecolor="none")
        axes[i].set_xticks(range(len(full_df)))
        axes[i].set_xticklabels([n.replace(" ","\n") for n in full_df["Model"]], fontsize=7)
        axes[i].set_title(m, fontweight="bold")
        axes[i].set_ylim(0, 1.15)
        for j,v in enumerate(full_df[m]):
            axes[i].text(j, v+0.01, f"{v:.2f}", ha="center", fontsize=7)
    plt.suptitle("Model Performance — Full Dataset", fontsize=12, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")
    st.subheader("🔹 Age-Based Evaluation")
    if not age_df.empty:
        st.dataframe(age_df, use_container_width=True, hide_index=True)

    st.subheader("🔹 Gender-Based Evaluation")
    if not gender_df.empty:
        st.dataframe(gender_df, use_container_width=True, hide_index=True)

    st.subheader("🔹 Hybrid Evaluation")
    if not hybrid_df.empty:
        st.dataframe(hybrid_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("⭐ Final Hybrid Comparison (Including Stacking Classifier)")
    st.dataframe(style_table(hyb_final), use_container_width=True, hide_index=True)
    bh = hyb_final.sort_values("ROC-AUC",ascending=False).iloc[0]
    st.success(f"🏆 Best Hybrid Model: **{bh['Model']}** — ROC-AUC {bh['ROC-AUC']}")

    st.markdown("---")
    st.subheader("💡 Best Model per Dataset Type")
    summary = []
    for label, df_r in [("Age-Based", age_df), ("Gender-Based", gender_df),
                         ("Hybrid", hybrid_df), ("Full Dataset", full_df)]:
        if not df_r.empty:
            b = df_r.sort_values("ROC-AUC", ascending=False).iloc[0]
            summary.append({"Dataset Type": label,
                             "Best Model": b.get("Model",""), "Group": b.get("Group","-"),
                             "ROC-AUC": b["ROC-AUC"], "Recall": b["Recall"]})
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown(SLOGAN_HTML, unsafe_allow_html=True)
    st.markdown('<div class="sec-head">📈 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    # Class distribution
    st.subheader("📌 Class Distribution")
    c1,c2 = st.columns(2)
    counts = df["Dataset"].value_counts()
    with c1:
        fig, ax = plt.subplots(figsize=(5,3.5))
        ax.bar(["No Disease","Liver Disease"], [counts.get(0,0), counts.get(1,0)],
               color=["#22c55e","#ef4444"], edgecolor="none", width=0.5)
        for i,v in enumerate([counts.get(0,0), counts.get(1,0)]):
            ax.text(i, v+2, str(v), ha="center", fontweight="bold")
        ax.set_title("Class Distribution", fontweight="bold")
        st.pyplot(fig, use_container_width=True); plt.close()
    with c2:
        fig, ax = plt.subplots(figsize=(5,3.5))
        ax.pie([counts.get(0,0), counts.get(1,0)], labels=["No Disease","Liver Disease"],
               colors=["#22c55e","#ef4444"], autopct="%1.1f%%", startangle=90,
               wedgeprops={"edgecolor":"#0f1117"})
        ax.set_title("Class Split", fontweight="bold")
        st.pyplot(fig, use_container_width=True); plt.close()
    st.caption("📌 Dataset is imbalanced (~71% disease). SMOTE balances this before training.")

    # Histograms
    st.markdown("---")
    st.subheader("📊 Feature Histograms")
    fig, axes = plt.subplots(2,3, figsize=(15,8))
    axes = axes.flatten()
    for i,(col,lbl) in enumerate(IMP_LABELS.items()):
        axes[i].hist(df[col], bins=30, color="#3b82f6", edgecolor="none", alpha=0.85)
        axes[i].set_title(lbl, fontweight="bold")
        axes[i].set_xlabel("Value"); axes[i].set_ylabel("Freq")
    axes[5].set_visible(False)
    plt.suptitle("Histograms — Key Features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()
    st.caption("📌 All enzyme values are right-skewed — extreme values indicate disease.")

    # Boxplots
    st.markdown("---")
    st.subheader("📦 Boxplots by Disease Class")
    fig, axes = plt.subplots(2,3, figsize=(15,8))
    axes = axes.flatten()
    for i,(col,lbl) in enumerate(IMP_LABELS.items()):
        d0 = df[df["Dataset"]==0][col]
        d1 = df[df["Dataset"]==1][col]
        bp = axes[i].boxplot([d0.values, d1.values], labels=["No Disease","Disease"],
                              patch_artist=True)
        bp["boxes"][0].set_facecolor("#22c55e")
        bp["boxes"][1].set_facecolor("#ef4444")
        for m in bp["medians"]: m.set_color("white")
        axes[i].set_title(lbl, fontweight="bold"); axes[i].set_xlabel("")
    axes[5].set_visible(False)
    plt.suptitle("Boxplots — Feature vs Disease Class", fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()
    st.caption("📌 Disease patients show significantly higher enzyme levels and more outliers.")

    # Heatmap
    st.markdown("---")
    st.subheader("🔥 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8,6))
    corr = df[IMPORTANT_FEATURES + ["Dataset"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, cbar_kws={"label":"Correlation"})
    ax.set_title("Correlation — Key Features + Target", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()
    st.caption("📌 Total & Direct Bilirubin are highly correlated. SGPT-SGOT pair is also strongly correlated.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — FEATURE RELATIONSHIPS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown(SLOGAN_HTML, unsafe_allow_html=True)
    st.markdown('<div class="sec-head">🔗 Feature Relationships</div>', unsafe_allow_html=True)

    st.subheader("Scatter Plot: Feature vs Feature")
    c1,c2 = st.columns(2)
    feat_opts = list(IMP_LABELS.keys())
    feat_x = c1.selectbox("X Axis", feat_opts, index=0, key="sc_x")
    feat_y = c2.selectbox("Y Axis", feat_opts, index=1, key="sc_y")

    fig, ax = plt.subplots(figsize=(8,5))
    for cls, col, lbl in [(0,"#22c55e","No Disease"),(1,"#ef4444","Liver Disease")]:
        sub = df[df["Dataset"]==cls]
        ax.scatter(sub[feat_x], sub[feat_y], c=col, alpha=0.5, label=lbl, s=25, edgecolors="none")
    ax.set_xlabel(IMP_LABELS[feat_x], fontsize=11)
    ax.set_ylabel(IMP_LABELS[feat_y], fontsize=11)
    ax.set_title(f"{IMP_LABELS[feat_x]} vs {IMP_LABELS[feat_y]}", fontweight="bold")
    ax.legend()
    st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("---")
    st.subheader("Feature vs Target (Mean Comparison)")
    means = df.groupby("Dataset")[IMPORTANT_FEATURES].mean().T
    means.columns = ["No Disease (0)", "Liver Disease (1)"]
    fig, ax = plt.subplots(figsize=(10,4))
    x = np.arange(len(means))
    w = 0.35
    ax.bar(x-w/2, means["No Disease (0)"], w, label="No Disease", color="#22c55e", alpha=0.85)
    ax.bar(x+w/2, means["Liver Disease (1)"], w, label="Liver Disease", color="#ef4444", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([IMP_LABELS[c] for c in means.index], fontsize=9)
    ax.set_title("Mean Feature Values by Class", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()
    st.caption("📌 All enzyme markers are significantly higher in liver disease patients.")

    st.markdown("---")
    st.subheader("ROC Curves — All Models")
    fig, ax = plt.subplots(figsize=(9,6))
    roc_colors = ["#3b82f6","#ef4444","#22c55e","#f59e0b","#a855f7","#06b6d4"]
    for (n,m), c in zip(D["full_trained"].items(), roc_colors):
        fpr, tpr, _ = roc_curve(D["yte"], m.predict_proba(D["Xte"])[:,1])
        auc = roc_auc_score(D["yte"], m.predict_proba(D["Xte"])[:,1])
        ax.plot(fpr, tpr, label=f"{n} (AUC={auc:.3f})", color=c, lw=2)
    ax.plot([0,1],[0,1],"w--", lw=1.2, label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves", fontweight="bold"); ax.legend(fontsize=8); ax.grid(alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — LIFESTYLE SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown(SLOGAN_HTML, unsafe_allow_html=True)
    st.markdown('<div class="sec-head">⚗️ Lifestyle Risk Simulator</div>', unsafe_allow_html=True)
    st.markdown("Simulate how changing your liver marker values affects disease risk.")

    best_m = D["full_trained"][best_name]

    c1, c2 = st.columns(2)
    with c1:
        s_tb   = st.slider("Total Bilirubin",    0.0,  30.0, 1.0, 0.1, key="sim_tb")
        s_db   = st.slider("Direct Bilirubin",   0.0,  15.0, 0.3, 0.1, key="sim_db")
        s_sgpt = st.slider("SGPT (ALT) U/L",     0,    500,  30,  5,   key="sim_sgpt")
    with c2:
        s_sgot = st.slider("SGOT (AST) U/L",     0,    500,  40,  5,   key="sim_sgot")
        s_alkp = st.slider("Alk. Phosphatase",    0,    2000, 200, 10,  key="sim_alkp")

    inp = np.array([[s_tb, s_db, s_sgpt, s_sgot, s_alkp]])
    inp_sc = D["sc_imp"].transform(inp)
    pred  = best_m.predict(inp_sc)[0]
    proba = best_m.predict_proba(inp_sc)[0]

    st.markdown("---")
    r1,r2,r3 = st.columns(3)
    if pred == 1:
        r1.error("🔴 HIGH RISK — Liver Disease")
    else:
        r1.success("🟢 LOW RISK — No Disease")
    r2.metric("Risk Score",       f"{proba[1]*100:.1f}%")
    r3.metric("Safety Score",     f"{proba[0]*100:.1f}%")

    # Gauge chart
    fig, ax = plt.subplots(figsize=(8,1.5))
    ax.barh(0, proba[1], color="#ef4444", height=0.6, label="Disease Risk")
    ax.barh(0, proba[0], left=proba[1], color="#22c55e", height=0.6, label="Safe")
    ax.axvline(0.5, color="white", ls="--", lw=2)
    ax.set_xlim(0,1); ax.set_yticks([])
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("Risk Gauge", fontsize=11, fontweight="bold")
    st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("---")
    st.info("🔁 **Tip:** Try increasing SGPT/SGOT above 200 to simulate hepatitis-like values and see risk jump.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — FUTURE PROGRESSION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown(SLOGAN_HTML, unsafe_allow_html=True)
    st.markdown('<div class="sec-head">📉 Future Risk Progression</div>', unsafe_allow_html=True)
    st.markdown("See how your disease risk changes as a marker value increases over time.")

    best_m = D["full_trained"][best_name]

    prog_feat = st.selectbox("Feature to Progress", list(IMP_LABELS.keys()), key="prog_feat")
    prog_max  = st.slider(f"Max value for {IMP_LABELS[prog_feat]}", 10, 2000, 300)

    # Baseline = median values
    baseline = {f: float(df[f].median()) for f in IMPORTANT_FEATURES}
    sweep     = np.linspace(float(df[prog_feat].min()), prog_max, 60)
    risks     = []
    for v in sweep:
        row = [baseline[f] if f != prog_feat else v for f in IMPORTANT_FEATURES]
        sc_row = D["sc_imp"].transform([row])
        risks.append(best_m.predict_proba(sc_row)[0][1])

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(sweep, risks, color="#ef4444", lw=2.5)
    ax.fill_between(sweep, risks, alpha=0.15, color="#ef4444")
    ax.axhline(0.5, color="white", ls="--", lw=1.5, label="Risk Threshold (50%)")
    ax.set_xlabel(IMP_LABELS[prog_feat], fontsize=11)
    ax.set_ylabel("Disease Probability", fontsize=11)
    ax.set_title(f"Risk Progression vs {IMP_LABELS[prog_feat]}", fontweight="bold")
    ax.set_ylim(0,1); ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    # Risk levels summary
    thresh_val = sweep[next((i for i,r in enumerate(risks) if r>=0.5), -1)]
    if any(r>=0.5 for r in risks):
        st.warning(f"⚠️ Risk crosses 50% when **{IMP_LABELS[prog_feat]}** exceeds **{thresh_val:.1f}**")
    else:
        st.success(f"✅ Risk stays below 50% across this range for {IMP_LABELS[prog_feat]}")

    st.caption("All other features are held at their median values during simulation.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — DIET & PREVENTION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[8]:
    st.markdown(SLOGAN_HTML, unsafe_allow_html=True)
    st.markdown('<div class="sec-head">🥗 Diet & Liver Health Prevention</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("✅ Foods That Support Liver Health")
        st.markdown("""
<div class="info-card">
🥦 <b>Cruciferous Vegetables</b> — Broccoli, cauliflower, Brussels sprouts<br>
Boost liver enzymes that flush out carcinogens
</div>
<div class="info-card">
🫒 <b>Olive Oil</b> — Reduces liver fat accumulation (NAFLD)
</div>
<div class="info-card">
🍵 <b>Green Tea</b> — Antioxidants (catechins) reduce liver inflammation
</div>
<div class="info-card">
🫐 <b>Berries</b> — Blueberries & cranberries protect liver from oxidative stress
</div>
<div class="info-card">
🥜 <b>Nuts</b> — High in Vitamin E, protect against NAFLD
</div>
<div class="info-card">
☕ <b>Coffee</b> — Studies show 2 cups/day lowers risk of liver cirrhosis by 44%
</div>
<div class="info-card">
🐟 <b>Fatty Fish</b> — Omega-3s reduce liver fat and inflammation
</div>
""", unsafe_allow_html=True)

    with c2:
        st.subheader("❌ Foods That Harm the Liver")
        st.markdown("""
<div class="info-card">
🍺 <b>Alcohol</b> — Even moderate consumption elevates SGPT/SGOT & bilirubin
</div>
<div class="info-card">
🍟 <b>Fried & Processed Foods</b> — Increase liver fat leading to fatty liver
</div>
<div class="info-card">
🧂 <b>High Salt Diet</b> — Leads to fluid retention and liver stress
</div>
<div class="info-card">
🍭 <b>Added Sugars</b> — Fructose overload causes non-alcoholic fatty liver disease
</div>
<div class="info-card">
💊 <b>Overuse of Painkillers</b> — NSAIDs & paracetamol overuse raises SGOT/SGPT
</div>
<div class="info-card">
🥩 <b>Red Meat Excess</b> — Saturated fat increases liver enzyme levels
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🏃 Lifestyle Recommendations")
    l1, l2, l3 = st.columns(3)
    l1.info("**💧 Hydration**\n\nDrink 8–10 glasses of water daily to help the liver flush toxins")
    l2.info("**🏋️ Exercise**\n\nAt least 30 min of moderate exercise 5 days/week reduces liver fat by 20–40%")
    l3.info("**😴 Sleep**\n\n7–8 hours of quality sleep helps liver cell regeneration")

    l4, l5, l6 = st.columns(3)
    l4.warning("**🚭 No Smoking**\n\nSmoking increases liver cancer risk and raises bilirubin levels")
    l5.warning("**⚖️ Maintain Healthy Weight**\n\nObesity is a leading cause of fatty liver disease (NAFLD)")
    l6.warning("**💉 Get Vaccinated**\n\nHepatitis A & B vaccines prevent the two most common liver infections")

    st.markdown("---")
    st.subheader("🩺 When to See a Doctor")
    st.error("""
⚠️ **Consult a doctor immediately if you notice:**
- Yellowing of skin or eyes (jaundice) — indicates high bilirubin
- Dark urine or pale stools
- Persistent fatigue and abdominal pain (upper right)
- Unexplained weight loss
- Swelling in legs and abdomen

These may indicate elevated Total/Direct Bilirubin, SGPT > 100, or SGOT > 80 — 
all key features in this prediction model.
""")

    st.markdown("---")
    st.caption("⚠️ This app is for educational purposes only. Always consult a licensed medical professional for diagnosis and treatment.")
