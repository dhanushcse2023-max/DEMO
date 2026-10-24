"""
Predicting Automobile Fuel Efficiency (MPG) with Machine Learning
-----------------------------------------------------------------
Single-file Streamlit application.

    pip install streamlit scikit-learn pandas numpy
    streamlit run mpg_app.py

Put auto-mpg.csv next to this file (or upload it inside the app).
On first start the app trains 4 algorithms, compares them with cross-validation
and deploys the best one. The trained model is cached, so later runs are instant.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

SEED = 42
NUMERIC = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]
CATEGORICAL = ["origin", "brand"]
FEATURES = NUMERIC + CATEGORICAL
ORIGIN_LABELS = {1: "USA", 2: "Europe", 3: "Japan"}
BRAND_FIXES = {"chevy": "chevrolet", "chevroelt": "chevrolet", "vw": "volkswagen",
               "vokswagen": "volkswagen", "toyouta": "toyota", "maxda": "mazda",
               "mercedes-benz": "mercedes"}


# ============================== 1. DATA ==============================
def extract_brand(car_name):
    """'chevrolet chevelle malibu' -> 'chevrolet' (also fixes common typos)."""
    if not isinstance(car_name, str) or not car_name.strip():
        return "unknown"
    first = car_name.strip().lower().split()[0]
    return BRAND_FIXES.get(first, first)


def load_data(source):
    """Read auto-mpg.csv (path or file object) into a clean DataFrame."""
    df = pd.read_csv(source, na_values=["?", "NA", ""])
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    for col in ["mpg"] + NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")       # '?' in horsepower -> NaN
    df["origin"] = pd.to_numeric(df["origin"], errors="coerce").map(ORIGIN_LABELS)
    df["brand"] = df["car_name"].apply(extract_brand)
    df = df.dropna(subset=["mpg"])                               # a row needs a target
    return df[FEATURES + ["mpg"]].reset_index(drop=True)


def build_input(cylinders, displacement, horsepower, weight, acceleration,
                model_year, origin, brand="unknown"):
    """One-row DataFrame in the format the pipeline expects (horsepower may be None)."""
    return pd.DataFrame([{
        "cylinders": cylinders, "displacement": displacement,
        "horsepower": np.nan if horsepower is None else horsepower,
        "weight": weight, "acceleration": acceleration, "model_year": model_year,
        "origin": origin, "brand": str(brand).strip().lower() or "unknown",
    }])[FEATURES]


# ============================== 2. MODEL ==============================
def make_preprocessor():
    """Missing values -> scaling (numbers) / most-frequent + one-hot (categories)."""
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                   min_frequency=5, sparse_output=False))]), CATEGORICAL),
    ])


def candidates():
    return {
        "Linear Regression": (LinearRegression(), {}),
        "Random Forest": (RandomForestRegressor(random_state=SEED),
                          {"model__n_estimators": [200, 300], "model__max_depth": [None, 12],
                           "model__min_samples_leaf": [1, 2]}),
        "Gradient Boosting": (GradientBoostingRegressor(random_state=SEED),
                              {"model__n_estimators": [200, 300], "model__learning_rate": [0.05, 0.1],
                               "model__max_depth": [2, 3]}),
        "Support Vector Regression": (SVR(), {"model__C": [1, 10, 50], "model__gamma": ["scale"]}),
    }


@st.cache_resource(show_spinner="Training and comparing 4 algorithms (first run only)…")
def train_all(df):
    """Tune 4 algorithms, pick the best by cross-validation, refit it on all data."""
    X, y = df[FEATURES], df["mpg"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    cv = KFold(5, shuffle=True, random_state=SEED)
    rows, fitted = [], {}
    for name, (est, grid) in candidates().items():
        pipe = Pipeline([("prep", make_preprocessor()), ("model", est)])
        gs = GridSearchCV(pipe, grid or {"model": [est]}, cv=cv,
                          scoring="neg_root_mean_squared_error", n_jobs=-1).fit(X_tr, y_tr)
        pred = gs.best_estimator_.predict(X_te)
        rows.append({"Model": name,
                     "CV RMSE": round(float(-gs.best_score_), 3),
                     "Test RMSE": round(float(np.sqrt(mean_squared_error(y_te, pred))), 3),
                     "Test MAE": round(float(mean_absolute_error(y_te, pred)), 3),
                     "Test R²": round(float(r2_score(y_te, pred)), 3)})
        fitted[name] = (gs.best_estimator_, pred)
    comp = pd.DataFrame(rows).sort_values("CV RMSE").reset_index(drop=True)
    best = comp.loc[0, "Model"]                       # chosen by CV, not by the test set
    final = fitted[best][0].fit(X, y)                 # deploy: refit on all rows
    counts = df["brand"].value_counts()
    return {
        "model": final, "best": best, "comparison": comp,
        "test_plot": pd.DataFrame({"Actual MPG": y_te.values, "Predicted MPG": fitted[best][1]}),
        "brands": sorted(counts[counts >= 5].index),
        "ranges": {c: (float(df[c].min()), float(df[c].max()), float(df[c].median())) for c in NUMERIC},
        "mean_mpg": float(y.mean()), "rows": len(df), "missing_hp": int(df["horsepower"].isna().sum()),
    }


# ============================== 3. APP ==============================
def main():
    st.set_page_config(page_title="MPG Predictor", page_icon="⛽", layout="wide")
    st.markdown("""<style>
    .stApp{background:linear-gradient(160deg,#f4f7ff 0%,#fff7f0 100%)}
    .hero{background:linear-gradient(90deg,#4f6bed,#7c5cff);color:#fff;padding:22px 28px;border-radius:18px;margin-bottom:18px}
    .hero h1{margin:0;font-size:2rem;color:#fff}.hero p{margin:4px 0 0;opacity:.9}
    .result{background:#fff;border-radius:18px;padding:22px;text-align:center;box-shadow:0 6px 20px rgba(79,107,237,.12)}
    .result .big{font-size:3.4rem;font-weight:800;color:#4f6bed;line-height:1}
    </style>""", unsafe_allow_html=True)
    st.markdown('<div class="hero"><h1>⛽ Car MPG Predictor</h1><p>Enter a vehicle\'s specifications and a machine-learning model estimates its fuel efficiency.</p></div>', unsafe_allow_html=True)

    csv = Path(__file__).with_name("auto-mpg.csv")
    if csv.exists():
        df = load_data(csv)
    else:
        up = st.file_uploader("Upload auto-mpg.csv", type="csv")
        if up is None:
            st.info("auto-mpg.csv was not found next to this file. Upload it to start.")
            st.stop()
        df = load_data(up)

    art = train_all(df)
    model, R = art["model"], art["ranges"]

    with st.sidebar:
        st.header("🚗 Vehicle details")
        cyl = st.selectbox("Cylinders", [3, 4, 5, 6, 8], index=1)
        disp = st.slider("Displacement (cu. in.)", int(R["displacement"][0]), int(R["displacement"][1]), int(R["displacement"][2]))
        unknown_hp = st.checkbox("I don't know the horsepower", help="The pipeline fills the gap automatically (median).")
        hp = None if unknown_hp else st.slider("Horsepower", int(R["horsepower"][0]), int(R["horsepower"][1]), int(R["horsepower"][2]))
        wt = st.slider("Weight (lbs)", int(R["weight"][0]), int(R["weight"][1]), int(R["weight"][2]), step=10)
        acc = st.slider("Acceleration (0-60 sec)", float(R["acceleration"][0]), float(R["acceleration"][1]), float(R["acceleration"][2]), step=0.1)
        yr = st.slider("Model year (19xx)", int(R["model_year"][0]), int(R["model_year"][1]), int(R["model_year"][2]))
        origin = st.radio("Origin", list(ORIGIN_LABELS.values()), horizontal=True)
        brand = st.selectbox("Brand (optional)", ["unknown"] + art["brands"])
        go = st.button("Predict MPG", type="primary")

    tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Model comparison", "⚙️ How it works"])

    with tab1:
        if go or "last" in st.session_state:
            if go:
                x = build_input(cyl, disp, hp, wt, acc, yr, origin, brand)
                st.session_state["last"] = (float(model.predict(x)[0]), x)
            mpg, x = st.session_state["last"]
            grade = (("🔴 Thirsty", "#e03131") if mpg < 18 else ("🟠 Average", "#f08c00") if mpg < 25
                     else ("🟢 Good", "#2f9e44") if mpg < 32 else ("🌟 Excellent", "#1c7ed6"))
            a, b = st.columns([1, 1.3])
            a.markdown(f'<div class="result"><div>Predicted fuel efficiency</div><div class="big">{mpg:.1f}</div>'
                       f'<div>miles per gallon</div><h3 style="color:{grade[1]}">{grade[0]}</h3></div>', unsafe_allow_html=True)
            with b:
                c1, c2 = st.columns(2)
                c1.metric("Kilometres per litre", f"{mpg * 0.4251:.1f}")
                c2.metric("Litres per 100 km", f"{235.215 / mpg:.1f}")
                st.metric("Compared with dataset average", f"{mpg:.1f} MPG", f"{mpg - art['mean_mpg']:+.1f} vs {art['mean_mpg']:.1f}")
                st.progress(min(mpg / 50, 1.0), text="Scale: 0 to 50 MPG")
            with st.expander("Input sent to the model"):
                st.dataframe(x.astype({"horsepower": "object"}).fillna({"horsepower": "missing (auto-filled)"}), hide_index=True)
        else:
            st.info("👈 Fill in the vehicle details in the sidebar and press **Predict MPG**.")

    with tab2:
        comp = art["comparison"]
        st.success(f"🏆 Best model (lowest cross-validation RMSE): **{art['best']}** – deployed in this app.")
        st.dataframe(comp, hide_index=True)
        c1, c2 = st.columns(2)
        c1.caption("Error – lower is better"); c1.bar_chart(comp.set_index("Model")[["CV RMSE", "Test RMSE"]])
        c2.caption("R² – higher is better"); c2.bar_chart(comp.set_index("Model")[["Test R²"]])
        st.caption("Actual vs predicted MPG on the held-out test cars (best model) – points near the diagonal are accurate")
        st.scatter_chart(art["test_plot"], x="Actual MPG", y="Predicted MPG")

    with tab3:
        st.markdown(f"""
**End-to-end pipeline** (one object, so training and prediction always match):
1. **Missing values:** numeric gaps (e.g. the {art['missing_hp']} missing horsepower values) → **median**; missing categories → **most frequent**.
2. **Feature scaling:** numeric features are standardised with `StandardScaler`.
3. **Categorical encoding:** `origin` and `brand` (taken from *car name*) are **one-hot encoded**; rare/unseen brands are handled safely.
4. **Model selection:** Linear Regression, Random Forest, Gradient Boosting and SVR are tuned with 5-fold cross-validation. The lowest CV error wins, is scored on a 20 % held-out test set, then re-fitted on all {art['rows']} rows for deployment.
""")


if __name__ == "__main__":
    main()
