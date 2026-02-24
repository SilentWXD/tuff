import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Movie Revenue Dashboard",
    layout="wide"
)

st.title("🎬 Movie Revenue Intelligence Dashboard")

# -------------------------------------------------
# SAFE MODEL LOADING
# -------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_revenue_regressor_pro.joblib")

try:
    model = load_model()
except Exception as e:
    st.error("❌ Model failed to load.")
    st.write("Files in repo:", os.listdir("."))
    st.exception(e)
    st.stop()

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 Industry Insights",
    "🤖 Revenue Predictor",
    "📈 Model Performance"
])

# =================================================
# TAB 1 — INSIGHTS
# =================================================
with tab1:
    st.subheader("Industry Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Median Revenue", "$42M")
    col2.metric("Hit Rate (ROI ≥ 2)", "31%")
    col3.metric("Model R²", "0.9949")

    st.markdown("""
    ### Key Insights
    - Revenue follows a heavy-tail distribution (winner-takes-most).
    - Mid-budget films often provide better risk-adjusted returns.
    - Horror genre shows highest median ROI.
    - July & December maximize revenue potential.
    """)

# =================================================
# TAB 2 — PREDICTION
# =================================================
with tab2:
    st.subheader("Predict Movie Revenue")

    left, right = st.columns(2)

    with left:
        budget_m = st.slider("Budget (Million $)", 1, 300, 50)
        runtime = st.slider("Runtime (minutes)", 60, 220, 120)
        vote_average = st.slider("Vote Average", 1.0, 10.0, 7.0)
        vote_count = st.number_input("Vote Count", 0, 1000000, 1000)
        popularity = st.slider("Popularity", 0.0, 200.0, 20.0)
        release_year = st.slider("Release Year", 1970, 2020, 2015)
        release_month = st.slider("Release Month", 1, 12, 7)

        genre = st.selectbox("Main Genre", [
            "Action","Adventure","Animation","Comedy",
            "Crime","Drama","Fantasy","Horror",
            "Romance","Thriller"
        ])

        run = st.button("Run Prediction")

    if run:
        budget = budget_m * 1_000_000

        input_df = pd.DataFrame({
            "log_budget": [np.log1p(budget)],
            "runtime": [runtime],
            "vote_average": [vote_average],
            "log_vote_count": [np.log1p(vote_count)],
            "log_popularity": [np.log1p(popularity)],
            "release_year": [release_year],
            "release_month": [release_month],
            "rev_per_budget": [1.2],
            "main_genre": [genre],
        })

        try:
            pred_log = model.predict(input_df)[0]
            pred_revenue = np.expm1(pred_log)
            pred_profit = pred_revenue - budget
            roi = pred_revenue / budget

            with right:
                st.metric("Predicted Revenue", f"${pred_revenue/1e6:,.2f}M")
                st.metric("Predicted Profit", f"${pred_profit/1e6:,.2f}M")
                st.metric("Predicted ROI", f"{roi:.2f}x")

                if pred_profit > 0:
                    st.success("Projected Profit 🎉")
                else:
                    st.error("Projected Loss ⚠️")

        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)

# =================================================
# TAB 3 — MODEL PERFORMANCE
# =================================================
with tab3:
    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)
    col1.metric("R² (log space)", "0.9949")
    col2.metric("MAPE", "8.76%")
    col3.metric("Median APE", "3.54%")

    st.markdown("""
    ### What This Means
    - Model explains 99.49% of revenue variance in log space.
    - Average prediction error is under 9%.
    - For half of films, error is under 3.5%.
    """)
