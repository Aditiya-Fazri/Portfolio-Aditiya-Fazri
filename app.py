import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go


# ══════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="My Portfolio with Streamlit",
    page_icon="🏠",
    layout="wide",
)


# ══════════════════════════════════════════════
# CONSTANTS, DATA LOADER, & MODEL TRAINER
# ══════════════════════════════════════════════
FITUR = [
    "OverallQual", "GrLivArea", "GarageCars", "GarageArea",
    "TotalBsmtSF", "1stFlrSF", "FullBath", "YearBuilt",
    "YearRemodAdd", "TotRmsAbvGrd", "Fireplaces", "LotArea",
]


@st.cache_data
def load_data():
    return pd.read_csv("data_house.csv")


def prepare_features(df):
    data = df[FITUR].copy()
    for kolom in FITUR:
        if data[kolom].isnull().sum() > 0:
            data[kolom] = data[kolom].fillna(data[kolom].median())
    data["TotalSF"] = data["TotalBsmtSF"] + data["GrLivArea"]
    return data


@st.cache_resource
def train_model():
    df = pd.read_csv("data_house.csv")
    X = prepare_features(df)
    y = df["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }
    feature_importance = dict(zip(X.columns, model.feature_importances_.tolist()))

    return model, metrics, feature_importance, y_test.tolist(), y_pred.tolist()


# load data + train model (cached, so cuma jalan sekali)
df_house = load_data()
model, metrics, feature_importance, y_test, y_pred = train_model()


# ══════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════
tab_home, tab_predict, tab_insight = st.tabs(
    ["🏠 Home", "🤖 House Price Predictor", "📊 Data & Model Insights"]
)


# ══════════════════════════════════════════════
# TAB 1 — HOME
# ══════════════════════════════════════════════
with tab_home:

    st.title("Hi, I'm Fazri 👋")
    st.subheader("Aspiring Data Scientist & Machine Learning Practitioner")

    col_photo, col_bio = st.columns([1, 3])

    with col_photo:
        st.image("Aditiya Fazri.jpg", width=200)

    with col_bio:
        st.write("""
        I'm a Computer Systems graduate from Universitas Gunadarma. My background
        started in IoT and embedded systems, but these days I spend most of my time
        on data science and machine learning, which I'm picking up through a bootcamp
        at Dibimbing.id.
        What I enjoy most is taking a project from start to finish, digging into the
        data, building a model that actually works, and turning it into something
        people can use. Right now I'm looking for an entry-level role in data science
        or machine learning.
        """)
        st.write("**Skills:** Data Analysis · Statistics · SQL · Python · "
                 "Machine Learning · Deep Learning · Data Visualization")

    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.link_button("📧 Email", "mailto:fazrihaditiya@gmail.com", use_container_width=True)
    col2.link_button("🔗 LinkedIn", "https://linkedin.com/in/aditiyafazri", use_container_width=True)
    col3.link_button("💻 GitHub", "https://github.com/Aditiya-Fazri", use_container_width=True)

    st.divider()

    st.subheader("Featured Projects")

    p1, p2, p3 = st.columns(3)

    with p1:
        with st.container(border=True):
            st.markdown("### 🏠 House Price Prediction")
            st.write("""
            I built a regression model that predicts house sale prices using the
            Kaggle House Prices dataset. Cleaned messy real-world data with missing
            values, picked the most useful features, and tuned a Random Forest to
            get the lowest error. You can test the live version in the Predictor tab.
            """)
            st.write("RMSE **$29.8K** · R² **0.88**")
            st.info("👉 Try it in the **🤖 House Price Predictor** tab.")

    with p2:
        with st.container(border=True):
            st.markdown("### 📉 Bank Customer Churn")
            st.write("""
            I built a model to spot which bank customers are about to leave. Worked
            with 10,000 records, cleaned the data, tried a few models, and checked
            which features matter most. You can test the live version in the Churn
            Predictor tab.
            """)
            st.write("Recall **0.65** · ROC-AUC **0.86**")

    with p3:
        with st.container(border=True):
            st.markdown("### 🎬 Recommendation System")
            st.write("""
            A recommender that mixes content-based and collaborative filtering to
            suggest items a user might like. Tested with RMSE on the rating
            predictions to see how close it got.
            """)
            st.write("Hybrid **CB + CF**")
            st.caption("Recommender · scikit-learn")


# ══════════════════════════════════════════════
# TAB 2 — HOUSE PRICE PREDICTOR
# ══════════════════════════════════════════════
with tab_predict:
    st.header("🏠 House Price Predictor")
    st.markdown(
        "Predict house sale price based on property features. "
        "Use single mode for one prediction, or batch mode for multiple houses via CSV."
    )

    mode = st.radio(
        "Prediction Mode:",
        ["Single Property", "Batch Upload (CSV)"],
        horizontal=True,
    )

    st.divider()

    # ============ MODE 1: SINGLE ============
    if mode == "Single Property":
        col1, col2 = st.columns(2)

        with col1:
            overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 7)
            gr_liv_area = st.number_input("Above-ground Living Area (sqft)", 500, 6000, 1500)
            garage_cars = st.slider("Garage Capacity (cars)", 0, 4, 2)
            garage_area = st.number_input("Garage Area (sqft)", 0, 1500, 500)
            total_bsmt_sf = st.number_input("Basement Area (sqft)", 0, 3000, 900)
            first_flr_sf = st.number_input("1st Floor Area (sqft)", 300, 4000, 1000)

        with col2:
            full_bath = st.slider("Full Bathrooms", 0, 4, 2)
            year_built = st.number_input("Year Built", 1870, 2024, 2000)
            year_remod = st.number_input("Year Remodeled", 1950, 2024, 2005)
            tot_rms = st.slider("Total Rooms Above Grade", 2, 14, 7)
            fireplaces = st.slider("Number of Fireplaces", 0, 4, 1)
            lot_area = st.number_input("Lot Area (sqft)", 1000, 50000, 9000)

        if st.button("Predict Price", type="primary"):
            input_data = pd.DataFrame([{
                "OverallQual": overall_qual,
                "GrLivArea": gr_liv_area,
                "GarageCars": garage_cars,
                "GarageArea": garage_area,
                "TotalBsmtSF": total_bsmt_sf,
                "1stFlrSF": first_flr_sf,
                "FullBath": full_bath,
                "YearBuilt": year_built,
                "YearRemodAdd": year_remod,
                "TotRmsAbvGrd": tot_rms,
                "Fireplaces": fireplaces,
                "LotArea": lot_area,
                "TotalSF": total_bsmt_sf + gr_liv_area,
            }])

            prediction = model.predict(input_data)[0]

            st.success(f"Predicted Sale Price: **${prediction:,.2f}**")

            median_price = df_house["SalePrice"].median()
            if prediction > median_price:
                st.info(f"💎 Above median (${median_price:,.0f}) — premium property")
            else:
                st.info(f"💰 Below median (${median_price:,.0f}) — affordable property")

    # ============ MODE 2: BATCH UPLOAD ============
    else:
        st.markdown("Upload CSV with these 12 columns:")
        st.code(", ".join(FITUR), language=None)

        sample_csv = df_house[FITUR].head(5).to_csv(index=False)
        st.download_button(
            "📥 Download Sample CSV",
            sample_csv,
            "sample_upload.csv",
            "text/csv",
        )

        uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

        if uploaded is not None:
            try:
                df_input = pd.read_csv(uploaded)

                missing = [c for c in FITUR if c not in df_input.columns]
                if missing:
                    st.error(f"Missing required columns: {', '.join(missing)}")
                else:
                    st.success(f"Loaded {len(df_input)} rows")
                    st.dataframe(df_input.head(), use_container_width=True)

                    if st.button("Run Batch Prediction", type="primary"):
                        data = df_input[FITUR].copy()
                        for col in FITUR:
                            if data[col].isnull().sum() > 0:
                                data[col] = data[col].fillna(data[col].median())
                        data["TotalSF"] = data["TotalBsmtSF"] + data["GrLivArea"]

                        predictions = model.predict(data)

                        result = df_input.copy()
                        result["Predicted_Price"] = predictions

                        st.success(f"Predicted {len(result)} properties")
                        st.dataframe(
                            result[["OverallQual", "GrLivArea", "YearBuilt", "Predicted_Price"]]
                            .style.format({"Predicted_Price": "${:,.2f}"}),
                            use_container_width=True,
                        )

                        csv_out = result.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results",
                            csv_out,
                            "predictions.csv",
                            "text/csv",
                        )

                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Avg Price", f"${predictions.mean():,.0f}")
                        col_b.metric("Min Price", f"${predictions.min():,.0f}")
                        col_c.metric("Max Price", f"${predictions.max():,.0f}")

            except Exception as e:
                st.error(f"Error reading file: {e}")


# ══════════════════════════════════════════════
# TAB 3 — DATA & MODEL INSIGHTS
# ══════════════════════════════════════════════
with tab_insight:
    st.header("📊 Data & Model Insights")
    st.markdown("Explore the dataset and model performance.")

    # ============ MODEL METRICS ============
    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"${metrics['rmse']:,.0f}")
    col2.metric("MAE", f"${metrics['mae']:,.0f}")
    col3.metric("R² Score", f"{metrics['r2']:.4f}")

    st.divider()

    # ============ DISTRIBUSI HARGA ============
    st.subheader("Sale Price Distribution")
    fig_dist = px.histogram(
        df_house, x="SalePrice", nbins=50,
        title="Distribution of House Prices",
        labels={"SalePrice": "Sale Price (USD)"},
    )
    fig_dist.update_layout(showlegend=False)
    st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()

    # ============ KORELASI 12 FITUR DENGAN HARGA ============
    st.subheader("Feature Correlation with Sale Price")
    corr_data = df_house[FITUR + ["SalePrice"]].corr()["SalePrice"].drop("SalePrice")
    corr_df = pd.DataFrame({
        "Feature": corr_data.index,
        "Correlation": corr_data.values,
    }).sort_values("Correlation", ascending=True)

    fig_corr = px.bar(
        corr_df, x="Correlation", y="Feature", orientation="h",
        title="Correlation: Features vs Sale Price",
        color="Correlation", color_continuous_scale="RdBu",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()

    # ============ FEATURE IMPORTANCE ============
    st.subheader("Feature Importance (Random Forest)")
    fi_df = pd.DataFrame({
        "Feature": list(feature_importance.keys()),
        "Importance": list(feature_importance.values()),
    }).sort_values("Importance", ascending=True)

    fig_fi = px.bar(
        fi_df, x="Importance", y="Feature", orientation="h",
        title="Which features matter most for prediction",
        color="Importance", color_continuous_scale="viridis",
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    st.divider()

    # ============ PREDICTED VS ACTUAL ============
    st.subheader("Predicted vs Actual Price")
    pred_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})

    fig_scatter = px.scatter(
        pred_df, x="Actual", y="Predicted",
        title="Model Predictions vs True Values",
        labels={"Actual": "Actual Price (USD)", "Predicted": "Predicted Price (USD)"},
        opacity=0.6,
    )
    min_val = min(pred_df["Actual"].min(), pred_df["Predicted"].min())
    max_val = max(pred_df["Actual"].max(), pred_df["Predicted"].max())
    fig_scatter.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines", name="Perfect Prediction",
        line=dict(dash="dash", color="red"),
    ))
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()

    # ============ RESIDUAL PLOT ============
    st.subheader("Residual Analysis")
    pred_df["Residual"] = pred_df["Actual"] - pred_df["Predicted"]

    fig_resid = px.scatter(
        pred_df, x="Predicted", y="Residual",
        title="Residuals: how far predictions are from actual",
        labels={"Predicted": "Predicted Price (USD)", "Residual": "Error (USD)"},
        opacity=0.6,
    )
    fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_resid, use_container_width=True)
