import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

st.set_page_config(
    page_title="Fazri | Data & AI Portfolio",
    page_icon="🤖",
    layout="wide",
)


@st.cache_data
def load_model():
    bundle = joblib.load("churn_model.joblib")
    return bundle["model"], bundle["scaler"], bundle["features"]


@st.cache_data
def load_data():
    return pd.read_csv("data_churn.csv")


def prepare_features(df_raw, feature_order):
    data = df_raw.copy()
    if "customer_id" in data.columns:
        data = data.drop(columns=["customer_id"])
    if "churn" in data.columns:
        data = data.drop(columns=["churn"])
    data["gender"] = data["gender"].map({"Male": 1, "Female": 0})
    data = pd.get_dummies(data, columns=["country"], drop_first=True)
    for col in feature_order:
        if col not in data.columns:
            data[col] = 0
    data = data[feature_order]
    return data


model, scaler, FEATURES = load_model()

tab_home, tab_ml, tab_eda = st.tabs([
    "🏠 Home",
    "🤖 Churn Predictor",
    "📊 Data & Model Insights",
])


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
            st.markdown("### 📉 Bank Customer Churn")
            st.write("""
            I built a model to spot which bank customers are about to leave. Worked
            with 10,000 records, cleaned the data, tried a few models, and checked
            which features matter most. You can test the live version in the Churn
            Predictor tab.
            """)
            st.write("Recall **0.65** · ROC-AUC **0.86**")
            st.info("👉 Try it in the **🤖 Churn Predictor** tab.")

    with p2:
        with st.container(border=True):
            st.markdown("### 📈 LSTM Stock Prediction")
            st.write("""
            A time series model that predicts stock closing prices using LSTM, built
            with TensorFlow. The tricky part was getting the data windowing right so
            the model learns from the right sequence.
            """)
            st.write("MAPE **under 3%**")
            st.caption("Deep Learning · TensorFlow · Time Series")

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
# TAB 2 — CHURN PREDICTOR
# ══════════════════════════════════════════════
with tab_ml:

    st.title("🤖 Bank Customer Churn Predictor")
    st.divider()

    mode = st.radio(
        "Choose prediction mode:",
        ["Single customer", "Batch upload (CSV)"],
        horizontal=True,
    )

    # ── MODE 1: satu customer ─────────────────
    if mode == "Single customer":
        col_inputs, col_result = st.columns(2)

        with col_inputs:
            st.subheader("Customer Details")
            credit_score = st.number_input("Credit Score", 300, 900, 650)
            age = st.slider("Age", 18, 95, 40)
            tenure = st.slider("Tenure (years with bank)", 0, 10, 5)
            balance = st.number_input("Balance", 0.0, 300000.0, 75000.0, step=1000.0)
            products_number = st.slider("Number of Products", 1, 4, 1)
            estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 100000.0, step=1000.0)
            gender = st.selectbox("Gender", ["Male", "Female"])
            country = st.selectbox("Country", ["France", "Germany", "Spain"])
            credit_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
            active_member = st.selectbox("Active Member?", ["Yes", "No"])

            predict_btn = st.button("Predict Churn", use_container_width=True)

        with col_result:
            st.subheader("Prediction Result")

            if predict_btn:
                row = pd.DataFrame([{
                    "credit_score": credit_score,
                    "country": country,
                    "gender": gender,
                    "age": age,
                    "tenure": tenure,
                    "balance": balance,
                    "products_number": products_number,
                    "credit_card": 1 if credit_card == "Yes" else 0,
                    "active_member": 1 if active_member == "Yes" else 0,
                    "estimated_salary": estimated_salary,
                }])

                X = prepare_features(row, FEATURES)
                proba = model.predict_proba(scaler.transform(X))[0][1]
                pred = model.predict(scaler.transform(X))[0]

                if pred == 1:
                    st.error("This customer is likely to CHURN")
                else:
                    st.success("This customer is likely to STAY")

                st.metric("Churn Probability", f"{proba*100:.1f}%")
                st.progress(float(proba))

                if proba > 0.5:
                    st.write("High risk of leaving. Might be worth a retention offer.")
                else:
                    st.write("Low risk. This customer looks stable.")
            else:
                st.info("Fill in the details on the left, then click Predict Churn.")

    # ── MODE 2: upload csv ────────────────────
    else:
        st.subheader("Upload Customer Data")
        st.write("Upload a CSV with customer data to score many customers at once. "
                 "It should have the same columns as the original dataset (you can "
                 "use sample_upload.csv from the project folder to try it).")

        uploaded = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded is not None:
            df_upload = pd.read_csv(uploaded)
            st.write(f"Loaded **{len(df_upload)} rows**")
            st.dataframe(df_upload.head(), use_container_width=True)

            if st.button("Run Prediction", use_container_width=True):
                X = prepare_features(df_upload, FEATURES)
                proba = model.predict_proba(scaler.transform(X))[:, 1]
                pred = model.predict(scaler.transform(X))

                result = df_upload.copy()
                result["Churn_Prediction"] = ["Churn" if p == 1 else "Stay" for p in pred]
                result["Churn_Probability"] = [f"{p*100:.1f}%" for p in proba]

                st.divider()
                st.subheader("Prediction Results")

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Customers", len(result))
                c2.metric("Predicted Churn", int(sum(pred)))
                c3.metric("Churn Rate", f"{sum(pred)/len(pred)*100:.1f}%")

                st.dataframe(result, use_container_width=True)

                csv_out = result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Results",
                    csv_out,
                    "churn_predictions.csv",
                    "text/csv",
                )
        else:
            st.info("No file uploaded yet.")


# ══════════════════════════════════════════════
# TAB 3 — DATA & MODEL INSIGHTS
# ══════════════════════════════════════════════
with tab_eda:

    df = load_data()

    st.title("📊 Data & Model Insights")
    st.divider()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Customers", len(df))
    m2.metric("Features", df.shape[1] - 2)
    m3.metric("Churn Rate", f"{df['churn'].mean()*100:.1f}%")
    m4.metric("Countries", df["country"].nunique())

    st.divider()

    st.subheader("Filters")
    selected_country = st.multiselect(
        "Country",
        options=df["country"].unique().tolist(),
        default=df["country"].unique().tolist(),
    )
    df_f = df[df["country"].isin(selected_country)]

    if df_f.empty:
        st.warning("No data. Select at least one country.")
        st.stop()

    st.divider()

    st.subheader("Churn Distribution")
    col_a, col_b = st.columns(2)

    with col_a:
        churn_counts = df_f["churn"].map({0: "Stay", 1: "Churn"}).value_counts().reset_index()
        churn_counts.columns = ["Status", "Count"]
        fig_pie = px.pie(
            churn_counts,
            names="Status", values="Count",
            hole=0.4,
            title="Overall Churn vs Stay",
            color="Status",
            color_discrete_map={"Stay": "#1D9E75", "Churn": "#E24B4A"},
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        by_country = df_f.groupby("country")["churn"].mean().reset_index()
        by_country["churn"] = by_country["churn"] * 100
        fig_bar = px.bar(
            by_country,
            x="country", y="churn",
            text_auto=".1f",
            title="Churn Rate by Country (%)",
            labels={"churn": "Churn Rate (%)", "country": "Country"},
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Feature Distribution")
    num_cols = ["credit_score", "age", "tenure", "balance",
                "products_number", "estimated_salary"]
    feature = st.selectbox("Select a feature", num_cols)
    fig_hist = px.histogram(
        df_f,
        x=feature,
        color=df_f["churn"].map({0: "Stay", 1: "Churn"}),
        nbins=30,
        barmode="overlay",
        opacity=0.7,
        title=f"Distribution of {feature} by Churn Status",
        labels={"color": "Status"},
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Feature Correlation")
    corr_df = df_f.drop(columns=["customer_id"]).copy()
    corr_df["gender"] = corr_df["gender"].map({"Male": 1, "Female": 0})
    corr_df = pd.get_dummies(corr_df, columns=["country"], drop_first=True)
    corr = corr_df.corr()
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()

    st.subheader("Model Performance")

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Accuracy", "0.84")
    p2.metric("Recall", "0.65")
    p3.metric("Precision", "0.60")
    p4.metric("ROC-AUC", "0.86")

    col_cm, col_fi = st.columns(2)

    with col_cm:
        st.write("**Confusion Matrix**")
        cm = np.array([[1416, 177], [144, 263]])
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual"),
            x=["Stay", "Churn"], y=["Stay", "Churn"],
            title="Confusion Matrix (Test Set)",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_fi:
        st.write("**Feature Importance**")
        importance = {
            "age": 0.322, "products_number": 0.201, "balance": 0.119,
            "estimated_salary": 0.084, "credit_score": 0.082,
            "active_member": 0.051, "country_Germany": 0.051,
            "tenure": 0.047, "gender": 0.022, "credit_card": 0.011,
            "country_Spain": 0.011,
        }
        imp_df = pd.DataFrame({
            "Feature": list(importance.keys()),
            "Importance": list(importance.values()),
        }).sort_values("Importance")
        fig_imp = px.bar(
            imp_df, x="Importance", y="Feature",
            orientation="h",
            title="What Drives Churn?",
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()
    if st.checkbox("Show raw data"):
        st.dataframe(df_f, use_container_width=True)
        st.caption(f"{len(df_f)} rows · {df_f.shape[1]} columns")
