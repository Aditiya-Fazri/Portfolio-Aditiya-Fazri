import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, confusion_matrix

# baca dataset
df = pd.read_csv("data_churn.csv")
print("data shape:", df.shape)

# buang customer_id, cuma nomor id ga ada arti buat prediksi
df = df.drop(columns=["customer_id"])

# ubah kolom teks jadi angka
df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
df = pd.get_dummies(df, columns=["country"], drop_first=True)

# pisah fitur sama target
X = df.drop(columns=["churn"])
y = df["churn"]
feature_order = X.columns.tolist()
print("fitur dipakai:", feature_order)

# bagi data train sama test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# scaling biar skala fitur seragam
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# training random forest, class_weight balanced buat data churn yang ga seimbang
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
)
model.fit(X_train_scaled, y_train)

# evaluasi
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print("\nhasil evaluasi di test set:")
print("accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("recall   :", round(recall_score(y_test, y_pred), 4))
print("precision:", round(precision_score(y_test, y_pred), 4))
print("roc_auc  :", round(roc_auc_score(y_test, y_proba), 4))
print("confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# simpan model, scaler, sama urutan fitur jadi satu file
joblib.dump(
    {"model": model, "scaler": scaler, "features": feature_order},
    "churn_model.joblib",
)
print("\nmodel disimpan ke churn_model.joblib")
