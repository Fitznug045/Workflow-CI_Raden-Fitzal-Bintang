import os
import gdown
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# 1. LOAD DATASET
# ===============================
FILE_ID = "1NRK4-UrsL-QVKz-H-fWWebbyEE2WXVAm"
url = f"https://drive.google.com/uc?id={FILE_ID}"

os.makedirs("data", exist_ok=True)
output_path = "data/hotel_bookings_clean.csv"

if not os.path.exists(output_path):
    gdown.download(url, output_path, quiet=False)

df = pd.read_csv(output_path)

X = df.drop(columns=["is_canceled"])
y = df["is_canceled"]

# ===============================
# 2. SPLIT DATA
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# 3. AUTOLOG
# ===============================
mlflow.autolog()

# ===============================
# 4. TRAIN
# ===============================
model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1
)
model.fit(X_train, y_train)

# ===============================
# 5. METRICS
# ===============================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"CI Training finished | accuracy={acc:.4f}, f1={f1:.4f}")

# ===============================
# 6. ARTIFACTS
# ===============================
os.makedirs("artifacts", exist_ok=True)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

cm_path = "artifacts/confusion_matrix.png"
plt.savefig(cm_path)
plt.close()

mlflow.log_artifact(cm_path)
