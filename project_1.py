
"""

# STEP 1: Upload cleaned dataset again
from google.colab import files
uploaded = files.upload()

import pandas as pd
df = pd.read_csv("cleaned_task_dataset.csv")
print("🔹 First 5 rows:")
print(df.head())

# STEP 2: Convert text into features using TF-IDF (Cleaned_Description → X)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
X_text = tfidf.fit_transform(df["Cleaned_Description"])

# STEP 3: Encode priority labels (Low/Medium/High → y)
y_priority = df["Priority"]

# STEP 4: Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_text, y_priority, test_size=0.2, random_state=42)

# STEP 5: Random Forest Classifier for Priority Prediction
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)
pred_rf = model_rf.predict(X_test)

print("\n✅ Random Forest Priority Prediction Report:")
print(classification_report(y_test, pred_rf))

# STEP 6 (Optional): Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
grid.fit(X_train, y_train)

print(f"\n🔍 Best Parameters: {grid.best_params_}")
print(f"📊 Best Score: {grid.best_score_:.4f}")

# STEP 7: Workload Balancing Logic (Simple Rule-Based Example)
# Count how many tasks each person has and suggest rebalancing
task_counts = df["Assigned To"].value_counts()
average_tasks = task_counts.mean()

print("\n📊 Workload Balancing Suggestion:")
for user, count in task_counts.items():
    if count > average_tasks:
        print(f"🔴 {user} has {count} tasks — consider reassigning.")
    else:
        print(f"🟢 {user} is balanced.")

# STEP 8: Summary Output
df_summary = df[["Description", "Cleaned_Description", "Priority", "Assigned To", "Category"]].copy()
df_summary["Predicted Priority"] = model_rf.predict(X_text)
df_summary.to_csv("task_priority_predictions.csv", index=False)
files.download("task_priority_predictions.csv")

print("\n📁 Final summary saved as task_priority_predictions.csv")
