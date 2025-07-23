import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import xgboost as xgb

# Load and merge data
labels = pd.read_csv("weather_prediction_bbq_labels.csv")
data = pd.read_csv("weather_prediction_dataset2.csv")
df = pd.merge(data, labels[['DATE', 'BASEL_BBQ_weather']], on='DATE')

# Split features and target
X = df.drop(columns=['DATE', 'BASEL_BBQ_weather'])
y = df['BASEL_BBQ_weather'].astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize XGBoost
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Hyperparameter grid
param_grid = {
    'max_depth': [3, 5],
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0]
}

# Grid search
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    verbose=2,
    n_jobs=-1
)

# Fit model
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predict on test set
y_pred = best_model.predict(X_test)

# Accuracy and report
acc = accuracy_score(y_test, y_pred)
print("\n✅ Best Parameters:", grid_search.best_params_)
print("✅ XGBoost Tuned Model Accuracy:", acc)
print(classification_report(y_test, y_pred))

# Save model
best_model.save_model('weather_xgboost_tuned_model.json')

# ---------------------------------------
# 📊 Graph 1: Confusion Matrix
# ---------------------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ---------------------------------------
# 📊 Graph 2: Feature Importance
# ---------------------------------------
xgb.plot_importance(best_model, importance_type='gain', max_num_features=10)
plt.title("Top Feature Importances (Gain)")
plt.tight_layout()
plt.show()
