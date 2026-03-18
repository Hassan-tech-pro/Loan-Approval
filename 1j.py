import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
import pickle as pkl

df = pd.read_csv('data/loan_data_combined.csv')
df = df.drop("loan_percent_income", axis=1)
X = df.drop('loan_status', axis=1)
y = df['loan_status']

X_train, X_20, y_train, y_20 = train_test_split(X, y, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_20, y_20, test_size=0.5, random_state=42)

scaler = MinMaxScaler()
scaler.fit(X_train) 

X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)

print(X["credit_score"].min(), X["credit_score"].max())
print(X_train.columns.tolist())
model_1 = xgb.XGBClassifier(n_estimators=300, learning_rate=0.25, random_state=42, use_label_encoder=False, eval_metric='logloss')
model_1.fit(X_train_scaled, y_train)
y_pred = model_1.predict(X_val_scaled)
print("Classification Report:")
print(classification_report(y_val, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))

y_pred_test = model_1.predict(X_test_scaled)
print("Classification Report on Test Set:")
print(classification_report(y_test, y_pred_test))
print("Confusion Matrix on Test Set:")
print(confusion_matrix(y_test, y_pred_test))

pkl.dump(model_1, open('model.pkl', 'wb'))
pkl.dump(scaler, open('scaler.pkl', 'wb'))
print(df.columns.to_list())


