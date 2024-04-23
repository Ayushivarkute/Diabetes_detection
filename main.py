import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


data = pd.read_csv('C:/Users/ayush/Downloads/archive (1)/diabetes_prediction_dataset.csv')

X = data[['gender', 'age', 'hypertension','heart_disease', 'smoking_history', 'bmi',  'HbA1c_level', 'blood_glucose_level']]
y = data['diabetes']

X = pd.get_dummies(X, columns=['gender'], drop_first=True)
X = pd.get_dummies(X, columns=['smoking_history'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = XGBClassifier()

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy*100}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')


joblib.dump(model, 'xgboost.joblib')