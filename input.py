from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QComboBox, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout
import pandas as pd

import joblib

class DiabetesPredictionApp(QWidget):
    def __init__(self):
        super().__init__()

        # Load your pre-trained XGBoost model
        # Replace 'your_model_path/xgboost_model.joblib' with the actual path to your model
        self.xgb_model= joblib.load('xgboost.joblib')
        self.feature_order = ['age', 'hypertension', 'bmi', 'heart_disease', 'hba1c', 'blood_glucose_level', 'gender', 'smoking_history']

        self.init_ui()

    def init_ui(self):
        # Create labels and input widgets
        self.gender_label = QLabel("Gender:")
        self.gender_entry = QComboBox()
        self.gender_entry.addItems(["Male", "Female"])

        self.age_label = QLabel("Age:")
        self.age_entry = QLineEdit()

        self.hypertension_label = QLabel("Hypertension:")
        self.hypertension_entry = QComboBox()
        self.hypertension_entry.addItems(["0", "1"])

        self.smoking_history_label = QLabel("Smoking History:")
        self.smoking_history_entry = QComboBox()
        self.smoking_history_entry.addItems(["Never", "Former", "Current"])

        self.bmi_label = QLabel("BMI:")
        self.bmi_entry = QLineEdit()

        self.heart_disease_label = QLabel("Heart Disease:")
        self.heart_disease_entry = QComboBox()
        self.heart_disease_entry.addItems(["0", "1"])

        self.hba1c_label = QLabel("HbA1c Level:")
        self.hba1c_entry = QLineEdit()

        self.blood_glucose_level_label = QLabel("Blood Glucose Level:")
        self.blood_glucose_level_entry = QLineEdit()

        # Create button to make predictions
        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.predict_diabetes)

        # Create label to display prediction result
        self.result_label = QLabel("Prediction: ")

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.gender_label)
        layout.addWidget(self.gender_entry)
        layout.addWidget(self.age_label)
        layout.addWidget(self.age_entry)
        layout.addWidget(self.hypertension_label)
        layout.addWidget(self.hypertension_entry)
        layout.addWidget(self.smoking_history_label)
        layout.addWidget(self.smoking_history_entry)
        layout.addWidget(self.bmi_label)
        layout.addWidget(self.bmi_entry)
        layout.addWidget(self.heart_disease_label)
        layout.addWidget(self.heart_disease_entry)
        layout.addWidget(self.hba1c_label)
        layout.addWidget(self.hba1c_entry)
        layout.addWidget(self.blood_glucose_level_label)
        layout.addWidget(self.blood_glucose_level_entry)
        layout.addWidget(self.predict_button)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def predict_diabetes(self):
        # Get user input from widgets
        gender = self.gender_entry.currentText()
        age = float(self.age_entry.text())
        hypertension = int(self.hypertension_entry.currentText())
        smoking_history = self.smoking_history_entry.currentText()
        bmi = float(self.bmi_entry.text())
        heart_disease = int(self.heart_disease_entry.currentText())
        hba1c = float(self.hba1c_entry.text())
        blood_glucose_level = float(self.blood_glucose_level_entry.text())

        # Convert categorical variables to numerical using One-Hot Encoding
        user_data = pd.DataFrame({
            'age': [age],
            'hypertension': [hypertension],
            'bmi': [bmi],
            'heart_disease': [heart_disease],
            'hba1c': [hba1c],
            'blood_glucose_level': [blood_glucose_level],
            'gender': [1 if gender == 'Male' else 0],
            'smoking_history': [1 if smoking_history == 'Former' else 0],
        }, columns=self.feature_order)
        # Make prediction
        print(user_data)
        prediction = self.xgb_model.predict(user_data)
        print(prediction)
        if prediction[0] == 1:
            result_text = "Risk of Diabetes"
        else:
            result_text = "No Risk of Diabetes"

        self.result_label.setText(f"Prediction: {result_text}")

if __name__ == '__main__':
    app = QApplication([])
    window = DiabetesPredictionApp()
    window.setWindowTitle("Diabetes Prediction")
    window.show()
    app.exec_()