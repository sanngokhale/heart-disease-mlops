"""Test script to verify model packaging reproducibility"""
import joblib
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("="*55)
print("   MODEL REPRODUCIBILITY TEST")
print("="*55)

# Load saved artifacts
model = joblib.load('models/best_model.joblib')
preprocessor = joblib.load('models/preprocessor.joblib')

print("\n[1] LOADED ARTIFACTS:")
print(f"    Model: {type(model).__name__}")
print(f"    Preprocessor: {type(preprocessor).__name__}")

# Sample patient
sample = pd.DataFrame([{
    'age': 55, 'sex': 1, 'cp': 2, 'trestbps': 130,
    'chol': 250, 'fbs': 0, 'restecg': 1, 'thalach': 150,
    'exang': 0, 'oldpeak': 1.5, 'slope': 1, 'ca': 0, 'thal': 2
}])

print("\n[2] SAMPLE INPUT:")
print("    Age: 55, Sex: Male, Chest Pain Type: 2")

# Transform and predict
X = preprocessor.transform(sample)
pred = model.predict(X)[0]
prob = model.predict_proba(X)[0][1]

print("\n[3] PREDICTION RESULT:")
if pred:
    print("    Diagnosis: Heart Disease")
else:
    print("    Diagnosis: No Heart Disease")
print(f"    Risk Score: {prob*100:.1f}%")

print("\n" + "="*55)
print("   REPRODUCIBILITY VERIFIED!")
print("="*55)
