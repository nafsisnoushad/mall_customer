import os
import joblib

# Go from src/ → project root (mall_customer)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Correct path (ONLY ONE results)
model_path = os.path.join(BASE_DIR, "results", "best_model.joblib")

print("Looking for model at:", model_path)

model = joblib.load(model_path)

print("✅ Model loaded successfully")
