from load_data import load_dataset
from preprocess import preprocess_data
from train_models import get_models
from evaluate import evaluate_models
import pandas as pd

def main():
    df = load_dataset("data/restaurant_classification_medium.csv")
    
    TARGET_COLUMN = "Satisfied"

    (
        X_train_resampling,
        X_test_sc,
        y_train_resampling,
        y_test
    ) = preprocess_data(df, TARGET_COLUMN)
    
    models = get_models()

    results_df = evaluate_models(
        models,
        X_train_resampling,
        X_test_sc,
        y_train_resampling,
        y_test
    )
    
    results_df = results_df.sort_values(by="ROC_AUC" and "F1 Score", ascending=False)
    
    print("\nModel Performance:\n")
    print(results_df)

    results_df.to_csv("results/model_scores.csv", index=False)

    print("\n✅ Best Model:", results_df.iloc[0]["Model"])
    
if __name__ == "__main__":
    main()