from sklearn.metrics import accuracy_score,f1_score,precision_score,roc_auc_score,recall_score
import pandas as pd 
import os
import joblib
import pandas as pd
from train_models import get_models
def evaluate_models(models,x_train_sc, x_test_sc,y_train, y_test):
    models=get_models()
    results=[]
    best_score=0
    best_model=None
    best_model_name=""
    
    for name,model in models.items():
        model.fit(x_train_sc,y_train)
        y_pred=model.predict(x_test_sc)
        acc=accuracy_score(y_test, y_pred)
        y_train_pred = model.predict(x_train_sc)
        train_acc = accuracy_score(y_train, y_train_pred)
        # Probabilities for ROC–AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(x_test_sc)[:, 1]
        else:
            y_prob = model.decision_function(x_test_sc)
        roc_auc = roc_auc_score(y_test, y_prob)
        results.append({
            "Model": name,
            "Train_Accuracy":train_acc,
            "Test_Accuracy": acc,
            "Precision": precision_score(y_test, y_pred, average="weighted"),
            "Recall": recall_score(y_test, y_pred, average="weighted"),
            "F1 Score": f1_score(y_test, y_pred, average="weighted"),
            "ROC_AUC": roc_auc
        })
        if acc > best_score:
            best_score = acc
            best_model = model
            best_model_name = name
    # Create results folder
    os.makedirs("results", exist_ok=True)
     # Save best model using joblib
    joblib.dump(best_model, "results/best_model.joblib")
    return pd.DataFrame(results)