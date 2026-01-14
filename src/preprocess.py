from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from imblearn.over_sampling import SMOTE


def preprocess_data(df, target_column, apply_smote=False):
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode categorical features
    label_encoders = {}

    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le 
    # -------------------------------
    # Encode TARGET if needed
    # -------------------------------
    if y.dtype == 'object':
        target_le = LabelEncoder()
        y = target_le.fit_transform(y)
    else:
        target_le = None

        
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

      # 🔥 SMOTE (ONLY training data)
    smote = SMOTE(random_state=42)
    X_train_resampling, y_train_resampling = smote.fit_resample(
        X_train_sc, y_train
    )

    # ✅ Verification
    print("Before SMOTE :", np.bincount(y_train))
    print("After SMOTE  :", np.bincount(y_train_resampling))

    return X_train_resampling, X_test_sc, y_train_resampling, y_test
