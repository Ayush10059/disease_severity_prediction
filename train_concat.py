import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import pickle
import numpy as np
import os

def train_model(data_path, model_save_path):
    """
    Loads the structured pickle data, performs a patient-level split,
    concatenates features, trains an XGBoost model, and saves the model.
    
    Args:
        data_path (str): Path to the pickle file with structured multimodal features.
        model_save_path (str): Path to save the trained model.
    """
    print("--- Starting Model Training ---")

    # 1. Load Data from Pickle
    print(f"Loading data from {data_path}...")
    try:
        with open(data_path, 'rb') as f:
            all_records = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return

    # 2. Prepare DataFrame and Perform Patient-Level Split
    print("Performing patient-level data split...")
    df = pd.DataFrame(all_records)
    seed = 42
    
    # Get a list of unique patient IDs
    patient_ids = df['subject_id'].unique()
    
    # Split the patient IDs into training and testing sets
    train_ids, test_ids = train_test_split(patient_ids, test_size=0.3, random_state=seed)
    
    # Create train and test sets by filtering the DataFrame based on patient IDs
    train_df = df[df['subject_id'].isin(train_ids)]
    test_df = df[df['subject_id'].isin(test_ids)]

    print(f"Data split complete. Training records: {len(train_df)}, Test records: {len(test_df)}")

    # 3. Create Feature Matrices from the Split DataFrames
    def create_feature_matrix(subset_df):
        """Helper function to concatenate features for a given dataframe."""
        X_demo = np.stack(subset_df['demographics'].values)
        X_notes = np.stack(subset_df['notes'].values)
        X_vdense = np.stack(subset_df['vision_dense'].values)
        X_vpred = np.stack(subset_df['vision_pred'].values)
        X_concat = np.concatenate([X_demo, X_notes, X_vdense, X_vpred], axis=1)
        return pd.DataFrame(X_concat), subset_df['label'].astype(int)

    X_train, y_train = create_feature_matrix(train_df)
    X_test, y_test = create_feature_matrix(test_df)
    
    print(f"Feature matrices created. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # Save the correctly-split test set for the evaluation script
    os.makedirs('data', exist_ok=True)
    test_set = {'X_test': X_test, 'y_test': y_test}
    with open('data/concatenated_test_set.pkl', 'wb') as f:
        pickle.dump(test_set, f)
    print("Test set saved to data/concatenated_test_set.pkl")

    # 4. Initialize and Train the XGBoost Model
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        seed=seed,
        use_label_encoder=False,
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05
    )
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 5. Save the Trained Model
    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved successfully to {model_save_path}")

    # 6. Quick Evaluation on the Test Set
    print("\n--- Quick Evaluation on Test Set ---")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    report = classification_report(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    
    print("Classification Report:")
    print(report)
    print(f"Test ROC-AUC (One-vs-Rest): {auc:.4f}")
    print("--------------------------------------")


if __name__ == '__main__':
    DATA_FILE_PATH = 'data/multimodal_features.pkl' 
    MODEL_SAVE_PATH = 'data/xgboost_concat_model.pkl'
    train_model(DATA_FILE_PATH, MODEL_SAVE_PATH)

