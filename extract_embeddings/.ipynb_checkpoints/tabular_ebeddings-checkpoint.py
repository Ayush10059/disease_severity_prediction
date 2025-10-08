import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.preprocessing import LabelEncoder

def load_patient_object(filepath):
    with open(filepath, 'rb') as input:  
        return pickle.load(input)
        
def get_demographic_embeddings(patient):
    """
    Performs label encoding on categorical columns using a pre-fitted
    dictionary of scikit-learn LabelEncoders.
    
    Args:
        patient (dict): A dictionary-like object with 'admissions' and 'patients' DataFrames.
        encoders (dict): A dictionary of pre-fitted LabelEncoder instances.

    Returns:
        np.ndarray: A 1D NumPy array of the label-encoded integer values.
    """
    
    # 1. Merge the patient's dataframes
    df = pd.merge(patient['admissions'], patient['patients'], on='subject_id')

    if df.empty:
        # Try to get the subject_id for a more informative warning message
        subject_id = "unknown"
        if not patient['admissions'].empty:
            subject_id = patient['admissions']['subject_id'].iloc[0]
        elif not patient['patients'].empty:
            subject_id = patient['patients']['subject_id'].iloc[0]

        print(f"Warning: Merge failed for subject_id '{subject_id}'. No matching data found. Skipping patient.")
        return None # Return None to signal that this patient should be skipped

    
    # --- Define the columns to be processed ---
    categorical_cols = [
        'race', 'marital_status', 'language', 'insurance',
        'admission_type', 'admission_location', 'discharge_location',
        'hospital_expire_flag', 'gender', 'anchor_year_group'
    ]
    
    # --- Create and fit a dictionary of LabelEncoders ---
    encoders = {}
    for col in categorical_cols:
        # Note: LabelEncoder cannot handle NaNs. You would need to fill them first, e.g., with .fillna('missing')
        le = LabelEncoder()
        le.fit(df[col])
        encoders[col] = le

    # 2. Initialize a list to store the encoded values
    encoded_values = []
    
    # 3. Iterate through the columns that have a fitted encoder
    for col, encoder in encoders.items():
        # Get the single value from the patient's dataframe for the current column
        value = df[col].iloc[0]
        
        # Transform the value. Handle cases where the value was not seen during fitting.
        try:
            encoded_value = encoder.transform([value])[0]
        except ValueError:
            encoded_value = -1
            
        encoded_values.append(encoded_value)
    
    # 4. Convert the list of encoded values to a NumPy array
    embedding = np.array(encoded_values)
    
    return embedding