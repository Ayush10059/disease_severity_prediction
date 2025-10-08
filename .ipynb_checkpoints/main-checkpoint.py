import os
import gc
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

from config.config import * 
from load_and_pickle import * 

from extract_embeddings.tabular_embeddings import *
from extract_embeddings.image_embeddings import *
from extract_embeddings.notes_embeddings import *

from transformers import AutoTokenizer, AutoModel

def process_patient_data(patient, xray_embedder):
    """
    Extracts features for EACH entry within a patient's data.
    Returns a LIST of dictionaries, where each dictionary is a complete record.
    """
    entry_records = []

    demo_embeddings = get_demographic_embeddings(patient)
    if demo_embeddings is None:
        return []

    notes_embeddings = get_all_notes_embeddings(patient, biobert_tokenizer, biobert_model)

    dense_vision_vecs, pred_vision_vecs = get_vision_embeddings(patient, xray_embedder)

    num_entries = len(patient['imcxr'])
    if len(dense_vision_vecs) != num_entries or len(pred_vision_vecs) != num_entries:
        print(f"Warning: Mismatch in entry count for subject {patient['imcxr']['subject_id'].iloc[0]}. "
              f"Found {num_entries} table entries but {len(dense_vision_vecs)} vision embeddings. Skipping patient.")
        return []

    for i, entry_row in patient['imcxr'].iterrows():
        entry_data = {
            'subject_id': entry_row['subject_id'],
            'label': np.int64(entry_row['label']),
            
            # Keep each modality as its own vector
            'demographics': demo_embeddings.astype(np.float32),
            'notes': notes_embeddings.flatten().astype(np.float32),
            'vision_dense': dense_vision_vecs[i].astype(np.float32),
            'vision_pred': pred_vision_vecs[i].astype(np.float32)
        }
        
        entry_records.append(entry_data)

    return entry_records


if __name__ == '__main__':
    # mimic_data = load_mimiciv()
    # create_patient_pickles(mimic_data)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    xray_embedder = XRayEmbedder(device=device)

    local_model_path = "models"

    print(f"Loading BioBERT model and tokenizer from local path: {local_model_path}...")
    biobert_tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    biobert_model = AutoModel.from_pretrained(local_model_path).to(device)
    print("BioBERT model loaded.")

    print(f"--- Starting Multimodal Feature Extraction to {FUSION_OUTPUT_FNAME} ---")
    sys.stdout.flush()

    all_patient_results = []
    
    for filename in tqdm(sorted(os.listdir(PICKLE_PATH)), desc="Processing Patient Data"):
        if filename.endswith(".pkl"):
            filepath = os.path.join(PICKLE_PATH, filename)
            patient = load_patient_object(filepath)
            
            patient_records = process_patient_data(patient, xray_embedder)
            
            if patient_records: # Check if the list is not empty
                all_patient_results.extend(patient_records)

    FUSION_OUTPUT_FNAME = 'data/multimodal_features.pkl'
    print(f"Saving final list with {len(all_patient_results)} records to {FUSION_OUTPUT_FNAME}...")
    with open(FUSION_OUTPUT_FNAME, 'wb') as f:
        pickle.dump(all_patient_results, f)

    print("--- Process Complete ---")
