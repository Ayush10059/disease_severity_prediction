import os
import datetime as dt

# Base
import pandas as pd
import numpy as np
import pydicom
import cv2
from collections import defaultdict

from dask import dataframe as dd
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar

import pickle
from tqdm import tqdm

# Register Dask progress bar for visualization
ProgressBar().register()

# --- Global Paths ---
mimic_iv_path = 'data/mimic_iv/'
mimic_cxr_path = 'data/mimic_cxr/'
mimic_note_path = 'data/mimic_note/'
pickle_path = 'data/pickle/'

class MIMICIVData:
    """
    A container class to hold all Dask DataFrames and any eagerly loaded data (like CXR images).
    """
    def __init__(self, **kwargs):
        # Dynamically assign all loaded DataFrames to instance attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

def load_mimiciv():
    """
    Loads all MIMIC-IV, CXR, and Note tables lazily using Dask.
    It then eagerly loads and processes a sample CXR image.
    It returns a fully initialized MIMICIVData object.
    """
    print('Starting lazy loading of MIMIC datasets...')
    
    # --- HOSP TABLES (Read Dask DataFrames) ---
    df_admissions = dd.read_csv(mimic_iv_path + 'hosp/expert_admissions.csv', assume_missing=True, 
                                dtype={'admission_location': 'object','deathtime': 'object','edouttime': 'object','edregtime': 'object'})
    df_patients = dd.read_csv(mimic_iv_path + 'hosp/expert_patients.csv', assume_missing=True, 
                              dtype={'dod': 'object'})
    df_transfers = dd.read_csv(mimic_iv_path + 'hosp/transfers.csv', assume_missing=True, 
                               dtype={'careunit': 'object'})
    df_diagnoses_icd = dd.read_csv(mimic_iv_path + 'hosp/diagnoses_icd.csv', assume_missing=True, 
                                   dtype={'icd_code': 'object', 'icd_version': 'object'})
    df_drgcodes = dd.read_csv(mimic_iv_path + 'hosp/drgcodes.csv', assume_missing=True)
    df_emar = dd.read_csv(mimic_iv_path + 'hosp/emar.csv', assume_missing=True)
    df_emar_detail = dd.read_csv(mimic_iv_path + 'hosp/emar_detail.csv', assume_missing=True, 
                                 low_memory=False, 
                                 dtype={'completion_interval': 'object','dose_due': 'object','dose_given': 'object',
                                        'infusion_complete': 'object','infusion_rate_adjustment': 'object','infusion_rate_unit': 'object',
                                        'new_iv_bag_hung': 'object','product_description_other': 'object','reason_for_no_barcode': 'object',
                                        'restart_interval': 'object','route': 'object','side': 'object','site': 'object',
                                        'continued_infusion_in_other_location': 'object','infusion_rate': 'object','non_formulary_visual_verification': 'object',
                                        'prior_infusion_rate': 'object','product_amount_given': 'object', 'infusion_rate_adjustment_amount': 'object'})
    df_hcpcsevents = dd.read_csv(mimic_iv_path + 'hosp/hcpcsevents.csv', assume_missing=True, 
                                 dtype={'hcpcs_cd': 'object'})
    df_labevents = dd.read_csv(mimic_iv_path + 'hosp/labevents.csv', assume_missing=True, 
                               dtype={'storetime': 'object', 'value': 'object', 'valueuom': 'object', 'flag': 'object', 'priority': 'object', 'comments': 'object'})
    df_microbiologyevents = dd.read_csv(mimic_iv_path + 'hosp/microbiologyevents.csv', assume_missing=True, 
                                        dtype={'comments': 'object', 'quantity': 'object', 'dilution_comparison': 'object', 'dilution_text': 'object'})
    df_poe = dd.read_csv(mimic_iv_path + 'hosp/poe.csv', assume_missing=True, 
                         dtype={'discontinue_of_poe_id': 'object','discontinued_by_poe_id': 'object','order_status': 'object'})
    df_poe_detail = dd.read_csv(mimic_iv_path + 'hosp/poe_detail.csv', assume_missing=True)
    df_prescriptions = dd.read_csv(mimic_iv_path + 'hosp/prescriptions.csv', assume_missing=True, 
                                   dtype={'form_rx': 'object','gsn': 'object'})
    df_procedures_icd = dd.read_csv(mimic_iv_path + 'hosp/procedures_icd.csv', assume_missing=True, 
                                    dtype={'icd_code': 'object', 'icd_version': 'object'})
    df_services = dd.read_csv(mimic_iv_path + 'hosp/services.csv', assume_missing=True, 
                              dtype={'prev_service': 'object'})

    # --- CXR TABLES ---
    df_cxr_split = dd.read_csv(mimic_cxr_path + 'subset/mimic-cxr-2.0.0-split.csv', assume_missing=True)
    df_cxr_negbio = dd.read_csv(mimic_cxr_path + 'subset/mimic-cxr-2.0.0-negbio.csv', assume_missing=True)
    df_cxr_chexpert = dd.read_csv(mimic_cxr_path + 'subset/mimic-cxr-2.0.0-chexpert.csv', assume_missing=True)
    df_cxr_metadata = dd.read_csv(mimic_cxr_path + 'subset/mimic-cxr-2.0.0-metadata.csv', assume_missing=True, 
                                        dtype={'dicom_id': 'object'}, blocksize=None)

    # --- NOTES TABLES ---
    df_dsnotes =  dd.from_pandas(pd.read_csv(mimic_note_path + 'discharge.csv', 
                             dtype={'charttime': 'object', 'storetime': 'object', 'text': 'object'}), chunksize=8)
    df_radnotes =  dd.from_pandas(pd.read_csv(mimic_note_path + 'radiology.csv', 
                              dtype={'charttime': 'object', 'storetime': 'object', 'text': 'object'}), chunksize=8)

    # --- LAZY DATA PREPARATION (Chaining Dask operations) ---
    print('Applying lazy cleaning operations...')

    # HOSP Cleaning (Stripping whitespace and dropping time columns)
    df_diagnoses_icd = df_diagnoses_icd.assign(
        icd_code=df_diagnoses_icd.icd_code.str.strip(),
        icd_version=df_diagnoses_icd.icd_version.str.strip()
    )
    df_procedures_icd = df_procedures_icd.assign(
        icd_code=df_procedures_icd.icd_code.str.strip(),
        icd_version=df_procedures_icd.icd_version.str.strip()
    )
    df_hcpcsevents = df_hcpcsevents.assign(hcpcs_cd=df_hcpcsevents.hcpcs_cd.str.strip())
    
    # Column Drops
    cols_to_drop = {
        'admissions': ['admittime', 'dischtime', 'edregtime', 'edouttime'],
        'transfers': ['intime', 'outtime'],
        'prescriptions': ['starttime', 'stoptime'],
        'emar': ['charttime', 'scheduletime', 'storetime'],
        'labevents': ['charttime', 'storetime'],
        'microbiologyevents': ['charttime', 'storedate', 'storetime'],
        'poe': ['ordertime'],
        'services': ['transfertime'],
        'dsnotes': ['charttime', 'storetime'],
        'radnotes': ['charttime', 'storetime'],
    }
    
    # Function to apply drops lazily
    def lazy_drop(df, drop_cols):
        current_cols = set(df.columns)
        valid_drops = [col for col in drop_cols if col in current_cols]
        return df.drop(columns=valid_drops)

    df_admissions = lazy_drop(df_admissions, cols_to_drop['admissions'])
    df_transfers = lazy_drop(df_transfers, cols_to_drop['transfers'])
    df_prescriptions = lazy_drop(df_prescriptions, cols_to_drop['prescriptions'])
    df_emar = lazy_drop(df_emar, cols_to_drop['emar'])
    df_labevents = lazy_drop(df_labevents, cols_to_drop['labevents'])
    df_microbiologyevents = lazy_drop(df_microbiologyevents, cols_to_drop['microbiologyevents'])
    df_poe = lazy_drop(df_poe, cols_to_drop['poe'])
    df_services = lazy_drop(df_services, cols_to_drop['services'])
    df_dsnotes = lazy_drop(df_dsnotes, cols_to_drop['dsnotes'])
    df_radnotes = lazy_drop(df_radnotes, cols_to_drop['radnotes'])
        
    print('Finished defining Dask graph. DataFrames are ready for use.')

    # --- EAGER CXR IMAGE PROCESSING (Non-Dask operation) ---
    print('Starting Eager CXR Image Processing...')
    imcxr_list = []
    img_cxr_shape = (224, 224)
    
    # Load patient subset IDs and desired severity column
    df_severity_label = pd.read_csv('data/severity_dataset/expert_report_edema_severity.csv')
    cxr_meta = pd.read_csv(os.path.join(mimic_cxr_path, 'mimic-cxr-2.0.0-metadata.csv.gz'), compression='gzip')

    df_severity_label = df_severity_label.rename(columns={'edema_severity': 'label'})

    cxr_filtered = pd.merge(
        cxr_meta,
        df_severity_label,
        on=['subject_id', 'study_id'],
        how='inner'
    )
    
    # Check if there is any imaging record for the patient
    for _, row in tqdm(cxr_filtered.iterrows(), desc="Processing Images"):
        # Construct folder structure
        study_id_str = "s" + str(int(row['study_id']))
        subject_id_str = "p" + str(int(row['subject_id']))
        top_folder = "p" + str(int(row['subject_id']))[:2]
        filename = row['dicom_id'] + ".dcm"
        
        # Build the full path
        img_path = os.path.join(mimic_cxr_path + 'subset/files/', top_folder, subject_id_str, study_id_str, filename)
        
        # Check if the DICOM file exists
        if os.path.exists(img_path):
            try:
                # Load the DICOM file with pydicom
                dicom_file = pydicom.dcmread(img_path)
                # Extract the pixel data as a NumPy array
                img_cxr = dicom_file.pixel_array
                
                # Normalize the image data to a usable range (e.g., 0-255 for 8-bit images)
                img_normalized = cv2.normalize(img_cxr, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    
                # Resize the normalized image using OpenCV
                img_resized = cv2.resize(img_normalized, img_cxr_shape)
    
                # Append the final image array to the list
                imcxr_list.append({
                    'subject_id': int(row['subject_id']),
                    'label': int(row['label']),
                    'imcxr': np.array(img_resized)
                })

                # print(f"Image file processed: {img_path}")
                
            except Exception as e:
                print(f"Error processing DICOM file at '{img_path}': {e}")
        else:
            print(f"Image file not found: {img_path}")
    else:
        print("No imaging record found for this patient subset.")

    df_imcxr = pd.DataFrame(imcxr_list)

    df_imcxr.drop(df_imcxr.columns[0], axis=1)
    
    print(f"Loaded {len(df_imcxr)} CXR images with subject IDs.")

    # Store all loaded DataFrames and the processed image data in a dictionary
    loaded_dfs = {
        'admissions': df_admissions, 'patients': df_patients, 'transfers': df_transfers,
        'diagnoses_icd': df_diagnoses_icd, 'drgcodes': df_drgcodes, 'emar': df_emar,
        'emar_detail': df_emar_detail, 'hcpcsevents': df_hcpcsevents, 'labevents': df_labevents,
        'microbiologyevents': df_microbiologyevents, 'poe': df_poe, 'poe_detail': df_poe_detail,
        'prescriptions': df_prescriptions, 'procedures_icd': df_procedures_icd, 'services': df_services,
        'cxr_metadata': df_cxr_metadata, 'cxr_split': df_cxr_split, 
        'cxr_chexpert': df_cxr_chexpert, 'cxr_negbio': df_cxr_negbio, 
        'dsnotes': df_dsnotes, 'radnotes': df_radnotes,
        'imcxr': df_imcxr,
    }

    # Return the MIMICIVData object directly using dictionary unpacking (**)
    return MIMICIVData(**loaded_dfs)

def create_patient_pickles(mimic_data: MIMICIVData, output_dir: str = 'data/pickle'):
    """
    Processes the Dask DataFrames in MIMICIVData patient-by-patient, 
    computes the result, and saves the structured data to individual pickle files.
    """
    print(f"\n--- Starting Patient Pickling Process ---")

    # Setup output directory
    os.makedirs(output_dir, exist_ok=True)

    # Identify all DataFrames (Dask and Pandas) for processing
    # We iterate over the instance's attributes to find all data structures
    all_dfs = {k: v for k, v in mimic_data.__dict__.items() if isinstance(v, (dd.DataFrame, pd.DataFrame))}
    
    # Get the master list of unique subject IDs from the main patient table
    # This must be computed first if patients is a Dask DataFrame.
    print("Computing unique subject IDs...")
    try:
        # Check if patients is Dask and compute subject IDs
        if isinstance(all_dfs['imcxr'], dd.DataFrame):
            unique_subjects = all_dfs['imcxr']['subject_id'].compute().tolist()
        else: # Assuming it's already a Pandas DataFrame (like the mock)
            unique_subjects = all_dfs['imcxr']['subject_id'].tolist()
    except Exception as e:
        print(f"Failed to compute unique subject IDs: {e}")
        return

    print(f"Found {len(unique_subjects)} unique patients to process.")

    subject_id_counts = defaultdict(int)    
    
    # Process and Pickle each patient, using tqdm for progress visualization
    for subject_id in tqdm(unique_subjects, desc="Pickling Patient Data"):
        patient_record = {'subject_id': subject_id}
        
        # A. Filter Dask DataFrames
        dask_filter_tasks = {}
        for name, df in all_dfs.items():
            if isinstance(df, dd.DataFrame):
                # Apply the filter lazily
                filtered_df = df[df['subject_id'] == subject_id]
                dask_filter_tasks[name] = filtered_df
            elif isinstance(df, pd.DataFrame) and 'subject_id' in df.columns:
                # Apply the filter eagerly for Pandas DataFrames
                patient_record[name] = df[df['subject_id'] == subject_id].reset_index(drop=True)
            elif name == 'df_severity_label' and 'subject_id' in df.columns:
                 # Ensure df_severity_labels is handled if not included in the generic check
                patient_record[name] = df[df['subject_id'] == subject_id].reset_index(drop=True)

        # B. Compute Dask results in one go for efficiency
        if dask_filter_tasks:
            try:
                # Compute all filtered Dask DataFrames simultaneously
                computed_results = dd.compute(dask_filter_tasks)[0]
                
                # Store computed Pandas DataFrames in the patient record
                for name, pdf in computed_results.items():
                    patient_record[name] = pdf.reset_index(drop=True)
                    
            except Exception as e:
                # Use logging/warning instead of print for cleaner tqdm output
                print(f"    Warning: Failed to compute Dask data for patient {subject_id}: {e}")
                # Continue to the next patient, but the record might be incomplete

        # Save the patient record
        if len(patient_record) > 1: # Check if any data was actually collected
            # Get the current count for this subject_id
            instance_count = subject_id_counts[subject_id]
        
            filename = os.path.join(output_dir, f'patient_{int(subject_id)}_{instance_count}.pkl')

            # Increment the counter for the next time we see this subject_id
            subject_id_counts[subject_id] += 1
            try:
                with open(filename, 'wb') as f:
                    pickle.dump(patient_record, f)
                # print(f"    Saved data to {filename}") # Removed verbose print
            except Exception as e:
                print(f"    Error saving pickle file for {int(subject_id)}: {e}")

    print("\n--- Patient Pickling Complete ---")
    print(f"Pickle files saved in the '{output_dir}' directory.")
