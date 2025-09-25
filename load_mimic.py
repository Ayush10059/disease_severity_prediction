# System                                                                                           
import os
import sys

# Base
import cv2
import math
import copy
import pickle
import numpy as np
import pandas as pd
import pandas.io.sql as psql
import datetime as dt
import plotly.express as px
import matplotlib.pyplot as plt
import missingno as msno
from tqdm import tqdm
from glob import glob
from shutil import copyfile

from dask import dataframe as dd
from dask.diagnostics import ProgressBar
ProgressBar().register()

# Core AI/ML
import tensorflow as tf
import torch
import torch.nn.functional as F
import torchvision, torchvision.transforms
from torch.utils.data import Dataset, DataLoader
import torchxrayvision as xrv

# Scipy
from scipy.stats import ks_2samp
from scipy.signal import find_peaks

# Scikit-learn
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

import pydicom  # Import the pydicom library

# Define MIMIC IV Image Data Location
mimic_iv_path = 'data/mimic_iv/'
mimic_cxr_path = 'data/mimic_cxr/'
mimic_note_path = 'data/mimic_note/'

pickle_path = 'data/pickle/'

class MIMICIVData:
    def __init__(
        self,
        df_admissions, df_patients, df_transfers,
        df_diagnoses_icd, df_drgcodes, df_emar, 
        df_emar_detail, df_hcpcsevents, df_labevents,
        df_microbiologyevents, df_poe, df_poe_detail, 
        df_prescriptions, df_procedures_icd, df_services,
        df_procedureevents, df_outputevents, df_inputevents,
        df_icustays, df_datetimeevents, df_chartevents,
        df_mimic_cxr_metadata, df_mimic_cxr_split, df_mimic_cxr_chexpert,
        df_mimic_cxr_negbio, df_dsnotes, df_radnotes
    ):
        self.df_admissions = df_admissions
        self.df_patients = df_patients
        self.df_transfers = df_transfers
        self.df_diagnoses_icd = df_diagnoses_icd
        self.df_drgcodes = df_drgcodes
        self.df_emar = df_emar
        self.df_emar_detail = df_emar_detail
        self.df_hcpcsevents = df_hcpcsevents
        self.df_labevents = df_labevents
        self.df_microbiologyevents = df_microbiologyevents
        self.df_poe = df_poe
        self.df_poe_detail = df_poe_detail
        self.df_prescriptions = df_prescriptions
        self.df_procedures_icd = df_procedures_icd
        self.df_services = df_services
        self.df_procedureevents = df_procedureevents
        self.df_outputevents = df_outputevents
        self.df_inputevents = df_inputevents
        self.df_icustays = df_icustays
        self.df_datetimeevents = df_datetimeevents
        self.df_chartevents = df_chartevents
        self.df_mimic_cxr_metadata = df_mimic_cxr_metadata
        self.df_mimic_cxr_split = df_mimic_cxr_split
        self.df_mimic_cxr_chexpert = df_mimic_cxr_chexpert
        self.df_mimic_cxr_negbio = df_mimic_cxr_negbio
        self.df_dsnotes = df_dsnotes
        self.df_radnotes = df_radnotes


# MIMICIV PATIENT CLASS STRUCTURE
class Patient_ICU(object):
    def __init__(self, admissions, demographics, transfers, core,\
        diagnoses_icd, drgcodes, emar, emar_detail, hcpcsevents,\
        labevents, microbiologyevents, poe, poe_detail,\
        prescriptions, procedures_icd, services, procedureevents,\
        outputevents, inputevents, icustays, datetimeevents,\
        chartevents, cxr, imcxr,\
        dsnotes, radnotes):
        
        ## CORE
        self.admissions = admissions
        self.demographics = demographics
        self.transfers = transfers
        self.core = core
        ## HOSP
        self.drgcodes = drgcodes
        self.emar = emar
        self.emar_detail = emar_detail
        self.labevents = labevents
        self.microbiologyevents = microbiologyevents
        self.poe = poe
        self.poe_detail = poe_detail
        self.prescriptions = prescriptions
        self.procedures_icd = procedures_icd
        self.services = services
        ## ICU
        self.procedureevents = procedureevents
        self.outputevents = outputevents
        self.inputevents = inputevents
        self.icustays = icustays
        self.datetimeevents = datetimeevents
        self.chartevents = chartevents
        ## CXR
        self.cxr = cxr
        self.imcxr = imcxr
        ## NOTES
        self.dsnotes = dsnotes
        self.radnotes = radnotes

# LOAD ALL MIMIC IV TABLES IN MEMORY (warning: High memory lengthy process)
def load_mimiciv():
    # Outputs:
    #   df's -> Many dataframes with all loaded MIMIC IV tables 
    
    ### -> Initializations & Data Loading
    ###    Resources to identify tables and variables of interest can be found in the MIMIC-IV official API (https://mimic-iv.mit.edu/docs/)
    
    ## CORE
    df_admissions = dd.read_csv(mimic_iv_path + 'hosp/admissions.csv', assume_missing=True, dtype={'admission_location': 'object','deathtime': 'object','edouttime': 'object','edregtime': 'object'})
    df_patients = dd.read_csv(mimic_iv_path + 'hosp/patients.csv', assume_missing=True, dtype={'dod': 'object'})  
    df_transfers = dd.read_csv(mimic_iv_path + 'hosp/transfers.csv', assume_missing=True, dtype={'careunit': 'object'})
  
    ## HOSP
    df_diagnoses_icd = dd.read_csv(mimic_iv_path + 'hosp/diagnoses_icd.csv', assume_missing=True, dtype={'icd_code': 'object', 'icd_version': 'object'})
    df_drgcodes = dd.read_csv(mimic_iv_path + 'hosp/drgcodes.csv', assume_missing=True)
    df_emar = dd.read_csv(mimic_iv_path + 'hosp/emar.csv', assume_missing=True)
    df_emar_detail = dd.read_csv(mimic_iv_path + 'hosp/emar_detail.csv', assume_missing=True, low_memory=False, dtype={'completion_interval': 'object','dose_due': 'object','dose_given': 'object','infusion_complete': 'object','infusion_rate_adjustment': 'object','infusion_rate_unit': 'object','new_iv_bag_hung': 'object','product_description_other': 'object','reason_for_no_barcode': 'object','restart_interval': 'object','route': 'object','side': 'object','site': 'object','continued_infusion_in_other_location': 'object','infusion_rate': 'object','non_formulary_visual_verification': 'object','prior_infusion_rate': 'object','product_amount_given': 'object', 'infusion_rate_adjustment_amount': 'object'})
    df_hcpcsevents = dd.read_csv(mimic_iv_path + 'hosp/hcpcsevents.csv', assume_missing=True, dtype={'hcpcs_cd': 'object'})
    df_labevents = dd.read_csv(mimic_iv_path + 'hosp/labevents.csv', assume_missing=True, dtype={'storetime': 'object', 'value': 'object', 'valueuom': 'object', 'flag': 'object', 'priority': 'object', 'comments': 'object'})
    df_microbiologyevents = dd.read_csv(mimic_iv_path + 'hosp/microbiologyevents.csv', assume_missing=True, dtype={'comments': 'object', 'quantity': 'object', 'dilution_comparison': 'object', 'dilution_text': 'object'})
    df_poe = dd.read_csv(mimic_iv_path + 'hosp/poe.csv', assume_missing=True, dtype={'discontinue_of_poe_id': 'object','discontinued_by_poe_id': 'object','order_status': 'object'})
    df_poe_detail = dd.read_csv(mimic_iv_path + 'hosp/poe_detail.csv', assume_missing=True)
    df_prescriptions = dd.read_csv(mimic_iv_path + 'hosp/prescriptions.csv', assume_missing=True, dtype={'form_rx': 'object','gsn': 'object'})
    df_procedures_icd = dd.read_csv(mimic_iv_path + 'hosp/procedures_icd.csv', assume_missing=True, dtype={'icd_code': 'object', 'icd_version': 'object'})
    df_services = dd.read_csv(mimic_iv_path + 'hosp/services.csv', assume_missing=True, dtype={'prev_service': 'object'})
  
    ## ICU
    df_procedureevents = dd.read_csv(mimic_iv_path + 'icu/procedureevents.csv', assume_missing=True, dtype={'value': 'object', 'secondaryordercategoryname': 'object', 'totalamountuom': 'object', 'location': 'object', 'locationcategory': 'object'})
    df_outputevents = dd.read_csv(mimic_iv_path + 'icu/outputevents.csv', assume_missing=True, dtype={'value': 'object'})
    df_inputevents = dd.read_csv(mimic_iv_path + 'icu/inputevents.csv', assume_missing=True, dtype={'value': 'object', 'secondaryordercategoryname': 'object', 'totalamountuom': 'object'})
    df_icustays = dd.read_csv(mimic_iv_path + 'icu/icustays.csv', assume_missing=True)
    df_datetimeevents = dd.read_csv(mimic_iv_path + 'icu/datetimeevents.csv', assume_missing=True, dtype={'value': 'object'})
    df_chartevents = dd.read_csv(mimic_iv_path + 'icu/chartevents.csv', assume_missing=True, low_memory=False, dtype={'value': 'object', 'valueuom': 'object'})
  
    ## CXR
    df_mimic_cxr_split = dd.read_csv(mimic_cxr_path + 'subset/mimic-cxr-2.0.0-split.csv', assume_missing=True)
    df_mimic_cxr_chexpert = dd.read_csv(mimic_cxr_path + 'subset/mimic-cxr-2.0.0-chexpert.csv', assume_missing=True)
    try:
        df_mimic_cxr_metadata = dd.read_csv(mimic_cxr_path + 'subset/mimic-cxr-2.0.0-metadata.csv', assume_missing=True, dtype={'dicom_id': 'object'}, blocksize=None)
    except:
        df_mimic_cxr_metadata = pd.read_csv(mimic_cxr_path + 'subset/mimic-cxr-2.0.0-metadata.csv', dtype={'dicom_id': 'object'})
        df_mimic_cxr_metadata = dd.from_pandas(df_mimic_cxr_metadata, npartitions=7)
    df_mimic_cxr_negbio = dd.read_csv(mimic_cxr_path + 'subset/mimic-cxr-2.0.0-negbio.csv', assume_missing=True)
  
    ## NOTES
    df_dsnotes = dd.from_pandas(pd.read_csv(mimic_note_path + 'discharge.csv', dtype={'charttime': 'object', 'storetime': 'object', 'text': 'object'}), chunksize=8)
    df_radnotes = dd.from_pandas(pd.read_csv(mimic_note_path + 'radiology.csv', dtype={'charttime': 'object', 'storetime': 'object', 'text': 'object'}), chunksize=8)
    
    
    ### -> Data Preparation (Create full database in dask format)
    ### Fix data type issues to allow for merging
    ## CORE
    df_admissions['admittime'] = dd.to_datetime(df_admissions['admittime'])
    df_admissions['dischtime'] = dd.to_datetime(df_admissions['dischtime'])
    df_admissions['deathtime'] = dd.to_datetime(df_admissions['deathtime'])
    df_admissions['edregtime'] = dd.to_datetime(df_admissions['edregtime'])
    df_admissions['edouttime'] = dd.to_datetime(df_admissions['edouttime'])
    
    df_transfers['intime'] = dd.to_datetime(df_transfers['intime'])
    df_transfers['outtime'] = dd.to_datetime(df_transfers['outtime'])
    
    ## HOSP
    df_diagnoses_icd.icd_code = df_diagnoses_icd.icd_code.str.strip()
    df_diagnoses_icd.icd_version = df_diagnoses_icd.icd_version.str.strip()
    
    df_procedures_icd.icd_code = df_procedures_icd.icd_code.str.strip()
    df_procedures_icd.icd_version = df_procedures_icd.icd_version.str.strip()
    
    df_hcpcsevents.hcpcs_cd = df_hcpcsevents.hcpcs_cd.str.strip()
    
    df_prescriptions['starttime'] = dd.to_datetime(df_prescriptions['starttime'])
    df_prescriptions['stoptime'] = dd.to_datetime(df_prescriptions['stoptime'])
    
    df_emar['charttime'] = dd.to_datetime(df_emar['charttime'])
    df_emar['scheduletime'] = dd.to_datetime(df_emar['scheduletime'])
    df_emar['storetime'] = dd.to_datetime(df_emar['storetime'])
    
    df_labevents['charttime'] = dd.to_datetime(df_labevents['charttime'])
    df_labevents['storetime'] = dd.to_datetime(df_labevents['storetime'])
    
    df_microbiologyevents['chartdate'] = dd.to_datetime(df_microbiologyevents['chartdate'])
    df_microbiologyevents['charttime'] = dd.to_datetime(df_microbiologyevents['charttime'])
    df_microbiologyevents['storedate'] = dd.to_datetime(df_microbiologyevents['storedate'])
    df_microbiologyevents['storetime'] = dd.to_datetime(df_microbiologyevents['storetime'])
    
    df_poe['ordertime'] = dd.to_datetime(df_poe['ordertime'])
    df_services['transfertime'] = dd.to_datetime(df_services['transfertime'])
    
    ## ICU
    df_procedureevents['starttime'] = dd.to_datetime(df_procedureevents['starttime'])
    df_procedureevents['endtime'] = dd.to_datetime(df_procedureevents['endtime'])
    df_procedureevents['storetime'] = dd.to_datetime(df_procedureevents['storetime'])
    
    df_outputevents['charttime'] = dd.to_datetime(df_outputevents['charttime'])
    df_outputevents['storetime'] = dd.to_datetime(df_outputevents['storetime'])
    
    df_inputevents['starttime'] = dd.to_datetime(df_inputevents['starttime'])
    df_inputevents['endtime'] = dd.to_datetime(df_inputevents['endtime'])
    df_inputevents['storetime'] = dd.to_datetime(df_inputevents['storetime'])
    
    df_icustays['intime'] = dd.to_datetime(df_icustays['intime'])
    df_icustays['outtime'] = dd.to_datetime(df_icustays['outtime'])
    
    df_datetimeevents['charttime'] = dd.to_datetime(df_datetimeevents['charttime'])
    df_datetimeevents['storetime'] = dd.to_datetime(df_datetimeevents['storetime'])
    
    df_chartevents['charttime'] = dd.to_datetime(df_chartevents['charttime'])
    df_chartevents['storetime'] = dd.to_datetime(df_chartevents['storetime'])
    
    ## CXR
    if (not 'cxrtime' in df_mimic_cxr_metadata.columns) or (not 'Img_Filename' in df_mimic_cxr_metadata.columns):
        # Create CXRTime variable if it does not exist already
        print("Processing CXRtime stamps")
        df_cxr = df_mimic_cxr_metadata.compute()
        df_cxr['StudyDateForm'] = pd.to_datetime(df_cxr['StudyDate'], format='%Y%m%d')
        df_cxr['StudyTimeForm'] = df_cxr.apply(lambda x : '%#010.3f' % x['StudyTime'] ,1)
        df_cxr['StudyTimeForm'] = pd.to_datetime(df_cxr['StudyTimeForm'], format='%H%M%S.%f').dt.time
        df_cxr['cxrtime'] = df_cxr.apply(lambda r : dt.datetime.combine(r['StudyDateForm'],r['StudyTimeForm']),1)
        try:
            df_mimic_cxr_metadata = dd.read_csv(mimic_cxr_path + 'mimic-cxr-2.0.0-metadata.csv', assume_missing=True, dtype={'dicom_id': 'object', 'Note': 'object'}, blocksize=None)
        except:
            df_mimic_cxr_metadata = pd.read_csv(mimic_cxr_path + 'mimic-cxr-2.0.0-metadata.csv', dtype={'dicom_id': 'object', 'Note': 'object'})
            df_mimic_cxr_metadata = dd.from_pandas(df_mimic_cxr_metadata, npartitions=7)
    ## NOTES
    df_dsnotes['charttime'] = dd.to_datetime(df_dsnotes['charttime'])
    df_dsnotes['storetime'] = dd.to_datetime(df_dsnotes['storetime'])
  
    df_radnotes['charttime'] = dd.to_datetime(df_radnotes['charttime'])
    df_radnotes['storetime'] = dd.to_datetime(df_radnotes['storetime'])
    
    
    ### -> SORT data
    ## CORE
    print('PROCESSING "CORE" DB...')
    df_admissions = df_admissions.compute().sort_values(by=['subject_id','hadm_id'])
    df_patients = df_patients.compute().sort_values(by=['subject_id'])
    df_transfers = df_transfers.compute().sort_values(by=['subject_id','hadm_id'])
    
    ## HOSP
    print('PROCESSING "HOSP" DB...')
    df_diagnoses_icd = df_diagnoses_icd.compute().sort_values(by=['subject_id'])
    df_drgcodes = df_drgcodes.compute().sort_values(by=['subject_id','hadm_id'])
    df_emar = df_emar.compute().sort_values(by=['subject_id','hadm_id'])
    df_emar_detail = df_emar_detail.compute().sort_values(by=['subject_id'])
    df_hcpcsevents = df_hcpcsevents.compute().sort_values(by=['subject_id','hadm_id'])
    df_labevents = df_labevents.compute().sort_values(by=['subject_id','hadm_id'])
    df_microbiologyevents = df_microbiologyevents.compute().sort_values(by=['subject_id','hadm_id'])
    df_poe = df_poe.compute().sort_values(by=['subject_id','hadm_id'])
    df_poe_detail = df_poe_detail.compute().sort_values(by=['subject_id'])
    df_prescriptions = df_prescriptions.compute().sort_values(by=['subject_id','hadm_id'])
    df_procedures_icd = df_procedures_icd.compute().sort_values(by=['subject_id','hadm_id'])
    df_services = df_services.compute().sort_values(by=['subject_id','hadm_id'])
    
    ## ICU
    print('PROCESSING "ICU" DB...')
    df_procedureevents = df_procedureevents.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
    df_outputevents = df_outputevents.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
    df_inputevents = df_inputevents.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
    df_icustays = df_icustays.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
    df_datetimeevents = df_datetimeevents.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
    df_chartevents = df_chartevents.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
    
    ## CXR
    print('PROCESSING "CXR" DB...')
    df_mimic_cxr_split = df_mimic_cxr_split.compute().sort_values(by=['subject_id'])
    df_mimic_cxr_chexpert = df_mimic_cxr_chexpert.compute().sort_values(by=['subject_id'])
    df_mimic_cxr_metadata = df_mimic_cxr_metadata.compute().sort_values(by=['subject_id'])
    df_mimic_cxr_negbio = df_mimic_cxr_negbio.compute().sort_values(by=['subject_id'])
    
    ## NOTES
    print('PROCESSING "NOTES" DB...')
    df_dsnotes = df_dsnotes.compute().sort_values(by=['subject_id','hadm_id'])
    df_radnotes = df_radnotes.compute().sort_values(by=['subject_id','hadm_id'])
    
    # Return
    return df_admissions, df_patients, df_transfers, df_diagnoses_icd, df_drgcodes, df_emar, df_emar_detail, df_hcpcsevents, df_labevents, df_microbiologyevents, df_poe, df_poe_detail, df_prescriptions, df_procedures_icd, df_services, df_procedureevents, df_outputevents, df_inputevents, df_icustays, df_datetimeevents, df_chartevents, df_mimic_cxr_metadata, df_mimic_cxr_split, df_mimic_cxr_chexpert, df_mimic_cxr_negbio, df_dsnotes, df_radnotes


# GET FULL MIMIC IV PATIENT RECORD USING DATABASE KEYS
def get_patient_icustay(key_subject_id, mimic_data):
    # Inputs:
    #   key_subject_id -> subject_id is unique to a patient
    #   
    #   NOTES: Identifiers which specify the patient. More information about 
    #   these identifiers is available at https://mimic-iv.mit.edu/basics/identifiers

    # Outputs:
    #   Patient_ICUstay -> ICU patient stay structure

    #-> FILTER data
    ##-> CORE
    f_df_admissions = mimic_data.df_admissions[(mimic_data.df_admissions.subject_id == key_subject_id)]
    f_df_patients = mimic_data.df_patients[(mimic_data.df_patients.subject_id == key_subject_id)]
    f_df_transfers = mimic_data.df_transfers[(mimic_data.df_transfers.subject_id == key_subject_id)]
    ###-> Merge data into single patient structure
    f_df_core = f_df_admissions
    f_df_core = f_df_core.merge(f_df_patients, how='left')
    f_df_core = f_df_core.merge(f_df_transfers, how='left')

    ##-> HOSP
    f_df_diagnoses_icd = mimic_data.df_diagnoses_icd[(mimic_data.df_diagnoses_icd.subject_id == key_subject_id)]
    f_df_drgcodes = mimic_data.df_drgcodes[(mimic_data.df_drgcodes.subject_id == key_subject_id)]
    f_df_emar = mimic_data.df_emar[(mimic_data.df_emar.subject_id == key_subject_id)]
    f_df_emar_detail = mimic_data.df_emar_detail[(mimic_data.df_emar_detail.subject_id == key_subject_id)]
    f_df_hcpcsevents = mimic_data.df_hcpcsevents[(mimic_data.df_hcpcsevents.subject_id == key_subject_id)]
    f_df_labevents = mimic_data.df_labevents[(mimic_data.df_labevents.subject_id == key_subject_id)]
    f_df_microbiologyevents = mimic_data.df_microbiologyevents[(mimic_data.df_microbiologyevents.subject_id == key_subject_id)]
    f_df_poe = mimic_data.df_poe[(mimic_data.df_poe.subject_id == key_subject_id)]
    f_df_poe_detail = mimic_data.df_poe_detail[(mimic_data.df_poe_detail.subject_id == key_subject_id)]
    f_df_prescriptions = mimic_data.df_prescriptions[(mimic_data.df_prescriptions.subject_id == key_subject_id)]
    f_df_procedures_icd = mimic_data.df_procedures_icd[(mimic_data.df_procedures_icd.subject_id == key_subject_id)]
    f_df_services = mimic_data.df_services[(mimic_data.df_services.subject_id == key_subject_id)]
    
    ##-> ICU
    f_df_procedureevents = mimic_data.df_procedureevents[(mimic_data.df_procedureevents.subject_id == key_subject_id)]
    f_df_outputevents = mimic_data.df_outputevents[(mimic_data.df_outputevents.subject_id == key_subject_id)]
    f_df_inputevents = mimic_data.df_inputevents[(mimic_data.df_inputevents.subject_id == key_subject_id)]
    f_df_icustays = mimic_data.df_icustays[(mimic_data.df_icustays.subject_id == key_subject_id)]
    f_df_datetimeevents = mimic_data.df_datetimeevents[(mimic_data.df_datetimeevents.subject_id == key_subject_id)]
    f_df_chartevents = mimic_data.df_chartevents[(mimic_data.df_chartevents.subject_id == key_subject_id)]
    
    ##-> CXR
    f_df_mimic_cxr_split = mimic_data.df_mimic_cxr_split[(mimic_data.df_mimic_cxr_split.subject_id == key_subject_id)]
    f_df_mimic_cxr_chexpert = mimic_data.df_mimic_cxr_chexpert[(mimic_data.df_mimic_cxr_chexpert.subject_id == key_subject_id)]
    f_df_mimic_cxr_metadata = mimic_data.df_mimic_cxr_metadata[(mimic_data.df_mimic_cxr_metadata.subject_id == key_subject_id)]
    f_df_mimic_cxr_negbio = mimic_data.df_mimic_cxr_negbio[(mimic_data.df_mimic_cxr_negbio.subject_id == key_subject_id)]
    
    ###-> Merge data into single patient structure
    f_df_cxr = f_df_mimic_cxr_split
    f_df_cxr = f_df_cxr.merge(f_df_mimic_cxr_chexpert, how='left')
    f_df_cxr = f_df_cxr.merge(f_df_mimic_cxr_metadata, how='left')
    f_df_cxr = f_df_cxr.merge(f_df_mimic_cxr_negbio, how='left')
    
    ###-> Get images of that timebound patient
    # This list will hold the image data (will contain only one item)

    subset_ids = pd.read_csv('data/severity_dataset/severity_balanced.csv', usecols=['subject_id'])

    subset_ids = subset_ids.drop_duplicates()

    # Load metadata
    cxr_meta = pd.read_csv('data/mimic_cxr/mimic-cxr-2.0.0-metadata.csv.gz', compression='gzip')
    
    # Filter metadata
    cxr_filtered = cxr_meta[cxr_meta['subject_id'].isin(subset_ids['subject_id'])]
    cxr_filtered = cxr_filtered.drop_duplicates(subset=['subject_id'])

    f_df_imcxr = []
    img_cxr_shape = (224, 224)
    
    # Check if there is any imaging record for the patient
    if not cxr_filtered.empty:
        # Since there's only one image per patient, we can get the first and only row
        row = cxr_filtered.iloc[0]
    
        # Construct folder structure
        study_id_str = "s" + str(int(row['study_id']))
        subject_id_str = "p" + str(int(row['subject_id']))
        top_folder = "p" + str(int(row['subject_id']))[:2]
        filename = row['dicom_id'] + ".dcm"
        
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
                f_df_imcxr.append(np.array(img_resized))
                print(f"Image file processed: {img_path}")
                
            except Exception as e:
                print(f"Error processing DICOM file at '{img_path}': {e}")
        else:
            print(f"Image file not found: {img_path}")
    else:
        print("No imaging record found for this patient.")
      
    ##-> NOTES
    f_df_dsnotes = mimic_data.df_dsnotes[(mimic_data.df_dsnotes.subject_id == key_subject_id)]
    f_df_radnotes = mimic_data.df_radnotes[(mimic_data.df_radnotes.subject_id == key_subject_id)]


    # -> Create & Populate patient structure
    ## CORE
    admissions = f_df_admissions
    demographics = f_df_patients
    transfers = f_df_transfers
    core = f_df_core

    ## HOSP
    diagnoses_icd = f_df_diagnoses_icd
    drgcodes = f_df_diagnoses_icd
    emar = f_df_emar
    emar_detail = f_df_emar_detail
    hcpcsevents = f_df_hcpcsevents
    labevents = f_df_labevents
    microbiologyevents = f_df_microbiologyevents
    poe = f_df_poe
    poe_detail = f_df_poe_detail
    prescriptions = f_df_prescriptions
    procedures_icd = f_df_procedures_icd
    services = f_df_services

    ## ICU
    procedureevents = f_df_procedureevents
    outputevents = f_df_outputevents
    inputevents = f_df_inputevents
    icustays = f_df_icustays
    datetimeevents = f_df_datetimeevents
    chartevents = f_df_chartevents

    ## CXR
    cxr = f_df_cxr 
    imcxr = f_df_imcxr

    ## NOTES
    dsnotes = f_df_dsnotes
    radnotes = f_df_radnotes

    # Create patient object and return
    Patient_ICUstay = Patient_ICU(admissions, demographics, transfers, core, \
                                  diagnoses_icd, drgcodes, emar, emar_detail, hcpcsevents, \
                                  labevents, microbiologyevents, poe, poe_detail, \
                                  prescriptions, procedures_icd, services, procedureevents, \
                                  outputevents, inputevents, icustays, datetimeevents, \
                                  chartevents, cxr, imcxr, dsnotes, radnotes)

    return Patient_ICUstay

# GET TIMEBOUND MIMIC-IV PATIENT RECORD BY DATABASE KEYS AND TIMESTAMPS
def get_timebound_patient_icustay(Patient_ICUstay, start_hr = None, end_hr = None):
    # Inputs:
    #   Patient_ICUstay -> Patient ICU stay structure
    #   start_hr -> start_hr indicates the first valid time (in hours) from the admition time "admittime" for all retreived features, input "None" to avoid time bounding
    #   end_hr -> end_hr indicates the last valid time (in hours) from the admition time "admittime" for all retreived features, input "None" to avoid time bounding
    #
    #   NOTES: Identifiers which specify the patient. More information about 
    #   these identifiers is available at https://mimic-iv.mit.edu/basics/identifiers

    # Outputs:
    #   Patient_ICUstay -> Timebound ICU patient stay structure filtered by max_time_stamp or min_time_stamp if any
    
    # %% EXAMPLE OF USE
    ## Let's select a single patient
    '''
    key_subject_id = 10000032
    start_hr = 0
    end_hr = 24
    patient = get_patient_icustay(key_subject_id)
    dt_patient = get_timebound_patient_icustay(patient, start_hr , end_hr)
    '''
    
    # Create a deep copy so that it is not the same object
    # Patient_ICUstay = copy.deepcopy(Patient_ICUstay)
    
    
    ## --> Process Event Structure Calculations
    admittime = Patient_ICUstay.core['admittime'].values[0]
    dischtime = Patient_ICUstay.core['dischtime'].values[0]
    Patient_ICUstay.emar.loc[:, 'deltacharttime'] = Patient_ICUstay.emar.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.labevents.loc[:, 'deltacharttime'] = Patient_ICUstay.labevents.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.microbiologyevents.loc[:, 'deltacharttime'] = Patient_ICUstay.microbiologyevents.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.outputevents.loc[:, 'deltacharttime'] = Patient_ICUstay.outputevents.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.datetimeevents.loc[:, 'deltacharttime'] = Patient_ICUstay.datetimeevents.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.chartevents.loc[:, 'deltacharttime'] = Patient_ICUstay.chartevents.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.dsnotes['deltacharttime'] = Patient_ICUstay.dsnotes.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.radnotes['deltacharttime'] = Patient_ICUstay.radnotes.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    
    # Re-calculate times of CXR database
    Patient_ICUstay.cxr['StudyDateForm'] = pd.to_datetime(Patient_ICUstay.cxr['StudyDate'], format='%Y%m%d')
    Patient_ICUstay.cxr['StudyTimeForm'] = Patient_ICUstay.cxr.apply(lambda x : '%#010.3f' % x['StudyTime'] ,1)
    Patient_ICUstay.cxr['StudyTimeForm'] = pd.to_datetime(Patient_ICUstay.cxr['StudyTimeForm'], format='%H%M%S.%f').dt.time
    Patient_ICUstay.cxr['charttime'] = Patient_ICUstay.cxr.apply(lambda r : dt.datetime.combine(r['StudyDateForm'],r['StudyTimeForm']),1)
    Patient_ICUstay.cxr['charttime'] = Patient_ICUstay.cxr['charttime'].dt.floor('Min')
    Patient_ICUstay.cxr['deltacharttime'] = Patient_ICUstay.cxr.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    
    ## --> Filter by allowable time stamps
    if not (start_hr == None):
        Patient_ICUstay.emar = Patient_ICUstay.emar[(Patient_ICUstay.emar.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.emar.deltacharttime)]
        Patient_ICUstay.labevents = Patient_ICUstay.labevents[(Patient_ICUstay.labevents.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.labevents.deltacharttime)]
        Patient_ICUstay.microbiologyevents = Patient_ICUstay.microbiologyevents[(Patient_ICUstay.microbiologyevents.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.microbiologyevents.deltacharttime)]
        Patient_ICUstay.outputevents = Patient_ICUstay.outputevents[(Patient_ICUstay.outputevents.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.outputevents.deltacharttime)]
        Patient_ICUstay.datetimeevents = Patient_ICUstay.datetimeevents[(Patient_ICUstay.datetimeevents.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.datetimeevents.deltacharttime)]
        Patient_ICUstay.chartevents = Patient_ICUstay.chartevents[(Patient_ICUstay.chartevents.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.chartevents.deltacharttime)]
        Patient_ICUstay.cxr = Patient_ICUstay.cxr[(Patient_ICUstay.cxr.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.cxr.deltacharttime)]
        Patient_ICUstay.imcxr = [Patient_ICUstay.imcxr[i] for i, x in enumerate((Patient_ICUstay.cxr.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.cxr.deltacharttime)) if x]
        #Notes
        Patient_ICUstay.dsnotes = Patient_ICUstay.dsnotes[(Patient_ICUstay.dsnotes.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.dsnotes.deltacharttime)]
        Patient_ICUstay.radnotes = Patient_ICUstay.radnotes[(Patient_ICUstay.radnotes.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.radnotes.deltacharttime)]
        
        
    if not (end_hr == None):
        Patient_ICUstay.emar = Patient_ICUstay.emar[(Patient_ICUstay.emar.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.emar.deltacharttime)]
        Patient_ICUstay.labevents = Patient_ICUstay.labevents[(Patient_ICUstay.labevents.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.labevents.deltacharttime)]
        Patient_ICUstay.microbiologyevents = Patient_ICUstay.microbiologyevents[(Patient_ICUstay.microbiologyevents.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.microbiologyevents.deltacharttime)]
        Patient_ICUstay.outputevents = Patient_ICUstay.outputevents[(Patient_ICUstay.outputevents.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.outputevents.deltacharttime)]
        Patient_ICUstay.datetimeevents = Patient_ICUstay.datetimeevents[(Patient_ICUstay.datetimeevents.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.datetimeevents.deltacharttime)]
        Patient_ICUstay.chartevents = Patient_ICUstay.chartevents[(Patient_ICUstay.chartevents.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.chartevents.deltacharttime)]
        Patient_ICUstay.cxr = Patient_ICUstay.cxr[(Patient_ICUstay.cxr.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.cxr.deltacharttime)]
        # Patient_ICUstay.imcxr = [Patient_ICUstay.imcxr[i] for i, x in enumerate((Patient_ICUstay.cxr.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.cxr.deltacharttime)) if x]
        Patient_ICUstay.imcxr = [
        img for img, include in zip(
            Patient_ICUstay.imcxr,
            (Patient_ICUstay.cxr.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.cxr.deltacharttime)
        ) if include
    ]
        #Notes
        Patient_ICUstay.dsnotes = Patient_ICUstay.dsnotes[(Patient_ICUstay.dsnotes.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.dsnotes.deltacharttime)]
        Patient_ICUstay.radnotes = Patient_ICUstay.radnotes[(Patient_ICUstay.radnotes.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.radnotes.deltacharttime)]
        
        # Filter CXR to match allowable patient stay
        Patient_ICUstay.cxr = Patient_ICUstay.cxr[(Patient_ICUstay.cxr.charttime <= dischtime)]
    
    return Patient_ICUstay

# EXTRACT ALL INFO OF A SINGLE PATIENT FROM MIMIC-IV DATASET USING HAIM ID
def extract_single_patient_records_mimiciv(haim_patient_idx, df_haim_ids, start_hr, end_hr, mimic_data):
    # Inputs:
    #   haim_patient_idx -> Ordered number of HAIM patient
    #   df_haim_ids -> Dataframe with all unique available HAIM_MIMICIV records by key identifiers
    #   start_hr -> start_hr indicates the first valid time (in hours) from the admition time "admittime" for all retreived features, input "None" to avoid time bounding
    #   end_hr -> end_hr indicates the last valid time (in hours) from the admition time "admittime" for all retreived features, input "None" to avoid time bounding
    #
    # Outputs:
    #   key_subject_id -> MIMIC-IV Subject ID of selected patient
    #   key_hadm_id -> MIMIC-IV Hospital Admission ID of selected patient
    #   key_stay_id -> MIMIC-IV ICU Stay ID of selected patient
    #   patient -> Full ICU patient ICU stay structure
    #   dt_patient -> Timebound ICU patient stay structure filtered by max_time_stamp or min_time_stamp if any
    
    # Extract information for patient
    key_subject_id = df_haim_ids.iloc[haim_patient_idx].subject_id
    start_hr = start_hr # Select timestamps
    end_hr = end_hr   # Select timestamps
    patient = get_patient_icustay(key_subject_id, mimic_data)
    dt_patient = get_timebound_patient_icustay(patient, start_hr , end_hr)
    
    return key_subject_id, patient, dt_patient

# GENERATE ALL SINGLE PATIENT ICU STAY RECORDS FOR ENTIRE MIMIC-IV DATABASE
def generate_all_mimiciv_patient_object(df_haim_ids, core_mimiciv_path, mimic_data):
    # Inputs:
    #   df_haim_ids -> Dataframe with all unique available HAIM_MIMICIV records by key identifiers
    #   core_mimiciv_path -> Path to structured MIMIC IV databases in CSV files
    #
    # Outputs:
    #   nfiles -> Number of single patient HAIM files produced
    
    # Extract information for patient
    nfiles = len(df_haim_ids)
    with tqdm(total = nfiles) as pbar:
        #Iterate through all patients
        for haim_patient_idx in range(nfiles):
            # Let's select each single patient and extract patient object
            start_hr = None # Select timestamps
            end_hr = None   # Select timestamps
            key_subject_id, patient, dt_patient = extract_single_patient_records_mimiciv(haim_patient_idx, df_haim_ids, start_hr, end_hr, mimic_data)

            # Create the output directory if it doesn't exist
            os.makedirs(pickle_path, exist_ok=True)
            
            # Save
            filename = f"{haim_patient_idx:08d}" + '.pkl'
            save_patient_object(dt_patient, pickle_path + filename)
            # Update process bar
            pbar.update(1)
    return nfiles

# DELTA TIME CALCULATOR FROM TWO TIMESTAMPS
def date_diff_hrs(t1, t0):
    # Inputs:
    #   t1 -> Final timestamp in a patient hospital stay
    #   t0 -> Initial timestamp in a patient hospital stay

    # Outputs:
    #   delta_t -> Patient stay structure bounded by allowed timestamps

    try:
        delta_t = (t1-t0).total_seconds()/3600 # Result in hrs
    except:
        delta_t = math.nan
    
    return delta_t

# SAVE SINGLE PATIENT ICU STAY RECORDS FOR MIMIC-IV 
def save_patient_object(obj, filepath):
    # Inputs:
    #   obj -> Timebound ICU patient stay object
    #   filepath -> Pickle file path to save object to
    #
    # Outputs:
    #   VOID -> Object is saved in filename path
    # Overwrites any existing file.
    with open(filepath, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

# LOAD CORE INFO OF MIMIC IV PATIENTS
def load_core_mimic_haim_info(core_mimiciv_path, df_haim_ids):
    # Inputs:
    #   core_mimiciv_path -> Base path of mimiciv
    #   df_haim_ids -> Table of HAIM ids and corresponding keys
    #
    # Outputs:
    #   df_haim_ids_core_info -> Updated dataframe with integer representations of core data

    # %% EXAMPLE OF USE
    # df_haim_ids_core_info = load_core_mimic_haim_info(core_mimiciv_path)

    # Load core table
    df_mimiciv_core = pd.read_csv(core_mimiciv_path + 'core/core.csv')

    # Generate integer representations of categorical variables in core
    core_var_select_list = ['gender', 'ethnicity', 'marital_status', 'language','insurance']
    core_var_select_int_list = ['gender_int', 'ethnicity_int', 'marital_status_int', 'language_int','insurance_int']
    df_mimiciv_core[core_var_select_list] = df_mimiciv_core[core_var_select_list].astype('category')
    df_mimiciv_core[core_var_select_int_list] = df_mimiciv_core[core_var_select_list].apply(lambda x: x.cat.codes)

    # Combine HAIM IDs with core data
    df_haim_ids_core_info = pd.merge(df_haim_ids, df_mimiciv_core, on=["subject_id", "hadm_id"])

    return df_haim_ids_core_info

# LOAD SINGLE PATIENT ICU STAY RECORDS FOR MIMIC-IV
def load_patient_object(filepath):
    # Inputs:
    #   filepath -> Pickle file path to save object to
    #
    # Outputs:
    #   obj -> Loaded timebound ICU patient stay object

    # Overwrites any existing file.
    with open(filepath, 'rb') as input:  
        return pickle.load(input)

from sklearn.preprocessing import LabelEncoder

# GET DEMOGRAPHICS EMBEDDINGS OF MIMIC IV PATIENT
def get_demographic_embeddings(dt_patient, verbose=0):
    """
    Inputs:
        dt_patient -> Timebound mimic patient structure
        verbose -> Flag to print found keyword outputs (0,1,2)

    Outputs:
        demo_embeddings -> Core demographic embeddings (numpy array)
    """

    # Columns to encode
    categorical_cols = ['gender', 'ethnicity', 'marital_status', 'language', 'insurance']

    # Encode categorical columns into *_int if not already encoded
    for col in categorical_cols:
        int_col = col + '_int'
        if int_col not in dt_patient.core.columns and col in dt_patient.core.columns:
            le = LabelEncoder()
            dt_patient.core[int_col] = le.fit_transform(dt_patient.core[col].astype(str))

    # Select embeddings
    embed_cols = ['anchor_age', 'gender_int', 'ethnicity_int', 
                  'marital_status_int', 'language_int', 'insurance_int']

    available_cols = [c for c in embed_cols if c in dt_patient.core.columns]
    demo_embeddings = dt_patient.core.loc[0, available_cols]

    if verbose >= 1:
        print("Demographic embeddings (raw):")
        print(demo_embeddings)

    return demo_embeddings.values


def get_ts_embeddings(dt_patient, event_type):
    # Inputs:
    #   dt_patient -> Timebound Patient ICU stay structure
    #
    # Outputs:
    #   ts_emb -> TSfresh-like generated Lab event features for each timeseries
    #
    # %% EXAMPLE OF USE
    # ts_emb = get_labevent_ts_embeddings(dt_patient)
    
    #Get chartevents
    
    if(event_type == 'procedure'):
        df = dt_patient.procedureevents
        #Define chart events of interest
        event_list = ['Foley Catheter', 'PICC Line', 'Intubation', 'Peritoneal Dialysis', 
                            'Bronchoscopy', 'EEG', 'Dialysis - CRRT', 'Dialysis Catheter', 
                            'Chest Tube Removed', 'Hemodialysis']
        df_pivot = pivot_procedureevent(df, event_list)
        
    elif(event_type == 'lab'):
        df = dt_patient.labevents
        #Define chart events of interest
        event_list = ['Glucose', 'Potassium', 'Sodium', 'Chloride', 'Creatinine',
           'Urea Nitrogen', 'Bicarbonate', 'Anion Gap', 'Hemoglobin', 'Hematocrit',
           'Magnesium', 'Platelet Count', 'Phosphate', 'White Blood Cells',
           'Calcium, Total', 'MCH', 'Red Blood Cells', 'MCHC', 'MCV', 'RDW', 
                      'Platelet Count', 'Neutrophils', 'Vancomycin']
        df_pivot = pivot_labevent(df, event_list)
        
    elif(event_type == 'chart'):
        # df = dt_patient.chartevents
        # #Define chart events of interest
        # event_list = ['Heart Rate','Non Invasive Blood Pressure systolic',
        #             'Non Invasive Blood Pressure diastolic', 'Non Invasive Blood Pressure mean', 
        #             'Respiratory Rate','O2 saturation pulseoxymetry', 
        #             'GCS - Verbal Response', 'GCS - Eye Opening', 'GCS - Motor Response'] 
        # df_pivot = pivot_chartevent(df, event_list)

        # Step 1: Load the d_items table to get the itemid-to-label mapping.
        d_items = pd.read_csv(mimic_iv_path + 'hosp/d_items.csv')
        
        # Step 2: The CRITICAL merge step.
        # This joins the patient's chartevents with d_items to add the 'label' column.
        merged_df = pd.merge(
            dt_patient.chartevents, 
            d_items[['itemid', 'label']], 
            on='itemid', 
            how='left'
        )

        # Step 3: Define the chart events of interest using labels.
        event_list = [
            'Heart Rate', 'Non Invasive Blood Pressure systolic',
            'Non Invasive Blood Pressure diastolic', 'Non Invasive Blood Pressure mean',
            'Respiratory Rate', 'O2 saturation pulseoxymetry',
            'GCS - Verbal Response', 'GCS - Eye Opening', 'GCS - Motor Response'
        ]
        
        # Step 4: Call your pivot function on the newly merged DataFrame.
        # This will now work without a KeyError because 'label' exists.
        df_pivot = pivot_chartevent(merged_df, event_list)
        return df_pivot
    
    #Pivote df to record these values
    
    ts_emb = get_ts_emb(df_pivot, event_list)
    try:
        ts_emb = ts_emb.drop(['subject_id', 'hadm_id']).fillna(value=0)
    except:
        ts_emb = pd.Series(0, index=ts_emb.columns).drop(['subject_id', 'hadm_id']).fillna(value=0)

    return ts_emb

# def pivot_chartevent(df, event_list):
#     # create a new table with additional columns with label list  
#     df1 = df[['subject_id', 'hadm_id', 'stay_id', 'charttime']] 
#     for event in event_list: 
#         df1[event] = np.nan
#          #search in the abbreviations column  
#         df1.loc[(df['label']==event), event] = df['valuenum'].astype(float)
#     df_out = df1.dropna(axis=0, how='all', subset=event_list)
#     return df_out 

def pivot_chartevent(df, event_list):
    """
    Pivots the chartevents DataFrame to create a wide-format table.
    """
    df_filtered = df[df['label'].isin(event_list)].copy()

    df_pivot = df_filtered.pivot_table(
        index=['subject_id', 'hadm_id', 'stay_id', 'charttime'],
        columns='label',
        values='valuenum',
        aggfunc='mean'
    ).reset_index()

    desired_cols = ['subject_id', 'hadm_id', 'stay_id', 'charttime'] + event_list
    df_out = df_pivot[df_pivot.columns.intersection(desired_cols)]
    
    df_out = df_out.dropna(axis=0, how='all', subset=event_list)

    return df_out

def pivot_labevent(df, event_list):
    # create a new table with additional columns with label list  
    df1 = df[['subject_id', 'hadm_id',  'charttime']] 
    for event in event_list: 
        df1[event] = np.nan
        #search in the label column 
        df1.loc[(df['label']==event), event] = df['valuenum'].astype(float) 
    df_out = df1.dropna(axis=0, how='all', subset=event_list)
    return df_out 

def pivot_procedureevent(df, event_list):
    # create a new table with additional columns with label list  
    df1 = df[['subject_id', 'hadm_id',  'storetime']] 
    for event in event_list: 
        df1[event] = np.nan
        #search in the label column 
        df1.loc[(df['label']==event), event] = df['value'].astype(float)  #Yu: maybe if not label use abbreviation 
    df_out = df1.dropna(axis=0, how='all', subset=event_list)
    return df_out 

#FUNCTION TO COMPUTE A LIST OF TIME SERIES FEATURES
def get_ts_emb(df_pivot, event_list):
    # Inputs:
    #   df_pivot -> Pivoted table
    #   event_list -> MIMIC IV Type of Event
    #
    # Outputs:
    #   df_out -> Embeddings
    
    # %% EXAMPLE OF USE
    # df_out = get_ts_emb(df_pivot, event_list)
    
    # Initialize table
    try:
        df_out = df_pivot[['subject_id', 'hadm_id']].iloc[0]
    except:
#         print(df_pivot)
        df_out = pd.DataFrame(columns = ['subject_id', 'hadm_id'])
#         df_out = df_pivot[['subject_id', 'hadm_id']]
        
     #Adding a row of zeros to df_pivot in case there is no value
    df_pivot = df_pivot.append(pd.Series(0, index=df_pivot.columns), ignore_index=True)
    
    #Compute the following features
    for event in event_list:
        series = df_pivot[event].dropna() #dropna rows
        if len(series) >0: #if there is any event
            df_out[event+'_max'] = series.max()
            df_out[event+'_min'] = series.min()
            df_out[event+'_mean'] = series.mean(skipna=True)
            df_out[event+'_variance'] = series.var(skipna=True)
            df_out[event+'_meandiff'] = series.diff().mean() #average change
            df_out[event+'_meanabsdiff'] =series.diff().abs().mean()
            df_out[event+'_maxdiff'] = series.diff().abs().max()
            df_out[event+'_sumabsdiff'] =series.diff().abs().sum()
            df_out[event+'_diff'] = series.iloc[-1]-series.iloc[0]
            #Compute the n_peaks
            peaks,_ = find_peaks(series) #, threshold=series.median()
            df_out[event+'_npeaks'] = len(peaks)
            #Compute the trend (linear slope)
            if len(series)>1:
                df_out[event+'_trend']= np.polyfit(np.arange(len(series)), series, 1)[0] #fit deg-1 poly
            else:
                 df_out[event+'_trend'] = 0
    return df_out

def get_chest_xray_embeddings(dt_patient, verbose=0):
    # Inputs:
    #   dt_patient -> Timebound ICU patient stay structure filtered by max_time_stamp or min_time_stamp if any
    #   verbose -> Level of printed output of function
    #
    # Outputs:
    #   aggregated_densefeature_embeddings -> CXR aggregated dense feature embeddings for all images in timebound patient
    #   densefeature_embeddings ->  List of CXR dense feature embeddings for all images
    #   aggregated_prediction_embeddings -> CXR aggregated embeddings of predictions for all images in timebound patient
    #   prediction_embeddings ->  List of CXR embeddings of predictions for all images
    #   imgs_weights ->  Array of weights for embedding aggregation


    # %% EXAMPLE OF USE
    # aggregated_densefeature_embeddings, densefeature_embeddings, aggregated_prediction_embeddings, prediction_embeddings, imgs_weights = get_chest_xray_embeddings(dt_patient, verbose=2)

    # Clean out process bar before starting
    sys.stdout.flush()

    # Select if you want to use CUDA support for GPU (optional as it is usually pretty fast even in CPUT)
    cuda = False

    # Select model with a String that determines the model to use for Chest Xrays according to https://github.com/mlmed/torchxrayvision
    #   model_weights_name = "densenet121-res224-all" # Every output trained for all models
    #   model_weights_name = "densenet121-res224-rsna" # RSNA Pneumonia Challenge
    #model_weights_name = "densenet121-res224-nih" # NIH chest X-ray8
    #model_weights_name = "densenet121-res224-pc") # PadChest (University of Alicante)
    model_weights_name = "densenet121-res224-chex" # CheXpert (Stanford)
    #   model_weights_name = "densenet121-res224-mimic_nb" # MIMIC-CXR (MIT)
    #model_weights_name = "densenet121-res224-mimic_ch") # MIMIC-CXR (MIT)
    #model_weights_name = "resnet50-res512-all" # Resnet only for 512x512 inputs
    # NOTE: The all model has every output trained. However, for the other weights some targets are not trained and will predict randomly becuase they do not exist in the training dataset.


    # Extract chest x-ray images from timebound patient and iterate through them
    imgs = dt_patient.imcxr
    densefeature_embeddings = []
    prediction_embeddings = []

    # Iterate
    nImgs = len(imgs)
    with tqdm(total = nImgs) as pbar:
        for idx, img in enumerate(imgs):
            #img = skimage.io.imread(img_path) # If importing from path use this
            img = xrv.datasets.normalize(img, 255)
          
            # For each image check if they are 2D arrays
            if len(img.shape) > 2:
                img = img[:, :, 0]
            if len(img.shape) < 2:
                print("Error: Dimension lower than 2 for image!")

            # Add color channel for prediction
            #Resize using OpenCV
            img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)   
            img = img[None, :, :]
            
            #Or resize using core resizer (thows error sometime)
            #transform = transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])
            #img = transform(img)
            model = xrv.models.DenseNet(weights = model_weights_name)
            # model = xrv.models.ResNet(weights="resnet50-res512-all") # ResNet is also available
            
            output = {}
            with torch.no_grad():
                img = torch.from_numpy(img).unsqueeze(0)
                if cuda:
                    img = img.cuda()
                    model = model.cuda()
              
                # Extract dense features
                feats = model.features(img)
                feats = F.relu(feats, inplace=True)
                feats = F.adaptive_avg_pool2d(feats, (1, 1))
                densefeatures = feats.cpu().detach().numpy().reshape(-1)
                densefeature_embeddings.append(densefeatures) # append to list of dense features for all images
                
                # Extract predicted probabilities of considered 18 classes:
                # Get by calling "xrv.datasets.default_pathologies" or "dict(zip(xrv.datasets.default_pathologies,preds[0].detach().numpy()))"
                # ['Atelectasis','Consolidation','Infiltration','Pneumothorax','Edema','Emphysema',Fibrosis',
                #  'Effusion','Pneumonia','Pleural_Thickening','Cardiomegaly','Nodule',Mass','Hernia',
                #  'Lung Lesion','Fracture','Lung Opacity','Enlarged Cardiomediastinum']
                preds = model(img).cpu()
                predictions = preds[0].detach().numpy()
                prediction_embeddings.append(predictions) # append to list of predictions for all images
            
                if verbose >=1:
                    # Update process bar
                    pbar.update(1)
        
        
    # Get image weights by hours passed from current time to image
    orig_imgs_weights = np.asarray(dt_patient.cxr.deltacharttime.values)
    adj_imgs_weights = orig_imgs_weights - orig_imgs_weights.min()
    imgs_weights = (adj_imgs_weights) / (adj_imgs_weights).max()
  
    # Aggregate with weighted average of ebedding vector across temporal dimension
    try:
        aggregated_densefeature_embeddings = np.average(densefeature_embeddings, axis=0, weights=imgs_weights)
        if np.isnan(np.sum(aggregated_densefeature_embeddings)):
            aggregated_densefeature_embeddings = np.zeros_like(densefeature_embeddings[0])
    except:
        aggregated_densefeature_embeddings = np.zeros_like(densefeature_embeddings[0])
      
    try:
        aggregated_prediction_embeddings = np.average(prediction_embeddings, axis=0, weights=imgs_weights)
        if np.isnan(np.sum(aggregated_prediction_embeddings)):
            aggregated_prediction_embeddings = np.zeros_like(prediction_embeddings[0])
    except:
        aggregated_prediction_embeddings = np.zeros_like(prediction_embeddings[0])
      
      
    if verbose >=2:
        x = orig_imgs_weights
        y = prediction_embeddings
        plt.xlabel("Time [hrs]")
        plt.ylabel("Disease probability [0-1]")
        plt.title("A test graph")
        for i in range(len(y[0])):
            plt.plot(x,[pt[i] for pt in y],'o', label = xrv.datasets.default_pathologies[i])
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.show()

    # Return embeddings
    return aggregated_densefeature_embeddings, densefeature_embeddings, aggregated_prediction_embeddings, prediction_embeddings, imgs_weights


def get_single_chest_xray_embeddings(img):
    # Inputs:
    #   img -> Image array
    #
    # Outputs:
    #   densefeature_embeddings ->  CXR dense feature embeddings for image
    #   prediction_embeddings ->  CXR embeddings of predictions for image
    
    
    # %% EXAMPLE OF USE
    # densefeature_embeddings, prediction_embeddings = get_single_chest_xray_embeddings(img)
    
    # Clean out process bar before starting
    sys.stdout.flush()
    
    # Select if you want to use CUDA support for GPU (optional as it is usually pretty fast even in CPUT)
    cuda = False
    
    # Select model with a String that determines the model to use for Chest Xrays according to https://github.com/mlmed/torchxrayvision
    #model_weights_name = "densenet121-res224-all" # Every output trained for all models
    #model_weights_name = "densenet121-res224-rsna" # RSNA Pneumonia Challenge
    #model_weights_name = "densenet121-res224-nih" # NIH chest X-ray8
    #model_weights_name = "densenet121-res224-pc") # PadChest (University of Alicante)
    model_weights_name = "densenet121-res224-chex" # CheXpert (Stanford)
    #model_weights_name = "densenet121-res224-mimic_nb" # MIMIC-CXR (MIT)
    #model_weights_name = "densenet121-res224-mimic_ch" # MIMIC-CXR (MIT)
    #model_weights_name = "resnet50-res512-all" # Resnet only for 512x512 inputs
    # NOTE: The all model has every output trained. However, for the other weights some targets are not trained and will predict randomly becuase they do not exist in the training dataset.
    
    # Extract chest x-ray image embeddings and preddictions
    densefeature_embeddings = []
    prediction_embeddings = []
    
    #img = skimage.io.imread(img_path) # If importing from path use this
    img = xrv.datasets.normalize(img, 255)

    # For each image check if they are 2D arrays
    if len(img.shape) > 2:
        img = img[:, :, 0]
    if len(img.shape) < 2:
        print("Error: Dimension lower than 2 for image!")
    
    # Add color channel for prediction
    #Resize using OpenCV
    img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)   
    img = img[None, :, :]

    #Or resize using core resizer (thows error sometime)
    #transform = transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])
    #img = transform(img)
    model = xrv.models.DenseNet(weights = model_weights_name)
    # model = xrv.models.ResNet(weights="resnet50-res512-all") # ResNet is also available

    output = {}
    with torch.no_grad():
        img = torch.from_numpy(img).unsqueeze(0)
        if cuda:
            img = img.cuda()
            model = model.cuda()
          
        # Extract dense features
        feats = model.features(img)
        feats = F.relu(feats, inplace=True)
        feats = F.adaptive_avg_pool2d(feats, (1, 1))
        densefeatures = feats.cpu().detach().numpy().reshape(-1)
        densefeature_embeddings = densefeatures

        # Extract predicted probabilities of considered 18 classes:
        # Get by calling "xrv.datasets.default_pathologies" or "dict(zip(xrv.datasets.default_pathologies,preds[0].detach().numpy()))"
        # ['Atelectasis','Consolidation','Infiltration','Pneumothorax','Edema','Emphysema',Fibrosis',
        #  'Effusion','Pneumonia','Pleural_Thickening','Cardiomegaly','Nodule',Mass','Hernia',
        #  'Lung Lesion','Fracture','Lung Opacity','Enlarged Cardiomediastinum']
        preds = model(img).cpu()
        predictions = preds[0].detach().numpy()
        prediction_embeddings = predictions  

    # Return embeddings
    return densefeature_embeddings, prediction_embeddings

# FOR NOTES EMBEDDING EXTRACTION
def get_notes_biobert_embeddings(dt_patient, note_type):
    # Inputs:
    #   dt_patient -> Timebound ICU patient stay structure filtered by max_time_stamp or min_time_stamp if any
    #   note_type -> Type of note to get
    #
    # Outputs:
    #   aggregated_embeddings -> Biobert event features for selected note
  
    # %% EXAMPLE OF USE
    # aggregated_embeddings = get_notes_biobert_embeddings(dt_patient, note_type = 'ecgnotes')
  
    admittime = dt_patient.core['admittime'].values[0]
    note_table = getattr(dt_patient, note_type).copy()
    note_table['deltacharttime'] = note_table['charttime'].apply(lambda x: (x.replace(tzinfo=None) - admittime).total_seconds()/3600)
    try:
        aggregated_embeddings, __, __ = get_biobert_embedding_from_events_list(note_table['text'], note_table['deltacharttime'])
    except:
        aggregated_embeddings, __, __ = get_biobert_embedding_from_events_list(pd.Series([""]), pd.Series([1]))
  
    return aggregated_embeddings