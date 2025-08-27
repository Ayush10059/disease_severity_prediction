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

# Scipy
from scipy.stats import ks_2samp
from scipy.signal import find_peaks

# Scikit-learn
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer


# MIMICIV PATIENT CLASS STRUCTURE
class Patient_ICU(object):
    def __init__(self, admissions, demographics, transfers, core,\
        diagnoses_icd, drgcodes, emar, emar_detail, hcpcsevents,\
        labevents, microbiologyevents, poe, poe_detail,\
        prescriptions, procedures_icd, services, procedureevents,\
        outputevents, inputevents, icustays, datetimeevents,\
        chartevents, cxr, imcxr, noteevents, dsnotes, ecgnotes, \
        echonotes, radnotes):
        
        ## CORE
        self.admissions = admissions
        self.demographics = demographics
        self.transfers = transfers
        self.core = core
        ## HOSP
        self.diagnoses_icd = diagnoses_icd
        self.drgcodes = drgcodes
        self.emar = emar
        self.emar_detail = emar_detail
        self.hcpcsevents = hcpcsevents
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
        # self.noteevents = noteevents
        # self.dsnotes = dsnotes
        # self.ecgnotes = ecgnotes
        # self.echonotes = echonotes
        # self.radnotes = radnotes

# LOAD ALL MIMIC IV TABLES IN MEMORY (warning: High memory lengthy process)
def load_mimiciv(core_mimiciv_path):
    # Inputs:
    #   core_mimiciv_path -> Path to structured MIMIC IV databases in CSV files
    #   filename -> Pickle filename to save object to
    #
    # Outputs:
    #   df's -> Many dataframes with all loaded MIMIC IV tables 
    
    ### -> Initializations & Data Loading
    ###    Resources to identify tables and variables of interest can be found in the MIMIC-IV official API (https://mimic-iv.mit.edu/docs/)
    
    ## CORE
    df_admissions = dd.read_csv(core_mimiciv_path + 'hosp/admissions.csv', assume_missing=True, dtype={'admission_location': 'object','deathtime': 'object','edouttime': 'object','edregtime': 'object'})
    df_patients = dd.read_csv(core_mimiciv_path + 'hosp/patients.csv', assume_missing=True, dtype={'dod': 'object'})  
    df_transfers = dd.read_csv(core_mimiciv_path + 'hosp/transfers.csv', assume_missing=True, dtype={'careunit': 'object'})
  
    ## HOSP
    df_d_labitems = dd.read_csv(core_mimiciv_path + 'hosp/d_labitems.csv', assume_missing=True, dtype={'loinc_code': 'object'})
    df_d_icd_procedures = dd.read_csv(core_mimiciv_path + 'hosp/d_icd_procedures.csv', assume_missing=True, dtype={'icd_code': 'object', 'icd_version': 'object'})
    df_d_icd_diagnoses = dd.read_csv(core_mimiciv_path + 'hosp/d_icd_diagnoses.csv', assume_missing=True, dtype={'icd_code': 'object', 'icd_version': 'object'})
    df_d_hcpcs = dd.read_csv(core_mimiciv_path + 'hosp/d_hcpcs.csv', assume_missing=True, dtype={'category': 'object', 'long_description': 'object'})
    df_diagnoses_icd = dd.read_csv(core_mimiciv_path + 'hosp/diagnoses_icd.csv', assume_missing=True, dtype={'icd_code': 'object', 'icd_version': 'object'})
    df_drgcodes = dd.read_csv(core_mimiciv_path + 'hosp/drgcodes.csv', assume_missing=True)
    df_emar = dd.read_csv(core_mimiciv_path + 'hosp/emar.csv', assume_missing=True)
    df_emar_detail = dd.read_csv(core_mimiciv_path + 'hosp/emar_detail.csv', assume_missing=True, low_memory=False, dtype={'completion_interval': 'object','dose_due': 'object','dose_given': 'object','infusion_complete': 'object','infusion_rate_adjustment': 'object','infusion_rate_unit': 'object','new_iv_bag_hung': 'object','product_description_other': 'object','reason_for_no_barcode': 'object','restart_interval': 'object','route': 'object','side': 'object','site': 'object','continued_infusion_in_other_location': 'object','infusion_rate': 'object','non_formulary_visual_verification': 'object','prior_infusion_rate': 'object','product_amount_given': 'object', 'infusion_rate_adjustment_amount': 'object'})
    df_hcpcsevents = dd.read_csv(core_mimiciv_path + 'hosp/hcpcsevents.csv', assume_missing=True, dtype={'hcpcs_cd': 'object'})
    df_labevents = dd.read_csv(core_mimiciv_path + 'hosp/labevents.csv', assume_missing=True, dtype={'storetime': 'object', 'value': 'object', 'valueuom': 'object', 'flag': 'object', 'priority': 'object', 'comments': 'object'})
    df_microbiologyevents = dd.read_csv(core_mimiciv_path + 'hosp/microbiologyevents.csv', assume_missing=True, dtype={'comments': 'object', 'quantity': 'object'})
    df_poe = dd.read_csv(core_mimiciv_path + 'hosp/poe.csv', assume_missing=True, dtype={'discontinue_of_poe_id': 'object','discontinued_by_poe_id': 'object','order_status': 'object'})
    df_poe_detail = dd.read_csv(core_mimiciv_path + 'hosp/poe_detail.csv', assume_missing=True)
    df_prescriptions = dd.read_csv(core_mimiciv_path + 'hosp/prescriptions.csv', assume_missing=True, dtype={'form_rx': 'object','gsn': 'object'})
    df_procedures_icd = dd.read_csv(core_mimiciv_path + 'hosp/procedures_icd.csv', assume_missing=True, dtype={'icd_code': 'object', 'icd_version': 'object'})
    df_services = dd.read_csv(core_mimiciv_path + 'hosp/services.csv', assume_missing=True, dtype={'prev_service': 'object'})
  
    ## ICU
    df_d_items = dd.read_csv(core_mimiciv_path + 'icu/d_items.csv', assume_missing=True)
    df_procedureevents = dd.read_csv(core_mimiciv_path + 'icu/procedureevents.csv', assume_missing=True, dtype={'value': 'object', 'secondaryordercategoryname': 'object', 'totalamountuom': 'object'})
    df_outputevents = dd.read_csv(core_mimiciv_path + 'icu/outputevents.csv', assume_missing=True, dtype={'value': 'object'})
    df_inputevents = dd.read_csv(core_mimiciv_path + 'icu/inputevents.csv', assume_missing=True, dtype={'value': 'object', 'secondaryordercategoryname': 'object', 'totalamountuom': 'object'})
    df_icustays = dd.read_csv(core_mimiciv_path + 'icu/icustays.csv', assume_missing=True)
    df_datetimeevents = dd.read_csv(core_mimiciv_path + 'icu/datetimeevents.csv', assume_missing=True, dtype={'value': 'object'})
    df_chartevents = dd.read_csv(core_mimiciv_path + 'icu/chartevents.csv', assume_missing=True, low_memory=False, dtype={'value': 'object', 'valueuom': 'object'})
  
    ## CXR
    df_mimic_cxr_split = dd.read_csv(core_mimiciv_path + 'mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv', assume_missing=True)
    df_mimic_cxr_chexpert = dd.read_csv(core_mimiciv_path + 'mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-chexpert.csv', assume_missing=True)
    try:
        df_mimic_cxr_metadata = dd.read_csv(core_mimiciv_path + 'mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv', assume_missing=True, dtype={'dicom_id': 'object'}, blocksize=None)
    except:
        df_mimic_cxr_metadata = pd.read_csv(core_mimiciv_path + 'mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv', dtype={'dicom_id': 'object'})
        df_mimic_cxr_metadata = dd.from_pandas(df_mimic_cxr_metadata, npartitions=7)
    df_mimic_cxr_negbio = dd.read_csv(core_mimiciv_path + 'mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-negbio.csv', assume_missing=True)
  
    ## NOTES
    # df_noteevents = dd.from_pandas(pd.read_csv(core_mimiciv_path + 'note/noteevents.csv', dtype={'charttime': 'object', 'storetime': 'object', 'text': 'object'}), chunksize=8)
    # df_dsnotes = dd.from_pandas(pd.read_csv(core_mimiciv_path + 'note/ds_icustay.csv', dtype={'charttime': 'object', 'storetime': 'object', 'text': 'object'}), chunksize=8)
    # df_ecgnotes = dd.from_pandas(pd.read_csv(core_mimiciv_path + 'note/ecg_icustay.csv', dtype={'charttime': 'object', 'storetime': 'object', 'text': 'object'}), chunksize=8)
    # df_echonotes = dd.from_pandas(pd.read_csv(core_mimiciv_path + 'note/echo_icustay.csv', dtype={'charttime': 'object', 'storetime': 'object', 'text': 'object'}), chunksize=8)
    # df_radnotes = dd.from_pandas(pd.read_csv(core_mimiciv_path + 'note/rad_icustay.csv', dtype={'charttime': 'object', 'storetime': 'object', 'text': 'object'}), chunksize=8)
    
    
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
    df_d_icd_diagnoses.icd_code = df_d_icd_diagnoses.icd_code.str.strip()
    df_d_icd_diagnoses.icd_version = df_d_icd_diagnoses.icd_version.str.strip()
    
    df_procedures_icd.icd_code = df_procedures_icd.icd_code.str.strip()
    df_procedures_icd.icd_version = df_procedures_icd.icd_version.str.strip()
    df_d_icd_procedures.icd_code = df_d_icd_procedures.icd_code.str.strip()
    df_d_icd_procedures.icd_version = df_d_icd_procedures.icd_version.str.strip()
    
    df_hcpcsevents.hcpcs_cd = df_hcpcsevents.hcpcs_cd.str.strip()
    df_d_hcpcs.code = df_d_hcpcs.code.str.strip()
    
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
    # df_procedureevents['comments_date'] = dd.to_datetime(df_procedureevents['comments_date'])
    
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
        # Add paths and info to images in cxr
        # df_mimic_cxr_jpg =pd.read_csv(core_mimiciv_path + 'mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-jpeg-txt.csv')
        # df_cxr = pd.merge(df_mimic_cxr_jpg, df_cxr, on='dicom_id')
        # # Save
        # df_cxr.to_csv(core_mimiciv_path + 'mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv', index=False)
        #Read back the dataframe
        try:
            df_mimic_cxr_metadata = dd.read_csv(core_mimiciv_path + 'mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv', assume_missing=True, dtype={'dicom_id': 'object', 'Note': 'object'}, blocksize=None)
        except:
            df_mimic_cxr_metadata = pd.read_csv(core_mimiciv_path + 'mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv', dtype={'dicom_id': 'object', 'Note': 'object'})
            df_mimic_cxr_metadata = dd.from_pandas(df_mimic_cxr_metadata, npartitions=7)
    # df_mimic_cxr_metadata['cxrtime'] = dd.to_datetime(df_mimic_cxr_metadata['cxrtime'])
    
    ## NOTES
    # df_noteevents['chartdate'] = dd.to_datetime(df_noteevents['chartdate'])
    # df_noteevents['charttime'] = dd.to_datetime(df_noteevents['charttime'])
    # df_noteevents['storetime'] = dd.to_datetime(df_noteevents['storetime'])
  
    # df_dsnotes['charttime'] = dd.to_datetime(df_dsnotes['charttime'])
    # df_dsnotes['storetime'] = dd.to_datetime(df_dsnotes['storetime'])
  
    # df_ecgnotes['charttime'] = dd.to_datetime(df_ecgnotes['charttime'])
    # df_ecgnotes['storetime'] = dd.to_datetime(df_ecgnotes['storetime'])
  
    # df_echonotes['charttime'] = dd.to_datetime(df_echonotes['charttime'])
    # df_echonotes['storetime'] = dd.to_datetime(df_echonotes['storetime'])
  
    # df_radnotes['charttime'] = dd.to_datetime(df_radnotes['charttime'])
    # df_radnotes['storetime'] = dd.to_datetime(df_radnotes['storetime'])
    
    
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
    #--> Unwrap dictionaries
    df_d_icd_diagnoses = df_d_icd_diagnoses.compute()
    df_d_icd_procedures = df_d_icd_procedures.compute()
    df_d_hcpcs = df_d_hcpcs.compute()
    df_d_labitems = df_d_labitems.compute()
    
    ## ICU
    print('PROCESSING "ICU" DB...')
    df_procedureevents = df_procedureevents.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
    df_outputevents = df_outputevents.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
    df_inputevents = df_inputevents.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
    df_icustays = df_icustays.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
    df_datetimeevents = df_datetimeevents.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
    df_chartevents = df_chartevents.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
    #--> Unwrap dictionaries
    df_d_items = df_d_items.compute()
    
    ## CXR
    print('PROCESSING "CXR" DB...')
    df_mimic_cxr_split = df_mimic_cxr_split.compute().sort_values(by=['subject_id'])
    df_mimic_cxr_chexpert = df_mimic_cxr_chexpert.compute().sort_values(by=['subject_id'])
    df_mimic_cxr_metadata = df_mimic_cxr_metadata.compute().sort_values(by=['subject_id'])
    df_mimic_cxr_negbio = df_mimic_cxr_negbio.compute().sort_values(by=['subject_id'])
    
    ## NOTES
    # print('PROCESSING "NOTES" DB...')
    # df_noteevents = df_noteevents.compute().sort_values(by=['subject_id','hadm_id'])
    # df_dsnotes = df_dsnotes.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
    # df_ecgnotes = df_ecgnotes.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
    # df_echonotes = df_echonotes.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
    # df_radnotes = df_radnotes.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
    
    # Return
    return df_admissions, df_patients, df_transfers, df_diagnoses_icd, df_drgcodes, df_emar, df_emar_detail, df_hcpcsevents, df_labevents, df_microbiologyevents, df_poe, df_poe_detail, df_prescriptions, df_procedures_icd, df_services, df_d_icd_diagnoses, df_d_icd_procedures, df_d_hcpcs, df_d_labitems, df_procedureevents, df_outputevents, df_inputevents, df_icustays, df_datetimeevents, df_chartevents, df_d_items, df_mimic_cxr_metadata, df_mimic_cxr_split, df_mimic_cxr_chexpert, df_mimic_cxr_negbio
    # df_noteevents, df_dsnotes, df_ecgnotes, df_echonotes, df_radnotes,



# GET FULL MIMIC IV PATIENT RECORD USING DATABASE KEYS
def get_patient_icustay(key_subject_id, key_hadm_id, key_stay_id):
    # Inputs:
    #   key_subject_id -> subject_id is unique to a patient
    #   key_hadm_id    -> hadm_id is unique to a patient hospital stay
    #   key_stay_id    -> stay_id is unique to a patient ward stay
    #   
    #   NOTES: Identifiers which specify the patient. More information about 
    #   these identifiers is available at https://mimic-iv.mit.edu/basics/identifiers

    # Outputs:
    #   Patient_ICUstay -> ICU patient stay structure

    #-> FILTER data
    ##-> CORE
    # f_df_base_core = df_base_core[(df_base_core.subject_id == key_subject_id) & (df_base_core.hadm_id == key_hadm_id)]
    f_df_admissions = df_admissions[(df_admissions.subject_id == key_subject_id) & (df_admissions.hadm_id == key_hadm_id)]
    f_df_patients = df_patients[(df_patients.subject_id == key_subject_id)]
    f_df_transfers = df_transfers[(df_transfers.subject_id == key_subject_id) & (df_transfers.hadm_id == key_hadm_id)]
    ###-> Merge data into single patient structure
    # f_df_core = f_df_base_core
    f_df_core = f_df_admissions
    f_df_core = f_df_core.merge(f_df_patients, how='left')
    f_df_core = f_df_core.merge(f_df_transfers, how='left')

    ##-> HOSP
    f_df_diagnoses_icd = df_diagnoses_icd[(df_diagnoses_icd.subject_id == key_subject_id)]
    f_df_drgcodes = df_drgcodes[(df_drgcodes.subject_id == key_subject_id) & (df_drgcodes.hadm_id == key_hadm_id)]
    f_df_emar = df_emar[(df_emar.subject_id == key_subject_id) & (df_emar.hadm_id == key_hadm_id)]
    f_df_emar_detail = df_emar_detail[(df_emar_detail.subject_id == key_subject_id)]
    f_df_hcpcsevents = df_hcpcsevents[(df_hcpcsevents.subject_id == key_subject_id) & (df_hcpcsevents.hadm_id == key_hadm_id)]
    f_df_labevents = df_labevents[(df_labevents.subject_id == key_subject_id) & (df_labevents.hadm_id == key_hadm_id)]
    f_df_microbiologyevents = df_microbiologyevents[(df_microbiologyevents.subject_id == key_subject_id) & (df_microbiologyevents.hadm_id == key_hadm_id)]
    f_df_poe = df_poe[(df_poe.subject_id == key_subject_id) & (df_poe.hadm_id == key_hadm_id)]
    f_df_poe_detail = df_poe_detail[(df_poe_detail.subject_id == key_subject_id)]
    f_df_prescriptions = df_prescriptions[(df_prescriptions.subject_id == key_subject_id) & (df_prescriptions.hadm_id == key_hadm_id)]
    f_df_procedures_icd = df_procedures_icd[(df_procedures_icd.subject_id == key_subject_id) & (df_procedures_icd.hadm_id == key_hadm_id)]
    f_df_services = df_services[(df_services.subject_id == key_subject_id) & (df_services.hadm_id == key_hadm_id)]
    ###-> Merge content from dictionaries
    f_df_diagnoses_icd = f_df_diagnoses_icd.merge(df_d_icd_diagnoses, how='left') 
    f_df_procedures_icd = f_df_procedures_icd.merge(df_d_icd_procedures, how='left')
    f_df_hcpcsevents = f_df_hcpcsevents.merge(df_d_hcpcs, how='left')
    f_df_labevents = f_df_labevents.merge(df_d_labitems, how='left')

    ##-> ICU
    f_df_procedureevents = df_procedureevents[(df_procedureevents.subject_id == key_subject_id) & (df_procedureevents.hadm_id == key_hadm_id) & (df_procedureevents.stay_id == key_stay_id)]
    f_df_outputevents = df_outputevents[(df_outputevents.subject_id == key_subject_id) & (df_outputevents.hadm_id == key_hadm_id) & (df_outputevents.stay_id == key_stay_id)]
    f_df_inputevents = df_inputevents[(df_inputevents.subject_id == key_subject_id) & (df_inputevents.hadm_id == key_hadm_id) & (df_inputevents.stay_id == key_stay_id)]
    f_df_icustays = df_icustays[(df_icustays.subject_id == key_subject_id) & (df_icustays.hadm_id == key_hadm_id) & (df_icustays.stay_id == key_stay_id)]
    f_df_datetimeevents = df_datetimeevents[(df_datetimeevents.subject_id == key_subject_id) & (df_datetimeevents.hadm_id == key_hadm_id) & (df_datetimeevents.stay_id == key_stay_id)]
    f_df_chartevents = df_chartevents[(df_chartevents.subject_id == key_subject_id) & (df_chartevents.hadm_id == key_hadm_id) & (df_chartevents.stay_id == key_stay_id)]
    ###-> Merge content from dictionaries
    f_df_procedureevents = f_df_procedureevents.merge(df_d_items, how='left')
    f_df_outputevents = f_df_outputevents.merge(df_d_items, how='left')
    f_df_inputevents = f_df_inputevents.merge(df_d_items, how='left')
    f_df_datetimeevents = f_df_datetimeevents.merge(df_d_items, how='left')
    f_df_chartevents = f_df_chartevents.merge(df_d_items, how='left')       

    ##-> CXR
    f_df_mimic_cxr_split = df_mimic_cxr_split[(df_mimic_cxr_split.subject_id == key_subject_id)]
    f_df_mimic_cxr_chexpert = df_mimic_cxr_chexpert[(df_mimic_cxr_chexpert.subject_id == key_subject_id)]
    f_df_mimic_cxr_metadata = df_mimic_cxr_metadata[(df_mimic_cxr_metadata.subject_id == key_subject_id)]
    f_df_mimic_cxr_negbio = df_mimic_cxr_negbio[(df_mimic_cxr_negbio.subject_id == key_subject_id)]
    ###-> Merge data into single patient structure
    f_df_cxr = f_df_mimic_cxr_split
    f_df_cxr = f_df_cxr.merge(f_df_mimic_cxr_chexpert, how='left')
    f_df_cxr = f_df_cxr.merge(f_df_mimic_cxr_metadata, how='left')
    f_df_cxr = f_df_cxr.merge(f_df_mimic_cxr_negbio, how='left')
    ###-> Get images of that timebound patient
    f_df_imcxr = []
    for img_idx, img_row in f_df_cxr.iterrows():
        img_path = core_mimiciv_imgcxr_path + str(img_row['Img_Folder']) + '/' + str(img_row['Img_Filename'])
        img_cxr_shape = [224, 224]
        img_cxr = cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), (img_cxr_shape[0], img_cxr_shape[1]))
        f_df_imcxr.append(np.array(img_cxr))
      
    ##-> NOTES
    # f_df_noteevents = df_noteevents[(df_noteevents.subject_id == key_subject_id) & (df_noteevents.hadm_id == key_hadm_id)]
    # f_df_dsnotes = df_dsnotes[(df_dsnotes.subject_id == key_subject_id) & (df_dsnotes.hadm_id == key_hadm_id) & (df_dsnotes.stay_id == key_stay_id)]
    # f_df_ecgnotes = df_ecgnotes[(df_ecgnotes.subject_id == key_subject_id) & (df_ecgnotes.hadm_id == key_hadm_id) & (df_ecgnotes.stay_id == key_stay_id)]
    # f_df_echonotes = df_echonotes[(df_echonotes.subject_id == key_subject_id) & (df_echonotes.hadm_id == key_hadm_id) & (df_echonotes.stay_id == key_stay_id)]
    # f_df_radnotes = df_radnotes[(df_radnotes.subject_id == key_subject_id) & (df_radnotes.hadm_id == key_hadm_id) & (df_radnotes.stay_id == key_stay_id)]

    ###-> Merge data into single patient structure
    #--None


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
    # noteevents = f_df_noteevents
    # dsnotes = f_df_dsnotes
    # ecgnotes = f_df_ecgnotes
    # echonotes = f_df_echonotes
    # radnotes = f_df_radnotes


    # Create patient object and return
    Patient_ICUstay = Patient_ICU(admissions, demographics, transfers, core, \
                                  diagnoses_icd, drgcodes, emar, emar_detail, hcpcsevents, \
                                  labevents, microbiologyevents, poe, poe_detail, \
                                  prescriptions, procedures_icd, services, procedureevents, \
                                  outputevents, inputevents, icustays, datetimeevents, \
                                  chartevents, cxr, imcxr, noteevents, dsnotes, ecgnotes, \
                                  echonotes, radnotes)

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
    key_hadm_id = 29079034
    key_stay_id = 39553978
    start_hr = 0
    end_hr = 24
    patient = get_patient_icustay(key_subject_id, key_hadm_id, key_stay_id)
    dt_patient = get_timebound_patient_icustay(patient, start_hr , end_hr)
    '''
    
    # Create a deep copy so that it is not the same object
    # Patient_ICUstay = copy.deepcopy(Patient_ICUstay)
    
    
    ## --> Process Event Structure Calculations
    admittime = Patient_ICUstay.core['admittime'].values[0]
    dischtime = Patient_ICUstay.core['dischtime'].values[0]
    Patient_ICUstay.emar['deltacharttime'] = Patient_ICUstay.emar.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.labevents['deltacharttime'] = Patient_ICUstay.labevents.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.microbiologyevents['deltacharttime'] = Patient_ICUstay.microbiologyevents.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.outputevents['deltacharttime'] = Patient_ICUstay.outputevents.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.datetimeevents['deltacharttime'] = Patient_ICUstay.datetimeevents.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.chartevents['deltacharttime'] = Patient_ICUstay.chartevents.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.noteevents['deltacharttime'] = Patient_ICUstay.noteevents.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.dsnotes['deltacharttime'] = Patient_ICUstay.dsnotes.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.ecgnotes['deltacharttime'] = Patient_ICUstay.ecgnotes.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.echonotes['deltacharttime'] = Patient_ICUstay.echonotes.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
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
        Patient_ICUstay.noteevents = Patient_ICUstay.noteevents[(Patient_ICUstay.noteevents.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.noteevents.deltacharttime)]
        Patient_ICUstay.dsnotes = Patient_ICUstay.dsnotes[(Patient_ICUstay.dsnotes.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.dsnotes.deltacharttime)]
        Patient_ICUstay.ecgnotes = Patient_ICUstay.ecgnotes[(Patient_ICUstay.ecgnotes.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.ecgnotes.deltacharttime)]
        Patient_ICUstay.echonotes = Patient_ICUstay.echonotes[(Patient_ICUstay.echonotes.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.echonotes.deltacharttime)]
        Patient_ICUstay.radnotes = Patient_ICUstay.radnotes[(Patient_ICUstay.radnotes.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.radnotes.deltacharttime)]
        
        
    if not (end_hr == None):
        Patient_ICUstay.emar = Patient_ICUstay.emar[(Patient_ICUstay.emar.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.emar.deltacharttime)]
        Patient_ICUstay.labevents = Patient_ICUstay.labevents[(Patient_ICUstay.labevents.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.labevents.deltacharttime)]
        Patient_ICUstay.microbiologyevents = Patient_ICUstay.microbiologyevents[(Patient_ICUstay.microbiologyevents.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.microbiologyevents.deltacharttime)]
        Patient_ICUstay.outputevents = Patient_ICUstay.outputevents[(Patient_ICUstay.outputevents.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.outputevents.deltacharttime)]
        Patient_ICUstay.datetimeevents = Patient_ICUstay.datetimeevents[(Patient_ICUstay.datetimeevents.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.datetimeevents.deltacharttime)]
        Patient_ICUstay.chartevents = Patient_ICUstay.chartevents[(Patient_ICUstay.chartevents.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.chartevents.deltacharttime)]
        Patient_ICUstay.cxr = Patient_ICUstay.cxr[(Patient_ICUstay.cxr.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.cxr.deltacharttime)]
        Patient_ICUstay.imcxr = [Patient_ICUstay.imcxr[i] for i, x in enumerate((Patient_ICUstay.cxr.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.cxr.deltacharttime)) if x]
        #Notes
        Patient_ICUstay.noteevents = Patient_ICUstay.noteevents[(Patient_ICUstay.noteevents.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.noteevents.deltacharttime)]
        Patient_ICUstay.dsnotes = Patient_ICUstay.dsnotes[(Patient_ICUstay.dsnotes.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.dsnotes.deltacharttime)]
        Patient_ICUstay.ecgnotes = Patient_ICUstay.ecgnotes[(Patient_ICUstay.ecgnotes.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.ecgnotes.deltacharttime)]
        Patient_ICUstay.echonotes = Patient_ICUstay.echonotes[(Patient_ICUstay.echonotes.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.echonotes.deltacharttime)]
        Patient_ICUstay.radnotes = Patient_ICUstay.radnotes[(Patient_ICUstay.radnotes.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.radnotes.deltacharttime)]
        
        # Filter CXR to match allowable patient stay
        Patient_ICUstay.cxr = Patient_ICUstay.cxr[(Patient_ICUstay.cxr.charttime <= dischtime)]
    
    return Patient_ICUstay

# EXTRACT ALL INFO OF A SINGLE PATIENT FROM MIMIC-IV DATASET USING HAIM ID
def extract_single_patient_records_mimiciv(haim_patient_idx, df_haim_ids, start_hr, end_hr):
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
    key_hadm_id = df_haim_ids.iloc[haim_patient_idx].hadm_id
    key_stay_id = df_haim_ids.iloc[haim_patient_idx].stay_id
    start_hr = start_hr # Select timestamps
    end_hr = end_hr   # Select timestamps
    patient = get_patient_icustay(key_subject_id, key_hadm_id, key_stay_id)
    dt_patient = get_timebound_patient_icustay(patient, start_hr , end_hr)
    
    return key_subject_id, key_hadm_id, key_stay_id, patient, dt_patient

# GENERATE ALL SINGLE PATIENT ICU STAY RECORDS FOR ENTIRE MIMIC-IV DATABASE
def generate_all_mimiciv_patient_object(df_haim_ids, core_mimiciv_path):
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
            key_subject_id, key_hadm_id, key_stay_id, patient, dt_patient = extract_single_patient_records_mimiciv(haim_patient_idx, df_haim_ids, start_hr, end_hr)
            
            # Save
            filename = f"{haim_patient_idx:08d}" + '.pkl'
            save_patient_object(dt_patient, core_mimiciv_path + 'pickle/' + filename)
            # Update process bar
            pbar.update(1)
    return nfiles