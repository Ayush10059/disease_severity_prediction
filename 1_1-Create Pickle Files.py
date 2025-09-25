from load_mimic import *

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

# Define MIMIC IV Data Location
core_mimiciv_path = 'data/mimic_iv/'

subset_ids = pd.read_csv('data/severity_dataset/severity_balanced.csv', usecols=['subject_id'])

subset_ids = subset_ids.drop_duplicates()

print(subset_ids.shape)

df_admissions, df_patients, df_transfers, df_diagnoses_icd, df_drgcodes, df_emar, df_emar_detail, df_hcpcsevents, df_labevents, df_microbiologyevents, df_poe, df_poe_detail, df_prescriptions, df_procedures_icd, df_services, df_procedureevents, df_outputevents, df_inputevents, df_icustays, df_datetimeevents, df_chartevents, df_mimic_cxr_metadata, df_mimic_cxr_split, df_mimic_cxr_chexpert, df_mimic_cxr_negbio, df_dsnotes, df_radnotes = load_mimiciv()

mimic_data = MIMICIVData(
    df_admissions, df_patients, df_transfers, df_diagnoses_icd, df_drgcodes, df_emar, df_emar_detail, df_hcpcsevents,
    df_labevents, df_microbiologyevents, df_poe, df_poe_detail, df_prescriptions, df_procedures_icd, df_services,
    df_procedureevents, df_outputevents, df_inputevents, df_icustays, df_datetimeevents, df_chartevents,
    df_mimic_cxr_metadata, df_mimic_cxr_split, df_mimic_cxr_chexpert, df_mimic_cxr_negbio, df_dsnotes, df_radnotes
)
				
## CORE
print('- CORE > df_admissions')
print('--------------------------------')
print(df_admissions.dtypes)
print('\n\n')

print('- CORE > df_patients')
print('--------------------------------')
print(df_patients.dtypes)
print('\n\n')

print('- CORE > df_transfers')
print('--------------------------------')
print(df_transfers.dtypes)
print('\n\n')


## HOSP
print('- HOSP > df_diagnoses_icd')
print('--------------------------------')
print(df_diagnoses_icd.dtypes)
print('\n\n')

print('- HOSP > df_drgcodes')
print('--------------------------------')
print(df_drgcodes.dtypes)
print('\n\n')

print('- HOSP > df_emar')
print('--------------------------------')
print(df_emar.dtypes)
print('\n\n')

print('- HOSP > df_emar_detail')
print('--------------------------------')
print(df_emar_detail.dtypes)
print('\n\n')

print('- HOSP > df_hcpcsevents')
print('--------------------------------')
print(df_hcpcsevents.dtypes)
print('\n\n')

print('- HOSP > df_labevents')
print('--------------------------------')
print(df_labevents.dtypes)
print('\n\n')

print('- HOSP > df_microbiologyevents')
print('--------------------------------')
print(df_microbiologyevents.dtypes)
print('\n\n')

print('- HOSP > df_poe')
print('--------------------------------')
print(df_poe.dtypes)
print('\n\n')

print('- HOSP > df_poe_detail')
print('--------------------------------')
print(df_poe_detail.dtypes)
print('\n\n')

print('- HOSP > df_prescriptions')
print('--------------------------------')
print(df_prescriptions.dtypes)
print('\n\n')

print('- HOSP > df_procedures_icd')
print('--------------------------------')
print(df_procedures_icd.dtypes)
print('\n\n')

print('- HOSP > df_services')
print('--------------------------------')
print(df_services.dtypes)
print('\n\n')


## ICU
print('- ICU > df_procedureevents')
print('--------------------------------')
print(df_procedureevents.dtypes)
print('\n\n')

print('- ICU > df_outputevents')
print('--------------------------------')
print(df_outputevents.dtypes)
print('\n\n')

print('- ICU > df_inputevents')
print('--------------------------------')
print(df_inputevents.dtypes)
print('\n\n')

print('- ICU > df_icustays')
print('--------------------------------')
print(df_icustays.dtypes)
print('\n\n')

print('- ICU > df_datetimeevents')
print('--------------------------------')
print(df_datetimeevents.dtypes)
print('\n\n')


print('- ICU > df_chartevents')
print('--------------------------------')
print(df_chartevents.dtypes)
print('\n\n')


# CXR
print('- CXR > df_mimic_cxr_split')
print('--------------------------------')
print(df_mimic_cxr_split.dtypes)
print('\n\n')

print('- CXR > df_mimic_cxr_chexpert')
print('--------------------------------')
print(df_mimic_cxr_chexpert.dtypes)
print('\n\n')

print('- CXR > df_mimic_cxr_metadata')
print('--------------------------------')
print(df_mimic_cxr_metadata.dtypes)
print('\n\n')

print('- CXR > df_mimic_cxr_negbio')
print('--------------------------------')
print(df_mimic_cxr_negbio.dtypes)
print('\n\n')


## NOTES
print('- NOTES > df_icunotes')
print('--------------------------------')
print(df_dsnotes.dtypes)
print('\n\n')

print('- NOTES > df_radnotes')
print('--------------------------------')
print(df_radnotes.dtypes)
print('\n\n')


# # ## -> GET LIST OF ALL UNIQUE ID COMBINATIONS IN MIMIC-IV (subject_id, hadm_id, stay_id)

# # Get Unique Subject/HospAdmission/Stay Combinations
# # This process collects all unique subject_id, hadm_id, and stay_id combinations
# # from the various clinical event tables into a single DataFrame.
# df_haim_ids = pd.concat(
#     [
#         df_procedureevents[['subject_id', 'hadm_id', 'stay_id']],
#         df_outputevents[['subject_id', 'hadm_id', 'stay_id']],
#         df_inputevents[['subject_id', 'hadm_id', 'stay_id']],
#         df_icustays[['subject_id', 'hadm_id', 'stay_id']],
#         df_datetimeevents[['subject_id', 'hadm_id', 'stay_id']],
#         df_chartevents[['subject_id', 'hadm_id', 'stay_id']]
#     ],
#     ignore_index=True,
# ).drop_duplicates().reset_index(drop=True)

# # At this point, df_haim_ids is already the final, correct cohort,
# # as the initial data has no records without a matching X-ray.

# # Save the master list of key IDs to a CSV file
# df_haim_ids.to_csv(core_mimiciv_path + 'haim_mimiciv_key_ids.csv', index=False)

# # Print the final counts
# print('Unique Subjects: ' + str(len(df_patients['subject_id'].unique())))
# print('Unique Subjects/HospAdmissions/Stays Combinations: ' + str(len(df_haim_ids)))

# # Load the saved file to confirm the count
# df_haim_ids = pd.read_csv(core_mimiciv_path + 'haim_mimiciv_key_ids.csv')
# print('Unique HAIM Records Available: ' + str(len(df_haim_ids)))

# ## -> GET LIST OF ALL UNIQUE SUBJECTS WITH BOTH CLINICAL AND IMAGE DATA

# First, create a master list of all unique subject IDs present in the clinical data tables
df_clinical_subjects = pd.concat(
    [
        df_procedureevents[['subject_id']],
        df_outputevents[['subject_id']],
        df_inputevents[['subject_id']],
        df_icustays[['subject_id']],
        df_datetimeevents[['subject_id']],
        df_chartevents[['subject_id']]
    ],
    ignore_index=True,
).drop_duplicates().reset_index(drop=True)

df_image_ids = subset_ids.drop_duplicates(subset=['subject_id'])

# Filter the list of clinical subjects to keep only those also present in the image ID list
df_haim_subjects = df_clinical_subjects[
    df_clinical_subjects['subject_id'].isin(df_image_ids['subject_id'])
].copy().reset_index(drop=True)

# Save the final, filtered list of unique subject IDs to a CSV file
# This will be your master key for processing patients
df_haim_subjects.to_csv(core_mimiciv_path + 'haim_mimiciv_key_subjects.csv', index=False)

# Print the final counts
print('Unique Subjects from Clinical Data: ' + str(len(df_clinical_subjects)))
print('Unique Subjects with Images: ' + str(len(df_image_ids)))
print('Final HAIM Subjects Available: ' + str(len(df_haim_subjects)))

# Load the saved file to confirm the count
df_haim_ids = pd.read_csv(core_mimiciv_path + 'haim_mimiciv_key_subjects.csv')

# GENERATE ALL SINGLE PATIENT ICU STAY RECORDS FOR ENTIRE MIMIC-IV DATABASE
nfiles = generate_all_mimiciv_patient_object(df_haim_ids, core_mimiciv_path, mimic_data)
