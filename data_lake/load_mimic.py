import os
import duckdb
import pandas as pd

# ==========================================
# 1. PATHS CONFIGURATION
# ==========================================
BRONZE_PE_DIR = "./bronze/pe_severity"

PATHS = {
    "consensus": f"{BRONZE_PE_DIR}/consensus_image_edema_severity.csv",
    "expert": f"{BRONZE_PE_DIR}/expert_report_edema_severity.csv",
    "regex": f"{BRONZE_PE_DIR}/regex_report_edema_severity.csv",
    "cxr_meta": "./bronze/mimic_cxr/mimic-cxr-2.0.0-metadata.csv",
    "patients": "./bronze/mimic_iv/patients.csv",
    "admissions": "./bronze/mimic_iv/admissions.csv",
    "discharge": "./bronze/mimic_note/discharge.csv",
}

SILVER_DIR = "silver"
os.makedirs(SILVER_DIR, exist_ok=True)


# ==========================================
# 2. UNIFY & DEDUPLICATE THE 3 SEVERITY FILES
# ==========================================
def load_and_standardize_pe(filepath: str, source_name: str, priority: int) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    # Normalize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Automatically identify the severity column (e.g., 'severity', 'edema_severity', 'consensus_severity')
    sev_col = [c for c in df.columns if "severity" in c or "label" in c or "grade" in c][0]
    df = df.rename(columns={sev_col: "edema_severity"})
    
    # Keep only valid severity grades (0, 1, 2, 3)
    df = df[df["edema_severity"].isin([0, 1, 2, 3])].copy()
    df["edema_severity"] = df["edema_severity"].astype(int)
    df["label_source"] = source_name
    df["source_priority"] = priority
    
    # Ensure dicom_id exists (expert/regex report files may only have subject_id & study_id)
    if "dicom_id" not in df.columns:
        df["dicom_id"] = None
        
    return df[["subject_id", "study_id", "dicom_id", "edema_severity", "label_source", "source_priority"]]


def build_balanced_multimodal_lake(samples_per_class: int = None, random_seed: int = 42):
    print("1. Merging the 3 PE-Severity files with priority hierarchy...")
    df_cons = load_and_standardize_pe(PATHS["consensus"], "consensus_image", priority=1)
    df_exp = load_and_standardize_pe(PATHS["expert"], "expert_report", priority=2)
    df_reg = load_and_standardize_pe(PATHS["regex"], "regex_report", priority=3)

    pe_all = pd.concat([df_cons, df_exp, df_reg], ignore_index=True)
    # Deduplicate at study_id level, keeping highest priority (1 > 2 > 3)
    pe_dedup = (
        pe_all.sort_values("source_priority", ascending=True)
        .drop_duplicates(subset=["subject_id", "study_id"], keep="first")
        .reset_index(drop=True)
    )
    print("   Raw merged study counts by source:\n", pe_dedup["label_source"].value_counts())

    # ==========================================
    # 3. ALIGN WITH CXR, ADMISSIONS & NOTES
    # ==========================================
    print("\n2. Aligning with MIMIC-CXR-JPG, MIMIC-IV Admissions, and Discharge Notes...")
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=8;")
    con.register("pe_table", pe_dedup)

    alignment_query = f"""
    WITH cxr_joined AS (
        SELECT 
            pe.subject_id,
            pe.study_id,
            COALESCE(pe.dicom_id, m.dicom_id) AS dicom_id,
            pe.edema_severity,
            pe.label_source,
            pe.source_priority,
            m.ViewPosition AS view_position,
            try_strptime(
                CAST(m.StudyDate AS VARCHAR) || lpad(SPLIT_PART(CAST(m.StudyTime AS VARCHAR), '.', 1), 6, '0'),
                '%Y%m%d%H%M%S'
            ) AS cxr_datetime,
            'files/p' || SUBSTRING(CAST(pe.subject_id AS VARCHAR), 1, 2) || 
            '/p' || CAST(pe.subject_id AS VARCHAR) || 
            '/s' || CAST(pe.study_id AS VARCHAR) || 
            '/' || COALESCE(pe.dicom_id, m.dicom_id) || '.jpg' AS jpg_rel_path,
            -- Prefer PA over AP if a study has multiple frontal views
            ROW_NUMBER() OVER (
                PARTITION BY pe.study_id 
                ORDER BY CASE WHEN m.ViewPosition = 'PA' THEN 1 ELSE 2 END
            ) AS view_rank
        FROM pe_table pe
        INNER JOIN read_csv_auto(
            '{PATHS["cxr_meta"]}', 
            quote='"', 
            escape='"', 
            strict_mode=false, 
            ignore_errors=true
        ) m
            ON pe.subject_id = m.subject_id 
            AND pe.study_id = m.study_id
            AND (pe.dicom_id IS NULL OR pe.dicom_id = m.dicom_id)
        WHERE m.ViewPosition IN ('PA', 'AP')
    ),
    adm_note_joined AS (
        SELECT 
            c.subject_id,
            a.hadm_id,
            c.study_id,
            c.dicom_id,
            c.edema_severity,
            c.label_source,
            c.source_priority,
            c.view_position,
            c.cxr_datetime,
            c.jpg_rel_path,
            a.admittime,
            a.dischtime,
            a.admission_type,
            a.race,
            a.hospital_expire_flag,
            p.gender,
            p.anchor_age + (YEAR(a.admittime) - p.anchor_year) AS age_at_admission,
            d.note_id,
            d.text AS discharge_text,
            -- Enforce 1-to-1 per admission: pick highest priority label first, then earliest CXR near admission
            ROW_NUMBER() OVER (
                PARTITION BY a.hadm_id 
                ORDER BY c.source_priority ASC, ABS(epoch(c.cxr_datetime - COALESCE(a.edregtime, a.admittime))) ASC
            ) AS hadm_rank
        FROM cxr_joined c
        INNER JOIN read_csv_auto(
            '{PATHS["admissions"]}', 
            quote='"', 
            escape='"', 
            strict_mode=false
        ) a
            ON c.subject_id = a.subject_id
            AND c.cxr_datetime BETWEEN (COALESCE(a.edregtime, a.admittime) - INTERVAL 24 HOUR) AND a.dischtime
        INNER JOIN read_csv_auto(
            '{PATHS["patients"]}', 
            quote='"', 
            escape='"', 
            strict_mode=false
        ) p
            ON c.subject_id = p.subject_id
        INNER JOIN read_csv_auto(
            '{PATHS["discharge"]}', 
            quote='"', 
            escape='"', 
            strict_mode=false,
             sample_size=100000
        ) d
            ON a.hadm_id = d.hadm_id
        WHERE c.view_rank = 1
          AND c.cxr_datetime IS NOT NULL
          AND d.text IS NOT NULL
    )
    SELECT * EXCLUDE (hadm_rank)
    FROM adm_note_joined
    WHERE hadm_rank = 1
    """
    
    multimodal_df = con.execute(alignment_query).df()
    print("\n3. Complete Multimodal Cases (1 per hadm_id) BEFORE Balancing:")
    print(pd.crosstab(multimodal_df["edema_severity"], multimodal_df["label_source"], margins=True))

    # ==========================================
    # 4. PRIORITY-AWARE CLASS BALANCING
    # ==========================================
    class_counts = multimodal_df["edema_severity"].value_counts()
    min_class_count = class_counts.min()
    target_n = min(samples_per_class, min_class_count) if samples_per_class else min_class_count
    print(f"\n4. Balancing classes to exactly N = {target_n} per severity level (Total = {target_n * 4})...")

    balanced_dfs = []
    for grade in [0, 1, 2, 3]:
        subset = multimodal_df[multimodal_df["edema_severity"] == grade].copy()
        # Shuffle regex rows with fixed seed, but ALWAYS keep priority 1 (consensus) and 2 (expert) at the top
        subset["random_order"] = subset.sample(frac=1.0, random_state=random_seed).index
        subset = subset.sort_values(by=["source_priority", "random_order"], ascending=[True, True])
        balanced_dfs.append(subset.head(target_n).drop(columns=["random_order"]))

    final_balanced_df = pd.concat(balanced_dfs, ignore_index=True)

    # Assign clean Train / Val / Test splits:
    # Put consensus & expert cases in Test/Val first, and prevent subject_id leakage across splits
    print("\n5. Final Balanced Cohort Distribution:")
    print(pd.crosstab(final_balanced_df["edema_severity"], final_balanced_df["label_source"], margins=True))

    out_path = f"{SILVER_DIR}/balanced_multimodal_cohort.parquet"
    final_balanced_df.to_parquet(out_path, compression="zstd", index=False)
    print(f"\nSaved unified, balanced dataset to: {out_path}")


if __name__ == "__main__":
    # Leave samples_per_class=None to automatically match the minority class (Grade 3),
    # or pass e.g. samples_per_class=250 for a fast 1,000-case benchmark.
    # build_balanced_multimodal_lake(samples_per_class=None, random_seed=42)

    import torch
    print("CUDA Available:", torch.cuda.is_available())
    print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")