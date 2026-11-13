import os
import argparse
import duckdb
import pandas as pd

# ==========================================
# 1. PATHS CONFIGURATION
# ==========================================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(ROOT_DIR, "data_lake") if os.path.isdir(os.path.join(ROOT_DIR, "data_lake")) else ROOT_DIR
BRONZE_PE_DIR = os.path.join(BASE_DIR, "bronze", "pe_severity")

PATHS = {
    "consensus": os.path.join(BRONZE_PE_DIR, "consensus_image_edema_severity.csv").replace("\\", "/"),
    "expert": os.path.join(BRONZE_PE_DIR, "expert_report_edema_severity.csv").replace("\\", "/"),
    "regex": os.path.join(BRONZE_PE_DIR, "regex_report_edema_severity.csv").replace("\\", "/"),
    "cxr_meta": os.path.join(BASE_DIR, "bronze", "mimic_cxr", "mimic-cxr-2.0.0-metadata.csv").replace("\\", "/"),
    "patients": os.path.join(BASE_DIR, "bronze", "mimic_iv", "patients.csv").replace("\\", "/"),
    "admissions": os.path.join(BASE_DIR, "bronze", "mimic_iv", "admissions.csv").replace("\\", "/"),
    "discharge": os.path.join(BASE_DIR, "bronze", "mimic_note", "discharge.csv").replace("\\", "/"),
    "radiology_gz": os.path.join(BASE_DIR, "bronze", "mimic_note", "radiology.csv.gz").replace("\\", "/"),
    "radiology_csv": os.path.join(BASE_DIR, "bronze", "mimic_note", "radiology.csv").replace("\\", "/"),
}

SILVER_DIR = os.path.join(BASE_DIR, "silver")
os.makedirs(SILVER_DIR, exist_ok=True)


def find_radiology_csv_path(custom_path: str = None) -> str:
    """
    Locates `radiology.csv.gz` or `radiology.csv` under `data_lake/bronze/mimic_note/`
    (or custom_path) and verifies that the download has completed (no active .crdownload/.part file).
    """
    candidates = []
    if custom_path:
        candidates.append(custom_path)
    candidates.extend([
        PATHS["radiology_gz"],
        PATHS["radiology_csv"],
        os.path.join(BASE_DIR, "bronze", "radiology.csv.gz"),
        os.path.join(BASE_DIR, "bronze", "radiology.csv"),
        os.path.join(BASE_DIR, "radiology.csv.gz"),
        os.path.join(BASE_DIR, "radiology.csv"),
        os.path.join(ROOT_DIR, "radiology.csv.gz"),
        os.path.join(ROOT_DIR, "radiology.csv"),
    ])
    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        if os.path.exists(path + ".crdownload") or os.path.exists(path + ".part"):
            continue
        if os.path.getsize(path) > 10_000_000:  # >10 MB sanity check for completed MIMIC radiology file
            return os.path.abspath(path)
    return None


def attach_radiology_to_cohort_df(cohort_df: pd.DataFrame, radiology_path: str) -> pd.DataFrame:
    """
    Aligns each study (`subject_id`, `hadm_id`, `study_id`, `cxr_datetime`) in `cohort_df`
    with its dedicated CXR radiology report from `radiology.csv.gz` / `radiology.csv` using DuckDB.
    Uses a LEFT JOIN so all existing cohort rows and downloaded CXR images are 100% preserved.
    """
    if not radiology_path or not os.path.exists(radiology_path):
        print("   No radiology.csv.gz / radiology.csv found yet; keeping discharge-only notes.")
        return cohort_df

    print(f"\nAligning cohort with dedicated CXR radiology reports from: {radiology_path} ...")
    drop_cols = [
        "radiology_note_id",
        "radiology_charttime",
        "radiology_time_diff_hrs",
        "radiology_text",
        "admission_radiology_text",
    ]
    clean_df = cohort_df.drop(columns=[c for c in drop_cols if c in cohort_df.columns]).copy()

    rad_path_sql = os.path.abspath(radiology_path).replace("\\", "/")
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=8;")
    con.register("cohort_table", clean_df)

    rad_join_query = f"""
    WITH cohort_subjects AS (
        SELECT DISTINCT TRY_CAST(subject_id AS BIGINT) AS subject_id
        FROM cohort_table
    ),
    rad_filtered AS (
        SELECT
            CAST(r.note_id AS VARCHAR) AS rad_note_id,
            TRY_CAST(r.subject_id AS BIGINT) AS subject_id,
            TRY_CAST(r.hadm_id AS BIGINT) AS hadm_id,
            CAST(r.note_type AS VARCHAR) AS note_type,
            TRY_CAST(r.note_seq AS INTEGER) AS note_seq,
            TRY_CAST(r.charttime AS TIMESTAMP) AS rad_charttime,
            CAST(r.text AS VARCHAR) AS rad_text
        FROM read_csv_auto(
            '{rad_path_sql}',
            quote='"',
            escape='"',
            strict_mode=false,
            ignore_errors=true,
            sample_size=100000
        ) r
        INNER JOIN cohort_subjects cs
            ON TRY_CAST(r.subject_id AS BIGINT) = cs.subject_id
        WHERE r.text IS NOT NULL
          AND LENGTH(TRIM(CAST(r.text AS VARCHAR))) > 20
    ),
    rad_candidates AS (
        SELECT
            c.study_id,
            c.hadm_id AS cohort_hadm_id,
            r.rad_note_id,
            r.rad_charttime,
            r.note_type,
            r.rad_text,
            ABS(epoch(r.rad_charttime - TRY_CAST(c.cxr_datetime AS TIMESTAMP))) / 3600.0 AS time_diff_hrs,
            CASE
                WHEN regexp_matches(
                    SUBSTRING(r.rad_text, 1, 500),
                    '(?i)(chest|portable\\s+ap|pa\\s+and\\s+lateral|ap\\s+and\\s+lateral|port\\.\\s*chest|cxr|thorax|lungs|cardiomediastinal|cardiopulmonary|pleural)'
                )
                AND NOT regexp_matches(
                    SUBSTRING(r.rad_text, 1, 250),
                    '(?i)(ct\\s+head|ct\\s+abdomen|ct\\s+pelvis|ct\\s+chest|cta\\s+chest|ct\\s+angio|mri\\s+|mr\\s+head|ultrasound|echocardiogram|extremity|knee|hip|shoulder|spine|abdomen\\s+ap|renal\\s+us|carotid|venous\\s+doppler)'
                )
                THEN 0
                ELSE 1
            END AS is_non_cxr,
            CASE
                WHEN ABS(epoch(r.rad_charttime - TRY_CAST(c.cxr_datetime AS TIMESTAMP))) <= 21600 THEN 0
                WHEN r.hadm_id = TRY_CAST(c.hadm_id AS BIGINT) THEN 1
                WHEN ABS(epoch(r.rad_charttime - TRY_CAST(c.cxr_datetime AS TIMESTAMP))) <= 86400 THEN 2
                ELSE 3
            END AS match_tier,
            ROW_NUMBER() OVER (
                PARTITION BY c.study_id, c.hadm_id
                ORDER BY
                    CASE
                        WHEN regexp_matches(
                            SUBSTRING(r.rad_text, 1, 500),
                            '(?i)(chest|portable\\s+ap|pa\\s+and\\s+lateral|ap\\s+and\\s+lateral|port\\.\\s*chest|cxr|thorax|lungs|cardiomediastinal|cardiopulmonary|pleural)'
                        )
                        AND NOT regexp_matches(
                            SUBSTRING(r.rad_text, 1, 250),
                            '(?i)(ct\\s+head|ct\\s+abdomen|ct\\s+pelvis|ct\\s+chest|cta\\s+chest|ct\\s+angio|mri\\s+|mr\\s+head|ultrasound|echocardiogram|extremity|knee|hip|shoulder|spine|abdomen\\s+ap|renal\\s+us|carotid|venous\\s+doppler)'
                        )
                        THEN 0
                        ELSE 1
                    END ASC,
                    CASE
                        WHEN ABS(epoch(r.rad_charttime - TRY_CAST(c.cxr_datetime AS TIMESTAMP))) <= 21600 THEN 0
                        WHEN r.hadm_id = TRY_CAST(c.hadm_id AS BIGINT) THEN 1
                        WHEN ABS(epoch(r.rad_charttime - TRY_CAST(c.cxr_datetime AS TIMESTAMP))) <= 86400 THEN 2
                        ELSE 3
                    END ASC,
                    CASE WHEN r.note_type = 'RR' THEN 0 ELSE 1 END ASC,
                    ABS(epoch(r.rad_charttime - TRY_CAST(c.cxr_datetime AS TIMESTAMP))) ASC NULLS LAST,
                    r.note_seq ASC NULLS LAST
            ) AS rad_rank
        FROM cohort_table c
        INNER JOIN rad_filtered r
            ON TRY_CAST(c.subject_id AS BIGINT) = r.subject_id
           AND (
               r.hadm_id = TRY_CAST(c.hadm_id AS BIGINT)
               OR ABS(epoch(r.rad_charttime - TRY_CAST(c.cxr_datetime AS TIMESTAMP))) <= 172800
               OR (
                   r.rad_charttime BETWEEN (TRY_CAST(c.admittime AS TIMESTAMP) - INTERVAL 24 HOUR)
                                       AND (TRY_CAST(c.dischtime AS TIMESTAMP) + INTERVAL 24 HOUR)
               )
           )
    ),
    best_rad AS (
        SELECT
            study_id,
            cohort_hadm_id,
            MAX(CASE WHEN rad_rank = 1 THEN rad_note_id END) AS radiology_note_id,
            MAX(CASE WHEN rad_rank = 1 THEN rad_charttime END) AS radiology_charttime,
            MAX(CASE WHEN rad_rank = 1 THEN time_diff_hrs END) AS radiology_time_diff_hrs,
            MAX(CASE WHEN rad_rank = 1 THEN rad_text END) AS radiology_text,
            STRING_AGG(
                CASE WHEN rad_rank <= 2 AND is_non_cxr = 0 AND COALESCE(time_diff_hrs, 0) <= 36 THEN rad_text END,
                '\n\n--- ADJACENT CXR REPORT ---\n\n'
                ORDER BY rad_rank ASC
            ) AS admission_radiology_text
        FROM rad_candidates
        WHERE is_non_cxr = 0 OR match_tier <= 1
        GROUP BY study_id, cohort_hadm_id
    )
    SELECT
        c.*,
        b.radiology_note_id,
        b.radiology_charttime,
        b.radiology_time_diff_hrs,
        b.radiology_text,
        COALESCE(b.admission_radiology_text, b.radiology_text) AS admission_radiology_text
    FROM cohort_table c
    LEFT JOIN best_rad b
        ON c.study_id = b.study_id
       AND c.hadm_id = b.cohort_hadm_id
    """
    enriched_df = con.execute(rad_join_query).df()
    matched_count = int(enriched_df["radiology_text"].notna().sum())
    print(f"   Matched dedicated CXR radiology reports for {matched_count} / {len(enriched_df)} cohort studies!")
    return enriched_df


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

    rad_path = find_radiology_csv_path()
    if rad_path:
        final_balanced_df = attach_radiology_to_cohort_df(final_balanced_df, rad_path)

    # Assign clean Train / Val / Test splits:
    # Put consensus & expert cases in Test/Val first, and prevent subject_id leakage across splits
    print("\n5. Final Balanced Cohort Distribution:")
    print(pd.crosstab(final_balanced_df["edema_severity"], final_balanced_df["label_source"], margins=True))

    out_path = os.path.join(SILVER_DIR, "balanced_multimodal_cohort.parquet")
    final_balanced_df.to_parquet(out_path, compression="zstd", index=False)
    print(f"\nSaved unified, balanced dataset to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build or enrich Silver balanced multimodal cohort with radiology.csv.gz.")
    parser.add_argument("--radiology-csv", type=str, default=None, help="Optional path to radiology.csv.gz or radiology.csv")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild silver cohort from scratch instead of enriching in-place")
    args = parser.parse_args()

    cohort_out_path = os.path.join(SILVER_DIR, "balanced_multimodal_cohort.parquet")
    if not args.rebuild and os.path.exists(cohort_out_path):
        rad_path = find_radiology_csv_path(args.radiology_csv)
        if rad_path:
            existing_df = pd.read_parquet(cohort_out_path)
            enriched_df = attach_radiology_to_cohort_df(existing_df, rad_path)
            enriched_df.to_parquet(cohort_out_path, compression="zstd", index=False)
            print(f"Updated existing silver cohort with radiology reports at: {cohort_out_path}")
        else:
            print(
                f"Existing silver cohort found at {cohort_out_path}, "
                f"waiting for radiology.csv.gz in bronze/mimic_note/."
            )
    else:
        build_balanced_multimodal_lake(samples_per_class=None, random_seed=42)