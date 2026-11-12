import os

# Disable hf_xet, symlink warnings, and BLAS/OpenMP thread contention on Windows
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import re
import argparse
import pickle
import requests
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_COHORT_PATH = os.path.join(BASE_DIR, "silver", "balanced_multimodal_cohort.parquet")
DEFAULT_CXR_DIR = os.path.join(BASE_DIR, "bronze", "mimic_cxr")
DEFAULT_ADM_PATH = os.path.join(BASE_DIR, "bronze", "mimic_iv", "admissions.csv")
DEFAULT_PAT_PATH = os.path.join(BASE_DIR, "bronze", "mimic_iv", "patients.csv")
DEFAULT_GOLD_DIR = os.path.join(BASE_DIR, "gold")
DEFAULT_OUTPUT_PKL = os.path.join(DEFAULT_GOLD_DIR, "multimodal_features.pkl")
DEFAULT_OUTPUT_PARQUET = os.path.join(DEFAULT_GOLD_DIR, "multimodal_features.parquet")
DEFAULT_XRV_CACHE_DIR = os.path.join(DEFAULT_GOLD_DIR, "model_cache", "torchxrayvision")
DEFAULT_BIOBERT_CACHE_DIR = os.path.join(DEFAULT_GOLD_DIR, "model_cache", "biobert")

# Local BioBERT path (if downloaded locally) or HuggingFace Hub model ID
LOCAL_BIOBERT_PATH = os.path.join(os.path.dirname(BASE_DIR), "models", "biobert")
HF_BIOBERT_MODEL = "dmis-lab/biobert-v1.1"

# Pre-compiled keyword pattern for extracting cardiopulmonary, CXR, and edema-relevant sentences
CARDIOPULMONARY_KEYWORDS = re.compile(
    r"\b(?:"
    r"edema|congestion|vascular|interstitial|alveolar|cephalization|kerley|"
    r"effusion|pleural|cardiomegaly|heart\s+failure|chf|overload|fluid|"
    r"lasix|furosemide|bumex|bumetanide|torsemide|diures\w*|iv\s+lasix|"
    r"cxr|chest\s+x-ray|radiograph\w*|portable\s+chest|pa\s+and\s+lateral|"
    r"lung\w*|pulmonary|bibasilar|basilar|perihilar|hilar|infiltrat\w*|opacit\w*|atelecta\w*|consolida\w*|"
    r"crackles|rales|rhonchi|wheez\w*|jvd|jugular|orthopnea|pnd|dyspnea|shortness\s+of\s+breath|sob|hypoxi\w*|o2|bipap|intubat\w*"
    r")\b",
    re.IGNORECASE,
)

PERIPHERAL_EDEMA_RE = re.compile(
    r"\b(?:lower\s+extremit\w+|bilateral\s+lower|upper\s+extremit\w+|pedal|pretibial|leg|ankle|foot|feet|sacral|scrotal|periorbital|pitting|non-pitting|trace|[1234]\+)\s+edema\b"
    r"|\bedema\s+(?:of|in)\s+(?:both\s+|the\s+)?(?:legs|lower\s+extremit\w+|ankles|feet)\b"
    r"|\bext(?:remities)?\s*:[^\n.]*",
    re.IGNORECASE,
)

RAD_KEYWORDS = re.compile(
    r"\b(?:"
    r"pulmonary\s+edema|vascular\s+congestion|interstitial\s+edema|alveolar\s+edema|"
    r"cephalization|kerley|peribronchial|perihilar|batwing|fluid\s+overload|"
    r"cxr|chest\s+x-ray|chest\s+xray|radiograph\w*|portable\s+chest|pa\s+and\s+lateral|port\s+chest|"
    r"edema|congestion|engorg\w*|redistribution|prominence|haziness|hazy|airspace|"
    r"effusion\w*|pleural|cardiomegaly|cardiopulmonary|intrathoracic|clear\s+lungs|lungs\s+are\s+clear|"
    r"bibasilar|basilar|hilar|opacit\w*|infiltrat\w*|atelecta\w*|consolida\w*"
    r")\b",
    re.IGNORECASE,
)

CXR_HEADER_RE = re.compile(
    r"\b(?:cxr|chest\s+x-?ray|portable\s+chest|port\s+chest|pa\s+and\s+lateral|chest\s+\(portable|chest\s+\(pa|findings|impression)\b",
    re.IGNORECASE,
)

MODIFIER_TAIL_RE = re.compile(
    r"\b(no|without|mild\w*|moderate\w*|severe\w*|marked\w*|slight\w*|minimal\w*|acute|frank|interval|with|and|or|of|to|in|for|on|as|is|are|was|were)\s*\n\s*([A-Z][a-z])",
    re.IGNORECASE,
)
WRAP_LOWER_RE = re.compile(r"([a-zA-Z0-9,\-])\s*\n\s*([a-z])")
HEADER_COLON_RE = re.compile(
    r"((?:cxr|chest|findings|impression)[^\n.]{0,30}[:_])\s*\n\s*([A-Za-z])",
    re.IGNORECASE,
)

CLINICAL_SEVERITY_CONCEPTS = [
    r"\bno\s+(?:acute\s+|frank\s+|overt\s+|evidence\s+of\s+)?(?:pulmonary\s+)?edema\b",
    r"\bwithout\s+(?:acute\s+|frank\s+|overt\s+)?(?:pulmonary\s+)?edema\b",
    r"\bno\s+(?:pulmonary\s+)?vascular\s+congestion\b",
    r"\b(?:lungs\s+are\s+clear|clear\s+lungs|no\s+acute\s+cardiopulmonary|no\s+acute\s+intrathoracic|unremarkable)\b",
    r"\b(?:resolved|resolution\s+of)\s+(?:pulmonary\s+)?edema\b",
    r"\b(?:mild|slight|minimal|trace|early)\s+(?:pulmonary\s+|vascular\s+|interstitial\s+)?(?:edema|congestion|prominence|overload|chf)\b",
    r"\b(?:cephalization|redistribution|pulmonary\s+vascular\s+congestion|vascular\s+congestion|vascular\s+prominence|vascular\s+engorgement|hilar\s+fullness)\b",
    r"\b(?:moderate|mild\s+to\s+moderate|mild-to-moderate)\s+(?:pulmonary\s+|interstitial\s+|vascular\s+)?(?:edema|congestion)\b",
    r"\b(?:interstitial\s+pulmonary\s+edema|interstitial\s+edema|kerley|peribronchial\s+cuffing|interstitial\s+markings|interstitial\s+opacit\w*)\b",
    r"\b(?:severe|moderate\s+to\s+severe|moderate-to-severe|marked|extensive|florid|flash|fulminant|worsening)\s+(?:pulmonary\s+|alveolar\s+|interstitial\s+)?(?:edema|congestion)\b",
    r"\b(?:alveolar\s+edema|alveolar\s+pulmonary\s+edema|flash\s+pulmonary\s+edema|perihilar\s+opacit\w*|perihilar\s+airspace|perihilar\s+haziness|batwing|airspace\s+edema)\b",
    r"\b(?:pleural\s+effusion\w*|bilateral\s+effusion\w*)\b",
    r"\b(?:cardiomegaly|enlarged\s+cardiac)\b",
    r"\b(?:iv\s+lasix|iv\s+furosemide|lasix\s+drip|bumex|diures\w*)\b",
]
COMPILED_CONCEPT_PATS = [re.compile(p, re.IGNORECASE) for p in CLINICAL_SEVERITY_CONCEPTS]


def extract_focused_clinical_text(text: str, max_chars: int = 4000) -> str:
    """
    Filters a long discharge summary (~13,500 chars) to sentences/lines discussing
    cardiopulmonary status, chest imaging (CXR), physical exam, and diuresis/heart failure.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    segments = re.split(r"[\n.]+", text)
    matched = []
    for seg in segments:
        cleaned = seg.strip()
        if len(cleaned) < 5:
            continue
        if CARDIOPULMONARY_KEYWORDS.search(cleaned):
            matched.append(cleaned)

    if matched:
        focused = " . ".join(matched)
        return focused[:max_chars]
    return text[:max_chars]


def extract_radiology_and_concepts(text: str):
    """
    1. Masks peripheral/extremity edema mentions so 'edema' strictly reflects cardiopulmonary edema.
    2. Repairs 70-character hard line-wraps inside sentences so negations and severity modifiers
       (e.g., 'without frank\\npulmonary edema', 'moderate\\ninterstitial edema') remain intact.
    3. Extracts full-stay radiology text (`rad_full`), early-stay/admission radiology text (`rad_early`),
       and a 28-d log-count vector of explicit clinical severity concepts (`concept_vec`).
    """
    if not isinstance(text, str) or not text.strip():
        return "", "", np.zeros(len(COMPILED_CONCEPT_PATS) * 2, dtype=np.float32)

    t = PERIPHERAL_EDEMA_RE.sub(" peripheral_swelling ", text)
    t = HEADER_COLON_RE.sub(r"\1 \2", t)
    for _ in range(3):
        t = WRAP_LOWER_RE.sub(r"\1 \2", t)
        t = MODIFIER_TAIL_RE.sub(r"\1 \2", t)

    segs = re.split(r"[\n.;]+", t)
    matched = []
    n_segs = len(segs)
    for i, seg in enumerate(segs):
        s = seg.strip()
        if len(s) < 4:
            continue
        if RAD_KEYWORDS.search(s):
            matched.append(s)
            if len(s) < 45 and CXR_HEADER_RE.search(s) and (i + 1) < n_segs:
                nxt = segs[i + 1].strip()
                if len(nxt) >= 4:
                    matched.append(nxt)

    if matched:
        rad_full = " . ".join(matched)[:8000]
        k_half = max(2, (len(matched) + 1) // 2)
        rad_early = " . ".join(matched[:k_half])[:4000]
    else:
        rad_full = t[:4000]
        rad_early = t[:2000]

    concept_counts = [len(p.findall(rad_full)) for p in COMPILED_CONCEPT_PATS] + [
        len(p.findall(rad_early)) for p in COMPILED_CONCEPT_PATS
    ]
    concept_vec = np.log1p(np.array(concept_counts, dtype=np.float32))
    return rad_full, rad_early, concept_vec


def ensure_biobert_local(model_id: str = HF_BIOBERT_MODEL, target_dir: str = DEFAULT_BIOBERT_CACHE_DIR) -> str:
    """
    Downloads BioBERT files directly via standard HTTPS streaming (requests) into target_dir,
    bypassing hf_xet / filelock deadlocks on Windows.
    """
    os.makedirs(target_dir, exist_ok=True)
    required_files = [
        "config.json",
        "vocab.txt",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "pytorch_model.bin",
    ]

    session = requests.Session()
    session.headers.update({"User-Agent": "Wget/1.21.4", "Accept": "*/*"})
    try:
        import urllib3.util.connection as urllib3_conn
        urllib3_conn.HAS_IPV6 = False
    except Exception:
        pass

    for fname in required_files:
        dest_path = os.path.join(target_dir, fname)
        min_bytes = 400_000_000 if fname == "pytorch_model.bin" else 10
        if os.path.exists(dest_path) and os.path.getsize(dest_path) >= min_bytes:
            continue

        url = f"https://huggingface.co/{model_id}/resolve/main/{fname}?download=true"
        temp_path = dest_path + ".part"
        print(f"Downloading {model_id}/{fname} via direct HTTPS stream...")

        with session.get(url, stream=True, timeout=(15, 120), allow_redirects=True) as resp:
            resp.raise_for_status()
            total_size = int(resp.headers.get("Content-Length", 0))
            with open(temp_path, "wb") as out_f, tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=fname,
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        out_f.write(chunk)
                        pbar.update(len(chunk))

        os.replace(temp_path, dest_path)

    return target_dir


class XRayEmbedder:
    """
    Generates 1024-d dense features and 18-d pathology prediction features
    from chest X-ray JPG images using torchxrayvision DenseNet121.
    """
    def __init__(self, weights_name: str = "densenet121-res224-chex", device: str = "cpu"):
        import torchxrayvision as xrv
        from torchvision import transforms

        self.xrv = xrv
        self.device = torch.device(device)
        print(f"Loading XRayEmbedder ('{weights_name}') onto {self.device}...")
        os.makedirs(DEFAULT_XRV_CACHE_DIR, exist_ok=True)
        self.model = xrv.models.DenseNet(weights=weights_name, cache_dir=DEFAULT_XRV_CACHE_DIR).to(self.device)
        self.model.eval()

        try:
            resizer = xrv.datasets.XRayResizer(224, engine="cv2")
        except TypeError:
            resizer = xrv.datasets.XRayResizer(224)

        self.transform = transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            resizer,
        ])

    def preprocess_image(self, jpg_path: str):
        if not os.path.exists(jpg_path) or os.path.getsize(jpg_path) == 0:
            return None

        img_array = cv2.imread(jpg_path, cv2.IMREAD_GRAYSCALE)
        if img_array is None:
            return None

        img = self.xrv.datasets.normalize(img_array, 255)
        if img.ndim > 2:
            img = img[:, :, 0]
        if img.shape[0] != 1:
            img = img[None, :, :]

        img = self.transform(img)
        return torch.from_numpy(img).float()

    def get_embeddings(self, jpg_path: str):
        tensor = self.preprocess_image(jpg_path)
        if tensor is None:
            return None, None

        img_tensor = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.model.features(img_tensor)
            feats = F.relu(feats, inplace=True)
            feats = F.adaptive_avg_pool2d(feats, (1, 1))
            dense_features = feats.cpu().numpy().flatten().astype(np.float32)

            preds = self.model(img_tensor).cpu()
            prediction_features = preds.numpy().flatten().astype(np.float32)

        return dense_features, prediction_features


class BioBERTEmbedder:
    """
    Generates 768-d embeddings from focused cardiopulmonary clinical notes using BioBERT
    with attention-masked mean pooling.
    """
    def __init__(self, model_name_or_path: str = None, device: str = "cpu", chunk_batch_size: int = 32):
        from transformers import AutoTokenizer, AutoModel

        self.device = torch.device(device)
        self.chunk_batch_size = chunk_batch_size

        if model_name_or_path is None:
            if os.path.isdir(LOCAL_BIOBERT_PATH):
                model_name_or_path = LOCAL_BIOBERT_PATH
            else:
                model_name_or_path = ensure_biobert_local(HF_BIOBERT_MODEL, DEFAULT_BIOBERT_CACHE_DIR)
        elif not os.path.isdir(model_name_or_path):
            model_name_or_path = ensure_biobert_local(model_name_or_path, DEFAULT_BIOBERT_CACHE_DIR)

        print(f"Loading BioBERT from '{model_name_or_path}' onto {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_name_or_path, local_files_only=True, use_safetensors=False).to(self.device)
        self.model.eval()

    def _mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask

    def get_batch_embeddings(self, texts: list, max_length: int = 512) -> np.ndarray:
        all_vecs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.chunk_batch_size), desc="BioBERT Note Batches"):
                batch_texts = [t if (isinstance(t, str) and t.strip()) else "none" for t in texts[i : i + self.chunk_batch_size]]
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                pooled = self._mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
                pooled = F.normalize(pooled, p=2, dim=1)
                all_vecs.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(all_vecs)


def build_demographic_features(
    cohort_df: pd.DataFrame,
    adm_path: str = DEFAULT_ADM_PATH,
    pat_path: str = DEFAULT_PAT_PATH,
) -> np.ndarray:
    df = cohort_df.copy()

    extra_adm_cols = ["hadm_id", "marital_status", "language", "insurance", "admission_location", "discharge_location"]
    if os.path.exists(adm_path):
        adm_df = pd.read_csv(adm_path, usecols=lambda c: c in extra_adm_cols)
        adm_df = adm_df.drop_duplicates(subset=["hadm_id"])
        df = df.merge(adm_df, on="hadm_id", how="left")
    else:
        for col in extra_adm_cols[1:]:
            df[col] = "UNKNOWN"

    if os.path.exists(pat_path):
        pat_df = pd.read_csv(pat_path, usecols=lambda c: c in ["subject_id", "anchor_year_group"])
        pat_df = pat_df.drop_duplicates(subset=["subject_id"])
        df = df.merge(pat_df, on="subject_id", how="left")
    else:
        df["anchor_year_group"] = "UNKNOWN"

    categorical_cols = [
        "race",
        "marital_status",
        "language",
        "insurance",
        "admission_type",
        "admission_location",
        "discharge_location",
        "gender",
        "anchor_year_group",
        "view_position",
    ]
    for col in categorical_cols:
        if col not in df.columns:
            df[col] = "MISSING"
        df[col] = df[col].fillna("MISSING").astype(str)

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    cat_encoded = encoder.fit_transform(df[categorical_cols]).astype(np.float32)

    age = df["age_at_admission"].fillna(df["age_at_admission"].median()).to_numpy(dtype=np.float32).reshape(-1, 1)
    expire_flag = df["hospital_expire_flag"].fillna(0).to_numpy(dtype=np.float32).reshape(-1, 1)

    if "admittime" in df.columns and "dischtime" in df.columns:
        los_days = (
            (pd.to_datetime(df["dischtime"]) - pd.to_datetime(df["admittime"])).dt.total_seconds() / 86400.0
        ).fillna(0.0).clip(lower=0.0, upper=120.0).to_numpy(dtype=np.float32).reshape(-1, 1)
    else:
        los_days = np.zeros((len(df), 1), dtype=np.float32)

    if "cxr_datetime" in df.columns and "admittime" in df.columns:
        cxr_hours = (
            (pd.to_datetime(df["cxr_datetime"]) - pd.to_datetime(df["admittime"])).dt.total_seconds() / 3600.0
        ).fillna(0.0).clip(lower=-48.0, upper=720.0).to_numpy(dtype=np.float32).reshape(-1, 1)
    else:
        cxr_hours = np.zeros((len(df), 1), dtype=np.float32)

    demo_matrix = np.hstack([cat_encoded, age, expire_flag, los_days, cxr_hours]).astype(np.float32)
    return demo_matrix


def extract_cohort_features(
    cohort_path: str = DEFAULT_COHORT_PATH,
    cxr_dir: str = DEFAULT_CXR_DIR,
    output_pkl: str = DEFAULT_OUTPUT_PKL,
    output_parquet: str = DEFAULT_OUTPUT_PARQUET,
    biobert_model: str = None,
    device: str = None,
    refresh_notes: bool = False,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Reading silver cohort from: {cohort_path}")
    df = pd.read_parquet(cohort_path)
    print(f"Total cohort rows: {len(df)}")

    print("1. Extracting demographic & clinical tabular features across full cohort...")
    full_demo_matrix = build_demographic_features(df)
    df["cohort_row_idx"] = np.arange(len(df))

    df["local_jpg_path"] = df["jpg_rel_path"].apply(lambda p: os.path.join(cxr_dir, p))
    df["jpg_exists"] = df["local_jpg_path"].apply(lambda p: os.path.exists(p) and os.path.getsize(p) > 0)

    num_available = int(df["jpg_exists"].sum())
    print(f"Downloaded CXR images found on disk: {num_available} / {len(df)}")
    if num_available == 0:
        raise FileNotFoundError(
            f"No downloaded images found under {cxr_dir}. Please finish running download_images.py first."
        )
    if num_available < len(df):
        print(
            f"WARNING: Only {num_available}/{len(df)} images are currently present on disk. "
            f"Proceeding with the {num_available} available cases."
        )
        df = df[df["jpg_exists"]].reset_index(drop=True)

    cached_by_hadm = {}
    if os.path.exists(output_pkl):
        try:
            with open(output_pkl, "rb") as f:
                prev_records = pickle.load(f)
            for rec in prev_records:
                if "hadm_id" in rec:
                    cached_by_hadm[int(rec["hadm_id"])] = rec
            print(f"Loaded {len(cached_by_hadm)} cached records from {output_pkl}")
        except Exception as e:
            print(f"Could not load cache ({e}), extracting all from scratch.")

    print("2. Extracting stitched radiology segments, cardiopulmonary sentences, and clinical severity concepts...")
    raw_texts = df["discharge_text"].tolist()
    focused_texts = [extract_focused_clinical_text(t) for t in raw_texts]
    rad_tuples = [extract_radiology_and_concepts(t) for t in raw_texts]

    need_vision = [
        (int(h) not in cached_by_hadm) or ("vision_dense" not in cached_by_hadm[int(h)])
        for h in df["hadm_id"]
    ]
    need_notes = refresh_notes or any(
        (int(h) not in cached_by_hadm) or ("focused_text" not in cached_by_hadm[int(h)])
        for h in df["hadm_id"]
    )

    xray_embedder = XRayEmbedder(device=device) if any(need_vision) else None
    if need_notes:
        biobert_embedder = BioBERTEmbedder(model_name_or_path=biobert_model, device=device)
        all_note_vecs = biobert_embedder.get_batch_embeddings(focused_texts)
    else:
        all_note_vecs = None

    records = []
    skipped = 0
    print("3. Assembling Gold multimodal records...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building Gold Multimodal Cohort"):
        hadm_id = int(row["hadm_id"])
        demo_vec = full_demo_matrix[int(row["cohort_row_idx"])].astype(np.float32)

        if hadm_id in cached_by_hadm and "vision_dense" in cached_by_hadm[hadm_id]:
            dense_vec = cached_by_hadm[hadm_id]["vision_dense"]
            pred_vec = cached_by_hadm[hadm_id]["vision_pred"]
        else:
            dense_vec, pred_vec = xray_embedder.get_embeddings(row["local_jpg_path"])
            if dense_vec is None or pred_vec is None:
                skipped += 1
                continue

        if all_note_vecs is not None:
            note_vec = all_note_vecs[idx]
        else:
            note_vec = cached_by_hadm[hadm_id]["notes"]

        rad_full, rad_early, concept_vec = rad_tuples[idx]

        record = {
            "subject_id": int(row["subject_id"]),
            "hadm_id": hadm_id,
            "study_id": int(row["study_id"]),
            "dicom_id": str(row["dicom_id"]),
            "label": np.int64(row["edema_severity"]),
            "label_source": str(row["label_source"]),
            "source_priority": int(row["source_priority"]),
            "focused_text": focused_texts[idx],
            "rad_full": rad_full,
            "rad_early": rad_early,
            "concept_vec": concept_vec,
            "demographics": demo_vec,
            "notes": note_vec.flatten().astype(np.float32),
            "vision_dense": dense_vec.astype(np.float32),
            "vision_pred": pred_vec.astype(np.float32),
        }
        records.append(record)

    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    with open(output_pkl, "wb") as f:
        pickle.dump(records, f)
    print(f"Saved {len(records)} multimodal records (skipped {skipped}) to: {output_pkl}")

    gold_df = pd.DataFrame([
        {
            "subject_id": r["subject_id"],
            "hadm_id": r["hadm_id"],
            "study_id": r["study_id"],
            "dicom_id": r["dicom_id"],
            "label": int(r["label"]),
            "label_source": r["label_source"],
            "source_priority": r["source_priority"],
            "demographics": r["demographics"].tolist(),
            "notes": r["notes"].tolist(),
            "vision_dense": r["vision_dense"].tolist(),
            "vision_pred": r["vision_pred"].tolist(),
        }
        for r in records
    ])
    gold_df.to_parquet(output_parquet, compression="zstd", index=False)
    print(f"Saved Gold parquet table to: {output_parquet}")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract multimodal embeddings from Silver cohort to Gold layer.")
    parser.add_argument("--cohort", type=str, default=DEFAULT_COHORT_PATH, help="Path to silver cohort parquet")
    parser.add_argument("--cxr-dir", type=str, default=DEFAULT_CXR_DIR, help="Path to bronze mimic_cxr directory")
    parser.add_argument("--out-pkl", type=str, default=DEFAULT_OUTPUT_PKL, help="Output pickle path")
    parser.add_argument("--out-parquet", type=str, default=DEFAULT_OUTPUT_PARQUET, help="Output gold parquet path")
    parser.add_argument("--biobert", type=str, default=None, help="BioBERT local path or HuggingFace model name")
    parser.add_argument("--refresh-notes", action="store_true", help="Re-extract focused BioBERT note embeddings")
    args = parser.parse_args()

    extract_cohort_features(
        cohort_path=args.cohort,
        cxr_dir=args.cxr_dir,
        output_pkl=args.out_pkl,
        output_parquet=args.out_parquet,
        biobert_model=args.biobert,
        refresh_notes=args.refresh_notes,
    )
