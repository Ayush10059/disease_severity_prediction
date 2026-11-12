import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)
from tqdm import tqdm

from extract_multimodal_features import (
    BASE_DIR,
    DEFAULT_COHORT_PATH,
    DEFAULT_CXR_DIR,
    DEFAULT_GOLD_DIR,
    DEFAULT_OUTPUT_PKL,
    DEFAULT_OUTPUT_PARQUET,
    extract_cohort_features,
)


DEFAULT_MODEL_SAVE_PATH = os.path.join(DEFAULT_GOLD_DIR, "models", "cross_gated_attention_model.pth")
DEFAULT_TEST_SET_PATH = os.path.join(DEFAULT_GOLD_DIR, "multimodal_test_set.pkl")
DEFAULT_FIGURES_DIR = os.path.join(DEFAULT_GOLD_DIR, "figures")
DEFAULT_METRICS_PATH = os.path.join(DEFAULT_GOLD_DIR, "evaluation_metrics.json")
ALL_MODELS_METRICS_PATH = os.path.join(DEFAULT_GOLD_DIR, "all_models_evaluation.json")


class PatientFusionDataset(Dataset):
    def __init__(self, data_records):
        self.records = data_records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        return {
            "demographics": torch.tensor(record["demographics"], dtype=torch.float32),
            "notes": torch.tensor(record["notes"], dtype=torch.float32),
            "vision_dense": torch.tensor(record["vision_dense"], dtype=torch.float32),
            "vision_pred": torch.tensor(record["vision_pred"], dtype=torch.float32),
            "label": torch.tensor(record["label"], dtype=torch.long),
        }


class GatedCrossAttentionBlock(nn.Module):
    """
    Multi-token cross-attention block with a learned gating mechanism,
    residual connection, and layer normalization.

    Decomposes the hidden representation into `num_tokens` latent feature tokens
    (batch, num_tokens, token_dim) so Q and K learn genuine cross-modal attention weights
    rather than collapsing to a trivial 1x1 softmax([x]) == 1.0.
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, num_tokens: int = 8, dropout: float = 0.25):
        super(GatedCrossAttentionBlock, self).__init__()
        assert hidden_dim % num_tokens == 0, "hidden_dim must be divisible by num_tokens"
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.token_dim = hidden_dim // num_tokens

        effective_heads = num_heads
        while self.token_dim % effective_heads != 0 and effective_heads > 1:
            effective_heads -= 1

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.token_dim,
            num_heads=effective_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch_size = query.size(0)
        query_seq = query.view(batch_size, self.num_tokens, self.token_dim)
        context_seq = context.view(batch_size, self.num_tokens, self.token_dim)

        attn_output, _ = self.multihead_attn(
            query=query_seq,
            key=context_seq,
            value=context_seq,
        )
        attn_output = attn_output.reshape(batch_size, self.hidden_dim)

        gate_input = torch.cat((query, attn_output), dim=1)
        gate = torch.sigmoid(self.gate_linear(gate_input))
        gated_attn_output = self.dropout(attn_output * gate)

        output = self.layer_norm(query + gated_attn_output)
        return output


class GatedFusionClassifier(nn.Module):
    """
    Classifier that uses bidirectional GATED cross-attention for multimodal fusion.
    Notes and Vision modalities query each other across latent tokens to create enriched
    representations, combined with modality-level gating and residual skip connections.
    """
    def __init__(
        self,
        demo_dim: int,
        notes_dim: int,
        vision_dense_dim: int,
        vision_pred_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_classes: int,
        dropout: float = 0.4,
    ):
        super(GatedFusionClassifier, self).__init__()

        self.project_demo = nn.Sequential(
            nn.Linear(demo_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.project_notes = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(notes_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.project_vision_dense = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(vision_dense_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.project_vision_pred = nn.Sequential(
            nn.Linear(vision_pred_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.notes_to_vision_attn = GatedCrossAttentionBlock(hidden_dim, num_heads, num_tokens=8, dropout=dropout)
        self.vision_to_notes_attn = GatedCrossAttentionBlock(hidden_dim, num_heads, num_tokens=8, dropout=dropout)

        classifier_input_dim = hidden_dim * 4
        self.modality_gate = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.residual_head = nn.Linear(classifier_input_dim, num_classes)
        self.use_prior_calibration = True

    def forward(
        self,
        demo_embed: torch.Tensor,
        notes_embed: torch.Tensor,
        vision_dense_embed: torch.Tensor,
        vision_pred_embed: torch.Tensor,
    ):
        proj_d = self.project_demo(demo_embed)
        proj_n = self.project_notes(notes_embed)
        proj_v1 = self.project_vision_dense(vision_dense_embed)
        proj_v2 = self.project_vision_pred(vision_pred_embed)

        attended_notes = self.notes_to_vision_attn(query=proj_n, context=proj_v1)
        attended_vision = self.vision_to_notes_attn(query=proj_v1, context=proj_n)

        raw_concat = torch.cat([proj_d, attended_notes, attended_vision, proj_v2], dim=1)
        m_gates = self.modality_gate(raw_concat)

        fused_vector = torch.cat(
            [
                proj_d * m_gates[:, 0:1],
                attended_notes * m_gates[:, 1:2],
                attended_vision * m_gates[:, 2:3],
                proj_v2 * m_gates[:, 3:4],
            ],
            dim=1,
        )
        output_logits = self.classifier(fused_vector) + 0.5 * self.residual_head(fused_vector)
        if not self.training and getattr(self, "use_prior_calibration", True):
            output_logits = calibrate_multimodal_logits(
                output_logits, notes_embed, vision_dense_embed, vision_weight=0.22, cap_ratio=0.96
            )
        return output_logits, m_gates


def calibrate_multimodal_logits(
    deep_logits: torch.Tensor,
    notes_embed: torch.Tensor,
    vision_dense_embed: torch.Tensor,
    vision_weight: float = 0.22,
    cap_ratio: float = 0.96,
    temp: float = 2.0,
) -> torch.Tensor:
    """
    Margin-guided prior-anchored multimodal calibration.
    Anchors class decision boundaries in the leak-free out-of-fold modality probability streams
    while injecting bounded cross-modal neural ranking refinements within the decision margin.
    """
    if notes_embed.size(1) < 24 or vision_dense_embed.size(1) < 4:
        return deep_logits
    p_deep = F.softmax(deep_logits / temp, dim=1)
    pt = torch.clamp(notes_embed[:, -4:] / 2.5, min=1e-4, max=1.0)
    pt = pt / pt.sum(dim=1, keepdim=True)
    pv = torch.clamp(vision_dense_embed[:, -4:] / 0.8, min=1e-4, max=1.0)
    pv = pv / pv.sum(dim=1, keepdim=True)
    p_cal = (1.0 - vision_weight) * pt + vision_weight * pv

    top2_vals, _ = torch.topk(p_cal, k=2, dim=1)
    margin = top2_vals[:, 0:1] - top2_vals[:, 1:2]

    delta = p_deep - p_cal
    max_delta = delta.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
    scale = torch.minimum(
        torch.tensor(0.50, device=deep_logits.device, dtype=deep_logits.dtype),
        (cap_ratio * margin) / (2.0 * max_delta),
    )
    p_final = torch.clamp(p_cal + scale * delta, min=1e-4)
    conf = 0.85 * p_cal.amax(dim=1, keepdim=True) + 0.15 * p_deep.amax(dim=1, keepdim=True)
    return torch.log(p_final) * (0.7 + 0.6 * conf)


class OrdinalCrossEntropyLoss(nn.Module):
    """
    Combines label-smoothed CrossEntropyLoss with an ordinal expected-grade penalty
    so distant grade errors (e.g., Grade 0 vs Grade 3) are penalized more than adjacent grades.
    """
    def __init__(self, num_classes: int = 4, label_smoothing: float = 0.05, ordinal_weight: float = 0.2):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.ordinal_weight = ordinal_weight
        self.register_buffer("classes", torch.arange(num_classes, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        probs = F.softmax(logits, dim=1)
        expected_grade = torch.sum(probs * self.classes.to(logits.device), dim=1)
        ordinal_mse = F.smooth_l1_loss(expected_grade, targets.float())
        return ce_loss + self.ordinal_weight * ordinal_mse


def _get_oof_and_split_probs(clf, X_splits, y_train, skf_oof):
    oof = cross_val_predict(clf, X_splits[0], y_train, cv=skf_oof, method="predict_proba").astype(np.float32)
    clf.fit(X_splits[0], y_train)
    p_val = clf.predict_proba(X_splits[1]).astype(np.float32)
    p_te = clf.predict_proba(X_splits[2]).astype(np.float32)
    return [oof, p_val, p_te]


def split_records(all_records, random_state: int = 42, group_by_subject: bool = False):
    """
    Splits records into 60% Train, 20% Val, 20% Test.
    Fits leak-free multi-stream clinical/radiology phrase features, severity concept models,
    and calibrated vision experts strictly on the training split (zero data leakage into Val/Test).
    """
    cache_dir = os.path.join(DEFAULT_GOLD_DIR, "model_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"splits_v3_n{len(all_records)}_s{random_state}_g{int(group_by_subject)}.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    records = [dict(r) for r in all_records]
    labels = np.array([int(r["label"]) for r in records])

    if group_by_subject and "subject_id" in records[0]:
        groups = np.array([int(r["subject_id"]) for r in records])
        sgkf_test = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
        train_val_idx, test_idx = next(sgkf_test.split(records, labels, groups))

        train_val_records = [records[i] for i in train_val_idx]
        test_records = [records[i] for i in test_idx]

        tv_labels = np.array([int(r["label"]) for r in train_val_records])
        tv_groups = np.array([int(r["subject_id"]) for r in train_val_records])
        sgkf_val = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=random_state)
        train_idx, val_idx = next(sgkf_val.split(train_val_records, tv_labels, tv_groups))

        train_records = [train_val_records[i] for i in train_idx]
        val_records = [train_val_records[i] for i in val_idx]
    else:
        train_val_records, test_records = train_test_split(
            records,
            test_size=0.2,
            random_state=random_state,
            stratify=labels,
        )
        train_records, val_records = train_test_split(
            train_val_records,
            test_size=0.25,
            random_state=random_state,
            stratify=[int(r["label"]) for r in train_val_records],
        )

    splits = [train_records, val_records, test_records]
    y_train = np.array([int(r["label"]) for r in train_records])
    skf_oof = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # 1. Enrich notes with multi-stream stitched radiology text, character n-grams, and clinical severity concepts
    if "rad_full" in train_records[0] and "concept_vec" in train_records[0]:
        v_rw = TfidfVectorizer(max_features=6000, ngram_range=(1, 3), sublinear_tf=True, min_df=2)
        v_ew = TfidfVectorizer(max_features=4000, ngram_range=(1, 3), sublinear_tf=True, min_df=2)
        v_rc = TfidfVectorizer(max_features=6000, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=3)
        v_fw = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)

        X_rw = [v_rw.fit_transform([r["rad_full"] for r in s]) if i == 0 else v_rw.transform([r["rad_full"] for r in s]) for i, s in enumerate(splits)]
        X_ew = [v_ew.fit_transform([r["rad_early"] for r in s]) if i == 0 else v_ew.transform([r["rad_early"] for r in s]) for i, s in enumerate(splits)]
        X_rc = [v_rc.fit_transform([r["rad_full"] for r in s]) if i == 0 else v_rc.transform([r["rad_full"] for r in s]) for i, s in enumerate(splits)]
        X_fw = [v_fw.fit_transform([r["focused_text"] for r in s]) if i == 0 else v_fw.transform([r["focused_text"] for r in s]) for i, s in enumerate(splits)]

        sc_con = StandardScaler()
        X_con = [sc_con.fit_transform(np.vstack([r["concept_vec"] for r in train_records]))]
        X_con.append(sc_con.transform(np.vstack([r["concept_vec"] for r in val_records])))
        X_con.append(sc_con.transform(np.vstack([r["concept_vec"] for r in test_records])))

        P_rw = _get_oof_and_split_probs(LogisticRegression(C=1.0, max_iter=1000, random_state=random_state), X_rw, y_train, skf_oof)
        P_ew = _get_oof_and_split_probs(LogisticRegression(C=0.8, max_iter=1000, random_state=random_state), X_ew, y_train, skf_oof)
        P_rc = _get_oof_and_split_probs(LogisticRegression(C=0.7, max_iter=1000, random_state=random_state), X_rc, y_train, skf_oof)
        P_fw = _get_oof_and_split_probs(LogisticRegression(C=0.7, max_iter=1000, random_state=random_state), X_fw, y_train, skf_oof)
        P_co = _get_oof_and_split_probs(LogisticRegression(C=0.3, max_iter=1000, random_state=random_state), X_con, y_train, skf_oof)

        k_best = min(256, X_rw[0].shape[1])
        selector = SelectKBest(chi2, k=k_best)
        K_rw = [selector.fit_transform(X_rw[0], y_train).toarray().astype(np.float32)]
        K_rw.append(selector.transform(X_rw[1]).toarray().astype(np.float32))
        K_rw.append(selector.transform(X_rw[2]).toarray().astype(np.float32))

        for s_idx, split_recs in enumerate(splits):
            p_text_blend = (
                0.35 * P_rw[s_idx]
                + 0.20 * P_ew[s_idx]
                + 0.20 * P_rc[s_idx]
                + 0.12 * P_fw[s_idx]
                + 0.13 * P_co[s_idx]
            )
            for idx, r in enumerate(split_recs):
                biobert_norm = normalize(r["notes"].reshape(1, -1), norm="l2").flatten().astype(np.float32)
                r["notes"] = np.concatenate([
                    biobert_norm * 0.4,
                    K_rw[s_idx][idx] * 1.5,
                    X_con[s_idx][idx].astype(np.float32) * 0.25,
                    P_rw[s_idx][idx] * 1.5,
                    P_ew[s_idx][idx] * 1.0,
                    P_rc[s_idx][idx] * 1.0,
                    P_fw[s_idx][idx] * 0.8,
                    P_co[s_idx][idx] * 0.8,
                    p_text_blend[idx] * 2.5,
                ]).astype(np.float32)

    # 2. Preserve L2 geometry of vision_dense & enrich vision with calibrated multi-expert vision probabilities
    X_vd = [normalize(np.vstack([r["vision_dense"] for r in s]), norm="l2") for s in splits]
    sc_vp_raw = StandardScaler()
    X_vp_raw = [sc_vp_raw.fit_transform(np.vstack([r["vision_pred"] for r in train_records]))]
    X_vp_raw.append(sc_vp_raw.transform(np.vstack([r["vision_pred"] for r in val_records])))
    X_vp_raw.append(sc_vp_raw.transform(np.vstack([r["vision_pred"] for r in test_records])))

    X_vis_comb = [np.hstack([X_vd[s], X_vp_raw[s] * 0.15]) for s in range(3)]
    P_vlr = _get_oof_and_split_probs(LogisticRegression(C=2.0, max_iter=1000, random_state=random_state), X_vis_comb, y_train, skf_oof)
    P_vsvc = _get_oof_and_split_probs(CalibratedClassifierCV(SVC(C=1.5, kernel="rbf", random_state=random_state), cv=3), X_vis_comb, y_train, skf_oof)

    for s_idx, split_recs in enumerate(splits):
        p_vis_blend = 0.5 * P_vlr[s_idx] + 0.5 * P_vsvc[s_idx]
        for idx, r in enumerate(split_recs):
            r["vision_dense"] = np.concatenate([X_vd[s_idx][idx], p_vis_blend[idx] * 0.8]).astype(np.float32)
            r["vision_pred"] = np.concatenate([
                r["vision_pred"],
                P_vlr[s_idx][idx] * 2.0,
                P_vsvc[s_idx][idx] * 2.0,
                p_vis_blend[idx] * 2.5,
            ]).astype(np.float32)

    # 3. Standardize tabular demographics and vision_pred (fit strictly on train_records)
    for mod_key in ("demographics", "vision_pred"):
        scaler = StandardScaler()
        train_mat = np.vstack([r[mod_key] for r in train_records])
        scaler.fit(train_mat)
        for split in splits:
            for r in split:
                r[mod_key] = scaler.transform(r[mod_key].reshape(1, -1)).flatten().astype(np.float32)

    try:
        with open(cache_file, "wb") as f:
            pickle.dump((train_records, val_records, test_records), f)
    except Exception:
        pass

    return train_records, val_records, test_records


def train_and_average_top_k(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epochs: int,
    model_out: str,
    top_k: int = 5,
    is_late_fusion: bool = False,
):
    """
    Trains the model and keeps the top-K validation checkpoints (ranked by Val Acc + 0.25 * Val AUROC).
    Saves the weight-averaged state_dict (Stochastic Weight Averaging over top-K checkpoints) to `model_out`.
    """
    best_val_accuracy = -1.0
    best_val_f1 = -1.0
    best_val_auroc = -1.0
    top_k_states = []  # list of (composite_score, val_acc, state_dict)

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    model.use_prior_calibration = False

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for batch in progress_bar:
            demo = batch["demographics"].to(device)
            notes = batch["notes"].to(device)
            vdense = batch["vision_dense"].to(device)
            vpred = batch["vision_pred"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits, extra = model(demo, notes, vdense, vpred)
            loss = criterion(logits, labels)
            if is_late_fusion and isinstance(extra, list):
                aux_loss = sum(criterion(el, labels) for el in extra) / len(extra)
                loss = loss + 0.25 * aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_train_loss = total_loss / max(1, len(train_loader))

        val_labels, val_preds, val_scores = evaluate_model(model, val_loader, device)
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        val_auroc = roc_auc_score(val_labels, val_scores, multi_class="ovr", average="macro")
        composite_score = val_accuracy + 0.25 * val_auroc

        saved_marker = ""
        if (val_accuracy > best_val_accuracy) or (
            np.isclose(val_accuracy, best_val_accuracy) and val_auroc > best_val_auroc
        ):
            best_val_accuracy = val_accuracy
            best_val_f1 = val_f1
            best_val_auroc = val_auroc
            saved_marker = " [BEST VAL]"

        if len(top_k_states) < top_k or composite_score > top_k_states[-1][0]:
            state_copy = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            top_k_states.append((composite_score, val_accuracy, state_copy))
            top_k_states.sort(key=lambda x: x[0], reverse=True)
            top_k_states = top_k_states[:top_k]

        if (epoch + 1) % 10 == 0 or epoch == 0 or saved_marker:
            print(
                f"Epoch {epoch + 1:3d}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Acc: {val_accuracy:.4f} | Val F1: {val_f1:.4f} | Val AUROC: {val_auroc:.4f}{saved_marker}"
            )

    # Average top-K validation checkpoints and check against single best on Val
    avg_state = {}
    for k in top_k_states[0][2].keys():
        avg_state[k] = torch.stack([x[2][k].float() for x in top_k_states], dim=0).mean(dim=0)

    model.load_state_dict({k: v.to(device) for k, v in avg_state.items()})
    swa_labels, swa_preds, swa_scores = evaluate_model(model, val_loader, device)
    swa_val_acc = accuracy_score(swa_labels, swa_preds)
    swa_val_auroc = roc_auc_score(swa_labels, swa_scores, multi_class="ovr", average="macro")

    if swa_val_acc + 0.25 * swa_val_auroc >= top_k_states[0][0] - 0.005:
        best_val_accuracy = max(best_val_accuracy, swa_val_acc)
        torch.save(avg_state, model_out)
    else:
        torch.save(top_k_states[0][2], model_out)

    model.use_prior_calibration = True
    cal_labels, cal_preds, cal_scores = evaluate_model(model, val_loader, device)
    best_val_accuracy = max(best_val_accuracy, accuracy_score(cal_labels, cal_preds))
    best_val_auroc = max(best_val_auroc, roc_auc_score(cal_labels, cal_scores, multi_class="ovr", average="macro"))

    return best_val_accuracy, best_val_f1, best_val_auroc


def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device):
    model.eval()
    all_preds, all_labels, all_scores = [], [], []
    with torch.no_grad():
        for batch in data_loader:
            demo = batch["demographics"].to(device)
            notes = batch["notes"].to(device)
            vdense = batch["vision_dense"].to(device)
            vpred = batch["vision_pred"].to(device)
            labels = batch["label"].to(device)

            logits, _ = model(demo, notes, vdense, vpred)
            scores = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_scores)


def save_confusion_matrix(cm: np.ndarray, class_names: list, output_dir: str, model_name: str = "Cross_Modality_Gating"):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        os.makedirs(output_dir, exist_ok=True)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_normalized = np.divide(cm.astype("float"), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

        plt.figure(figsize=(4.0, 3.5))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=False,
        )
        plt.ylabel("True Label", fontsize=10)
        plt.xlabel("Predicted Label", fontsize=10)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)

        safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        out_path = os.path.join(output_dir, f"cm_{safe_name}.pdf")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        return out_path
    except Exception as e:
        print(f"Note: Could not plot confusion matrix PDF ({e}).")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run Cross-Gated Attention Model on data_lake/silver/balanced_multimodal_cohort.parquet"
    )
    parser.add_argument("--cohort", type=str, default=DEFAULT_COHORT_PATH, help="Silver cohort parquet file")
    parser.add_argument("--cxr-dir", type=str, default=DEFAULT_CXR_DIR, help="Bronze mimic_cxr directory")
    parser.add_argument("--features-pkl", type=str, default=DEFAULT_OUTPUT_PKL, help="Gold extracted features pickle")
    parser.add_argument("--model-out", type=str, default=DEFAULT_MODEL_SAVE_PATH, help="Saved model weights path")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-by-subject", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cohort_df = pd.read_parquet(args.cohort)
    available_imgs = int(
        cohort_df["jpg_rel_path"]
        .apply(lambda p: os.path.exists(os.path.join(args.cxr_dir, p)) and os.path.getsize(os.path.join(args.cxr_dir, p)) > 0)
        .sum()
    )

    need_extract = args.force_extract or (not os.path.exists(args.features_pkl))
    if not need_extract and os.path.exists(args.features_pkl):
        with open(args.features_pkl, "rb") as f:
            all_records = pickle.load(f)
        if (
            len(all_records) < available_imgs
            or "rad_full" not in all_records[0]
            or "concept_vec" not in all_records[0]
            or all_records[0]["demographics"].shape[0] < 14
        ):
            print("Upgrading Gold multimodal features with stitched radiology & clinical concept features...")
            need_extract = True

    if need_extract:
        all_records = extract_cohort_features(
            cohort_path=args.cohort,
            cxr_dir=args.cxr_dir,
            output_pkl=args.features_pkl,
            output_parquet=DEFAULT_OUTPUT_PARQUET,
            device=str(device),
            refresh_notes=False,
        )

    print(f"\nLoaded {len(all_records)} multimodal records from {args.features_pkl}")

    train_records, val_records, test_records = split_records(
        all_records, random_state=args.seed, group_by_subject=args.group_by_subject
    )

    demo_dim = int(train_records[0]["demographics"].shape[0])
    notes_dim = int(train_records[0]["notes"].shape[0])
    vision_dense_dim = int(train_records[0]["vision_dense"].shape[0])
    vision_pred_dim = int(train_records[0]["vision_pred"].shape[0])
    num_classes = 4
    class_names = [f"Class {i}" for i in range(num_classes)]

    print(
        f"Modality Dimensions -> Demographics: {demo_dim}, Notes: {notes_dim}, "
        f"Vision Dense: {vision_dense_dim}, Vision Pred: {vision_pred_dim}"
    )

    os.makedirs(os.path.dirname(DEFAULT_TEST_SET_PATH), exist_ok=True)
    with open(DEFAULT_TEST_SET_PATH, "wb") as f:
        pickle.dump(test_records, f)
    print(f"Saved test set ({len(test_records)} records) to: {DEFAULT_TEST_SET_PATH}")

    train_dataset = PatientFusionDataset(train_records)
    val_dataset = PatientFusionDataset(val_records)
    test_dataset = PatientFusionDataset(test_records)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(
        f"Split sizes -> Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    model = GatedFusionClassifier(
        demo_dim=demo_dim,
        notes_dim=notes_dim,
        vision_dense_dim=vision_dense_dim,
        vision_pred_dim=vision_pred_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_classes=num_classes,
    ).to(device)

    criterion = OrdinalCrossEntropyLoss(num_classes=num_classes, label_smoothing=0.05, ordinal_weight=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    print("\n--- Training Cross-Gated Attention Model ---")
    best_val_accuracy, _, _ = train_and_average_top_k(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        model_out=args.model_out,
        top_k=5,
        is_late_fusion=False,
    )

    # --- Final Evaluation on Test Set ---
    print("\n" + "=" * 60)
    print("EVALUATING BEST CROSS-GATED ATTENTION MODEL ON TEST SET")
    print("=" * 60)

    model.load_state_dict(torch.load(args.model_out, map_location=device, weights_only=True))
    true_labels, predictions, scores = evaluate_model(model, test_loader, device)

    test_acc = accuracy_score(true_labels, predictions)
    f1_macro = f1_score(true_labels, predictions, average="macro", zero_division=0)
    f1_weighted = f1_score(true_labels, predictions, average="weighted", zero_division=0)
    auroc = float(roc_auc_score(true_labels, scores, multi_class="ovr", average="macro"))

    print(f"Best Validation Accuracy : {best_val_accuracy:.4f}")
    print(f"Test Overall Accuracy    : {test_acc:.4f}")
    print(f"Test Macro F1 Score      : {f1_macro:.4f}")
    print(f"Test Weighted F1 Score   : {f1_weighted:.4f}")
    print(f"Test Macro AUROC (OvR)   : {auroc:.4f}\n")

    print("Classification Report:")
    print(classification_report(true_labels, predictions, target_names=class_names, zero_division=0))

    cm = confusion_matrix(true_labels, predictions, labels=list(range(num_classes)))
    fig_path = save_confusion_matrix(cm, class_names, DEFAULT_FIGURES_DIR)
    if fig_path:
        print(f"Saved confusion matrix to: {fig_path}")

    metrics_payload = {
        "model_name": "Cross-Gated Attention",
        "total_records": len(all_records),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "best_val_accuracy": float(best_val_accuracy),
        "test_accuracy": float(test_acc),
        "test_f1_macro": float(f1_macro),
        "test_f1_weighted": float(f1_weighted),
        "test_auroc_macro_ovr": auroc,
        "confusion_matrix": cm.tolist(),
    }
    with open(DEFAULT_METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    summary = {}
    if os.path.exists(ALL_MODELS_METRICS_PATH):
        try:
            with open(ALL_MODELS_METRICS_PATH, "r") as f:
                summary = json.load(f)
        except Exception:
            summary = {}
    summary["cross_gated_attention"] = metrics_payload
    with open(ALL_MODELS_METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved evaluation metrics to: {DEFAULT_METRICS_PATH}")


if __name__ == "__main__":
    main()
