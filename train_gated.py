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
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)

from extract_multimodal_features import (
    DEFAULT_COHORT_PATH,
    DEFAULT_CXR_DIR,
    DEFAULT_GOLD_DIR,
    DEFAULT_OUTPUT_PKL,
    DEFAULT_OUTPUT_PARQUET,
    extract_cohort_features,
)
from train_cross_gated import (
    PatientFusionDataset,
    OrdinalCrossEntropyLoss,
    calibrate_multimodal_logits,
    load_or_extract_records,
    split_records,
    train_and_average_top_k,
    evaluate_model,
    save_confusion_matrix,
)
from train_cross import update_all_models_summary


DEFAULT_MODEL_SAVE_PATH = os.path.join(DEFAULT_GOLD_DIR, "models", "modality_gating_model.pth")
DEFAULT_FIGURES_DIR = os.path.join(DEFAULT_GOLD_DIR, "figures")
DEFAULT_METRICS_PATH = os.path.join(DEFAULT_GOLD_DIR, "evaluation_metrics_gated.json")


class ModalityGatingClassifier(nn.Module):
    """
    Classifier that uses a Modality Gating mechanism to fuse modalities.
    Projects each modality to a shared hidden dimension and learns sample-specific
    gate weights (0 to 1) to scale each modality's contribution independently.
    """
    def __init__(
        self,
        demo_dim: int,
        notes_dim: int,
        vision_dense_dim: int,
        vision_pred_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.4,
    ):
        super(ModalityGatingClassifier, self).__init__()
        self.num_modalities = 4

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

        classifier_input_dim = hidden_dim * self.num_modalities
        self.gating_network = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_modalities),
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

        concat_features = torch.cat([proj_d, proj_n, proj_v1, proj_v2], dim=1)
        gate_logits = self.gating_network(concat_features)
        gates = torch.sigmoid(gate_logits)

        gated_d = proj_d * gates[:, 0].unsqueeze(-1)
        gated_n = proj_n * gates[:, 1].unsqueeze(-1)
        gated_v1 = proj_v1 * gates[:, 2].unsqueeze(-1)
        gated_v2 = proj_v2 * gates[:, 3].unsqueeze(-1)

        fused_vector = torch.cat([gated_d, gated_n, gated_v1, gated_v2], dim=1)
        output_logits = self.classifier(fused_vector) + 0.5 * self.residual_head(fused_vector)
        if not self.training and getattr(self, "use_prior_calibration", True):
            output_logits = calibrate_multimodal_logits(
                output_logits, notes_embed, vision_dense_embed, vision_weight=0.21, cap_ratio=1.02
            )
        return output_logits, gates


def main():
    parser = argparse.ArgumentParser(
        description="Run Modality Gating Model on data_lake/silver/balanced_multimodal_cohort.parquet"
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-by-subject", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_records = load_or_extract_records(
        cohort_path=args.cohort,
        cxr_dir=args.cxr_dir,
        features_pkl=args.features_pkl,
        device=str(device),
        force_extract=args.force_extract,
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

    train_dataset = PatientFusionDataset(train_records)
    val_dataset = PatientFusionDataset(val_records)
    test_dataset = PatientFusionDataset(test_records)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = ModalityGatingClassifier(
        demo_dim=demo_dim,
        notes_dim=notes_dim,
        vision_dense_dim=vision_dense_dim,
        vision_pred_dim=vision_pred_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
    ).to(device)

    criterion = OrdinalCrossEntropyLoss(num_classes=num_classes, label_smoothing=0.05, ordinal_weight=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    print("\n--- Training Modality Gating Model ---")
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

    print("\n" + "=" * 60)
    print("EVALUATING BEST MODALITY GATING MODEL ON TEST SET")
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
    fig_path = save_confusion_matrix(cm, class_names, DEFAULT_FIGURES_DIR, model_name="Modality_Gating")
    if fig_path:
        print(f"Saved confusion matrix to: {fig_path}")

    metrics_payload = {
        "model_name": "Modality Gating",
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
    update_all_models_summary("modality_gating", metrics_payload)
    print(f"Saved evaluation metrics to: {DEFAULT_METRICS_PATH}")


if __name__ == "__main__":
    main()
