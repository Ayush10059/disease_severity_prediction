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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)

import load_mimic
import download_images
from extract_multimodal_features import (
    DEFAULT_COHORT_PATH,
    DEFAULT_CXR_DIR,
    DEFAULT_GOLD_DIR,
    DEFAULT_OUTPUT_PKL,
    DEFAULT_OUTPUT_PARQUET,
    extract_cohort_features,
    find_radiology_csv_path,
    attach_radiology_to_cohort_df,
)
import train_cross_gated
import train_cross
import train_early
import train_gated
import train_late
from train_cross_gated import (
    GatedFusionClassifier,
    PatientFusionDataset,
    OrdinalCrossEntropyLoss,
    load_or_extract_records,
    split_records,
    train_and_average_top_k,
    evaluate_model,
    save_confusion_matrix,
    DEFAULT_TEST_SET_PATH,
    DEFAULT_FIGURES_DIR,
    ALL_MODELS_METRICS_PATH,
)
from train_cross import CrossAttentionFusionClassifier, update_all_models_summary
from train_early import EarlyFusionClassifier
from train_gated import ModalityGatingClassifier
from train_late import LateFusionClassifier


MODEL_REGISTRY = {
    "cross_gated": {
        "summary_key": "cross_gated_attention",
        "display_name": "Cross-Gated Attention",
        "fig_name": "Cross_Modality_Gating",
        "model_out": train_cross_gated.DEFAULT_MODEL_SAVE_PATH,
        "metrics_out": train_cross_gated.DEFAULT_METRICS_PATH,
        "is_late_fusion": False,
    },
    "cross": {
        "summary_key": "cross_attention",
        "display_name": "Cross-Attention",
        "fig_name": "Cross_Attention",
        "model_out": train_cross.DEFAULT_MODEL_SAVE_PATH,
        "metrics_out": train_cross.DEFAULT_METRICS_PATH,
        "is_late_fusion": False,
    },
    "early": {
        "summary_key": "early_fusion",
        "display_name": "Early Fusion",
        "fig_name": "Early_Fusion",
        "model_out": train_early.DEFAULT_MODEL_SAVE_PATH,
        "metrics_out": train_early.DEFAULT_METRICS_PATH,
        "is_late_fusion": False,
    },
    "gated": {
        "summary_key": "modality_gating",
        "display_name": "Modality Gating",
        "fig_name": "Modality_Gating",
        "model_out": train_gated.DEFAULT_MODEL_SAVE_PATH,
        "metrics_out": train_gated.DEFAULT_METRICS_PATH,
        "is_late_fusion": False,
    },
    "late": {
        "summary_key": "late_fusion",
        "display_name": "Late Fusion",
        "fig_name": "Late_Fusion",
        "model_out": train_late.DEFAULT_MODEL_SAVE_PATH,
        "metrics_out": train_late.DEFAULT_METRICS_PATH,
        "is_late_fusion": True,
    },
}


def build_model_instance(
    model_key: str,
    demo_dim: int,
    notes_dim: int,
    vision_dense_dim: int,
    vision_pred_dim: int,
    hidden_dim: int,
    num_heads: int,
    num_classes: int,
    device: torch.device,
):
    if model_key == "cross_gated":
        model = GatedFusionClassifier(
            demo_dim=demo_dim,
            notes_dim=notes_dim,
            vision_dense_dim=vision_dense_dim,
            vision_pred_dim=vision_pred_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_classes=num_classes,
        )
    elif model_key == "cross":
        model = CrossAttentionFusionClassifier(
            demo_dim=demo_dim,
            notes_dim=notes_dim,
            vision_dense_dim=vision_dense_dim,
            vision_pred_dim=vision_pred_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_classes=num_classes,
        )
    elif model_key == "early":
        model = EarlyFusionClassifier(
            demo_dim=demo_dim,
            notes_dim=notes_dim,
            vision_dense_dim=vision_dense_dim,
            vision_pred_dim=vision_pred_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
        )
    elif model_key == "gated":
        model = ModalityGatingClassifier(
            demo_dim=demo_dim,
            notes_dim=notes_dim,
            vision_dense_dim=vision_dense_dim,
            vision_pred_dim=vision_pred_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
        )
    elif model_key == "late":
        model = LateFusionClassifier(
            demo_dim=demo_dim,
            notes_dim=notes_dim,
            vision_dense_dim=vision_dense_dim,
            vision_pred_dim=vision_pred_dim,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown model key: {model_key}")
    return model.to(device)


def prepare_silver_cohort(
    cohort_path: str,
    radiology_csv: str = None,
    rebuild_cohort: bool = False,
    refresh_radiology: bool = False,
):
    """
    Ensures the Silver balanced multimodal cohort exists and attaches `radiology.csv.gz`
    reports via DuckDB whenever available on disk.
    """
    if rebuild_cohort or not os.path.exists(cohort_path):
        print("\n=== Step 1: Building Silver Balanced Multimodal Cohort ===")
        load_mimic.build_balanced_multimodal_lake(samples_per_class=None, random_seed=42)
    else:
        rad_path = find_radiology_csv_path(radiology_csv)
        df = pd.read_parquet(cohort_path)
        has_rad_col = "radiology_text" in df.columns and int(df["radiology_text"].notna().sum()) > 0
        if rad_path and (refresh_radiology or not has_rad_col):
            print("\n=== Step 1: Enriching Silver Cohort with Dedicated CXR Radiology Reports ===")
            df = attach_radiology_to_cohort_df(df, rad_path)
            df.to_parquet(cohort_path, compression="zstd", index=False)
            print(f"Saved radiology-enriched Silver cohort to: {cohort_path}")
        else:
            status = (
                f"with radiology.csv.gz ({int(df['radiology_text'].notna().sum())}/{len(df)} matched)"
                if has_rad_col
                else "with discharge.csv notes (waiting for radiology.csv.gz)"
            )
            print(f"\n=== Step 1: Silver Cohort Ready ({len(df)} studies, {status}) ===")


def run_pipeline():
    parser = argparse.ArgumentParser(
        description="Unified End-to-End Pipeline for Multimodal Pulmonary Edema Severity Prediction"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=["all", "cross_gated", "cross", "early", "gated", "late"],
        help="Models to train and evaluate (default: all 5 models)",
    )
    parser.add_argument("--cohort", type=str, default=DEFAULT_COHORT_PATH, help="Silver cohort parquet path")
    parser.add_argument("--cxr-dir", type=str, default=DEFAULT_CXR_DIR, help="Bronze mimic_cxr directory")
    parser.add_argument("--radiology-csv", type=str, default=None, help="Optional path to radiology.csv.gz or radiology.csv")
    parser.add_argument("--features-pkl", type=str, default=DEFAULT_OUTPUT_PKL, help="Gold extracted features pickle")
    parser.add_argument("--rebuild-cohort", action="store_true", help="Rebuild Silver cohort from Bronze tables")
    parser.add_argument("--refresh-radiology", action="store_true", help="Re-join radiology.csv.gz to Silver cohort")
    parser.add_argument("--download-images", action="store_true", help="Run CXR image downloader before feature extraction")
    parser.add_argument("--force-extract", action="store_true", help="Force re-extraction of Gold multimodal features")
    parser.add_argument("--extract-only", action="store_true", help="Run cohort alignment and feature extraction only")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-by-subject", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("MULTIMODAL PULMONARY EDEMA SEVERITY PREDICTION PIPELINE")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    print("=" * 72)

    # 1. Prepare Silver Cohort & Align radiology.csv.gz (if present)
    prepare_silver_cohort(
        cohort_path=args.cohort,
        radiology_csv=args.radiology_csv,
        rebuild_cohort=args.rebuild_cohort,
        refresh_radiology=args.refresh_radiology,
    )

    # 2. Optional CXR Image Download
    if args.download_images:
        print("\n=== Step 2: Checking / Downloading CXR Images ===")
        download_images.image_path()
        download_images.main()

    # 3. Extract / Load Gold Multimodal Features
    print("\n=== Step 2: Loading / Extracting Gold Multimodal Features ===")
    all_records = load_or_extract_records(
        cohort_path=args.cohort,
        cxr_dir=args.cxr_dir,
        features_pkl=args.features_pkl,
        device=str(device),
        force_extract=args.force_extract or args.refresh_radiology,
    )
    has_rad = bool(all_records[0].get("has_radiology_csv", False)) if all_records else False
    print(
        f"Loaded {len(all_records)} Gold records from {args.features_pkl} "
        f"(Dedicated CXR Radiology Reports Active: {has_rad})"
    )

    if args.extract_only:
        print("\n--extract-only specified. Stopping after Gold feature extraction.")
        return

    # 4. Split Cohort & Fit Leak-Free Multi-Stream Modality Experts Once
    print("\n=== Step 3: Preparing Leak-Free Train / Val / Test Splits ===")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
    print(f"Split Sizes -> Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")

    os.makedirs(os.path.dirname(DEFAULT_TEST_SET_PATH), exist_ok=True)
    with open(DEFAULT_TEST_SET_PATH, "wb") as f:
        pickle.dump(test_records, f)

    train_dataset = PatientFusionDataset(train_records)
    val_dataset = PatientFusionDataset(val_records)
    test_dataset = PatientFusionDataset(test_records)

    selected_models = (
        ["cross_gated", "cross", "early", "gated", "late"]
        if "all" in args.models
        else args.models
    )

    # 5. Train & Evaluate Each Selected Model
    results_table = []
    for model_key in selected_models:
        cfg = MODEL_REGISTRY[model_key]
        display_name = cfg["display_name"]
        print("\n" + "=" * 72)
        print(f"TRAINING & EVALUATING: {display_name.upper()}")
        print("=" * 72)

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        model = build_model_instance(
            model_key=model_key,
            demo_dim=demo_dim,
            notes_dim=notes_dim,
            vision_dense_dim=vision_dense_dim,
            vision_pred_dim=vision_pred_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_classes=num_classes,
            device=device,
        )

        criterion = OrdinalCrossEntropyLoss(num_classes=num_classes, label_smoothing=0.05, ordinal_weight=0.2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

        best_val_acc, _, _ = train_and_average_top_k(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=args.epochs,
            model_out=cfg["model_out"],
            top_k=5,
            is_late_fusion=cfg["is_late_fusion"],
        )

        model.load_state_dict(torch.load(cfg["model_out"], map_location=device, weights_only=True))
        true_labels, predictions, scores = evaluate_model(model, test_loader, device)

        test_acc = float(accuracy_score(true_labels, predictions))
        f1_macro = float(f1_score(true_labels, predictions, average="macro", zero_division=0))
        f1_weighted = float(f1_score(true_labels, predictions, average="weighted", zero_division=0))
        auroc = float(roc_auc_score(true_labels, scores, multi_class="ovr", average="macro"))

        print(f"\n[{display_name} Test Results]")
        print(f"Best Validation Accuracy : {best_val_acc:.4f}")
        print(f"Test Overall Accuracy    : {test_acc:.4f}")
        print(f"Test Macro F1 Score      : {f1_macro:.4f}")
        print(f"Test Weighted F1 Score   : {f1_weighted:.4f}")
        print(f"Test Macro AUROC (OvR)   : {auroc:.4f}\n")
        print(classification_report(true_labels, predictions, target_names=class_names, zero_division=0))

        cm = confusion_matrix(true_labels, predictions, labels=list(range(num_classes)))
        save_confusion_matrix(cm, class_names, DEFAULT_FIGURES_DIR, model_name=cfg["fig_name"])

        metrics_payload = {
            "model_name": display_name,
            "has_radiology_csv": has_rad,
            "total_records": len(all_records),
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
            "best_val_accuracy": float(best_val_acc),
            "test_accuracy": test_acc,
            "test_f1_macro": f1_macro,
            "test_f1_weighted": f1_weighted,
            "test_auroc_macro_ovr": auroc,
            "confusion_matrix": cm.tolist(),
        }
        if model_key == "late" and hasattr(model, "expert_logits_weight"):
            lw = F.softmax(model.expert_logits_weight.detach().cpu(), dim=0).numpy()
            metrics_payload["expert_weights"] = {
                "demographics": float(lw[0]),
                "notes": float(lw[1]),
                "vision_dense": float(lw[2]),
                "vision_pred": float(lw[3]),
            }

        with open(cfg["metrics_out"], "w") as f:
            json.dump(metrics_payload, f, indent=2)
        update_all_models_summary(cfg["summary_key"], metrics_payload)

        results_table.append({
            "Model": display_name,
            "Val Acc": f"{best_val_acc * 100:.2f}%",
            "Test Acc": f"{test_acc * 100:.2f}%",
            "Macro F1": f"{f1_macro:.4f}",
            "Weighted F1": f"{f1_weighted:.4f}",
            "Macro AUROC": f"{auroc:.4f}",
        })

    print("\n" + "=" * 72)
    print("FINAL MULTIMODAL MODEL COMPARISON SUMMARY")
    print("=" * 72)
    summary_df = pd.DataFrame(results_table)
    print(summary_df.to_string(index=False))
    print("=" * 72)
    print(f"All metrics saved to: {ALL_MODELS_METRICS_PATH}")


if __name__ == "__main__":
    run_pipeline()
