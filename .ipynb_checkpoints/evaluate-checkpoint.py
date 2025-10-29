import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, f1_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from fusion_dataset import *

from train import *
from train_early import *
from train_late import *
from train_gated import *
from train_cross import *
from train_cross_gated import *

def evaluate_model(model, test_loader, device):
    """Runs a model on the test set and returns predictions and true labels."""
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []
    with torch.no_grad():
        for batch in test_loader:
            demo = batch['demographics'].to(device)
            notes = batch['notes'].to(device)
            vdense = batch['vision_dense'].to(device)
            vpred = batch['vision_pred'].to(device)
            labels = batch['label'].to(device)

            logits, _ = model(demo, notes, vdense, vpred)
            scores = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
            
    return all_labels, all_preds, np.array(all_scores)

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def save_cm_for_paper(cm, class_names, model_name):
    """
    Saves a normalized, paper-ready confusion matrix as a PDF.
    """
    # --- Create a 'figures' directory if it doesn't exist ---
    output_dir = 'figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # --- Normalize the confusion matrix ---
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # --- Set up the plot ---
    plt.figure(figsize=(3.5, 3)) 
    
    heatmap = sns.heatmap(
        cm_normalized, 
        annot=True,            # Show numbers in cells
        fmt=".2f",             # Format as 2-decimal floats (e.g., 0.90)
        cmap='Blues',          # Use a simple, print-friendly colormap
        xticklabels=class_names, 
        yticklabels=class_names,
        cbar=False             # No colorbar to save space
    )
    
    # --- Set labels (use a font size that will be readable) ---
    plt.ylabel('True Label', fontsize=10)
    plt.xlabel('Predicted Label', fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    
    # --- Generate a safe filename and save as PDF ---
    safe_filename = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    output_path = os.path.join(output_dir, f"cm_{safe_filename}.pdf")
    
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    return output_path


if __name__ == '__main__':
    # --- Configuration ---
    TEST_SET_PATH = 'data/multimodal_features.pkl'
    BATCH_SIZE = 32
    
    # These dimensions MUST match the dimensions used for training
    DEMO_DIM = 10 
    NOTES_DIM = 768 
    VISION_DENSE_DIM = 1024
    VISION_PRED_DIM = 18
    HIDDEN_DIM = 128
    NUM_CLASSES = 4
    NUM_HEADS = 8
    CLASS_NAMES = [f'Class {i}' for i in range(NUM_CLASSES)] # Or replace with actual names

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Test Data ---
    print(f"Loading test data from {TEST_SET_PATH}...")
    with open(TEST_SET_PATH, 'rb') as f:
        test_records = pickle.load(f)
    
    test_dataset = PatientFusionDataset(test_records)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test data loaded. Test size: {len(test_dataset)}")

    # --- Models to Evaluate ---
    models_to_evaluate = {
        "Late Fusion": {
            "path": "models/late_fusion_model.pth",
            "class": LateFusionClassifier(DEMO_DIM, NOTES_DIM, VISION_DENSE_DIM, VISION_PRED_DIM, NUM_CLASSES)
        },
        "Early Fusion": {
            "path": "models/early_fusion_model.pth",
            "class": EarlyFusionClassifier(DEMO_DIM, NOTES_DIM, VISION_DENSE_DIM, VISION_PRED_DIM, HIDDEN_DIM, NUM_CLASSES)
        },
        "Attention Fusion": {
            "path": "models/regex_attention_model.pth",
            "class": MultimodalClassifier(DEMO_DIM, NOTES_DIM, VISION_DENSE_DIM, VISION_PRED_DIM, HIDDEN_DIM, NUM_CLASSES)
        },
        "Modality Gating": {
            "path": "models/modality_gating_model.pth",
            "class": ModalityGatingClassifier(DEMO_DIM, NOTES_DIM, VISION_DENSE_DIM, VISION_PRED_DIM, HIDDEN_DIM, NUM_CLASSES)
        },
        "Cross-Attention Fusion": {
            "path": "models/cross_attention_model.pth",
            "class": CrossAttentionFusionClassifier(DEMO_DIM, NOTES_DIM, VISION_DENSE_DIM, VISION_PRED_DIM, HIDDEN_DIM, NUM_HEADS, NUM_CLASSES)
        },
        "Cross Modality Gating": {
            "path": "models/cross_gated_attention_model.pth",
            "class": GatedFusionClassifier(DEMO_DIM, NOTES_DIM, VISION_DENSE_DIM, VISION_PRED_DIM, HIDDEN_DIM, NUM_HEADS, NUM_CLASSES)
        }
    }

    # --- Loop and Evaluate Each Model ---
    for model_name, model_info in models_to_evaluate.items():
        print("\n" + "="*50)
        print(f"EVALUATING MODEL: {model_name}")
        print("="*50)

        # Initialize and load model
        model = model_info['class'].to(device)
        try:
            model.load_state_dict(torch.load(model_info['path'], map_location=device))
        except FileNotFoundError:
            print(f"ERROR: Model file not found at {model_info['path']}. Skipping this model.")
            continue
            
        # Get predictions
        true_labels, predictions, scores = evaluate_model(model, test_loader, device)
        
        # Calculate and print metrics
        accuracy = accuracy_score(true_labels, predictions)

        # F1 SCORE
        f1_macro = f1_score(true_labels, predictions, average='macro')
        f1_weighted = f1_score(true_labels, predictions, average='weighted')
        
        print(f"Overall Accuracy: {accuracy:.4f}\n")
        print(f"Macro F1 Score: {f1_macro:.4f}")
        print(f"Weighted F1 Score: {f1_weighted:.4f}\n")

        # Calculate and print AUROC for multiclass
        try:
            # Check if all classes are present in the true labels
            if len(np.unique(true_labels)) == NUM_CLASSES:
                 auroc = roc_auc_score(true_labels, scores, multi_class='ovr', average='macro')
                 print(f"Macro AUROC (One-vs-Rest): {auroc:.4f}\n")
            else:
                 print("AUROC not calculated: not all classes were present in the test set.\n")
        except ValueError as e:
            print(f"Could not calculate AUROC: {e}\n")
        
        print("Classification Report:")
        print(classification_report(true_labels, predictions, target_names=CLASS_NAMES))
        
        # Calculate and plot confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        fig_path = save_cm_for_paper(cm, CLASS_NAMES, model_name)
        print(f"Saved confusion matrix to: {fig_path}")