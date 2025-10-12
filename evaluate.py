import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. COPY ALL MODEL AND DATASET DEFINITIONS ---
# We need to define the classes so PyTorch can load the saved models.

# --- Attention Fusion Model Classes ---
class AttentionFusion(nn.Module):
    def __init__(self, demo_dim, notes_dim, vision_dense_dim, vision_pred_dim, hidden_dim, num_modalities=4):
        super(AttentionFusion, self).__init__()
        self.num_modalities = num_modalities
        self.hidden_dim = hidden_dim
        self.project_demo = nn.Linear(demo_dim, hidden_dim)
        self.project_notes = nn.Linear(notes_dim, hidden_dim)
        self.project_vision_dense = nn.Linear(vision_dense_dim, hidden_dim)
        self.project_vision_pred = nn.Linear(vision_pred_dim, hidden_dim)
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modalities)
        )

    def forward(self, demo_embed, notes_embed, vision_dense_embed, vision_pred_embed):
        proj_d = F.relu(self.project_demo(demo_embed))
        proj_n = F.relu(self.project_notes(notes_embed))
        proj_v1 = F.relu(self.project_vision_dense(vision_dense_embed))
        proj_v2 = F.relu(self.project_vision_pred(vision_pred_embed))
        concat_features = torch.cat([proj_d, proj_n, proj_v1, proj_v2], dim=1)
        attention_logits = self.attention_net(concat_features)
        attention_weights = F.softmax(attention_logits, dim=1)
        projected_modalities = torch.stack([proj_d, proj_n, proj_v1, proj_v2], dim=1)
        weights = attention_weights.unsqueeze(-1)
        fused_vector = torch.sum(weights * projected_modalities, dim=1)
        return fused_vector, attention_weights

class MultimodalClassifier(nn.Module):
    def __init__(self, demo_dim, notes_dim, vision_dense_dim, vision_pred_dim, hidden_dim, num_classes):
        super(MultimodalClassifier, self).__init__()
        self.fusion_module = AttentionFusion(
            demo_dim, notes_dim, vision_dense_dim, vision_pred_dim, hidden_dim
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, demo_embed, notes_embed, vision_dense_embed, vision_pred_embed):
        fused_representation, attention_weights = self.fusion_module(
            demo_embed, notes_embed, vision_dense_embed, vision_pred_embed
        )
        output_logits = self.classifier(fused_representation)
        return output_logits, attention_weights

# --- Early Fusion Model Class ---
class EarlyFusionClassifier(nn.Module):
    def __init__(self, demo_dim, notes_dim, vision_dense_dim, vision_pred_dim, hidden_dim, num_classes):
        super(EarlyFusionClassifier, self).__init__()
        total_input_dim = demo_dim + notes_dim + vision_dense_dim + vision_pred_dim
        self.classifier = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, demo_embed, notes_embed, vision_dense_embed, vision_pred_embed):
        fused_representation = torch.cat(
            [demo_embed, notes_embed, vision_dense_embed, vision_pred_embed], dim=1
        )
        output_logits = self.classifier(fused_representation)
        return output_logits, None

# --- Late Fusion Model Class ---
class LateFusionClassifier(nn.Module):
    def __init__(self, demo_dim, notes_dim, vision_dense_dim, vision_pred_dim, num_classes):
        super(LateFusionClassifier, self).__init__()
        self.demo_classifier = nn.Linear(demo_dim, num_classes)
        self.notes_classifier = nn.Linear(notes_dim, num_classes)
        self.vision_dense_classifier = nn.Linear(vision_dense_dim, num_classes)
        self.vision_pred_classifier = nn.Linear(vision_pred_dim, num_classes)
    def forward(self, demo_embed, notes_embed, vision_dense_embed, vision_pred_embed):
        logits_d = self.demo_classifier(demo_embed)
        logits_n = self.notes_classifier(notes_embed)
        logits_v1 = self.vision_dense_classifier(vision_dense_embed)
        logits_v2 = self.vision_pred_classifier(vision_pred_embed)
        final_logits = (logits_d + logits_n + logits_v1 + logits_v2) / 4.0
        return final_logits, None

# --- Dataset Class (Unchanged) ---
class PatientFusionDataset(Dataset):
    def __init__(self, data_records):
        self.records = data_records
    def __len__(self):
        return len(self.records)
    def __getitem__(self, idx):
        record = self.records[idx]
        return {
            'demographics': torch.tensor(record['demographics'], dtype=torch.float32),
            'notes': torch.tensor(record['notes'], dtype=torch.float32),
            'vision_dense': torch.tensor(record['vision_dense'], dtype=torch.float32),
            'vision_pred': torch.tensor(record['vision_pred'], dtype=torch.float32),
            'label': torch.tensor(record['label'], dtype=torch.long)
        }

# --- 2. THE MAIN EVALUATION SCRIPT ---

def evaluate_model(model, test_loader, device):
    """Runs a model on the test set and returns predictions and true labels."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            demo = batch['demographics'].to(device)
            notes = batch['notes'].to(device)
            vdense = batch['vision_dense'].to(device)
            vpred = batch['vision_pred'].to(device)
            labels = batch['label'].to(device)

            logits, _ = model(demo, notes, vdense, vpred)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds

def plot_confusion_matrix(cm, class_names, title):
    """Plots a confusion matrix using seaborn."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


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
        "Attention Fusion": {
            "path": "models/regex_attention_model.pth",
            "class": MultimodalClassifier(DEMO_DIM, NOTES_DIM, VISION_DENSE_DIM, VISION_PRED_DIM, HIDDEN_DIM, NUM_CLASSES)
        },
        "Early Fusion": {
            "path": "models/early_fusion_model.pth",
            "class": EarlyFusionClassifier(DEMO_DIM, NOTES_DIM, VISION_DENSE_DIM, VISION_PRED_DIM, HIDDEN_DIM, NUM_CLASSES)
        },
        "Late Fusion": {
            "path": "models/late_fusion_model.pth",
            "class": LateFusionClassifier(DEMO_DIM, NOTES_DIM, VISION_DENSE_DIM, VISION_PRED_DIM, NUM_CLASSES)
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
        true_labels, predictions = evaluate_model(model, test_loader, device)
        
        # Calculate and print metrics
        accuracy = accuracy_score(true_labels, predictions)
        print(f"Overall Accuracy: {accuracy:.4f}\n")
        
        print("Classification Report:")
        print(classification_report(true_labels, predictions, target_names=CLASS_NAMES))
        
        # Calculate and plot confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        plot_confusion_matrix(cm, CLASS_NAMES, f'Confusion Matrix - {model_name}')
