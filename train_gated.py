import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from fusion_dataset import *

class ModalityGatingClassifier(nn.Module):
    """
    A classifier that uses a gating mechanism to fuse modalities.
    It learns to scale the contribution of each modality independently.
    """
    def __init__(self, demo_dim, notes_dim, vision_dense_dim, vision_pred_dim, hidden_dim, num_classes):
        super(ModalityGatingClassifier, self).__init__()
        self.num_modalities = 4
        
        # --- Projection Layers (to create a common hidden dimension) ---
        self.project_demo = nn.Linear(demo_dim, hidden_dim)
        self.project_notes = nn.Linear(notes_dim, hidden_dim)
        self.project_vision_dense = nn.Linear(vision_dense_dim, hidden_dim)
        self.project_vision_pred = nn.Linear(vision_pred_dim, hidden_dim)

        # --- Gating Network ---
        # Takes the concatenated projected features and outputs a gate value for each modality
        self.gating_network = nn.Sequential(
            nn.Linear(hidden_dim * self.num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_modalities),
        )

        # --- Final Classifier ---
        # Input will be the concatenation of the gated modality vectors
        classifier_input_dim = hidden_dim * self.num_modalities
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, demo_embed, notes_embed, vision_dense_embed, vision_pred_embed):
        # 1. Project all modalities to the common hidden dimension
        proj_d = F.relu(self.project_demo(demo_embed))
        proj_n = F.relu(self.project_notes(notes_embed))
        proj_v1 = F.relu(self.project_vision_dense(vision_dense_embed))
        proj_v2 = F.relu(self.project_vision_pred(vision_pred_embed))

        # 2. Calculate gate values
        concat_features = torch.cat([proj_d, proj_n, proj_v1, proj_v2], dim=1)
        gate_logits = self.gating_network(concat_features)
        gates = torch.sigmoid(gate_logits) # Use sigmoid for independent gates (0 to 1)

        # 3. Apply gates to each projected modality
        # unsqueeze(-1) allows broadcasting the gate value across the feature dimension
        gated_d = proj_d * gates[:, 0].unsqueeze(-1)
        gated_n = proj_n * gates[:, 1].unsqueeze(-1)
        gated_v1 = proj_v1 * gates[:, 2].unsqueeze(-1)
        gated_v2 = proj_v2 * gates[:, 3].unsqueeze(-1)
        
        # 4. Concatenate the gated vectors for final classification
        fused_vector = torch.cat([gated_d, gated_n, gated_v1, gated_v2], dim=1)
        
        # 5. Pass through the final classifier
        output_logits = self.classifier(fused_vector)
        
        # We can return 'gates' instead of None if we want to inspect them during evaluation
        return output_logits, gates

if __name__ == '__main__':
    # --- Configuration ---
    PREPROCESSED_DATA_PATH = 'data/multimodal_features.pkl'
    MODEL_SAVE_PATH = 'models/modality_gating_model.pth'
    
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 150
    
    # Dimensions MUST match the output of your embedding functions
    DEMO_DIM = 10 
    NOTES_DIM = 768 
    VISION_DENSE_DIM = 1024
    VISION_PRED_DIM = 18
    HIDDEN_DIM = 128
    NUM_CLASSES = 4

    best_val_accuracy = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load and Split Data (Unchanged) ---
    print(f"Loading data from {PREPROCESSED_DATA_PATH}...")
    with open(PREPROCESSED_DATA_PATH, 'rb') as f:
        all_records = pickle.load(f)
    train_val_records, test_records = train_test_split(all_records, test_size=0.2, random_state=42, stratify=[r['label'] for r in all_records])
    TEST_SET_PATH = 'data/multimodal_test_set.pkl'
    with open(TEST_SET_PATH, 'wb') as f:
        pickle.dump(test_records, f)
    print(f"Test set saved to {TEST_SET_PATH}")
    train_records, val_records = train_test_split(train_val_records, test_size=0.25, random_state=42, stratify=[r['label'] for r in train_val_records])
    train_dataset = PatientFusionDataset(train_records)
    val_dataset = PatientFusionDataset(val_records)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Data loaded. Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # --- Initialize Model, Loss, Optimizer ---
    model = ModalityGatingClassifier(
        demo_dim=DEMO_DIM,
        notes_dim=NOTES_DIM,
        vision_dense_dim=VISION_DENSE_DIM,
        vision_pred_dim=VISION_PRED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training & Validation Loops (Unchanged logic) ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch in progress_bar:
            demo, notes, vdense, vpred, labels = [b.to(device) for b in batch.values()]
            optimizer.zero_grad()
            logits, _ = model(demo, notes, vdense, vpred)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Average Training Loss: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss, correct_predictions, total_samples = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                demo, notes, vdense, vpred, labels = [b.to(device) for b in batch.values()]
                logits, _ = model(demo, notes, vdense, vpred)
                loss = criterion(logits, labels)
                total_val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct_predictions += (preds == labels).sum().item()
                total_samples += labels.size(0)
        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = correct_predictions / total_samples
        print(f"Epoch {epoch+1} | Validation Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.4f}")

        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved to {MODEL_SAVE_PATH} with accuracy: {accuracy:.4f}")
