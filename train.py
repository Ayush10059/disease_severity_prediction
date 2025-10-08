import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- 1. COPY THE MODEL DEFINITIONS FROM THE PREVIOUS ANSWER ---

class AttentionFusion(nn.Module):
    """An attention module to learn a weighted fusion of different modality embeddings."""
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


# --- 2. CREATE A CUSTOM PYTORCH DATASET ---

class PatientFusionDataset(Dataset):
    """Dataset to load the pre-processed multimodal data."""
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

# --- 3. THE MAIN TRAINING AND EVALUATION SCRIPT ---

if __name__ == '__main__':
    # --- Configuration ---
    PREPROCESSED_DATA_PATH = 'data/multimodal_features.pkl'
    MODEL_SAVE_PATH = 'models/best_attention_model.pth'
    
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 150
    
    # IMPORTANT: These dimensions MUST match the output of your embedding functions
    DEMO_DIM = 10 # Replace with your actual dimension
    NOTES_DIM = 768 # BioBERT base
    VISION_DENSE_DIM = 1024 # Example: ResNet avgpool
    VISION_PRED_DIM = 18 # Example: CheXpert predictions
    HIDDEN_DIM = 128 # The common projection dimension
    NUM_CLASSES = 4

    best_val_accuracy = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load and Split Data ---
    print(f"Loading data from {PREPROCESSED_DATA_PATH}...")
    with open(PREPROCESSED_DATA_PATH, 'rb') as f:
        all_records = pickle.load(f)

    # First, split into training+validation (80%) and test (20%)
    train_val_records, test_records = train_test_split(all_records, test_size=0.2, random_state=42)

    # Save the test set for later
    TEST_SET_PATH = 'data/multimodal_test_set.pkl'
    with open(TEST_SET_PATH, 'wb') as f:
        pickle.dump(test_records, f)
    print(f"Test set saved to {TEST_SET_PATH}")

    # Now, split the remaining data into training (80% of original) and validation (20% of original)
    # The new test_size should be 0.25 (since 0.25 * 0.8 = 0.2)
    train_records, val_records = train_test_split(train_val_records, test_size=0.25, random_state=42)
    
    train_dataset = PatientFusionDataset(train_records)
    val_dataset = PatientFusionDataset(val_records)
    
    train_dataset = PatientFusionDataset(train_records)
    val_dataset = PatientFusionDataset(val_records)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Data loaded. Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # --- Initialize Model, Loss, Optimizer ---
    model = MultimodalClassifier(
        demo_dim=DEMO_DIM,
        notes_dim=NOTES_DIM,
        vision_dense_dim=VISION_DENSE_DIM,
        vision_pred_dim=VISION_PRED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for batch in progress_bar:
            # Move data to the correct device
            demo = batch['demographics'].to(device)
            notes = batch['notes'].to(device)
            vdense = batch['vision_dense'].to(device)
            vpred = batch['vision_pred'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits, _ = model(demo, notes, vdense, vpred)
            loss = criterion(logits, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Average Training Loss: {avg_train_loss:.4f}")

        # --- Validation Loop ---
        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        with torch.no_grad():
            for batch in val_loader:
                # ... (the inside of the validation loop remains the same)
                
                preds = torch.argmax(logits, dim=1)
                correct_predictions += (preds == labels).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = correct_predictions / len(val_dataset)
        print(f"Epoch {epoch+1} | Validation Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.4f}")

        # --- SAVE THE BEST MODEL ---
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved to {MODEL_SAVE_PATH} with accuracy: {accuracy:.4f}")

        # You can also inspect `attention_weights` here to see what the model is focusing on