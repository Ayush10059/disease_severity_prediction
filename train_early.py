import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- 1. DEFINE THE EARLY FUSION MODEL ---

class EarlyFusionClassifier(nn.Module):
    """
    A classifier that uses early fusion.
    It concatenates all modality embeddings into a single large vector
    and then passes it through a classifier network.
    """
    def __init__(self, demo_dim, notes_dim, vision_dense_dim, vision_pred_dim, hidden_dim, num_classes):
        super(EarlyFusionClassifier, self).__init__()
        
        # Calculate the total dimension after concatenating all feature vectors
        total_input_dim = demo_dim + notes_dim + vision_dense_dim + vision_pred_dim
        
        # Define the classifier that takes the single concatenated vector as input.
        # This is a simple Multi-Layer Perceptron (MLP).
        self.classifier = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5), # Dropout is a regularization technique to prevent overfitting
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, demo_embed, notes_embed, vision_dense_embed, vision_pred_embed):
        """
        The forward pass concatenates the features and then classifies them.
        """
        # 1. Concatenate all modality features along the feature dimension (dim=1)
        fused_representation = torch.cat(
            [demo_embed, notes_embed, vision_dense_embed, vision_pred_embed], 
            dim=1
        )
        
        # 2. Pass the single fused vector through the classifier
        output_logits = self.classifier(fused_representation)
        
        # We return `None` for the second value to maintain a consistent
        # return signature with the attention model, making the training loop compatible.
        return output_logits, None


# --- 2. CREATE A CUSTOM PYTORCH DATASET (Unchanged) ---

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
    # IMPORTANT: Changed the model save path to reflect the new model type
    MODEL_SAVE_PATH = 'models/early_fusion_model.pth'
    
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 150
    
    # These dimensions MUST match the output of your embedding functions
    DEMO_DIM = 10 
    NOTES_DIM = 768 
    VISION_DENSE_DIM = 1024
    VISION_PRED_DIM = 18
    HIDDEN_DIM = 128 # The hidden dimension for the classifier MLP
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
    # The only change here is instantiating our new EarlyFusionClassifier
    model = EarlyFusionClassifier(
        demo_dim=DEMO_DIM,
        notes_dim=NOTES_DIM,
        vision_dense_dim=VISION_DENSE_DIM,
        vision_pred_dim=VISION_PRED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop (Unchanged logic) ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for batch in progress_bar:
            demo = batch['demographics'].to(device)
            notes = batch['notes'].to(device)
            vdense = batch['vision_dense'].to(device)
            vpred = batch['vision_pred'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits, _ = model(demo, notes, vdense, vpred) # The call remains the same
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Average Training Loss: {avg_train_loss:.4f}")

        # --- Validation Loop (Unchanged logic) ---
        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                demo = batch['demographics'].to(device)
                notes = batch['notes'].to(device)
                vdense = batch['vision_dense'].to(device)
                vpred = batch['vision_pred'].to(device)
                labels = batch['label'].to(device)
                
                logits, _ = model(demo, notes, vdense, vpred)
                loss = criterion(logits, labels)
                total_val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                correct_predictions += (preds == labels).sum().item()
                total_samples += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = correct_predictions / total_samples
        print(f"Epoch {epoch+1} | Validation Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.4f}")

        # --- SAVE THE BEST MODEL (Unchanged logic) ---
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved to {MODEL_SAVE_PATH} with accuracy: {accuracy:.4f}")
