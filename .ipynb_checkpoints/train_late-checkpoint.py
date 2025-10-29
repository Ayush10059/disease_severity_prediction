import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from fusion_dataset import *

class LateFusionClassifier(nn.Module):
    """
    A classifier that uses late fusion (decision-level fusion).
    It trains a separate "expert" classifier for each modality and then
    averages their output logits to make a final prediction.
    """
    def __init__(self, demo_dim, notes_dim, vision_dense_dim, vision_pred_dim, num_classes):
        super(LateFusionClassifier, self).__init__()
        
        # Create a separate, simple linear classifier for each modality.
        # Each expert maps its specific input dimension directly to the number of classes.
        self.demo_classifier = nn.Linear(demo_dim, num_classes)
        self.notes_classifier = nn.Linear(notes_dim, num_classes)
        self.vision_dense_classifier = nn.Linear(vision_dense_dim, num_classes)
        self.vision_pred_classifier = nn.Linear(vision_pred_dim, num_classes)

    def forward(self, demo_embed, notes_embed, vision_dense_embed, vision_pred_embed):
        """
        The forward pass gets predictions from each expert and averages them.
        """
        # 1. Get the output logits from each modality-specific classifier
        logits_d = self.demo_classifier(demo_embed)
        logits_n = self.notes_classifier(notes_embed)
        logits_v1 = self.vision_dense_classifier(vision_dense_embed)
        logits_v2 = self.vision_pred_classifier(vision_pred_embed)

        # 2. Fuse the predictions by taking the average of the logits
        # This is a simple but effective late fusion strategy.
        final_logits = (logits_d + logits_n + logits_v1 + logits_v2) / 4.0
        
        # We return `None` for the second value to maintain a consistent
        # return signature, making the training loop compatible.
        return final_logits, None


if __name__ == '__main__':
    # --- Configuration ---
    PREPROCESSED_DATA_PATH = 'data/multimodal_features.pkl'
    MODEL_SAVE_PATH = 'models/late_fusion_model.pth'
    
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 150
    
    # These dimensions MUST match the output of your embedding functions
    DEMO_DIM = 10 
    NOTES_DIM = 768 
    VISION_DENSE_DIM = 1024
    VISION_PRED_DIM = 18
    NUM_CLASSES = 4
    # Note: HIDDEN_DIM is not needed for this model architecture

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
    # The only change here is instantiating our new LateFusionClassifier
    model = LateFusionClassifier(
        demo_dim=DEMO_DIM,
        notes_dim=NOTES_DIM,
        vision_dense_dim=VISION_DENSE_DIM,
        vision_pred_dim=VISION_PRED_DIM,
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
            logits, _ = model(demo, notes, vdense, vpred)
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
