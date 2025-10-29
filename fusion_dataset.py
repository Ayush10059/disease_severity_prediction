import torch
from torch.utils.data import Dataset, DataLoader

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
