import torch
import torch.nn.functional as F
import torchxrayvision as xrv
import numpy as np
import cv2
from torchvision import transforms

class XRayEmbedder:
    """
    A reusable class to efficiently generate embeddings from chest X-ray images
    using the torchxrayvision library.
    """
    def __init__(self, weights_name="densenet121-res224-chex", device="cpu"):
        """
        Initializes the embedder, loading the model and preprocessing transforms.
        This is the slow part that should only be run once.
        
        Args:
            weights_name (str): The name of the pre-trained weights to load.
            device (str): The device to run the model on ("cpu" or "cuda").
        """
        self.device = torch.device(device)
        print(f"Loading XRayEmbedder model '{weights_name}' onto {self.device}...")
        
        # 1. Load the pre-trained model
        self.model = xrv.models.DenseNet(weights=weights_name)
        self.model.to(self.device)
        self.model.eval() # Set the model to evaluation mode (very important!)

        # 2. Define the image preprocessing pipeline
        # Using the library's recommended transforms ensures consistency.
        self.transform = transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224) # Resizes, adds channel dim, and scales to [0, 1]
        ])
        
        print("XRayEmbedder initialized successfully.")

    def get_embeddings(self, img_array):
        """
        Processes a single image array and returns its embeddings.
        This is the fast part that can be called in a loop.

        Args:
            img_array (np.ndarray): The input image as a NumPy array.

        Returns:
            tuple: A tuple containing (dense_features, prediction_features)
                   or (None, None) if the input is invalid.
        """
        if not isinstance(img_array, np.ndarray):
            print("Error: Input must be a NumPy array.")
            return None, None
            
        # 1. Preprocessing
        # Ensure image is 2D grayscale and normalized to [0, 255] range
        img = xrv.datasets.normalize(img_array, 255)
        if len(img.shape) > 2:
            img = img[:, :, 0] # Select first channel if it's RGB
        if img.shape[0] != 1: # XRayResizer expects a single channel dimension
             img = img[None, :, :]

        # Apply the transforms
        img = self.transform(img)
        
        # Add the batch dimension and move to the correct device
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
        
        # 2. Inference
        with torch.no_grad(): # Disable gradient calculation for speed
            # Extract dense features from the convolutional base
            feats = self.model.features(img_tensor)
            feats = F.relu(feats, inplace=True)
            feats = F.adaptive_avg_pool2d(feats, (1, 1))
            dense_features = feats.cpu().numpy().flatten()

            # Get the final prediction probabilities
            preds = self.model(img_tensor).cpu()
            prediction_features = preds.numpy().flatten()
            
        return dense_features, prediction_features

def get_vision_embeddings(patient, embedder):
    """
    Processes all X-ray images for a patient and returns lists of their embeddings.
    
    Returns:
        tuple: A tuple containing (list_of_dense_features, list_of_prediction_features).
               Each item in the list corresponds to one image entry.
    """
    if 'imcxr' not in patient or patient['imcxr'].empty:
        # Return empty lists if no x-ray data is available.
        return [], []

    all_dense_feats = []
    all_pred_feats = []

    for img_array in patient['imcxr']['imcxr']:
        dense_feats, pred_feats = embedder.get_embeddings(img_array)
        if dense_feats is not None and pred_feats is not None:
            all_dense_feats.append(dense_feats)
            all_pred_feats.append(pred_feats)
    
    return all_dense_feats, all_pred_feats