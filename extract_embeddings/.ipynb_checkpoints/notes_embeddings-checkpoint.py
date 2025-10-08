import numpy as np
import torch

def get_biobert_embeddings(text, tokenizer, model):
    """
    Helper function to get BioBERT embeddings for a single text chunk.
    This assumes text is already less than 512 tokens.
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Tokenize and create tensors, moving them to the correct device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Get model output without calculating gradients
    with torch.no_grad():
        outputs = model(**inputs)

    # Use the [CLS] token's embedding as the sentence representation
    cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    return cls_embedding, None # Returning None for the second value to match the call signature

def get_all_notes_embeddings(patient, biobert_tokenizer, biobert_model):
    """
    Aggregates text from 'dsnotes' and 'radnotes', handles long
    texts by creating overlapping chunks, and averages their embeddings.
    """
    # 1. Define all possible note sources
    note_keys = ['dsnotes', 'radnotes']
    all_notes_text = []

    # 2. Loop through the sources and collect all available text
    for key in note_keys:
        if key in patient and not patient[key].empty:
            all_notes_text.extend(patient[key]['text'].fillna('').tolist())

    # 3. If no text was found, return a single zero vector for the patient
    if not all_notes_text:
        return np.zeros((1, 768))

    # 4. Join all collected notes into a single string
    combined_text = "\n\n--- NOTE SEPARATOR ---\n\n".join(all_notes_text)
    
    if not combined_text or not combined_text.strip():
        return np.zeros((1, 768))

    # 5. Tokenize the entire text without truncation to check its length
    tokens = biobert_tokenizer.encode(combined_text)
    
    # 6. Process using the chunking strategy
    chunk_size = 512
    overlap = 50
    
    if len(tokens) <= chunk_size:
        # If the text is short enough, process it directly
        embeddings, _ = get_biobert_embeddings(combined_text, biobert_tokenizer, biobert_model)
        return embeddings

    # Otherwise, create and process chunks
    all_chunk_embeddings = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = biobert_tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        
        chunk_embedding, _ = get_biobert_embeddings(chunk_text, biobert_tokenizer, biobert_model)
        
        if chunk_embedding is not None:
            all_chunk_embeddings.append(chunk_embedding)
    
    if not all_chunk_embeddings:
        return np.zeros((1, 768))
        
    # 7. Average the embeddings of all chunks to get a single representative vector
    final_embedding = np.mean(np.vstack(all_chunk_embeddings), axis=0, keepdims=True)
    
    return final_embedding