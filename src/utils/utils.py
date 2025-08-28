"""
Fonctions utilitaires pour l'API d'embeddings
"""
import io
import csv
import torch
import numpy as np
from typing import List, Union


def format_embeddings_as_csv(embeddings_tensor: torch.Tensor, dimension: int) -> str:
    """
    Convertit un tensor d'embeddings en format CSV.
    
    Args:
        embeddings_tensor: Tensor PyTorch contenant les embeddings
        dimension: Dimension des embeddings
        
    Returns:
        String contenant les données CSV
    """
    output = io.StringIO()
    writer = csv.writer(output)
    
    # En-tête avec les dimensions
    writer.writerow([f"dim_{i}" for i in range(dimension)])
    
    # Écrire les embeddings
    embeddings_list = embeddings_tensor.cpu().numpy().tolist()
    writer.writerows(embeddings_list)
    
    return output.getvalue()


def serialize_tensor(embeddings_tensor: torch.Tensor) -> bytes:
    """
    Sérialise un tensor PyTorch en bytes.
    
    Args:
        embeddings_tensor: Tensor PyTorch à sérialiser
        
    Returns:
        Bytes du tensor sérialisé
    """
    buffer = io.BytesIO()
    torch.save(embeddings_tensor, buffer)
    buffer.seek(0)
    return buffer.read()


def convert_embeddings_to_list(embeddings_tensor: torch.Tensor) -> List[List[float]]:
    """
    Convertit un tensor d'embeddings en liste Python.
    
    Args:
        embeddings_tensor: Tensor PyTorch contenant les embeddings
        
    Returns:
        Liste de listes contenant les embeddings
    """
    return embeddings_tensor.cpu().numpy().tolist()


def format_embeddings_as_csv_with_texts(texts: List[str], embeddings: Union[torch.Tensor, np.ndarray]) -> str:
    """
    Convertit des textes et leurs embeddings en format CSV.
    
    Args:
        texts: Liste des textes
        embeddings: Tensor PyTorch ou array numpy contenant les embeddings
        
    Returns:
        String contenant les données CSV avec texte et embeddings
    """
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Convertir en numpy array si c'est un tensor
    if isinstance(embeddings, torch.Tensor):
        embeddings_array = embeddings.cpu().numpy()
    else:
        embeddings_array = embeddings
    
    # En-tête avec 'text' et les dimensions
    dimension = embeddings_array.shape[1]
    header = ['text'] + [f"dim_{i}" for i in range(dimension)]
    writer.writerow(header)
    
    # Écrire les textes avec leurs embeddings
    for text, embedding in zip(texts, embeddings_array):
        row = [text] + embedding.tolist()
        writer.writerow(row)
    
    return output.getvalue()


def format_token_embeddings_as_csv(data: List) -> str:
    """
    Convertit des embeddings de tokens en format CSV.
    
    Args:
        data: Liste de TextTokenEmbeddings
        
    Returns:
        String contenant les données CSV avec texte, token et embeddings
    """
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Déterminer la dimension des embeddings
    if data and data[0].tokens:
        dimension = len(data[0].tokens[0].embedding)
    else:
        return ""
    
    # En-tête avec 'text', 'token', 'position' et les dimensions
    header = ['text', 'token', 'position'] + [f"dim_{i}" for i in range(dimension)]
    writer.writerow(header)
    
    # Écrire les données
    for text_data in data:
        for token_data in text_data.tokens:
            row = [
                text_data.text,
                token_data.token,
                token_data.position
            ] + token_data.embedding
            writer.writerow(row)
    
    return output.getvalue()