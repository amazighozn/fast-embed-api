"""
Fonctions de traitement par batch pour les embeddings
"""
from typing import List
import numpy as np
from tqdm import tqdm
from ..core import get_model


def batch_encode_large_dataset(
    texts: List[str],
    model_name: str = "modernbert",
    batch_size: int = 64,
    num_workers: int = 4
) -> np.ndarray:
    """
    Encode efficacement un grand nombre de textes (ex: 11k textes).
    
    Args:
        texts: Liste des textes à encoder
        model_name: Nom du modèle à utiliser
        batch_size: Taille des batchs
        num_workers: Nombre de workers pour le traitement parallèle (pour future implémentation)
        
    Returns:
        Array numpy contenant tous les embeddings
    """
    model = get_model(model_name)
    
    # Pour de très grandes quantités, on pourrait utiliser le multiprocessing
    # Ici on utilise une approche simple mais efficace
    embeddings = model.encode(texts, batch_size=batch_size)
    
    return embeddings.cpu().numpy()