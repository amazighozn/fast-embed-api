"""
Module core contenant la logique métier des embeddings
"""
from .embeddings import EmbeddingModel, get_model
from .config import MODEL_CONFIGS, DEFAULT_MODEL, DEFAULT_BATCH_SIZE, MAX_BATCH_SIZE, MAX_TEXT_LENGTH

__all__ = [
    'EmbeddingModel',
    'get_model',
    'MODEL_CONFIGS',
    'DEFAULT_MODEL',
    'DEFAULT_BATCH_SIZE',
    'MAX_BATCH_SIZE',
    'MAX_TEXT_LENGTH'
]