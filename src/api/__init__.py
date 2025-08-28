"""
Module API contenant les endpoints et modèles FastAPI
"""
from .models import TextInput, EmbeddingResponse, BatchEncodeResponse
from .endpoints import root, list_models, encode_texts, batch_encode_endpoint, process_item_profiles

__all__ = [
    'TextInput',
    'EmbeddingResponse',
    'BatchEncodeResponse',
    'root',
    'list_models',
    'encode_texts',
    'batch_encode_endpoint',
    'process_item_profiles'
]