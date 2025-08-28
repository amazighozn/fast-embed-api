"""
Modèles Pydantic pour l'API
"""
from pydantic import BaseModel
from typing import List, Optional, Dict

class TextInput(BaseModel):
    texts: List[str]
    model: Optional[str] = "modernbert"
    output_format: Optional[str] = "json"  # json, csv, tensor
    batch_size: Optional[int] = 32
    include_texts: Optional[bool] = False  # Inclure les textes avec les embeddings
    token_embeddings: Optional[bool] = False  # Retourner les embeddings par token plutôt que par texte

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimension: int
    processing_time: float

class BatchEncodeResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimension: int
    num_texts: int
    processing_time: float
    texts_per_second: float

class TextWithEmbedding(BaseModel):
    text: str
    embedding: List[float]

class BatchEncodeWithTextsResponse(BaseModel):
    data: List[TextWithEmbedding]
    model: str
    dimension: int
    num_texts: int
    processing_time: float
    texts_per_second: float

class TokenEmbedding(BaseModel):
    token: str
    embedding: List[float]
    position: int

class TextTokenEmbeddings(BaseModel):
    text: str
    tokens: List[TokenEmbedding]

class TokenEmbeddingsResponse(BaseModel):
    data: List[TextTokenEmbeddings]
    model: str
    dimension: int
    num_texts: int
    total_tokens: int
    processing_time: float

class ItemProfile(BaseModel):
    item_original_id: int
    item_profile: str

class ItemProfileInput(BaseModel):
    model: Optional[str] = "modernbert"
    output_format: Optional[str] = "json"  # json, csv
    batch_size: Optional[int] = 32

class ItemEmbedding(BaseModel):
    item_original_id: int
    embedding: List[float]

class ItemProfileResponse(BaseModel):
    embeddings: List[ItemEmbedding]
    model: str
    dimension: int
    num_items: int
    processing_time: float