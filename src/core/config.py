"""
Configuration pour les modèles d'embeddings
"""

MODEL_CONFIGS = {
    "modernbert": {
        "name": "answerdotai/ModernBERT-base",
        "dimension": 768  # Dimension pour ModernBERT base
    },
    "qwen3": {
        "name": "Qwen/Qwen3-Embedding-0.6B",
        "dimension": 1024  # Dimension pour Qwen3 Embedding
    }
}

DEFAULT_MODEL = "modernbert"
DEFAULT_BATCH_SIZE = 32
MAX_BATCH_SIZE = 128
MAX_TEXT_LENGTH = 512