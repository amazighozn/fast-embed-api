"""
Point d'entrée principal de l'API d'embeddings
"""
from fastapi import FastAPI
from src.core import get_model
from src.api import root, list_models, encode_texts, batch_encode_endpoint, process_item_profiles
from src.api.middleware import SaveResponseMiddleware

app = FastAPI(title="Text Embeddings API", version="1.0.0")
app.add_middleware(SaveResponseMiddleware)

# Configuration des routes
app.get("/")(root)
app.get("/models")(list_models)
app.post("/encode")(encode_texts)
app.post("/batch_encode")(batch_encode_endpoint)
app.post("/process_item_profiles")(process_item_profiles)

if __name__ == "__main__":
    import uvicorn
    
    # Précharger le modèle par défaut
    print("Préchargement du modèle par défaut...")
    get_model("modernbert")
    
    # Lancer le serveur
    uvicorn.run(app, host="0.0.0.0", port=8000)