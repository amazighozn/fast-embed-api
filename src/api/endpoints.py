"""
Endpoints FastAPI pour l'API d'embeddings
"""
from fastapi import HTTPException, File, UploadFile, Form
from fastapi.responses import Response
import time
import json
import pandas as pd
from typing import Optional
from tqdm import tqdm
from .models import TextInput, EmbeddingResponse, BatchEncodeResponse, TextWithEmbedding, BatchEncodeWithTextsResponse, TokenEmbedding, TextTokenEmbeddings, TokenEmbeddingsResponse, ItemProfile, ItemProfileInput, ItemEmbedding, ItemProfileResponse
from ..core import get_model, MODEL_CONFIGS
from ..utils import batch_encode_large_dataset, format_embeddings_as_csv, serialize_tensor, convert_embeddings_to_list, format_embeddings_as_csv_with_texts, format_token_embeddings_as_csv


async def root():
    """Endpoint racine avec informations sur l'API."""
    return {
        "message": "API d'embeddings de texte",
        "endpoints": {
            "/encode": "POST - Encode des textes en embeddings",
            "/models": "GET - Liste des modèles disponibles",
            "/batch_encode": "POST - Encode un grand nombre de textes efficacement",
            "/process_item_profiles": "POST - Process JSON file with item profiles"
        }
    }


async def list_models():
    """Liste les modèles disponibles avec leurs configurations."""
    return {
        "models": MODEL_CONFIGS,
        "default": "modernbert"
    }


async def encode_texts(input_data: TextInput):
    """
    Encode une liste de textes en embeddings.
    
    Retourne les embeddings au format spécifié (json, csv, ou tensor).
    """
    try:
        start_time = time.time()
        
        # Obtenir le modèle
        model = get_model(input_data.model)
        
        # Encoder les textes
        embeddings_tensor = model.encode(input_data.texts, batch_size=input_data.batch_size)
        
        processing_time = time.time() - start_time
        
        # Formater la sortie selon le format demandé
        if input_data.output_format == "tensor":
            # Retourner directement le tensor PyTorch sérialisé
            content = serialize_tensor(embeddings_tensor)
            return Response(
                content=content,
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": "attachment; filename=embeddings.pt",
                    "X-Model": input_data.model,
                    "X-Dimension": str(model.dimension),
                    "X-Processing-Time": str(processing_time)
                }
            )
        
        elif input_data.output_format == "csv":
            # Convertir en CSV
            csv_content = format_embeddings_as_csv(embeddings_tensor, model.dimension)
            return Response(
                content=csv_content,
                media_type="text/csv",
                headers={
                    "Content-Disposition": "attachment; filename=embeddings.csv",
                    "X-Model": input_data.model,
                    "X-Dimension": str(model.dimension),
                    "X-Processing-Time": str(processing_time)
                }
            )
        
        else:  # Format JSON par défaut
            embeddings_list = convert_embeddings_to_list(embeddings_tensor)
            return EmbeddingResponse(
                embeddings=embeddings_list,
                model=input_data.model,
                dimension=model.dimension,
                processing_time=processing_time
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def batch_encode_endpoint(input_data: TextInput):
    """
    Endpoint optimisé pour encoder un grand nombre de textes.
    Utilise la fonction batch_encode_large_dataset pour de meilleures performances.
    """
    try:
        start_time = time.time()
        
        # Utiliser la fonction optimisée
        embeddings_array = batch_encode_large_dataset(
            texts=input_data.texts,
            model_name=input_data.model,
            batch_size=input_data.batch_size or 64
        )
        
        processing_time = time.time() - start_time
        
        # Statistiques de performance
        texts_per_second = len(input_data.texts) / processing_time
        
        # Si l'utilisateur veut les embeddings par token
        if input_data.token_embeddings:
            # Obtenir le modèle
            model = get_model(input_data.model)
            # Utiliser la méthode encode_with_tokens
            results = model.encode_with_tokens(input_data.texts, batch_size=input_data.batch_size or 64)
            
            # Convertir en format approprié
            data = []
            total_tokens = 0
            
            for result in tqdm(results, desc="Processing token results", unit="text"):
                tokens = []
                for token_info in result['tokens']:
                    tokens.append(TokenEmbedding(
                        token=token_info['token'],
                        embedding=token_info['embedding'],
                        position=token_info['position']
                    ))
                    total_tokens += 1
                
                data.append(TextTokenEmbeddings(
                    text=result['text'],
                    tokens=tokens
                ))
            
            if input_data.output_format == "csv":
                # CSV avec tokens et embeddings
                csv_content = format_token_embeddings_as_csv(data)
                return Response(
                    content=csv_content,
                    media_type="text/csv",
                    headers={
                        "Content-Disposition": "attachment; filename=token_embeddings.csv",
                        "X-Model": input_data.model,
                        "X-Total-Tokens": str(total_tokens),
                        "X-Processing-Time": str(processing_time)
                    }
                )
            else:  # JSON
                return TokenEmbeddingsResponse(
                    data=data,
                    model=input_data.model,
                    dimension=model.dimension,
                    num_texts=len(input_data.texts),
                    total_tokens=total_tokens,
                    processing_time=processing_time
                )
        
        # Si l'utilisateur veut inclure les textes avec les embeddings (comportement existant)
        elif input_data.include_texts:
            # Formater la sortie selon le format demandé
            if input_data.output_format == "csv":
                # Créer un CSV avec texte et embeddings
                csv_content = format_embeddings_as_csv_with_texts(
                    texts=input_data.texts,
                    embeddings=embeddings_array
                )
                return Response(
                    content=csv_content,
                    media_type="text/csv",
                    headers={
                        "Content-Disposition": "attachment; filename=text_embeddings.csv",
                        "X-Model": input_data.model,
                        "X-Dimension": str(embeddings_array.shape[1]),
                        "X-Processing-Time": str(processing_time)
                    }
                )
            else:  # Format JSON par défaut
                # Créer une liste de TextWithEmbedding
                data = []
                for text, embedding in tqdm(zip(input_data.texts, embeddings_array.tolist()), desc="Creating text-embedding pairs", unit="pair", total=len(input_data.texts)):
                    data.append(TextWithEmbedding(text=text, embedding=embedding))
                
                return BatchEncodeWithTextsResponse(
                    data=data,
                    model=input_data.model,
                    dimension=embeddings_array.shape[1],
                    num_texts=len(input_data.texts),
                    processing_time=processing_time,
                    texts_per_second=texts_per_second
                )
        else:
            # Comportement original - seulement les embeddings
            return BatchEncodeResponse(
                embeddings=embeddings_array.tolist(),
                model=input_data.model,
                dimension=embeddings_array.shape[1],
                num_texts=len(input_data.texts),
                processing_time=processing_time,
                texts_per_second=texts_per_second
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def process_item_profiles(
    file: UploadFile = File(...),
    model: Optional[str] = Form("modernbert"),
    output_format: Optional[str] = Form("json"),
    batch_size: Optional[int] = Form(32)
):
    """
    Process a JSON file containing item profiles and generate embeddings.
    
    Expected JSON format:
    [
        {
            "item_original_id": 10125,
            "item_profile": "[Summarization] Target in Tucson..."
        },
        ...
    ]
    
    Returns embeddings mapped to item IDs.
    """
    try:
        start_time = time.time()
        
        # Read the uploaded file
        content = await file.read()
        
        # Parse JSON content
        try:
            # Try to parse as JSON array
            data = json.loads(content)
        except json.JSONDecodeError:
            # Try to parse as pandas-style JSON (with index)
            try:
                df = pd.read_json(content.decode('utf-8'))
                # Convert to list of dictionaries
                data = df.to_dict('records')
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid JSON format")
        
        # Validate the data format
        if not isinstance(data, list):
            raise HTTPException(status_code=400, detail="JSON must be an array of items")
        
        # Extract item profiles
        item_profiles = []
        item_ids = []
        texts = []
        
        for item in tqdm(data, desc="Processing items", unit="item"):
            if not isinstance(item, dict):
                raise HTTPException(status_code=400, detail="Each item must be an object")
            
            if 'item_original_id' not in item or 'item_profile' not in item:
                raise HTTPException(status_code=400, detail="Each item must have 'item_original_id' and 'item_profile'")
            
            item_ids.append(item['item_original_id'])
            texts.append(item['item_profile'])
        
        # Get the model
        model_instance = get_model(model)
        
        # Encode the texts using batch encoding for efficiency
        embeddings_array = batch_encode_large_dataset(
            texts=texts,
            model_name=model,
            batch_size=batch_size
        )
        
        processing_time = time.time() - start_time
        
        # Create response based on format
        if output_format == "csv":
            # Create CSV with item_id and embeddings
            csv_data = "item_original_id"
            for i in range(embeddings_array.shape[1]):
                csv_data += f",embedding_{i}"
            csv_data += "\n"
            
            for item_id, embedding in tqdm(zip(item_ids, embeddings_array.tolist()), desc="Generating CSV", unit="row", total=len(item_ids)):
                csv_data += f"{item_id}"
                for value in embedding:
                    csv_data += f",{value}"
                csv_data += "\n"
            
            return Response(
                content=csv_data,
                media_type="text/csv",
                headers={
                    "Content-Disposition": "attachment; filename=item_embeddings.csv",
                    "X-Model": model,
                    "X-Dimension": str(embeddings_array.shape[1]),
                    "X-Processing-Time": str(processing_time)
                }
            )
        else:  # JSON format
            # Create list of ItemEmbedding objects
            item_embeddings = []
            for item_id, embedding in tqdm(zip(item_ids, embeddings_array.tolist()), desc="Creating JSON response", unit="item", total=len(item_ids)):
                item_embeddings.append(ItemEmbedding(
                    item_original_id=item_id,
                    embedding=embedding
                ))
            
            return ItemProfileResponse(
                embeddings=item_embeddings,
                model=model,
                dimension=embeddings_array.shape[1],
                num_items=len(item_ids),
                processing_time=processing_time
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))