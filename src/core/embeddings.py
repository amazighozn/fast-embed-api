"""
Module pour la gestion des modèles d'embeddings
"""
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List
from tqdm import tqdm
from .config import MODEL_CONFIGS, MAX_TEXT_LENGTH

class EmbeddingModel:
    def __init__(self, model_name: str = "modernbert"):
        """
        Initialise le modèle d'embeddings.
        
        Args:
            model_name: Nom du modèle à utiliser (qwen ou distilbert)
        """
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Modèle non supporté: {model_name}. Modèles disponibles: {list(MODEL_CONFIGS.keys())}")
        
        self.model_name = model_name
        self.config = MODEL_CONFIGS[model_name]
        self.dimension = self.config["dimension"]
        
        # Charger le tokenizer et le modèle
        print(f"Chargement du modèle {self.config['name']}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['name'])
        self.model = AutoModel.from_pretrained(self.config['name'])
        
        # Utiliser GPU si disponible et compatible
        try:
            if torch.cuda.is_available():
                # Tester si le GPU est compatible avec le modèle
                self.device = torch.device("cuda")
                self.model = self.model.to(self.device)
                # Test simple pour vérifier la compatibilité
                test_input = torch.zeros(1, 1).to(self.device)
                _ = self.model.embeddings(test_input.long()) if hasattr(self.model, 'embeddings') else None
                print(f"Modèle chargé sur GPU (CUDA)")
            else:
                self.device = torch.device("cpu")
                self.model = self.model.to(self.device)
                print(f"Modèle chargé sur CPU")
        except Exception as e:
            # En cas d'erreur avec le GPU, utiliser le CPU
            print(f"Erreur avec GPU: {e}. Utilisation du CPU.")
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)
            
        self.model.eval()
    
    def mean_pooling(self, model_output, attention_mask):
        """Effectue un mean pooling sur les outputs du modèle."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def clean_token(self, token: str) -> str:
        """
        Nettoie un token en supprimant les symboles spéciaux selon le tokenizer.
        
        Args:
            token: Le token à nettoyer
            
        Returns:
            Le token nettoyé
        """
        # Identifier le type de tokenizer et appliquer le nettoyage approprié
        tokenizer_class_name = self.tokenizer.__class__.__name__.lower()
        
        # Pour les tokenizers type GPT/RoBERTa qui utilisent Ġ pour les espaces
        if 'gpt' in tokenizer_class_name or 'roberta' in tokenizer_class_name:
            if token.startswith('Ġ'):
                return token[1:]  # Enlever le Ġ
        
        # Pour les tokenizers type SentencePiece (T5, ALBERT, etc.) qui utilisent ▁
        elif 'sentencepiece' in tokenizer_class_name or 't5' in tokenizer_class_name:
            if token.startswith('▁'):
                return token[1:]  # Enlever le ▁
        
        # Pour les tokenizers type WordPiece (BERT) qui utilisent ##
        elif 'bert' in tokenizer_class_name and not 'roberta' in tokenizer_class_name:
            if token.startswith('##'):
                return token[2:]  # Enlever ## (mais c'est un cas spécial pour les sous-mots)
        
        # Pour ModernBERT, vérifier le type de tokenizer utilisé
        elif 'modern' in self.model_name.lower():
            # ModernBERT pourrait utiliser GPT-style tokenization
            if token.startswith('Ġ'):
                return token[1:]
        
        # Par défaut, retourner le token tel quel
        return token
    
    def encode(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Encode une liste de textes en embeddings.
        
        Args:
            texts: Liste des textes à encoder
            batch_size: Taille des batchs pour le traitement
            
        Returns:
            Tensor contenant les embeddings
        """
        all_embeddings = []
        
        # Traitement par batch pour optimiser les performances
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts", unit="batch"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenization
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=MAX_TEXT_LENGTH
            ).to(self.device)
            
            # Génération des embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
                
                # Normalisation des embeddings
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
            all_embeddings.append(embeddings)
        
        # Concaténer tous les embeddings
        return torch.cat(all_embeddings, dim=0)
    
    def encode_with_tokens(self, texts: List[str], batch_size: int = 32) -> List[dict]:
        """
        Encode une liste de textes en retournant les embeddings pour chaque token.
        
        Args:
            texts: Liste des textes à encoder
            batch_size: Taille des batchs pour le traitement
            
        Returns:
            Liste de dictionnaires contenant les tokens et leurs embeddings
        """
        all_results = []
        
        # Traitement par batch pour optimiser les performances
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding with tokens", unit="batch"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenization
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=MAX_TEXT_LENGTH
            ).to(self.device)
            
            # Génération des embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                # On prend directement les embeddings des tokens (sans mean pooling)
                token_embeddings = model_output.last_hidden_state
                
                # Normalisation des embeddings token par token
                token_embeddings = torch.nn.functional.normalize(token_embeddings, p=2, dim=2)
            
            # Extraire les tokens et leurs embeddings pour chaque texte du batch
            for idx, text in enumerate(batch_texts):
                # Les token IDs incluent [CLS] et [SEP], on doit les gérer
                token_ids = encoded_input['input_ids'][idx]
                
                # Trouver où se termine le texte réel (avant le padding)
                attention_mask = encoded_input['attention_mask'][idx]
                seq_len = attention_mask.sum().item()
                
                # Récupérer les embeddings pour ce texte (sans padding)
                text_embeddings = token_embeddings[idx, :seq_len, :].cpu().numpy()
                
                # Convertir les token IDs en tokens lisibles pour ce texte spécifique
                readable_tokens = self.tokenizer.convert_ids_to_tokens(token_ids[:seq_len])
                
                # Créer la structure de résultat
                token_data = []
                for pos, (token, embedding) in enumerate(zip(readable_tokens, text_embeddings)):
                    # Ignorer les tokens spéciaux si nécessaire
                    if token not in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
                        # Nettoyer le token selon le tokenizer
                        cleaned_token = self.clean_token(token)
                        token_data.append({
                            'token': cleaned_token,
                            'embedding': embedding.tolist(),
                            'position': pos
                        })
                
                # Ajouter ce texte avec ses tokens à la liste des résultats
                all_results.append({
                    'text': text,
                    'tokens': token_data
                })
        
        return all_results

# Instance globale du modèle
current_model = None
current_model_name = None

def get_model(model_name: str = "modernbert") -> EmbeddingModel:
    """Obtient ou charge le modèle demandé."""
    global current_model, current_model_name
    
    if current_model_name != model_name or current_model is None:
        current_model = EmbeddingModel(model_name)
        current_model_name = model_name
    
    return current_model