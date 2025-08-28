# Fast Embed API

**Fast Embed API** is a high-performance, scalable, and easy-to-use API for generating text embeddings using state-of-the-art sentence-transformer models. Built with FastAPI, it provides a simple and efficient way to integrate text embeddings into your applications.

## Features

- **High Performance:** Built on top of FastAPI and sentence-transformers for fast and efficient embedding generation.
- **Multiple Models:** Supports various pre-trained models from the sentence-transformers library.
- **Flexible Output Formats:** Get your embeddings in JSON, CSV, or even as a raw PyTorch tensor.
- **Batch Processing:** Optimized endpoint for encoding large datasets efficiently.
- **Token-level Embeddings:** Get embeddings for each token in your text.
- **Easy to Deploy:** Can be deployed as a standalone service using Docker.
- **Response Caching:** Automatically saves all responses to the `results/` directory for logging and analysis.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd fast-embed-api
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt 
    ```
    *(Note: You will need to create a `requirements.txt` file from `pyproject.toml` dependencies)*

    Or using `uv`:
    ```bash
    uv pip install -r requirements.txt
    ```

    The dependencies are:
    ```
    fastapi>=0.116.1
    numpy>=2.3.2
    pandas>=2.3.2
    pydantic>=2.11.7
    python-multipart>=0.0.20
    requests>=2.32.5
    sentence-transformers>=5.1.0
    torch>=2.8.0
    transformers>=4.55.4
    uvicorn[standard]>=0.35.0
    ```

## Usage

Run the API server with:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
The server will start, and the default model will be pre-loaded.

### API Endpoints

#### `GET /`

Returns basic information about the API.

- **Response:**
  ```json
  {
    "message": "API d\'embeddings de texte",
    "endpoints": {
      "/encode": "POST - Encode des textes en embeddings",
      "/models": "GET - Liste des modèles disponibles",
      "/batch_encode": "POST - Encode un grand nombre de textes efficacement",
      "/process_item_profiles": "POST - Process JSON file with item profiles"
    }
  }
  ```

#### `GET /models`

Lists the available embedding models.

- **Response:**
  ```json
  {
    "models": {
        "modernbert": {
            "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "description": "A fast and multilingual model.",
            "default": true
        },
        "distilbert": {
            "model_name": "sentence-transformers/distiluse-base-multilingual-cased-v1",
            "description": "A distilled version of BERT, faster and smaller."
        }
    },
    "default": "modernbert"
  }
  ```

#### `POST /encode`

Encodes a list of texts into embeddings.

- **Request Body:**
  ```json
  {
    "texts": ["Hello world", "This is a test"],
    "model": "modernbert",
    "output_format": "json",
    "batch_size": 32
  }
  ```
- **Parameters:**
    - `texts` (list[str]): A list of texts to encode.
    - `model` (str, optional): The model to use. Defaults to `modernbert`.
    - `output_format` (str, optional): The output format. Can be `json`, `csv`, or `tensor`. Defaults to `json`.
    - `batch_size` (int, optional): The batch size for encoding. Defaults to 32.

- **Example with `curl`:**
  ```bash
  curl -X POST "http://localhost:8000/encode" \
  -H "Content-Type: application/json" \
  -d \'\'\'{"texts": ["This is an example sentence", "Each sentence is converted"]}\'\'\'
  ```

#### `POST /batch_encode`

Optimized for encoding a large number of texts.

- **Request Body:**
  ```json
  {
    "texts": ["text1", "text2", ...],
    "model": "modernbert",
    "output_format": "json",
    "batch_size": 64,
    "include_texts": true,
    "token_embeddings": false
  }
  ```
- **Parameters:**
    - `include_texts` (bool, optional): If `true`, the response will include the original texts alongside their embeddings.
    - `token_embeddings` (bool, optional): If `true`, returns embeddings for each token instead of the whole sentence.

#### `POST /process_item_profiles`

Process a JSON file containing item profiles and generate embeddings.

- **Request:** `multipart/form-data`
    - `file`: The JSON file to process.
    - `model`: The model to use.
    - `output_format`: `json` or `csv`.
    - `batch_size`: The batch size.

- **Example with `curl`:**
  ```bash
  curl -X POST "http://localhost:8000/process_item_profiles" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@items.json" \
  -F "model=modernbert" \
  -F "output_format=csv"
  ```

## Response Saving

This API is configured to automatically save the response of every request to the `results/` directory.
- **JSON responses** are saved as `.json` and automatically converted to `.csv`.
- **CSV responses** are saved as `.csv`.

This is useful for logging, analysis, and debugging.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
