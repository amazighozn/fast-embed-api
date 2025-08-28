
import time
import json
import pandas as pd
from fastapi import Request
from starlette.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
import os

class SaveResponseMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Create directory if it doesn't exist
        if not os.path.exists("results"):
            os.makedirs("results")

        # Generate a unique filename
        timestamp = int(time.time())
        endpoint_name = request.url.path.replace('/', '_').strip('_')
        filename_base = f"results/{endpoint_name}_{timestamp}"

        # Get response body
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk
        
        # Save JSON response
        if "application/json" in response.headers.get("content-type", ""):
            json_filename = f"{filename_base}.json"
            with open(json_filename, "wb") as f:
                f.write(response_body)
            
            # Convert JSON to CSV and save
            try:
                data = json.loads(response_body)
                
                # Normalize JSON data to a flat table
                if isinstance(data, list):
                    df = pd.json_normalize(data)
                elif isinstance(data, dict):
                    # Handle nested JSON
                    df = pd.json_normalize(data, record_path='embeddings', meta=['model', 'dimension', 'num_items', 'processing_time'], errors='ignore')
                    if df.empty:
                         df = pd.json_normalize(data, record_path='data', meta=['model', 'dimension', 'num_texts', 'processing_time', 'texts_per_second'], errors='ignore')
                    if df.empty:
                        df = pd.DataFrame([data])

                else:
                    df = pd.DataFrame()

                if not df.empty:
                    csv_filename = f"{filename_base}.csv"
                    df.to_csv(csv_filename, index=False)
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                print(f"Could not convert JSON to CSV for {filename_base}: {e}")

        # Save CSV response
        elif "text/csv" in response.headers.get("content-type", ""):
            csv_filename = f"{filename_base}.csv"
            with open(csv_filename, "wb") as f:
                f.write(response_body)

        # Return a new response with the original body, so it can be sent to the client
        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )
