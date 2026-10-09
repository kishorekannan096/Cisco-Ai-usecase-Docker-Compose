import os
import requests
import logging
import json
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class VSSEngineClient:
    def __init__(self, engine_url: str = None):
        self.engine_url = engine_url or os.getenv("VSS_ENGINE_URL", "http://10.79.252.16:8076")
        self.engine_url = self.engine_url.rstrip("/")

    def register_file(self, file_path: str, purpose: str = "vision", is_remote_path: bool = False) -> Optional[str]:
        """
        Register a video file with the VSS Engine.
        
        Args:
            file_path: Absolute path to the video file. 
                      If is_remote_path=True, this is the path AS SEEN BY THE VSS CONTAINER.
                      If is_remote_path=False, this is the local path on the application host.
            purpose: Purpose of the file
            is_remote_path: If True, registers the path directly without uploading bytes (Zero-Copy).
            
        Returns:
            resource_id: The UUID mapping to this file, or None if failed
        """
        url = f"{self.engine_url}/files"
        
        if is_remote_path:
            # Zero-Copy registration: Tell VSS Engine where to find the file in its own /tmp mount
            data = {
                'purpose': purpose,
                'media_type': 'video',
                'camera_id': 'default',
                'filename': file_path
            }
            logger.info(f"Registering existing container path with VSS Engine: {file_path}")
            try:
                response = requests.post(url, data=data, timeout=60)
                if response.status_code == 200:
                    result = response.json()
                    resource_id = result.get("id")
                    logger.info(f"Path registered successfully! Resource ID: {resource_id}")
                    return resource_id
                else:
                    logger.error(f"Failed to register path: {response.status_code} {response.text}")
                    return None
            except Exception as e:
                logger.error(f"Error during path registration: {e}")
                return None
        else:
            # Upload-based registration (Original logic)
            if not os.path.exists(file_path):
                logger.error(f"File not found for registration: {file_path}")
                return None
                
            file_size = os.path.getsize(file_path)
            filename = os.path.basename(file_path)
            
            with open(file_path, 'rb') as f:
                files = {'file': (filename, f, 'video/mp4')}
                data = {'purpose': purpose, 'media_type': 'video', 'camera_id': 'default'}
                
                logger.info(f"Uploading file to VSS Engine: {filename} ({file_size} bytes)")
                try:
                    response = requests.post(url, files=files, data=data, timeout=60)
                    if response.status_code == 200:
                        result = response.json()
                        resource_id = result.get("id")
                        logger.info(f"File uploaded successfully! Resource ID: {resource_id}")
                        return resource_id
                    else:
                        logger.error(f"Failed to upload file: {response.status_code} {response.text}")
                        return None
                except Exception as e:
                    logger.error(f"Error during file upload: {e}")
                    return None

    def generate_dense_captions(self, asset_id: str, prompt: str, system_prompt: str = None, 
                                chunk_duration: int = 60, chunk_overlap_duration: int = 10) -> List[Dict[str, Any]]:
        """
        Fetches second-by-second dense captions from the VSS Engine.
        """
        url = f"{self.engine_url}/generate_vlm_captions"
        
        
        payload = {
            "id": asset_id,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "model": "Cosmos-Reason2-8B",
            "api_type": "internal",
            "response_format": {"type": "text"},
            "stream": True,
            # "stream_options": {"include_usage": True}, # Optional
            "max_tokens": 512,
            "temperature": 0.2,
            "top_p": 1,
            "top_k": 100,
            "seed": 10,
            "chunk_duration": chunk_duration,
            "chunk_overlap_duration": chunk_overlap_duration,
            "media_info": {
                "type": "offset",
                "start_offset": 0,
                "end_offset": 4000000000
            },
            "num_frames_per_chunk": 10,
            "enable_cv_metadata": True,
            "cv_pipeline_prompt": "package . person;0.3",
            "vlm_input_width": 256,
            "vlm_input_height": 256,
            "enable_reasoning": True,
            "tools": [],
            "user": "qa-inspector"
        }

        try:
            logger.info(f"Requesting dense captions for asset: {asset_id}")
            logger.debug(f"VLM Payload: {json.dumps(payload, indent=2)}")
            
            # Using stream=True requires manual error checking to see body
            response = requests.post(url, json=payload, timeout=300, stream=True)
            
            if response.status_code != 200:
                # Capture the full error body for debugging
                error_body = ""
                try:
                    error_body = response.text
                except:
                    error_body = "Could not read error body"
                    
                logger.error(f"VLM API returned {response.status_code}: {error_body}")
                return []
                # Do NOT call raise_for_status here, it hides the body
                error_body = response.text
                logger.error(f"VLM API returned {response.status_code}: {error_body}")
                return []
            
            # Parse SSE streaming response
            chunk_responses = []
            current_chunk_content = ""
            current_start_time = 0
            current_end_time = 0
            
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                    
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                    
                try:
                    data = json.loads(data_str)
                    
                    # VSS Engine wraps in chunk_responses
                    if "chunk_responses" in data:
                        chunk_responses.extend(data["chunk_responses"])
                        continue
                    
                    # Handle direct list responses
                    if isinstance(data, list):
                        chunk_responses.extend(data)
                        continue
                    
                    # Handle OpenAI-style streaming delta
                    choices = data.get("choices", [])
                    for choice in choices:
                        delta = choice.get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            current_chunk_content += content
                        
                        # Check for timing info
                        if "start_time" in data:
                            current_start_time = data["start_time"]
                        if "end_time" in data:
                            current_end_time = data["end_time"]
                        
                        # If finish_reason is set, this chunk is done
                        finish_reason = choice.get("finish_reason")
                        if finish_reason:
                            logger.debug(f"Chunk finished. Reason: {finish_reason}. Content: {current_chunk_content}")
                            if current_chunk_content:
                                chunk_responses.append({
                                    "content": current_chunk_content,
                                    "start_time": current_start_time,
                                    "end_time": current_end_time
                                })
                                current_chunk_content = ""
                                
                except json.JSONDecodeError:
                    logger.debug(f"Skipping non-JSON SSE line: {data_str[:100]}")
                    continue
            
            # If there's leftover content that wasn't finalized
            if current_chunk_content and not chunk_responses:
                chunk_responses.append({
                    "content": current_chunk_content,
                    "start_time": current_start_time,
                    "end_time": current_end_time
                })
            
            logger.info(f"Received {len(chunk_responses)} caption chunks")
            logger.debug(f"Caption chunks: {json.dumps(chunk_responses)}")
            return chunk_responses
            
        except Exception as e:
            logger.error(f"Failed to fetch dense captions: {e}")
            return []

    def format_timeline(self, chunk_responses: List[Dict[str, Any]]) -> str:
        """
        Converts raw VSS Engine chunk responses into a readable timeline string.
        """
        timeline = []
        for i, chunk in enumerate(chunk_responses):
            # Extract text from chunk (API returns 'content' for cosmos-reason1, 'text' for others)
            text = chunk.get("content") or chunk.get("text", "")
            
            start_time = chunk.get("start_time", "N/A")
            end_time = chunk.get("end_time", "N/A")
            
            if text:
                timeline.append(f"[{start_time}s - {end_time}s]: {text}")
        
        return "\n\n".join(timeline) if timeline else "No dense narrative available."
