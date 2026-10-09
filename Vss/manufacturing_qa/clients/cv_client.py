import os
import requests
import logging
import json
import time
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

class CVClient:
    def __init__(self, api_url: str = None):
        self.api_url = api_url or os.getenv("CV_DETECTOR_API_URL", "http://localhost:23491")
        # Ensure no trailing slash
        self.api_url = self.api_url.rstrip("/")

    def health_check(self) -> bool:
        """Checks if the CV Event Detector service is reachable."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:\
            return False

    def list_pipelines(self) -> List[Dict[str, Any]]:
        """Lists all available pipelines."""
        try:
            response = requests.get(f"{self.api_url}/api/pipelines")
            response.raise_for_status()
            return response.json().get("pipelines", [])
        except requests.RequestException as e:
            logger.error(f"Failed to list pipelines: {e}")
            return []

    def create_pipeline(self, name: str, params: Dict[str, Any] = None) -> Optional[str]:
        """Creates a new pipeline if it doesn't exist."""
        # Check if pipeline already exists
        pipelines = self.list_pipelines()
        for p in pipelines:
            if p.get("id") == name:
                 return name

        # Default params if not provided
        default_params = { 
            "min_clip_duration": 5,
            "max_clip_duration": 30,
            "frame_skip_interval": 1,
            "minimum_detection_threshold": 3
        }
        
        # Merge provided params with defaults
        if params:
            final_params = default_params.copy()
            final_params.update(params)
        else:
            final_params = default_params
        
        payload = {
            "name": name,
            "type": "object_detection",
            "params": final_params
        }
        
        try:
            logger.info(f"Creating pipeline '{name}' at {self.api_url}/api/pipeline")
            logger.debug(f"Pipeline config: {json.dumps(payload)}")
            
            response = requests.post(f"{self.api_url}/api/pipeline", json=payload, timeout=10)
            
            if response.status_code >= 400:
                logger.error(f"Failed to create pipeline. Status: {response.status_code}, Body: {response.text}")
            
            response.raise_for_status()
            pipeline_id = response.json().get("id")
            logger.info(f"Pipeline created successfully: {pipeline_id}")
            return pipeline_id
        except requests.RequestException as e:
            logger.error(f"Failed to create pipeline '{name}': {e}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Error Response: {e.response.text}")
            return None

    def add_stream(
        self, 
        video_url: str, 
        pipeline_id: str, 
        prompt: str = None, 
        threshold: float = 0.3,
        rois: List[List[int]] = None,
        output_folder: str = "/tmp",
        overlay: bool = True,
        stream_name: str = None
    ) -> Optional[str]:
        """
        Starts processing a video stream with the given CV parameters.
        
        Args:
            video_url: RTSP URL or file path (e.g. file:///path/to/video.mp4)
            pipeline_id: ID of the pipeline to use
            prompt: Text prompt for Grounding DINO (e.g. "person with helmet")
            threshold: Confidence threshold (0.0 - 1.0)
            rois: List of ROIs [[test_point_x, test_point_y, width, height]]
                  (Note: API expects this format for ROI-based counting/detection)
            output_folder: Folder where event clips will be saved.
            overlay: Whether to burn bounding boxes into the output video.
        """
        
        # Default ROI if not provided:
        # Official UI sends [[]] which seems to imply full frame or "no specific ROI constraint"
        if not rois:
             rois = [[]]
             
        cv_params = {
            "gdinoprompt": prompt,
            "gdinothreshold": threshold,
            "gdino_rois": rois,
            "overlay": overlay
        }
        
        import uuid
        
        payload = {
            "version": "1.0",
            "stream_url": video_url,
            "pipeline_id": pipeline_id,
            "stream_name": f"stream_{uuid.uuid4().hex[:8]}", # Required to prevent 500 error on backend
            "output_folder": output_folder, # Official UI uses subfolders, but /tmp is the shared mount root. Stick to /tmp for now or make configurable.
            "cv_params": cv_params,
        }
        
        try:
            logger.info(f"Adding stream to pipeline '{pipeline_id}' at {self.api_url}/api/addstream")
            logger.info(f"Stream config: {json.dumps(payload, indent=2)}")
            
            response = requests.post(f"{self.api_url}/api/addstream", json=payload, timeout=10)
            
            if response.status_code >= 400:
                logger.error(f"Failed to add stream. Status: {response.status_code}, Body: {response.text}")
                
            response.raise_for_status()
            stream_id = response.json().get("stream_id")
            logger.info(f"Stream added successfully: {stream_id}")
            return stream_id
        except requests.RequestException as e:
            logger.error(f"Failed to add stream: {e}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response: {e.response.text}")
            return None

    def stop_stream(self, stream_id: str) -> bool:
        """Stops a running stream."""
        payload = {"stream_id": stream_id}
        # Note: API uses DELETE /api/stream but takes a body? 
        # Requests doesn't support body in delete by default commonly, 
        # but `app.py` shows: @app.delete("/api/stream") but wait...
        # Let's check main.py again. 
        # main.py does NOT explicitly show a DELETE endpoint for stream in the `root()` function summary!
        # It says: "/api/stream": "DELETE endpoint to remove a video stream"
        # And usually in FastAPI DELETE can take a body (Pydantic model `RemoveStreamRequest`).
        
        try:
            response = requests.request("DELETE", f"{self.api_url}/api/stream", json=payload)
            return response.status_code == 200
        except requests.RequestException as e:
            logger.error(f"Failed to stop stream {stream_id}: {e}")
            return False

    def get_stream_status(self, stream_id: str) -> Dict[str, Any]:
        """Gets the status of a stream."""
        try:
            response = requests.get(f"{self.api_url}/api/streams/{stream_id}/status")
            return response.json()
        except requests.RequestException:
            return {"status": "unknown"}
    
    def delete_stream(self, stream_id: str) -> bool:
        """Delete a stream by ID."""
        try:
            response = requests.delete(
                f"{self.api_url}/api/stream",
                json={"stream_id": stream_id, "version": "1.0"}
            )
            response.raise_for_status()
            logger.info(f"Deleted stream: {stream_id}")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to delete stream {stream_id}: {e}")
            return False
    
    def delete_pipeline(self, pipeline_id: str) -> bool:
        """Delete a pipeline by ID."""
        try:
            response = requests.delete(
                f"{self.api_url}/api/pipeline",
                json={"id": pipeline_id, "cleanup_resources": True}
            )
            response.raise_for_status()
            logger.info(f"Deleted pipeline: {pipeline_id}")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to delete pipeline {pipeline_id}: {e}")
            return False
    
    def wait_for_completion(
        self,
        stream_id: str,
        timeout_seconds: int = 300,
        poll_interval: int = 2
    ) -> Dict[str, Any]:
        """
        Poll stream status until processing completes or timeout.
        
        Args:
            stream_id: Stream identifier
            timeout_seconds: Maximum wait time (default: 300s / 5min)
            poll_interval: Seconds between status checks (default: 2s)
            
        Returns:
            Final status dict with events array
            
        Raises:
            TimeoutError: If stream doesn't complete within timeout
        """
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            
            if elapsed > timeout_seconds:
                raise TimeoutError(
                    f"Stream {stream_id} did not complete within {timeout_seconds}s"
                )
            
            status_response = self.get_stream_status(stream_id)
            processing_state = status_response.get("processing_state", "")
            
            logger.info(
                f"Stream {stream_id} status: {processing_state} "
                f"({elapsed:.1f}s elapsed)"
            )
            
            # Check for completion
            if processing_state == "completed":
                logger.info(f"Stream {stream_id} completed successfully")
                return status_response
            
            # Check for errors
            if processing_state == "failed" or status_response.get("status") == "error":
                error_details = status_response.get("error_details", "Unknown error")
                raise RuntimeError(f"Stream processing failed: {error_details}")
            
            # Wait before next poll
            time.sleep(poll_interval)

if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.INFO)
    client = CVClient()
    if client.health_check():
        print("CV Detector is Healthy")
        print("Pipelines:", client.list_pipelines())
    else:
        print("CV Detector is Unreachable")
