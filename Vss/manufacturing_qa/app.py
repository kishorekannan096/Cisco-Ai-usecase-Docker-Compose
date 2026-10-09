import gradio as gr
import os
import json
import cv2
import re
import logging
from dotenv import load_dotenv
from ui.theme import theme
from ui import components
from clients.cv_client import CVClient

from clients.llm_client import LLMClient

from clients.vss_engine_client import VSSEngineClient
from utils import report_generator
from video_presets import VIDEO_PRESETS
import time
import pandas as pd
import uuid
from datetime import datetime, timezone
from utils.prompt_loader import load_prompts
# from pypdf import PdfReader # Removed

# Load configuration
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base directory for all media (mounted to /tmp in containers)
MEDIA_BASE_DIR = os.getenv("MEDIA_BASE_DIR", "/tmp")
os.makedirs(MEDIA_BASE_DIR, exist_ok=True)

APP_STORAGE_BASE_DIR = os.getenv("APP_STORAGE_BASE_DIR", "./storage")
OVERLAY_DIR = os.path.join(APP_STORAGE_BASE_DIR, "overlays")
REPORTS_DIR = os.path.join(APP_STORAGE_BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True) 
os.makedirs(OVERLAY_DIR, exist_ok=True)

# ALERT_MEDIA_DIR is now just a subfolder in MEDIA_BASE_DIR
ALERT_MEDIA_DIR = os.path.join(MEDIA_BASE_DIR, "clips")
os.makedirs(ALERT_MEDIA_DIR, exist_ok=True)

# Initialize Clients
cv_client = CVClient()
vss_engine_client = VSSEngineClient()

# Load Prompts
app_prompts = load_prompts()

def on_preset_change(preset_name):
    """Updates UI components when a video preset is selected."""
    if preset_name == "Custom" or preset_name not in VIDEO_PRESETS:
        # Return existing values (no-op) or clear? 
        # User said "if we are not going with the video that's listed then user can change the value"
        # So "Custom" should probably just leave things as they are.
        return [gr.update()] * 16
    
    config = VIDEO_PRESETS[preset_name]
    
    samples_dir = os.getenv("SAMPLE_VIDEOS_DIR", "./videos")
    video_path = os.path.join(samples_dir, preset_name)
    
    vlm = config.get("vlm_params", {})
    
    return [
        "Video File", # input_type
        gr.update(value=video_path),   # video_file - Use gr.update() to properly clear previous uploads
        config["det_classes"],
        config["box_thresh"],
        config["frame_skip"],
        config["obj_thresh"],
        config["alert_prompts"],
        # VLM States
        vlm.get("chunk_duration", 0),
        vlm.get("temperature", 0.2),
        vlm.get("top_p", 1.0),
        vlm.get("max_tokens", 256),
        vlm.get("frames_per_chunk", 10),
        vlm.get("enable_reasoning", True),
        vlm.get("vlm_width", 0),
        vlm.get("vlm_height", 0),
        # ROIs
        config.get("rois", [[]])
    ]

def start_inspection_handler(
    input_type, video_file, rtsp_url, 
    det_classes, box_thresh, 
    frame_skip, obj_thresh,
    alert_prompts,
    # VLM parameters
    vlm_chunk_dur, vlm_temp, vlm_top_p, vlm_max_tokens, vlm_frames, vlm_reasoning, vlm_width, vlm_height,
    # ROI parameters
    rois=[[]]
):
    """Starts the CV Pipeline."""
    try:
        if input_type == "Video File":
            if not video_file:
                raise gr.Error("Please upload a video file.")
            
            import shutil
            
            # Create a unique folder for this uploaded video as requested
            upload_token = uuid.uuid4().hex[:8]
            upload_folder_name = f"upload_{upload_token}"
            upload_folder_host = os.path.join(MEDIA_BASE_DIR, upload_folder_name)
            os.makedirs(upload_folder_host, exist_ok=True)
            
            # Extract base filename and sanitize
            base_filename = os.path.basename(video_file)
            base_filename = "".join(c for c in base_filename if c.isalnum() or c in "._- ")
            
            # Host path for the video
            local_dest_path = os.path.join(upload_folder_host, base_filename)
            
            logger.info(f"Copying uploaded video to shared path: {local_dest_path}")
            shutil.copy(video_file, local_dest_path)
            
            # Path for the CV service
            video_source = f"{MEDIA_BASE_DIR}/{upload_folder_name}/{base_filename}"
        else:
            video_source = rtsp_url
            
        if not video_source:
            raise gr.Error("Please provide a valid video source.")
        
        # Official UI uses space-dot-space separator ' . '
        gdinoprompt = det_classes.replace("\n", " . ")
        # Ensure it ends with a dot if strictly needed? Log showed "person ."
        # Examples: "person . helmet" or "person ."
        # If the user enters one class "person", we get "person". 
        # Let's verify if trailing dot is required. The log showed "person .".
        # Safe bet: If single class, add dot? Or just leave it. 
        
        # Use unique pipeline name to prevent CV service from using cached pipeline state
        pipeline_timestamp = int(time.time() * 1000)
        pipeline_name = f"manufacturing-qa-pipeline-{pipeline_timestamp}"
        logger.info(f"Creating unique pipeline: {pipeline_name}")
        
        # Capture the actual UUID returned by the API
        pipeline_id = cv_client.create_pipeline(pipeline_name, params={
            "frame_skip_interval": int(frame_skip),
             "minimum_detection_threshold": int(obj_thresh)
        })
        
        if not pipeline_id:
            raise gr.Error("Failed to create CV pipeline.")

        # Use unique stream name to prevent conflicts
        stream_token = uuid.uuid4().hex[:8]
        unique_stream_name = f"stream_{stream_token}"
        # output_folder must be a path AS SEEN BY THE CONTAINER
        # Since MEDIA_BASE_DIR is mounted to /tmp, this corresponds to {MEDIA_BASE_DIR}/clips/stream_{unique_stream_name}
        container_output_folder = f"{MEDIA_BASE_DIR}/clips/{unique_stream_name}"
        
        logger.info(f"Targeting container output folder: {container_output_folder}")

        stream_id = cv_client.add_stream(
            video_url=video_source,
            pipeline_id=pipeline_id,
            prompt=gdinoprompt,
            threshold=box_thresh,
            rois=rois,
            output_folder=container_output_folder,
            overlay=False,  # Disable burnt-in overlays for better VLM accuracy
            stream_name=unique_stream_name
        )
        
        if not stream_id:
            cv_client.delete_pipeline(pipeline_id)
            raise gr.Error("Failed to start stream.")
    
        # Poll for completion status
        logger.info(f"Polling stream {stream_id} for completion...")
        max_wait_time = 300  # 5 minutes timeout
        poll_interval = 2    # Check every 2 seconds
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            status_data = cv_client.get_stream_status(stream_id)
            status = status_data.get("status")
            processing_state = status_data.get("processing_state", "")

            logger.info(f"Stream {stream_id} status: {status}, processing_state: {processing_state}")

            if status == "completed":
                # Extract events
                events = status_data.get("events", [])
                if events is None:
                    events = []
                clip_count = len(events)
                
                logger.info(f"Stream completed with {clip_count} clips. Submitting alerts...")
                
                # Log the clips being returned
                for i, event in enumerate(events):
                    clip_name = event.get("clip", "unknown")
                    logger.info(f"Event {i+1}/{clip_count}: {clip_name}")
                
                # Submit all clips to VST and Alert Bridge
                # Process all c lips
                for event in events:
                    raw_clip_name = event.get("clip")
                    if not raw_clip_name:
                        continue
                    
                    clip_filename = os.path.basename(raw_clip_name)
                    # Host path for verification/access (clips are in the 'clips' subfolder)
                    clip_path = os.path.join(ALERT_MEDIA_DIR, unique_stream_name, clip_filename)
                    
                    # Verify clip exists on host
                    if not os.path.exists(clip_path):
                        logger.warning(f"Clip not found on host: {clip_path}")
                        continue
                    
                    # IMPORTANT: Update event with host path for UI and report generation
                    event['clip'] = clip_path

                # Cleanup stream and pipeline
                cv_client.delete_stream(stream_id)
                cv_client.delete_pipeline(pipeline_id)
                
                logger.info(f"Returning {clip_count} events to UI")
                return f"✅ Processing complete. {clip_count} clips extracted.", None, alert_prompts, events
            
            elif status == "error":
                error_msg = status_data.get("error_details", "Unknown error")
                logger.error(f"Stream processing failed: {error_msg}")
                
                # Cleanup on error
                cv_client.delete_stream(stream_id)
                cv_client.delete_pipeline(pipeline_id)
                
                raise gr.Error(f"Processing failed: {error_msg}")
        
            # Still processing
            time.sleep(poll_interval)
            elapsed_time += poll_interval
    
        # Timeout - cleanup anyway
        logger.warning(f"Stream processing timed out after {max_wait_time}s")
        cv_client.delete_stream(stream_id)
        cv_client.delete_pipeline(pipeline_id)
        raise gr.Error(f"Processing timed out after {max_wait_time} seconds")

    except Exception as e:
        logger.error(f"Error starting inspection: {e}")
        logger.exception("Full traceback:")
        # Try to clean up if variables exist
        try:
            if 'stream_id' in locals() and stream_id:
                cv_client.delete_stream(stream_id)
            if 'pipeline_id' in locals() and pipeline_id:
                cv_client.delete_pipeline(pipeline_id)
        except:
            pass
        # Re-raise as user-friendly error
        raise gr.Error(f"Inspection failed: {str(e)}")


def find_hero_timestamp(vlm_description):
    """Parses VLM text to find the best evidence moment. Returns None if no clear timestamp found."""
    if not vlm_description:
        return None
    
    # 1. Try to find explicit range from new timeline format: [0.0s - 5.0s]
    # We look for the first non-empty description block
    # Regex: [(\d+\.?\d*)s\s*-\s*(\d+\.?\d*)s]
    match = re.search(r'\[(\d+\.?\d*)s\s*-\s*(\d+\.?\d*)s\]', vlm_description)
    if match:
        start_time = float(match.group(1))
        end_time = float(match.group(2))
        return (start_time + end_time) / 2  # Return midpoint
        
    # 2. Support old formats '15-17 sec: Verdict: Yes'
    match = re.search(r'(\d+)-(\d+)\s+sec:.*?[Vv]erdict:\s*[**]*[Yy]es', vlm_description, re.DOTALL)
    if match:
        return float(match.group(1))
        
    # 3. Fallback to just searching for Yes if Verdict keyword is missing
    match = re.search(r'(\d+)-(\d+)\s+sec:.*?[**]*[Yy]es', vlm_description, re.DOTALL)
    if match:
         return float(match.group(1))
         
    return None

def render_cv_overlay(clip_path, metadata_path, output_img_path, target_time=None):
    """
    Extracts the 'Hero' frame from the clip and renders CV bounding boxes.
    If target_time is provided (seconds), it seeks to that moment.
    Otherwise, it finds the highest confidence frame.
    """
    if not os.path.exists(clip_path) or not os.path.exists(metadata_path):
        logger.warning(f"Clip or metadata missing for overlay: {clip_path}, {metadata_path}")
        return None
    try:
        # 1. Parse Metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        if not metadata or not isinstance(metadata, list):
            logger.warning(f"Metadata is empty or invalid format: {metadata_path}")
            return None

        # JSON frames are absolute from stream start, but clips start from frame 0
        frame_offset = metadata[0].get("frameNo", 0)
            
        # 2. Hero Frame Selection
        target_frame_no = frame_offset
        objects_to_draw = []
        
        # Strategy A: Use VLM hinted timestamp
        if target_time is not None:
            # VLM time is relative to clip start. Metadata timestamp is absolute.
            base_ts = metadata[0].get("timestamp", 0)
            target_ts_nanos = base_ts + int(target_time * 1e9)
            
            best_diff = float('inf')
            for frame in metadata:
                diff = abs(frame.get("timestamp", 0) - target_ts_nanos)
                if diff < best_diff and frame.get("objects"):
                    best_diff = diff
                    target_frame_no = frame.get("frameNo", 0)
                    objects_to_draw = frame.get("objects")
        
        # Strategy B: If no time provided or no objects at that time, find highest confidence frame
        if not objects_to_draw:
            max_conf = -1.0
            for frame in metadata:
                objs = frame.get("objects", [])
                if objs:
                    current_conf = max([o.get("conf", 0) for o in objs])
                    if current_conf > max_conf:
                        max_conf = current_conf
                        target_frame_no = frame.get("frameNo", 0)
                        objects_to_draw = objs
        
        # 3. Extract Frame
        cap = cv2.VideoCapture(clip_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate relative frame index
        relative_frame_no = max(0, min(target_frame_no - frame_offset, total_frames - 1))
        
        logger.info(f"Extracting relative frame {relative_frame_no} (absolute {target_frame_no}, offset {frame_offset}) from {clip_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, relative_frame_no)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            logger.error(f"Failed to extract frame {target_frame_no} from {clip_path}")
            return None
            
        # 4. Draw Overlays
        for obj in objects_to_draw:
            bbox = obj.get("bbox", {})
            lx, ty, rx, by = int(bbox.get("lX", 0)), int(bbox.get("tY", 0)), int(bbox.get("rX", 0)), int(bbox.get("bY", 0))
            label = f"{obj.get('type', 'object')} ({int(obj.get('conf', 0)*100)}%)"
            
            # Draw box
            cv2.rectangle(frame, (lx, ty), (rx, by), (0, 255, 0), 3)
            # Draw label background
            cv2.rectangle(frame, (lx, ty - 30), (lx + 200, ty), (0, 255, 0), -1)
            cv2.putText(frame, label, (lx + 5, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
        # 5. Save
        cv2.imwrite(output_img_path, frame)
        return output_img_path
        
    except Exception as e:
        logger.error(f"Error rendering CV overlay: {e}")
        return None

def generate_report_btn_handler(session_events, alert_prompts_text, preset_name):
    """
    Generates a PDF report by creating NEW Guided Dense Captions for every session clip
    and validating them against the User Criteria (alert_prompts) using an LLM.
    Bypasses the Alert Inspector Table entirely.
    """
    if not session_events or len(session_events) == 0:
        gr.Warning("No clips found in this session to report on.")
        return None
        
    gr.Info(f"📊 Generating Premium Report for {len(session_events)} clips...")
    logger.info(f"Starting Report Generation. Clips: {len(session_events)}. Criteria: {alert_prompts_text}")
    
    report_events = []
    
    # Base directory for clips (shared volume)

    llm_client = LLMClient()
    
    for i, event in enumerate(session_events):
        try:
            if isinstance(event, str):
                logger.warning(f"Skipping malformed event (string): {event}")
                continue
                
            raw_clip_path = event.get("clip")
            if not raw_clip_path:
                continue
                
            clip_path = raw_clip_path
            clip_basename = os.path.basename(clip_path)
            
            if not os.path.exists(clip_path):
                logger.warning(f"Clip not found for report: {clip_path}")
                continue
                
            logger.info(f"Processing Clip {i+1}/{len(session_events)}: {clip_basename}")
            
            # 1. Guided Dense Captioning 
            # Strategy: Ask VLM for pure description of physical attributes (Fact Gathering)
            # The compliance check (Rule Application) happens in the LLM step.
            # 1. Guided Dense Captioning 
            # 1. Guided Dense Captioning (The "Eyes")
            # Strategy: "Observer Mode". Ask VLM to describe facts related to the focus areas.
            # It must NOT give a verdict.
            guided_prompt_tpl = app_prompts.get("guided_observer_prompt", "")
            if not guided_prompt_tpl:
                logger.warning("guided_observer_prompt not found in YAML, fallback may be needed.")
            
            guided_prompt = guided_prompt_tpl.replace("{alert_prompts_text}", alert_prompts_text)
            
            # Register & Generate
            # The VSS container sees MEDIA_BASE_DIR.
            # clips are stored in {MEDIA_BASE_DIR}/clips/{stream_name}/{clip_name}
            clip_basename = os.path.basename(clip_path)
            # Find which stream folder this clip belongs to
            stream_folder_name = os.path.basename(os.path.dirname(clip_path))
            container_clip_path = f"{MEDIA_BASE_DIR}/clips/{stream_folder_name}/{clip_basename}"
            
            logger.info(f"Registering clip with VSS using container path: {container_clip_path}")
            resource_id = vss_engine_client.register_file(container_clip_path, is_remote_path=True)
            if not resource_id:
                logger.warning(f"Failed to register clip {container_clip_path}. Skipping.")
                continue
                
            # System prompt reinforces the "Observer" role
            inspection_system_prompt = app_prompts.get("industrial_observer_system", "")
            if not inspection_system_prompt:
                logger.warning("industrial_observer_system prompt not found in YAML, fallback may be needed.")
            # Get VLM parameters from preset
            preset_config = VIDEO_PRESETS.get(preset_name, {})
            # Get SOP Section Name for Report (or default)
            sop_section_title = preset_config.get("sop_section_name", "Safety Protocols")
            
            vlm_config = preset_config.get("vlm_params", {})
            chunk_duration = vlm_config.get("chunk_duration", 7)
            chunk_overlap_duration = vlm_config.get("chunk_overlap_duration", 3)
            
            dense_captions = vss_engine_client.generate_dense_captions(
                resource_id, 
                prompt=guided_prompt,
                system_prompt=inspection_system_prompt,
                chunk_duration=chunk_duration,
                chunk_overlap_duration=chunk_overlap_duration
            )
            dense_timeline = vss_engine_client.format_timeline(dense_captions)
            
            logger.info(f"Generated Guided Timeline for {clip_basename}")
            logger.info(f"TIMELINE CONTENT: {dense_timeline}")
            
            # 2. Logic-Based Validation (LLM)
            # Compare the Guided Timeline (Facts) vs Alert Prompts (Rules)
            validation = llm_client.validate_compliance(
                dense_timeline=dense_timeline,
                user_criteria=alert_prompts_text
            )
            
            # 3. Evidence Extraction (Hero Frame)
            # Strategy: Use LLM reasoning or VLM timeline to find the key moment
            metadata_path = clip_path.replace(".mp4", ".json")
            overlay_filename = clip_basename.replace(".mp4", "_overlay.jpg")
            screenshot_path = os.path.join(OVERLAY_DIR, overlay_filename)
            
            # If overlay doesn't exist, create it
            if not os.path.exists(screenshot_path):
                 target_time = None
                 # Try to extract timestamp from LLM reasoning
                 try:
                     # validation.get("reasoning") might contain "at 12 seconds..."
                     target_time = find_hero_timestamp(validation.get("reasoning", ""))
                     if target_time is None or target_time == 0.0:
                         target_time = find_hero_timestamp(dense_timeline)
                 except:
                     pass
                     
                 # Render the overlay
                 render_cv_overlay(clip_path, metadata_path, screenshot_path, target_time=target_time)

            # 4. Compile Event Data
            finding = {
                "timestamp": datetime.now().strftime("%H:%M:%S"), # TODO: Use real timestamp from clip name
                "type": "Inspection Finding",
                "description": "See detailed timeline.",
                "compliance_score": validation.get("compliance_score", 0),
                "verification_reasoning": validation.get("reasoning", "No reasoning provided."),
                "dense_timeline": dense_timeline,
                "screenshot_path": screenshot_path,
                "sop_section": sop_section_title,
                "video_file": clip_basename
            }
            report_events.append(finding)
            
        except Exception as e:
            logger.error(f"Error processing clip {event}: {e}")
            continue

    # Generate PDF
    if not report_events:
        gr.Warning("No valid findings generated. Report will be empty.")
        
    # Generate Executive Summary & Final Verdict
    summary_data = {
        "overall_verdict": "FAIL" if any(e['compliance_score'] < 50 for e in report_events) else "PASS",
        "executive_summary": "Manual summary required (LLM generation failed).",
        "improvements": [],
        "positive_highlights": []
    }
    
    if report_events:
        try:
            summary_data = llm_client.generate_final_summary(report_events, alert_prompts_text)
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")

    # Prepare Video Source List
    # Extract from findings metadata
    clip_names = [e.get("video_file", "unknown") for e in report_events]
    video_source_str = ", ".join(clip_names)

    report_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "video_source": video_source_str,
        "sop_summary": alert_prompts_text,
        "verdict": summary_data.get("overall_verdict", "UNKNOWN"),
        "executive_summary": summary_data.get("executive_summary", ""),
        "improvements": summary_data.get("improvements", []),
        "positive_highlights": summary_data.get("positive_highlights", []),
        "events": report_events
    }
    timestamp = int(datetime.now().timestamp())
    unique_id = uuid.uuid4().hex[:6]
    report_filename = f"QA_Report_{timestamp}_{unique_id}.pdf"
    report_path = os.path.join(REPORTS_DIR, report_filename)
    
    try:
        final_pdf = report_generator.generate_pdf_report(report_data, report_path)
        if final_pdf and os.path.exists(final_pdf):
            gr.Info(f"Report Generated: {final_pdf}")
            # Also provide a download link if possible, or just the path for now
            return final_pdf
        else:
            gr.Error("PDF Generation Failed (No output file).")
            return None
    except Exception as e:
        logger.error(f"Critical PDF Generation Error: {e}")
        gr.Error(f"PDF Generation Failed: {e}")

# --- UI Layout ---
with gr.Blocks(title="Manufacturing QA") as app:
    components.header()
    # State variables
    stream_id_state = gr.State()
    # We keep a dummy state just to satisfy function signatures if needed, 

    sop_text_state = gr.State("") 
    # State to track processed clips
    # Mapping: filename -> { "result": "...", "vlm": "..." }
    processed_alerts_state = gr.State({})
    
    # NEW: State to store the raw list of session events for the final report
    session_events_state = gr.State([])
    
    # VLM Parameter States (accessible across tabs)
    vlm_chunk_dur_state = gr.State(0)
    vlm_temp_state = gr.State(0.2)
    vlm_top_p_state = gr.State(1.0)
    vlm_max_tokens_state = gr.State(256)
    vlm_frames_state = gr.State(10)
    vlm_reasoning_state = gr.State(True)
    vlm_width_state = gr.State(0)
    vlm_height_state = gr.State(0)
    rois_state = gr.State([[]])

    with gr.Tabs():
        # --- TAB 1: Knowledge Base (Removed) ---

        # --- TAB 2: CV Vision Pipeline Manager ---
        with gr.TabItem("CV Vision Pipeline Manager"):
            with gr.Row():
                with gr.Column():
                    video_preset, input_type, video_file, rtsp_url = components.video_input_component()
                    process_btn = gr.Button("🚀 Process Video", variant="primary", size="lg")
                    status_output = gr.Textbox(label="Status", interactive=False)
                    
                with gr.Column():
                    frame_skip, obj_thresh = components.cv_pipeline_parameters_component()
                    alert_prompts = components.vss_alert_parameters_component()
                    det_classes, box_thresh = components.detection_parameters_component()
            
            # Register Preset Change Handler
            video_preset.change(
                on_preset_change,
                inputs=[video_preset],
                outputs=[
                    input_type, video_file, det_classes, box_thresh, 
                    frame_skip, obj_thresh, alert_prompts,
                    vlm_chunk_dur_state, vlm_temp_state, vlm_top_p_state,
                    vlm_max_tokens_state, vlm_frames_state, vlm_reasoning_state,
                    vlm_width_state, vlm_height_state, rois_state
                ]
            )
            
            process_btn.click(
                start_inspection_handler,
                inputs=[
                    input_type, video_file, rtsp_url,
                    det_classes, box_thresh,
                    frame_skip, obj_thresh,
                    alert_prompts,
                    vlm_chunk_dur_state, vlm_temp_state, vlm_top_p_state,
                    vlm_max_tokens_state, vlm_frames_state, vlm_reasoning_state,
                    vlm_width_state, vlm_height_state, rois_state
                ],
                outputs=[status_output, stream_id_state, sop_text_state, session_events_state]
            )




        # --- TAB 4: Reports ---
        with gr.TabItem("Reports"):
            gr.Markdown("### 📑 Generation Reports")
            gen_report_btn = gr.Button("Generate PDF Report", variant="primary")
            report_output = gr.File(label="Download Report")
            
            gen_report_btn.click(
                generate_report_btn_handler,
                inputs=[session_events_state, alert_prompts, video_preset],
                outputs=[report_output]
            )

if __name__ == "__main__":
    port = int(os.getenv("APP_PORT", 7865))
    # CSS for VLM parameters modal overlay
    css = """
    #vlm-params-overlay {
        background: var(--background-fill-primary) !important;
        padding: 2rem !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3) !important;
        border: 2px solid var(--border-color-primary) !important;
        margin: 1rem auto !important;
        max-width: 90% !important;
    }
    #alert-table {
        overflow-x: auto !important;
        max-height: 500px !important;
    }
    #alert-table table {
        table-layout: auto !important;
        width: 100% !important;
    }
    """
    
    app.launch(server_name="0.0.0.0", server_port=port, allowed_paths=[MEDIA_BASE_DIR], theme=theme, css=css)
