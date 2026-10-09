import gradio as gr

def header():
    """Renders the application header."""
    gr.Markdown(
        """
        # Manufacturing Quality Assurance System
        ### AI-Powered Inspections
        """
    )



def video_input_component():
    """Component for Video Input Selection."""
    with gr.Group():
        gr.Markdown("### 📂 Input Source")
        
        # Add Video Preset Dropdown
        import os
        videos_dir = os.getenv("SAMPLE_VIDEOS_DIR", "./videos")
        video_files = [f for f in os.listdir(videos_dir) if f.endswith(".mp4")] if os.path.exists(videos_dir) else []
        
        video_preset = gr.Dropdown(
            choices=["Custom"] + video_files, 
            value="Custom", 
            label="Video Preset",
            info="Select a pre-configured sample video to auto-populate all parameters below."
        )
        
        input_type = gr.Dropdown(choices=["Video File", "RTSP Stream"], value="Video File", label="Input Source")
        video_file = gr.Video(label="Upload Video", sources=["upload"])
        rtsp_url = gr.Textbox(label="RTSP URL", placeholder="rtsp://...", visible=False)
        
        def toggle_input(choice):
            return {
                video_file: gr.Video(visible=(choice == "Video File")),
                rtsp_url: gr.Textbox(visible=(choice == "RTSP Stream"))
            }
            
        input_type.change(toggle_input, inputs=[input_type], outputs=[video_file, rtsp_url])
        
        return video_preset, input_type, video_file, rtsp_url

def detection_parameters_component():
    """Component for CV Detection Parameters."""
    with gr.Group():
        gr.Markdown("### 🎯 Detection Parameters")
        detection_classes = gr.Textbox(
            label="Detection Classes (one per line)", 
            placeholder="person\nforklift\nhelmet",
            lines=3,
            value="person",
            info="Object classes to detect in video. One class per line (e.g., person, forklift, helmet)"
        )
        box_threshold = gr.Slider(
            minimum=0.1, maximum=1.0, value=0.3, 
            label="Box Threshold", 
            info="Minimum confidence score (0-1) for detection. Higher = fewer but more confident detections"
        )
        return detection_classes, box_threshold

def cv_pipeline_parameters_component():
    """Component for Pipeline Parameters."""
    with gr.Group():
        gr.Markdown("### 🎞️ CV Pipeline Parameters")
        frame_skip = gr.Slider(
            minimum=0, maximum=60, value=5, step=1, 
            label="Frame Skip", 
            info="Process every Nth frame. Higher = faster processing but might miss events"
        )
        obj_threshold = gr.Slider(
            minimum=1, maximum=100, value=1, step=1, 
            label="Object Detection Threshold", 
            info="Minimum number of detected objects to trigger clip recording"
        )
        return frame_skip, obj_threshold

def vss_alert_parameters_component():
    """Component for VSS Alert Logic."""
    with gr.Group():
        gr.Markdown("### 📋 Inspection Criteria")
        

        alert_prompts = gr.Textbox(
            label="QA Checklist (one per line)",
            placeholder="e.g., Is the worker wearing a safety helmet?\nIs the package sealed correctly?",
            lines=3,
            info="Define the safety protocols or quality standards for the VLM to verify against detected events."
        )
        
        return alert_prompts


