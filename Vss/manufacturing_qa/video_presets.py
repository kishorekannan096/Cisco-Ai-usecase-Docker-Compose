VIDEO_PRESETS = {
    "conveyor_belt_inspection_sdg_1080p.mp4": {
        "det_classes": "box\npackage",
        "box_thresh": 0.65,
        "frame_skip": 2,
        "obj_thresh": 1,
        "sop_section_name": "Packaging & Shipping Standards",
        "alert_prompts": "Inspect the cardboard boxes on the conveyor belt for: 1) Crumpling, 2) Tearing, 3) Dents, 4) Creases, 5) Open flaps/boxes.",
        "vlm_params": {
            "chunk_duration": 5,
            "chunk_overlap_duration": 1,
            "temperature": 0.2,
            "top_p": 1.0,
            "max_tokens": 256,
            "frames_per_chunk": 10,
            "enable_reasoning": True,
            "vlm_width": 0,
            "vlm_height": 0
        },
        "rois": [[]]
    },
    "warehouse_multistream_3.mp4": {
        "det_classes": "person",
        "box_thresh": 0.65,
        "frame_skip": 2,
        "obj_thresh": 1,
        "sop_section_name": "PPE & Safety Protocols",
        "alert_prompts": "Inspect all individuals in the scene for Personal Protective Equipment (PPE) compliance. Specifically describe: 1) Presence of high-visibility safety vests (e.g. orange/yellow vests). 2) Presence of safety helmets. 3) Proximity to the moving forklift.",
        "vlm_params": {
            "chunk_duration": 5,
            "chunk_overlap_duration": 1,
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 512,
            "frames_per_chunk": 15,
            "enable_reasoning": True,
            "vlm_width": 0,
            "vlm_height": 0
        },
        "rois": [[]]
    }
    
}
