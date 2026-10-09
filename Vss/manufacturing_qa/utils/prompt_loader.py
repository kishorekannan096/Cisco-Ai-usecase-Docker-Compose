import os
import yaml
import logging

logger = logging.getLogger(__name__)

def load_prompts():
    """Loads prompts from prompt.yaml."""
    # Find prompt.yaml relative to this file
    # utils/prompt_loader.py -> ../prompt.yaml
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_file = os.path.join(base_dir, "prompt.yaml")
    
    if not os.path.exists(prompt_file):
        logger.error(f"Prompt file not found: {prompt_file}")
        return {}
        
    try:
        with open(prompt_file, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load prompts: {e}")
        return {}
