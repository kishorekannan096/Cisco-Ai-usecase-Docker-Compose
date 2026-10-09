import os
import logging
import requests
import json
from typing import Dict, Any, List, Optional
from utils.prompt_loader import load_prompts

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, api_url: Optional[str] = None):
        self.api_url = api_url or os.getenv("LLM_API_URL", "http://10.79.252.16:8999/v1/chat/completions")
        self.api_url = self.api_url.rstrip("/")
        self.model = os.getenv("LLM_MODEL_NAME", "nvidia/llama-3.3-nemotron-super-49b-v1.5")
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        self.prompts = load_prompts()

    def generate_final_summary(self, findings: List[Dict[str, Any]], user_criteria: str) -> Dict[str, Any]:
        """
        Generates an Executive Summary, Improvements, and Highlights based on all findings.
        Acts as the 'Chief Inspector'.
        """
        system_prompt_tpl = self.prompts.get("final_summary_system", "")
        if not system_prompt_tpl:
            logger.warning("final_summary_system prompt not found in YAML, fallback may be needed.")
            
        system_prompt = system_prompt_tpl.replace("{user_criteria}", user_criteria)
        
        # Prepare findings summary for context window efficiency
        findings_text = ""
        for i, f in enumerate(findings):
            score = f.get('compliance_score', 0)
            verdict = "PASS" if score >= 50 else "FAIL"
            reasoning = f.get('verification_reasoning', 'No details provided.')
            findings_text += f"\nEvent {i+1}: Verdict={verdict}, Score={score}, Reasoning={reasoning}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"INSPECTION FINDINGS:\n{findings_text}"}
            ],
            "max_tokens": 1024,
            "temperature": 0.3,
            "stream": False,
            "response_format": {"type": "json_object"}
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=45)
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e:
            logger.error(f"Executive Summary generation failed: {e}")
            return {
                "overall_verdict": "FAIL" if any(f.get('params', {}).get('compliance_score', 0) < 50 for f in findings) else "PASS",
                "executive_summary": "Automated summary failed. Please review individual events.",
                "improvements": ["Review system logs."],
                "positive_highlights": ["Manual review required."]
            }

    def validate_compliance(self, dense_timeline: str, user_criteria: str) -> Dict[str, Any]:
        """
        Validates compliance based on guided dense captions and user criteria.
        """
        system_prompt_tpl = self.prompts.get("validate_compliance_system", "")
        if not system_prompt_tpl:
            logger.warning("validate_compliance_system prompt not found in YAML, fallback may be needed.")

        system_prompt = system_prompt_tpl.replace("{user_criteria}", user_criteria)
        
        
        user_content = f"OBSERVER REPORT:\n{dense_timeline}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "max_tokens": 1024,
            "temperature": 0.2,
            "stream": False,
            "response_format": {"type": "json_object"}
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=45)
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e:
            logger.error(f"Compliance validation failed: {e}")
            return {
                "verdict": "ERROR",
                "compliance_score": 0,
                "reasoning": f"Validation failed: {str(e)}",
                "positive_findings": [],
                "negative_findings": ["System Error"]
            }

if __name__ == "__main__":
    # Test
    client = LLMClient()
    res = client.audit_timeline_chunk(
        "A brown cardboard box with a large tear on the side moves along the belt.",
        "Final Product Inspection: Boxes must be free of tears, dents, and creases."
    )
    print(res)
