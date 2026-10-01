import os
import json
import requests

class ForensicReportGenerator:
    def __init__(self):
        self.api_key = os.environ.get("NVIDIA_NEMOTRON_API_KEY") # User provided key

        
    def generate_report(self, profile_data: dict) -> str:
        if not self.api_key:
            return "Insufficient evidence / LLM service unavailable. The detection pipeline remains fully operational. Please refer to the MediaDNA scores for forensic metrics."
            
        prompt = f"""
        You are the reporting layer for a digital forensic analysis system.
        Use only the supplied MediaDNA evidence below.
        Do not invent facts, scores, timestamps, provenance, evidence, or model outputs.
        Do not independently classify the media.
        Do not override model results.
        Clearly distinguish measured evidence, model outputs, heuristic assessments, and uncertainty.
        When evidence is insufficient, explicitly state that.
        
        Profile Data:
        {json.dumps(profile_data, indent=2)}
        
        Report Structure:
        1. Executive Summary
        2. Visual & Temporal Evidence Analysis
        3. Audio Evidence Analysis
        4. Cloud Forensics (NVIDIA NIM)
        5. Limitations
        """
        
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # We use a standard text-based LLM payload for now
        payload = {
            "model": "nvidia/nemotron-4-340b-instruct", # Using a highly capable text model since omni is VLM, but we just need text here. Wait, user specified nvidia/nemotron-3-nano-omni-30b-a3b-reasoning. I'll use exactly that!
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 1024
        }
        
        # Override to user requested model
        payload["model"] = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                res_json = response.json()
                return res_json["choices"][0]["message"]["content"]
            else:
                print(f"[LLM] Nemotron Error: {response.status_code} - {response.text}")
                return "Error generating report: LLM service returned an error."
        except Exception as e:
            print(f"[LLM] Nemotron Exception: {e}")
            return f"Error generating LLM report: {e}"
