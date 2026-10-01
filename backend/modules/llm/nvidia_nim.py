import os
import requests
import json

class ForensicLLM:
    def __init__(self):
        self.nvidia_key = os.environ.get("NVIDIA_API_KEY")
        self.provider = "nvidia" if self.nvidia_key else "none"

    def generate_explanation(self, profile_data: dict) -> str:
        prompt = f"""You are a forensic report writer.
Use only the supplied MediaDNA structured evidence.
Do not invent facts.
Do not invent evidence.
Do not invent timestamps.
Do not infer unsupported provenance.
Do not modify model scores.
Do not determine classification independently.
State uncertainty and limitations clearly.

Profile Data:
{json.dumps(profile_data, indent=2)}

Generate these sections exactly (as raw text or Markdown):
1. Forensic Summary
2. Evidence Interpretation
3. Manipulation Interpretation
4. Provenance Interpretation
5. Limitations
6. Recommended Interpretation
"""
        
        if self.provider == "nvidia":
            try:
                invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.nvidia_key}",
                    "Accept": "application/json",
                }
                payload = {
                  "messages": [
                    {
                      "role": "user",
                      "content": prompt
                    }
                  ],
                  "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
                  "max_tokens": 1024,
                  "temperature": 0.6,
                  "top_p": 0.95
                }
                
                response = requests.post(invoke_url, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"NVIDIA API Error: {e}")
                self.provider = "none"
                # Fallback to none below
                
        if self.provider == "none":
            return self._generate_deterministic(profile_data)
            
    def _generate_deterministic(self, data: dict) -> str:
        c = data.get("classification", {})
        visual_anomaly = data.get('visual', {}).get('anomaly_score', 0)
        audio_anomaly = data.get('audio', {}).get('anomaly_score', 0)
        temporal_anomaly = data.get('temporal', {}).get('anomaly_score', 0)
        return f"""
1. Forensic Summary
The MediaDNA system classified the input as {c.get('label', 'UNKNOWN').upper()} with a probability of {c.get('fake_probability', 0):.2f}. 

2. Evidence Interpretation
Visual anomaly score: {visual_anomaly}. 
Audio anomaly score: {audio_anomaly}.
Temporal anomaly score: {temporal_anomaly}.

3. Manipulation Interpretation
Manipulation category: {data.get('manipulation', {}).get('category', 'Unknown')}.

4. Provenance Interpretation
Provenance category: {data.get('provenance', {}).get('category', 'Unknown')}.

5. Limitations
{', '.join(data.get('limitations', []))}

6. Recommended Interpretation
This assessment is fully automated. Human review is required.
"""
