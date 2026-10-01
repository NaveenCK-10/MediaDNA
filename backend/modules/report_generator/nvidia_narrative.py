import os
import json
import logging
from openai import OpenAI
import httpx

logger = logging.getLogger("mediadna.nvidia")

class NvidiaNarrativeGenerator:
    def __init__(self):
        # Allow override from env, but do not fail if missing
        self.api_key = os.getenv("NVIDIA_API_KEY")
        self.base_url = "https://integrate.api.nvidia.com/v1"
        self.model = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
        self.timeout = int(os.getenv("NVIDIA_TIMEOUT_SECONDS", "20"))
        
    def _create_fallback_narrative(self, normalized_data: dict) -> dict:
        decision_state = normalized_data.get("decision", {}).get("state", "UNCERTAIN")
        return {
            "executive": f"MediaDNA produced an {decision_state.lower()} assessment from the available multimodal evidence. The reported scores and metadata below are derived from the current analysis run.",
            "evidence": "Analysis was successfully executed on the provided media asset.",
            "modality": "Specialist models produced the raw scores reflected in the data table.",
            "context": "This is a deterministic fallback narrative generated without an LLM.",
            "limitations": "The report was generated without natural language enrichment due to an upstream model error or timeout."
        }

    def generate_narrative(self, normalized_data: dict) -> dict:
        if not self.api_key:
            logger.warning("NVIDIA_API_KEY not found. Using fallback narrative.")
            return self._create_fallback_narrative(normalized_data)
            
        system_prompt = """You are the narrative assistant for MediaDNA, a multimodal media authenticity-analysis prototype.

You are NOT the detector.
You MUST NOT invent measurements, probabilities, model outputs, metadata, provenance, timestamps, evidence, manipulated regions, or technical findings.
You must only explain the structured evidence supplied in the input.

Treat:
- calibrated probability as a calibrated model output
- raw model scores as scores, not probabilities
- model-sensitive attribution as attribution, not ground-truth manipulation localization
- SHA-256 as file-byte identity, not proof of authenticity or origin
- provenance as unsupported unless explicitly validated
- uncertainty as a decision state, not proof that the media is real or fake

Never claim:
- guaranteed fake
- guaranteed authentic
- exact generator attribution
- manipulated pixel ground truth
- legal proof
- investigative proof
- causal explanation unsupported by the supplied evidence

Return concise professional forensic-style prose with these sections:
1. Executive Interpretation
2. Evidence Summary
3. Modality Interpretation
4. Assessment Context
5. Limitations

Every statement must be traceable to the supplied structured data.
If evidence is insufficient, explicitly say so.
Do not override the final MediaDNA decision.

Provide your response strictly as a JSON object with keys: "executive", "evidence", "modality", "context", "limitations"."""

        try:
            client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=httpx.Timeout(self.timeout)
            )
            
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(normalized_data)}
                ],
                temperature=0.1,
                top_p=0.95,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            content = completion.choices[0].message.content
            narrative = json.loads(content)
            
            # Verify structure
            required_keys = ["executive", "evidence", "modality", "context", "limitations"]
            for k in required_keys:
                if k not in narrative:
                    narrative[k] = ""
            
            return narrative
            
        except Exception as e:
            logger.error(f"NVIDIA API failed: {e}. Falling back to deterministic narrative.")
            return self._create_fallback_narrative(normalized_data)
