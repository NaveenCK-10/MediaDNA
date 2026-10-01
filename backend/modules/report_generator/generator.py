import os
import json
import logging
import datetime
import tempfile
import asyncio
from jinja2 import Environment, FileSystemLoader
from playwright.sync_api import sync_playwright
from backend.modules.report_generator.nvidia_narrative import NvidiaNarrativeGenerator

logger = logging.getLogger("mediadna.report_generator")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

class ReportGenerator:
    def __init__(self):
        self.template_dir = os.path.join(PROJECT_ROOT, "backend", "templates", "forensic_report")
        self.env = Environment(loader=FileSystemLoader(self.template_dir))
        self.narrative_gen = NvidiaNarrativeGenerator()
        
    def _normalize_data(self, profile_data: dict) -> dict:
        """Create a strict deterministic normalized report object."""
        case_id = profile_data.get("id", "UNKNOWN_CASE")
        
        # Determine state
        state = profile_data.get("status", "FAILED")
        if state == "COMPLETED":
            state = profile_data.get("trust", {}).get("abstention_state", "UNCERTAIN").upper()
            
        normalized = {
            "case": {
                "case_id": case_id,
                "analysis_id": profile_data.get("analysis_id", case_id),
                "filename": profile_data.get("media_filename", "Unknown"),
                "timestamp": profile_data.get("completed_at", datetime.datetime.now().isoformat())
            },
            "asset": {
                "sha256": profile_data.get("media_sha256", "Unavailable"),
                "duration_seconds": profile_data.get("media_info", {}).get("video_duration", 0),
                "fps": profile_data.get("media_info", {}).get("fps", 0),
                "width": profile_data.get("media_info", {}).get("width", 0),
                "height": profile_data.get("media_info", {}).get("height", 0),
                "audio_available": profile_data.get("media_info", {}).get("audio_presence", False)
            },
            "models": {
                "visual": "V22.3 Visual Specialist",
                "audio": "V22.3 Audio Specialist",
                "fusion": "0.5 visual + 0.5 audio",
                "calibration": "V22.3 Platt Calibrator"
            },
            "scores": {
                "visual": profile_data.get("visual", {}).get("raw_model_score", None),
                "audio": profile_data.get("audio", {}).get("raw_model_score", None),
                "fusion": profile_data.get("fusion", {}).get("raw_model_score", None),
                "calibrated_probability": profile_data.get("trust", {}).get("calibrated_probability", None)
            },
            "decision": {
                "state": state,
                "reason": "Calibrated probability threshold applied."
            },
            "evidence": {
                "modality_agreement": "N/A" if not profile_data.get("media_info", {}).get("audio_presence", False) else (
                    "Agreed" if abs((profile_data.get("visual", {}).get("raw_model_score") or 0) - (profile_data.get("audio", {}).get("raw_model_score") or 0)) < 0.2 else "Disagreed"
                ),
                "audio_status": "Present" if profile_data.get("media_info", {}).get("audio_presence", False) else "Missing/Unavailable",
                "attribution_type": "model-sensitive attribution",
                "provenance_status": "unsupported"
            },
            "limitations": [
                "The MediaDNA output is a model-assisted authenticity assessment and should not be interpreted as definitive proof of authenticity, manipulation, provenance, or source attribution."
            ],
            "narrative": {}
        }
        
        return normalized

    def generate_pdf(self, profile_data: dict, output_path: str):
        try:
            # 1. Normalize data
            normalized = self._normalize_data(profile_data)
            
            # 2. Get LLM Narrative (Optional/Graceful fallback)
            narrative = self.narrative_gen.generate_narrative(normalized)
            normalized["narrative"] = narrative
            
            # 3. Render HTML
            template = self.env.get_template("report.html")
            
            # Prevent injection by passing strictly structured normalized object
            html_out = template.render(
                report=normalized,
                date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                project_root=PROJECT_ROOT
            )
            
            # Inject CSS directly into HTML for Playwright
            css_path = os.path.join(self.template_dir, "style.css")
            if os.path.exists(css_path):
                with open(css_path, "r", encoding='utf-8') as f:
                    css_content = f.read()
                html_out = html_out.replace("</head>", f"<style>{css_content}</style></head>")
            
            # 4. Save PDF using Playwright
            temp_html_fd, temp_html_path = tempfile.mkstemp(suffix=".html")
            with os.fdopen(temp_html_fd, 'w', encoding='utf-8') as f:
                f.write(html_out)
                
            try:
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()
                    # Fix Windows file URI handling
                    file_uri = f"file:///{temp_html_path.replace(os.sep, '/')}"
                    page.goto(file_uri)
                    
                    page.pdf(
                        path=output_path, 
                        format="A4",
                        print_background=True,
                        margin={"top": "15mm", "bottom": "15mm", "left": "15mm", "right": "15mm"}
                    )
                    browser.close()
            finally:
                if os.path.exists(temp_html_path):
                    try:
                        os.remove(temp_html_path)
                    except Exception as ex:
                        logger.warning(f"Could not remove temp HTML: {ex}")
                        
            # Validate output
            if not os.path.exists(output_path) or os.path.getsize(output_path) < 100:
                raise RuntimeError("PDF generation produced empty or invalid file.")
                
            with open(output_path, "rb") as f:
                header = f.read(5)
                if not header.startswith(b"%PDF-"):
                    raise RuntimeError("Output file is not a valid PDF.")
                    
            return output_path
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            raise RuntimeError(f"Report generation failed: {e}")
