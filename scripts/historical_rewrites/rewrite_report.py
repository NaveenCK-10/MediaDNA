import os

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MediaDNA Forensic Report</title>
</head>
<body>

    <!-- Cover Page -->
    <div class="cover-page">
        <h1 class="cover-title">MEDIA DNA</h1>
        <h2 class="cover-subtitle">Deepfake Provenance and Authenticity Analysis</h2>
        <div class="panel">
            <p><strong>Case ID:</strong> {{ profile.case_id }}</p>
            <p><strong>Asset ID:</strong> {{ profile.asset_id }}</p>
            <p><strong>Run ID:</strong> {{ profile.run_id }}</p>
            <p><strong>SHA-256:</strong> {{ profile.asset_hash }}</p>
            <p><strong>Analysis Date:</strong> {{ date }}</p>
            <p><strong>Report Version:</strong> V20 Final</p>
        </div>
    </div>

    <!-- Executive Summary -->
    <div class="page-break"></div>
    <h1>Executive Forensic Summary</h1>
    
    <div class="panel">
        <h2 style="text-align:center; font-size: 28pt; margin:0;" class="{% if profile.classification.label == 'fake' %}text-fake{% else %}text-real{% endif %}">
            {{ profile.model_finding | upper }}
        </h2>
        <p style="text-align:center; font-size: 14pt;">HUMAN DETERMINATION: {{ profile.human_determination | upper }}</p>
    </div>

    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Model: {{ profile.model_name }} ({{ profile.model_version }})</div>
            <div class="metric-value">Threshold: {{ profile.threshold }}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Raw Model Score</div>
            <div class="metric-value">{{ "%.4f"|format(profile.trust.raw_model_score) }}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Calibrated Prob / Status</div>
            <div class="metric-value">{{ "%.2f"|format(profile.trust.calibrated_probability * 100) }}% ({{ profile.trust.calibration_status }})</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">OOD Status</div>
            <div class="metric-value">{{ profile.trust.ood_status }}</div>
        </div>
    </div>

    <h2>Final System State</h2>
    <div class="panel">
        <p><strong>Abstention State:</strong> {{ profile.trust.abstention_state }}</p>
        <p><strong>Model Confidence:</strong> {{ profile.trust.model_confidence }}</p>
        <p><strong>Evidence Agreement:</strong> {{ profile.trust.evidence_agreement }}</p>
    </div>

    <!-- Media Profile -->
    <div class="page-break"></div>
    <h1>Media Profile & Quality Gate</h1>
    <div class="panel">
        <table>
            <tr><th>Filename</th><td>{{ profile.case_id }}</td></tr>
            <tr><th>SHA-256</th><td>{{ profile.asset_hash }}</td></tr>
            <tr><th>Size (Bytes)</th><td>{{ profile.file_size_bytes }}</td></tr>
            <tr><th>Resolution</th><td>{{ profile.quality.resolution }}</td></tr>
            <tr><th>Duration (s)</th><td>{{ profile.quality.duration }}</td></tr>
            <tr><th>FPS</th><td>{{ profile.quality.frame_rate }}</td></tr>
            <tr><th>Audio Presence</th><td>{{ profile.quality.audio_presence }}</td></tr>
            <tr><th>Quality Findings</th><td>{{ profile.quality.findings | join(", ") if profile.quality.findings else "None" }}</td></tr>
        </table>
    </div>
    
    <!-- Evidence -->
    <div class="page-break"></div>
    <h1>Evidence Items</h1>
    {% for ev in profile.evidence_items %}
    <div class="panel" style="margin-bottom: 20px;">
        <h3>{{ ev.type | replace("_", " ") | title }} ({{ ev.modality | title }})</h3>
        <p><strong>Model:</strong> {{ ev.model }} ({{ ev.model_version }}) - Method: {{ ev.method }}</p>
        <p><strong>Value:</strong> {{ "%.4f"|format(ev.value) }} {{ ev.unit }}</p>
        <p><strong>Reliability:</strong> {{ ev.reliability }} | <strong>Calibration:</strong> {{ ev.calibration_status }}</p>
        <p><strong>Interpretation:</strong> {{ ev.interpretation }}</p>
        <p><strong>Limitations:</strong> {{ ev.limitations }}</p>
    </div>
    {% endfor %}

    <!-- Manipulation & Provenance -->
    <div class="page-break"></div>
    <h1>Manipulation & Provenance</h1>
    <div class="panel">
        <h3>Broad Manipulation Family</h3>
        <p>{{ profile.provenance_v20.broad_manipulation_family }}</p>
        
        <h3>Heuristic Provenance</h3>
        <p>{{ profile.provenance_v20.heuristic_provenance }}</p>
        
        <h3>Exact Generator Attribution</h3>
        <p>{{ profile.provenance_v20.exact_generator_attribution }}</p>
        
        <h3>Metadata / Cryptographic</h3>
        <p>{{ profile.provenance_v20.metadata_provenance }} / {{ profile.provenance_v20.cryptographic_provenance }}</p>
    </div>

    <!-- AI Forensic Explanation -->
    <div class="page-break"></div>
    <h1>AI Forensic Explanation</h1>
    <div class="llm-report">
{{ llm_text }}
    </div>
    
    <!-- System Limitations -->
    <div class="page-break"></div>
    <h1 style="color: #ef4444;">SYSTEM LIMITATIONS</h1>
    <div class="panel" style="border-left: 4px solid #ef4444; background: rgba(239, 68, 68, 0.1);">
        <p><strong>CRITICAL WARNING:</strong> Current benchmark performance is based on the validated research evaluation protocol and should not be represented as universal real-world accuracy.</p>
        <p>Limitations noted in this analysis:</p>
        <ul>
            {% for limitation in profile.limitations %}
            <li>{{ limitation }}</li>
            {% endfor %}
        </ul>
        <p>This system performs attribution/occlusion sensitivity and does not perform exact pixel-level localization.</p>
        <p>Raw fake probability must not be treated as human confidence.</p>
    </div>

    <div class="footer-disclaimer">
        Processing Timestamp: {{ date }}<br>
        This report was generated autonomously by the MediaDNA V20 Forensic Pipeline. 
        It is intended for research and demonstration purposes only.
    </div>

</body>
</html>"""

with open('c:/Users/navee/Desktop/Projects/MediaDna/OpenAVFF/backend/templates/forensic_report/report.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("Updated report.html successfully.")
