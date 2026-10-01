import os

def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

# HTML
html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MediaDNA | Forensic Command Center</title>
    <link rel="stylesheet" href="css/style.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
</head>
<body>
    <div class="grid-overlay"></div>
    
    <nav class="navbar">
        <div class="nav-brand">
            <span class="logo-mark">🧬</span>
            <span class="logo-text">MEDIADNA</span>
        </div>
        <div class="nav-links">
            <a href="#architecture">ARCHITECTURE</a>
            <a href="#research">RESEARCH</a>
            <a href="#evidence">EVIDENCE</a>
            <a href="https://github.com/NaveenCK-10/MediaDNA" target="_blank" class="btn-outline">GITHUB</a>
        </div>
    </nav>

    <header class="hero">
        <div class="hero-content">
            <div class="status-badge">[ SYSTEM ONLINE ]</div>
            <h1>MEDIA AUTHENTICITY<br><span class="accent-text">FORENSIC ANALYSIS</span></h1>
            <p class="subtitle">An evidence-oriented audiovisual forensic workflow combining multimodal analysis, calibration, uncertainty-aware decisions, provenance metadata, and structured reporting.</p>
            
            <div class="hero-actions">
                <a href="#pipeline" class="btn-primary">EXPLORE PIPELINE</a>
                <a href="#results" class="btn-secondary">VERIFIED RESULTS</a>
            </div>
        </div>
    </header>

    <section class="system-hud">
        <div class="container">
            <div class="hud-grid">
                <div class="hud-item active"><span class="dot"></span>MEDIA INGEST</div>
                <div class="hud-item active"><span class="dot"></span>VISUAL ANALYSIS</div>
                <div class="hud-item active"><span class="dot"></span>AUDIO ANALYSIS</div>
                <div class="hud-item active"><span class="dot"></span>MULTIMODAL FUSION</div>
                <div class="hud-item active"><span class="dot"></span>CALIBRATION</div>
                <div class="hud-item active"><span class="dot"></span>UNCERTAINTY</div>
                <div class="hud-item active"><span class="dot"></span>EVIDENCE</div>
                <div class="hud-item active"><span class="dot"></span>REPORTING</div>
            </div>
        </div>
    </section>

    <section id="architecture" class="section-dark">
        <div class="container">
            <h2 class="section-title">CORE ARCHITECTURE</h2>
            <div class="architecture-diagram">
                <!-- SVG Architecture Placeholder -->
                <div class="arch-flow">
                    <div class="arch-box">Frontend<br><span class="tech-stack">React / Vite / TS</span></div>
                    <div class="arch-arrow">↓</div>
                    <div class="arch-box">FastAPI Backend<br><span class="tech-stack">Python / Uvicorn</span></div>
                    <div class="arch-arrow">↓</div>
                    <div class="arch-box">Media Validation &amp; Extraction<br><span class="tech-stack">FFmpeg / ffprobe</span></div>
                    <div class="arch-arrow">↓</div>
                    <div class="arch-split">
                        <div class="arch-box">Visual Analysis<br><span class="tech-stack">16 Frames (224x224)</span></div>
                        <div class="arch-box">Audio Analysis<br><span class="tech-stack">16kHz Mel-Spectrogram</span></div>
                    </div>
                    <div class="arch-arrow">↓</div>
                    <div class="arch-box highlight">AVFF Multimodal Inference<br><span class="tech-stack">VideoCAVMAEFT Cross-Attention</span></div>
                    <div class="arch-arrow">↓</div>
                    <div class="arch-box highlight-purple">Platt Calibration<br><span class="tech-stack">Logistic Regression</span></div>
                    <div class="arch-arrow">↓</div>
                    <div class="arch-box">3-State Decision Policy</div>
                    <div class="arch-arrow">↓</div>
                    <div class="arch-box">Playwright PDF Report</div>
                </div>
            </div>
        </div>
    </section>

    <section id="decision-engine" class="section-light">
        <div class="container">
            <h2 class="section-title">DECISION ENGINE</h2>
            <p class="section-desc">Traditional deepfake detectors force hard binary decisions, destroying forensic credibility on ambiguous media. MediaDNA translates raw model logits into calibrated probabilities and applies an explicit uncertainty bracket.</p>
            
            <div class="decision-grid">
                <div class="decision-card authentic">
                    <h3>AUTHENTIC</h3>
                    <p>Evidence supports authenticity under the selected DEV operating policy (P ≤ 0.95).</p>
                </div>
                <div class="decision-card uncertain">
                    <h3>UNCERTAIN</h3>
                    <p>Evidence is insufficient for a confident binary conclusion (0.95 &lt; P &lt; 0.96).</p>
                </div>
                <div class="decision-card synthetic">
                    <h3>SYNTHETIC</h3>
                    <p>Evidence crosses the selected synthetic decision boundary (P ≥ 0.96).</p>
                </div>
            </div>
        </div>
    </section>

    <section id="results" class="section-dark">
        <div class="container">
            <h2 class="section-title">VERIFIED RESULTS</h2>
            <p class="warning-text">⚠️ Metrics are protocol-specific and operating-point dependent.</p>
            
            <div class="metrics-dashboard">
                <div class="metric-card">
                    <h4>DEV RANKING</h4>
                    <div class="metric-value">0.8974</div>
                    <div class="metric-label">ROC-AUC</div>
                </div>
                <div class="metric-card highlight-card">
                    <h4>LOCKED TEST RANKING</h4>
                    <div class="metric-value">0.9094</div>
                    <div class="metric-label">ROC-AUC</div>
                </div>
                <div class="metric-card">
                    <h4>LOCKED SPECIFICITY</h4>
                    <div class="metric-value">0.0000</div>
                    <div class="metric-label">At historical operating point</div>
                </div>
            </div>
            
            <div class="research-note">
                <h4>🔬 Scientific Context: The Specificity Collapse</h4>
                <p>The locked-test evaluation produced a high ROC-AUC (0.9094), indicating excellent threshold-independent ranking discrimination. However, at the historically selected uncalibrated operating point, the model labeled every single test sample as synthetic (Specificity = 0, MCC = 0). This pathological behavior under severe class imbalance is the exact reason MediaDNA mandates proper Platt calibration and uncertainty policies.</p>
            </div>
        </div>
    </section>

    <section id="research" class="section-light">
        <div class="container">
            <h2 class="section-title">RESEARCH FOUNDATION</h2>
            <p>MediaDNA is an applied forensic-analysis workflow built entirely around the <strong>AVFF</strong> audiovisual detector family.</p>
            
            <div class="citation-box">
                <p class="citation-title">AVFF: Audio-Visual Feature Fusion for Video Deepfake Detection</p>
                <p class="citation-authors">Oorloff, T. A., Masi, I., &amp; Hosseini, R. (CVPR 2024)</p>
            </div>
            
            <p class="section-desc mt-4">MediaDNA extends this core research architecture with real-time SSE extraction pipelines, strict calibration loops, modality ablation diagnostics, and deterministic PDF forensic reporting.</p>
        </div>
    </section>

    <footer class="footer">
        <div class="container">
            <div class="footer-grid">
                <div>
                    <h4>MEDIADNA</h4>
                    <p>Multimodal Deepfake Provenance</p>
                </div>
                <div>
                    <h4>DISCLAIMER</h4>
                    <p class="small-text">MediaDNA is a research prototype. Its outputs and PDF reports should not be treated as definitive proof of authenticity, physical provenance, or legal evidence.</p>
                </div>
            </div>
        </div>
    </footer>

    <script src="js/app.js"></script>
</body>
</html>
"""

# CSS
css_content = """:root {
    --bg-dark: #07090f;
    --bg-card: #0d121c;
    --bg-card-hover: #151d2b;
    --accent-cyan: #00e5ff;
    --accent-purple: #9d4edd;
    --accent-blue: #2563eb;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --border-color: rgba(0, 229, 255, 0.15);
    
    --authentic: #10b981;
    --uncertain: #f59e0b;
    --synthetic: #ef4444;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    background-color: var(--bg-dark);
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
    line-height: 1.6;
    overflow-x: hidden;
}

.grid-overlay {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image: 
        linear-gradient(rgba(0, 229, 255, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 229, 255, 0.03) 1px, transparent 1px);
    background-size: 30px 30px;
    z-index: -1;
    pointer-events: none;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 2rem;
}

h1, h2, h3, h4, .font-mono {
    font-family: 'JetBrains Mono', monospace;
}

.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.5rem 2rem;
    border-bottom: 1px solid var(--border-color);
    background: rgba(7, 9, 15, 0.8);
    backdrop-filter: blur(10px);
    position: sticky;
    top: 0;
    z-index: 100;
}

.logo-brand { display: flex; align-items: center; gap: 0.5rem; }
.logo-mark { font-size: 1.5rem; }
.logo-text { font-family: 'JetBrains Mono', monospace; font-weight: 700; letter-spacing: 2px; }

.nav-links { display: flex; gap: 2rem; align-items: center; }
.nav-links a { color: var(--text-secondary); text-decoration: none; font-size: 0.875rem; font-weight: 600; letter-spacing: 1px; transition: color 0.2s; }
.nav-links a:hover { color: var(--accent-cyan); }

.btn-outline { border: 1px solid var(--accent-cyan); padding: 0.5rem 1.25rem; border-radius: 4px; color: var(--accent-cyan) !important; }
.btn-outline:hover { background: rgba(0, 229, 255, 0.1); }

.hero {
    padding: 8rem 0 6rem;
    text-align: center;
}

.status-badge {
    display: inline-block;
    color: var(--accent-cyan);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.875rem;
    letter-spacing: 2px;
    padding: 0.5rem 1rem;
    border: 1px solid var(--accent-cyan);
    background: rgba(0, 229, 255, 0.05);
    border-radius: 4px;
    margin-bottom: 2rem;
}

.hero h1 { font-size: 3.5rem; line-height: 1.2; margin-bottom: 1.5rem; letter-spacing: -1px; }
.accent-text { color: var(--accent-cyan); text-shadow: 0 0 20px rgba(0, 229, 255, 0.3); }

.subtitle {
    font-size: 1.125rem;
    color: var(--text-secondary);
    max-width: 800px;
    margin: 0 auto 3rem;
}

.hero-actions { display: flex; justify-content: center; gap: 1rem; }
.btn-primary { background: var(--accent-cyan); color: #000; padding: 0.75rem 2rem; border-radius: 4px; text-decoration: none; font-weight: 600; font-family: 'JetBrains Mono', monospace; transition: all 0.2s; }
.btn-primary:hover { box-shadow: 0 0 20px rgba(0, 229, 255, 0.4); }
.btn-secondary { background: transparent; color: var(--text-primary); border: 1px solid var(--text-secondary); padding: 0.75rem 2rem; border-radius: 4px; text-decoration: none; font-weight: 600; font-family: 'JetBrains Mono', monospace; transition: all 0.2s; }
.btn-secondary:hover { border-color: var(--text-primary); }

.system-hud { padding: 2rem 0; border-top: 1px solid var(--border-color); border-bottom: 1px solid var(--border-color); background: rgba(0, 229, 255, 0.02); }
.hud-grid { display: flex; flex-wrap: wrap; justify-content: center; gap: 2rem; }
.hud-item { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: var(--text-secondary); display: flex; align-items: center; gap: 0.5rem; }
.hud-item.active { color: var(--accent-cyan); }
.hud-item .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--text-secondary); }
.hud-item.active .dot { background: var(--accent-cyan); box-shadow: 0 0 10px var(--accent-cyan); }

.section-dark { padding: 6rem 0; background: transparent; }
.section-light { padding: 6rem 0; background: rgba(255, 255, 255, 0.02); border-top: 1px solid var(--border-color); border-bottom: 1px solid var(--border-color); }

.section-title { font-size: 2rem; margin-bottom: 1rem; text-align: center; letter-spacing: 2px; }
.section-desc { text-align: center; color: var(--text-secondary); max-width: 800px; margin: 0 auto 3rem; }

.arch-flow { display: flex; flex-direction: column; align-items: center; gap: 1rem; margin-top: 3rem; }
.arch-box { border: 1px solid var(--border-color); background: var(--bg-card); padding: 1rem 2rem; border-radius: 4px; text-align: center; width: 100%; max-width: 400px; font-family: 'JetBrains Mono', monospace; transition: all 0.3s; }
.arch-box:hover { border-color: var(--accent-cyan); background: var(--bg-card-hover); }
.arch-box.highlight { border-color: var(--accent-cyan); box-shadow: 0 0 20px rgba(0, 229, 255, 0.1); }
.arch-box.highlight-purple { border-color: var(--accent-purple); box-shadow: 0 0 20px rgba(157, 78, 221, 0.1); }
.arch-split { display: flex; gap: 2rem; width: 100%; max-width: 600px; justify-content: center; }
.tech-stack { font-size: 0.75rem; color: var(--text-secondary); display: block; margin-top: 0.5rem; }
.arch-arrow { color: var(--text-secondary); font-family: 'JetBrains Mono', monospace; }

.decision-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem; margin-top: 3rem; }
.decision-card { padding: 2rem; border-radius: 4px; background: var(--bg-card); border-top: 4px solid; }
.decision-card.authentic { border-color: var(--authentic); }
.decision-card.uncertain { border-color: var(--uncertain); }
.decision-card.synthetic { border-color: var(--synthetic); }
.decision-card h3 { margin-bottom: 1rem; letter-spacing: 2px; }
.decision-card p { color: var(--text-secondary); font-size: 0.875rem; }

.metrics-dashboard { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 2rem; margin-top: 3rem; }
.metric-card { background: var(--bg-card); border: 1px solid var(--border-color); padding: 2rem; border-radius: 4px; text-align: center; }
.metric-card.highlight-card { border-color: var(--accent-purple); background: rgba(157, 78, 221, 0.05); }
.metric-value { font-size: 3rem; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: var(--accent-cyan); margin: 1rem 0; }
.highlight-card .metric-value { color: var(--accent-purple); }
.metric-label { color: var(--text-secondary); font-size: 0.875rem; }
.metric-card h4 { letter-spacing: 1px; font-size: 0.875rem; }

.warning-text { color: var(--uncertain); font-family: 'JetBrains Mono', monospace; font-size: 0.875rem; text-align: center; }

.research-note { margin-top: 3rem; padding: 2rem; border: 1px dashed var(--text-secondary); border-radius: 4px; background: rgba(255, 255, 255, 0.02); }
.research-note h4 { color: var(--text-primary); margin-bottom: 1rem; }
.research-note p { color: var(--text-secondary); font-size: 0.875rem; }

.citation-box { padding: 2rem; border-left: 4px solid var(--accent-cyan); background: var(--bg-card); margin: 2rem auto; max-width: 800px; }
.citation-title { font-weight: 600; font-size: 1.125rem; margin-bottom: 0.5rem; }
.citation-authors { color: var(--text-secondary); font-size: 0.875rem; }
.mt-4 { margin-top: 2rem; }

.footer { padding: 4rem 0; border-top: 1px solid var(--border-color); background: #000; }
.footer-grid { display: grid; grid-template-columns: 1fr 2fr; gap: 4rem; }
.footer h4 { margin-bottom: 1rem; letter-spacing: 2px; }
.small-text { font-size: 0.75rem; color: var(--text-secondary); }

@media (max-width: 768px) {
    .hero h1 { font-size: 2.5rem; }
    .nav-links { display: none; }
    .arch-split { flex-direction: column; gap: 1rem; align-items: center; }
    .footer-grid { grid-template-columns: 1fr; gap: 2rem; }
}
"""

# JS
js_content = """// MediaDNA Forensic UI Interactions
document.addEventListener('DOMContentLoaded', () => {
    console.log('[SYSTEM ONLINE] MediaDNA Frontend Analytics Loaded');
    
    // Smooth scrolling
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });
});
"""

# GitHub Pages Action
workflow_content = """name: Deploy MediaDNA Pages

on:
  push:
    branches: ["main"]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      - name: Setup Pages
        uses: actions/configure-pages@v4
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: './site'
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
"""

write_file('site/index.html', html_content)
write_file('site/css/style.css', css_content)
write_file('site/js/app.js', js_content)
write_file('.github/workflows/deploy-pages.yml', workflow_content)

print("Site built successfully.")
