import os
import shutil
import glob

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def move_files(pattern, dest_dir):
    ensure_dir(dest_dir)
    files = glob.glob(pattern)
    for f in files:
        if os.path.isfile(f) and not os.path.basename(f) == "organize_workspace.py":
            try:
                shutil.move(f, os.path.join(dest_dir, os.path.basename(f)))
            except:
                pass

if __name__ == "__main__":
    # Markdown Audits and Reports
    move_files("CODEX_*.md", "docs/audits")
    move_files("CODEX_*.txt", "docs/audits")
    move_files("CODEX_*.json", "docs/audits")
    
    move_files("MEDIADNA_*.md", "docs/reports")
    move_files("MEDIADNA_*.json", "docs/reports")
    move_files("MEDIADNA_*.png", "docs/reports")
    
    move_files("V20_*.md", "docs/reports/v20")
    move_files("V21_*.md", "docs/reports/v21")
    move_files("V22_*.md", "docs/reports/v22")
    move_files("V22_*.json", "docs/reports/v22_data")
    move_files("V22_*.csv", "docs/reports/v22_data")
    move_files("V22_*.png", "docs/reports/v22_data")
    move_files("V22_*.sha256", "docs/reports/v22_data")
    move_files("V22_*.pkl", "docs/reports/v22_data")
    move_files("V22_*.pth", "checkpoints/v22_legacy")
    
    move_files("V21_*.json", "docs/reports/v21_data")
    move_files("V21_*.csv", "docs/reports/v21_data")
    
    move_files("ANTIGRAVITY_CHANGELOG.md", "docs/")
    move_files("MODEL_CARD.md", "docs/")
    
    # Scripts
    move_files("rewrite_*.py", "scripts/historical_rewrites")
    move_files("verify_*.py", "scripts/historical_verification")
    move_files("test_*.py", "scripts/historical_tests")
    move_files("generate_*.py", "scripts/historical_generators")
    move_files("repair_*.py", "scripts/historical_repairs")
    move_files("v21_*.py", "scripts/historical_v21")
    move_files("v22_*.py", "scripts/historical_v22")
    move_files("v15_*.py", "scripts/historical_v15")
    move_files("v16_*.py", "scripts/historical_v16")
    
    move_files("evaluate_*.py", "scripts/evaluation")
    move_files("fast_extractor.py", "scripts/utils")
    move_files("evidence_extractor.py", "scripts/utils")
    move_files("inventory_script.py", "scripts/utils")
    move_files("prov_script.py", "scripts/utils")
    move_files("create_v21_4_manifest.py", "scripts/utils")
    move_files("phase23_test.py", "scripts/historical_tests")
    move_files("final_baseline_eval.py", "scripts/evaluation")
    move_files("reconstruction_demo.ipynb", "notebooks")
    
    print("Workspace organized.")
