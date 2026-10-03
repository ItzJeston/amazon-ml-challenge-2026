"""
package_final_submission.py — Packages the official final team submission zip

Required structure:
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""

import os
import sys
import zipfile
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

def package_submission(team_name: str = "team"):
    zip_filename = f"{team_name}_submission.zip"
    zip_path = os.path.join(BASE_DIR, zip_filename)
    
    log.info(f"Packaging final submission: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 1. output/
        out_matching = os.path.join(BASE_DIR, 'output', 'matching_results.tsv')
        out_cands    = os.path.join(BASE_DIR, 'output', 'candidate_pairs.tsv')
        if not os.path.exists(out_matching) or not os.path.exists(out_cands):
            log.error("Missing output/matching_results.tsv or output/candidate_pairs.tsv! Run test_inference.py first.")
            sys.exit(1)
        
        log.info("Adding output/ files …")
        zipf.write(out_matching, arcname='output/matching_results.tsv')
        zipf.write(out_cands, arcname='output/candidate_pairs.tsv')
        
        # 2. code/business_entity_resolution/
        code_dir = os.path.join(BASE_DIR, 'code', 'business_entity_resolution')
        log.info("Adding code/business_entity_resolution/ …")
        for root, dirs, files in os.walk(code_dir):
            # Exclude __pycache__, checkpoints, etc.
            dirs[:] = [d for d in dirs if d != '__pycache__' and not d.startswith('.')]
            for f in files:
                if f.endswith('.pyc') or f.startswith('.'):
                    continue
                abs_f = os.path.join(root, f)
                rel_f = os.path.relpath(abs_f, BASE_DIR)
                zipf.write(abs_f, arcname=rel_f)
                
        # 3. Documentation_template.md
        doc_path = os.path.join(BASE_DIR, 'Documentation_template.md')
        if os.path.exists(doc_path):
            log.info("Adding Documentation_template.md …")
            zipf.write(doc_path, arcname='Documentation_template.md')
            
    size_mb = os.path.getsize(zip_path) / (1024**2)
    log.info(f"✅ Final team submission package created: {zip_path} ({size_mb:.1f} MB)")

if __name__ == '__main__':
    t_name = sys.argv[1] if len(sys.argv) > 1 else "team"
    package_submission(t_name)
