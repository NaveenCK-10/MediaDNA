import os
import re

with open('c:/Users/navee/Desktop/Projects/MediaDna/OpenAVFF/backend/main.py', 'r') as f:
    main_content = f.read()

if 'import uuid' not in main_content:
    main_content = main_content.replace('import json', 'import json\nimport uuid')

# Replace demo job ID logic
old_demo = """            context = {
                "case_id": "case_" + asset_hash[:12],
                "asset_id": "asset_" + asset_hash[:12],
                "run_id": jid,
                "asset_hash": asset_hash,
                "file_size_bytes": file_size
            }"""
new_demo = """            context = {
                "case_id": "case_" + asset_hash[:16],
                "asset_id": "asset_" + asset_hash[:16],
                "run_id": jid,
                "asset_hash": asset_hash,
                "file_size_bytes": file_size,
                "ingestion_timestamp": time.time()
            }"""
main_content = main_content.replace(old_demo, new_demo)

# Replace demo jid creation
old_demo_jid = """    job_id = "run_" + str(int(time.time()))
    job_manager.create_job(job_id)"""
new_demo_jid = """    job_id = "run_" + uuid.uuid4().hex
    job_manager.create_job(job_id)"""
main_content = main_content.replace(old_demo_jid, new_demo_jid)

# Replace upload ID logic
old_upload = """        job_id = "run_" + str(int(time.time()))
        asset_id = "asset_" + asset_hash[:12]
        case_id = "case_" + asset_hash[:12] # One asset = one case for now
        
        job_manager.create_job(job_id)
        
        async def run_analysis_job(jid, t_path, fname):"""
new_upload = """        job_id = "run_" + uuid.uuid4().hex
        asset_id = "asset_" + asset_hash[:16]
        case_id = "case_" + asset_hash[:16] # One asset = one case for now
        
        job_manager.create_job(job_id)
        
        async def run_analysis_job(jid, t_path, fname):"""
main_content = main_content.replace(old_upload, new_upload)

old_upload_ctx = """                context = {
                    "case_id": case_id,
                    "asset_id": asset_id,
                    "run_id": jid,
                    "asset_hash": asset_hash,
                    "file_size_bytes": file_size
                }"""
new_upload_ctx = """                context = {
                    "case_id": case_id,
                    "asset_id": asset_id,
                    "run_id": jid,
                    "asset_hash": asset_hash,
                    "file_size_bytes": file_size,
                    "ingestion_timestamp": time.time()
                }"""
main_content = main_content.replace(old_upload_ctx, new_upload_ctx)

with open('c:/Users/navee/Desktop/Projects/MediaDna/OpenAVFF/backend/main.py', 'w') as f:
    f.write(main_content)

print("Updated main.py")
