import re
import os

with open('c:/Users/navee/Desktop/Projects/MediaDna/OpenAVFF/backend/main.py', 'r') as f:
    content = f.read()

# Add hashlib
if 'import hashlib' not in content:
    content = content.replace('import json', 'import json\nimport hashlib')

# Update analyze_video upload reading to compute sha256
old_upload = """        file_size = 0
        with open(temp_path, "wb") as f:
            while True:
                chunk = await video.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File too large. Maximum size: {MAX_FILE_SIZE_MB} MB",
                    )
                f.write(chunk)"""

new_upload = """        file_size = 0
        sha256_hash = hashlib.sha256()
        with open(temp_path, "wb") as f:
            while True:
                chunk = await video.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                file_size += len(chunk)
                sha256_hash.update(chunk)
                if file_size > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File too large. Maximum size: {MAX_FILE_SIZE_MB} MB",
                    )
                f.write(chunk)
        
        asset_hash = sha256_hash.hexdigest()"""
content = content.replace(old_upload, new_upload)

# Update job context logic
old_job_exec = """        job_id = str(time.time())
        job_manager.create_job(job_id)
        
        async def run_analysis_job(jid, t_path, fname):
            loop = asyncio.get_running_loop()
            def progress_cb(stage, msg, pct):
                asyncio.run_coroutine_threadsafe(
                    job_manager.update_job(jid, "processing", stage, msg, pct, completed_stage=stage),
                    loop
                )
                
            try:
                # Run heavy inference in threadpool
                result = await loop.run_in_executor(None, lambda: service.analyze_video(t_path, progress_cb))"""

new_job_exec = """        job_id = "run_" + str(int(time.time()))
        asset_id = "asset_" + asset_hash[:12]
        case_id = "case_" + asset_hash[:12] # One asset = one case for now
        
        job_manager.create_job(job_id)
        
        async def run_analysis_job(jid, t_path, fname):
            loop = asyncio.get_running_loop()
            def progress_cb(stage, msg, pct):
                asyncio.run_coroutine_threadsafe(
                    job_manager.update_job(jid, "processing", stage, msg, pct, completed_stage=stage),
                    loop
                )
                
            try:
                context = {
                    "case_id": case_id,
                    "asset_id": asset_id,
                    "run_id": jid,
                    "asset_hash": asset_hash,
                    "file_size_bytes": file_size
                }
                # Run heavy inference in threadpool
                result = await loop.run_in_executor(None, lambda: service.analyze_video(t_path, progress_cb, context))"""
content = content.replace(old_job_exec, new_job_exec)

# Update analyze_demo
old_demo = """    job_id = str(time.time())
    job_manager.create_job(job_id)
    
    async def run_demo_job(jid, t_path, fname):
        loop = asyncio.get_running_loop()
        def progress_cb(stage, msg, pct):
            asyncio.run_coroutine_threadsafe(
                job_manager.update_job(jid, "processing", stage, msg, pct, completed_stage=stage),
                loop
            )
            
        try:
            result = await loop.run_in_executor(None, lambda: service.analyze_video(t_path, progress_cb))"""

new_demo = """    job_id = "run_" + str(int(time.time()))
    job_manager.create_job(job_id)
    
    async def run_demo_job(jid, t_path, fname):
        loop = asyncio.get_running_loop()
        def progress_cb(stage, msg, pct):
            asyncio.run_coroutine_threadsafe(
                job_manager.update_job(jid, "processing", stage, msg, pct, completed_stage=stage),
                loop
            )
            
        try:
            # Hash the demo file
            sha256_hash = hashlib.sha256()
            with open(t_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    sha256_hash.update(chunk)
            asset_hash = sha256_hash.hexdigest()
            file_size = os.path.getsize(t_path)
            
            context = {
                "case_id": "case_" + asset_hash[:12],
                "asset_id": "asset_" + asset_hash[:12],
                "run_id": jid,
                "asset_hash": asset_hash,
                "file_size_bytes": file_size
            }
            
            result = await loop.run_in_executor(None, lambda: service.analyze_video(t_path, progress_cb, context))"""
content = content.replace(old_demo, new_demo)

with open('c:/Users/navee/Desktop/Projects/MediaDna/OpenAVFF/backend/main.py', 'w') as f:
    f.write(content)

print("Updated main.py successfully.")
