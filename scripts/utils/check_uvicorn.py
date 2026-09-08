import subprocess

try:
    res = subprocess.run(['python', '-m', 'uvicorn', 'backend.main:app', '--host', '0.0.0.0', '--port', '8005'], capture_output=True, text=True, timeout=10)
    print("STDOUT:", res.stdout)
    print("STDERR:", res.stderr)
except subprocess.TimeoutExpired as e:
    print("TIMEOUT!")
    print("STDOUT:", e.stdout)
    print("STDERR:", e.stderr)
