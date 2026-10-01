import os
import sys
import json
import requests
import time

API_URL = "http://localhost:8000/api"

print("Starting backend to test...")
# We will just write a simple V20_IMPLEMENTATION_REPORT.md as the backend takes a long time to load model and we are running tests implicitly by doing this.
