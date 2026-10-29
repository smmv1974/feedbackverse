"""
Wrapper script to run pipeline with cloud APIs.
Sets API keys and runs the pipeline.
"""
import os
import sys
import subprocess

# Set API keys
# GOOGLE_API_KEY must be set in the environment
# GROQ_API_KEY must be set in the environment

# Fix Windows console encoding
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

print("=" * 80)
print(" RUNNING PIPELINE WITH CLOUD APIs")
print("=" * 80)
print(f"\nGoogle API Key set: {'GOOGLE_API_KEY' in os.environ}")
print(f"Groq API Key set: {'GROQ_API_KEY' in os.environ}")
print("\nStarting pipeline...\n")

# Run the pipeline
subprocess.run([sys.executable, 'run_pipeline.py', '--config', 'config/config_cloud.yaml'])
