"""Monitor pipeline progress"""
import time
import os
from pathlib import Path

print("Monitoring pipeline progress...")
print("=" * 80)

output_file = Path("data/results/analysis_ollama_cloud.json")
start_time = time.time()

while True:
    elapsed = time.time() - start_time
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)

    if output_file.exists():
        size = output_file.stat().st_size
        print(f"\n[{mins:02d}:{secs:02d}] Pipeline COMPLETE!")
        print(f"Output file: {output_file} ({size:,} bytes)")
        break
    else:
        print(f"\r[{mins:02d}:{secs:02d}] Processing... (expected: 30-40 mins)", end="", flush=True)

    time.sleep(5)

print("\nDone!")
