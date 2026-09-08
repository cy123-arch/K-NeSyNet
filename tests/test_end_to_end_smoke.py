import subprocess, sys

def test_end_to_end_smoke():
    subprocess.run([sys.executable,'scripts/smoke_pipeline.py'],check=True)
