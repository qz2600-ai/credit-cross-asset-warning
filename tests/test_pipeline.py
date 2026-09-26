from pathlib import Path
def test_required_phase_a_files_exist():
    root=Path(__file__).resolve().parents[1]
    for p in ["config.yaml","scripts/download_data.py","scripts/run_pipeline.py","src/credit_warning/data.py"]:
        assert (root/p).exists()
