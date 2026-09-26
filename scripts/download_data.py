from pathlib import Path
import argparse, sys, yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from credit_warning.data import load_etf_history, load_vix_history, save_raw_series

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    start, end = cfg["project"]["analysis_start"], cfg["project"]["analysis_end"]
    etfs = cfg["assets"]["etfs"]
    vix_id = cfg["assets"]["vix_series"]
    raw_dir = ROOT / "data" / "raw"
    px = load_etf_history(etfs, start, end)
    save_raw_series(px, raw_dir, "Yahoo Finance via yfinance", "Adj Close",
                    start, end, "Yahoo provider Adj Close (splits and cash distributions)")
    vix = load_vix_history(vix_id, start, end)
    save_raw_series(vix, raw_dir, "FRED; underlying source CBOE", "Daily close",
                    start, end, "not applicable")
    print(f"Saved raw cache to {raw_dir}")

if __name__ == "__main__":
    main()
