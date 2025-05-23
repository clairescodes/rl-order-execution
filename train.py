"""
train.py

Entry point for running a trading-RL experiment.
Reads configuration, initializes QLib, loads the universe, and
prints out the first few tickers as a sanity check.
"""
import os 
import yaml 
import qlib 
from qlib.data import D

def main(): 
    # load experiment config from yaml file. 
    # should specify at least: 
    #      - provider_uri: path to QLib data directory
    #      - region: data region (e.g. "cn" or "us")
    #      - universe_file: relative path under instruments/ defining stock universe
    #      - start_date, end_date: backtest period
    cfg_path = "configs/config.yaml"
    with open(cfg_path) as f: 
        config = yaml.safe_load(f)
    
    # init QLib to point at data directory 
    # Qlib uses this to serve all calendar, universe, and feature queries
    uri = os.path.expanduser(config["provider_uri"])
    qlib.init(provider_uri=uri, region=config["region"])

    # read universe file manually 
    # universe_file should be something like "csiall.txt" 
    uni_path = os.path.join(uri, "instruments", config["universe_file"])
    with open(uni_path) as f: 
        tickers = [line.strip().split("\t")[0] for line in f if line.strip()]
    print(f"Universe: {len(tickers)} tickers, sample: {tickers[:5]}")

    # pick top-N for testing (or use all)
    N = config.get("sample_size", 10) 
    sample = tickers[:N] 

    # fetch features 
    fields = config["fields"]
    df = D.features(
        instruments=sample,
        fields=fields,
        start_time=config["start_date"],
        end_time=config["end_date"],
    )

    # sanity check dataframe 
    print("feature dataframe shape:", df.shape) 
    print(df.head())

if __name__ == "__main__":
    main()

