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
    with open(cfg_path, "r") as f: 
        config = yaml.safe_load(f)
    
    # init QLib to point at data directory 
    # Qlib uses this to serve all calendar, universe, and feature queries
    provider_uri = os.path.expanduser(config["provider_uri"])
    qlib.init(provider_uri=provider_uri, region=config["region"])

    # read universe file manually 
    # universe_file should be something like "csiall.txt" 
    uni_path = os.path.join(provider_uri, "instruments", config["universe_file"])
    with open(uni_path, "r") as f: 
        tickers = [line.strip().split("\t") for line in f if line.strip()]

    # sanity check. print universe size and first five tickers 
    print(f"Loaded universe '{config['universe_file']}': {len(tickers)} tickers")
    print("First 5 tickers:", tickers[:5])

if __name__ == "__main__":
    main()

