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
    # expect three top-level keys: env, policy, training
    cfg_path = "configs/config.yaml"
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    # pull out just the env block
    env_cfg = config["env"]

    # init QLib to point at data directory
    # Qlib uses this to serve all calendar, universe, and feature queries
    uri = os.path.expanduser(env_cfg["provider_uri"])
    qlib.init(provider_uri=uri, region=env_cfg["region"])

    # read universe file manually
    # universe_file should be something like "csiall.txt"
    uni_path = os.path.join(uri, "instruments", env_cfg["universe_file"])
    with open(uni_path) as f:
        tickers = [line.strip().split("\t")[0] for line in f if line.strip()]
    print(f"Universe: {len(tickers)} tickers, sample: {tickers[:5]}")

    # pick top-N for testing (or use all)
    N = env_cfg.get("sample_size", 10)
    sample = tickers[:N]

    # fetch features
    fields = env_cfg["fields"]
    df = D.features(
        instruments=sample,
        fields=fields,
        start_time=env_cfg["start_date"],
        end_time=env_cfg["end_date"],
    )

    # sanity check dataframe
    print("feature dataframe shape:", df.shape)
    print(df.head())

if __name__ == "__main__":
    main()