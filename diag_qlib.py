import os
import pandas as pd
import qlib
from qlib.config import C
from qlib.data import D

# disable multiprocessing so that the "spawn" bootstrap bug doesn't occur
C.joblib_backend = "threading"
C.joblib_n_jobs = 1

def main():
    # init
    qlib.init(provider_uri=os.path.expanduser("~/.qlib/qlib_data/cn_data"))

    # load calendar
    cal = D.calendar(start_time="2005-01-01", end_time="2025-05-22")
    # The 4949 trading days from 2005–01–01 through 2025–05–22 look right
    print("Calendar length:", len(cal)) 
    # heterogenous gaps (1, 2, … 12 days) are just weekends & holidays
    print("Day-to-day gaps:", (pd.to_datetime(cal[1:]) - pd.to_datetime(cal[:-1])).unique())

    # read universe manually
    instr_path = os.path.expanduser(
        "~/.qlib/qlib_data/cn_data/instruments/csiall.txt")
    with open(instr_path) as f:
        raw = [line.strip() for line in f if line.strip()]
        tickers = [item.split("\t")[0] for item in raw]  # keep only the symbol
    print("Universe size:", len(tickers), "   First 5:", tickers[:5])

    # sample a few symbols and pull OHLCV
    sample = tickers[:3]
    df = D.features(
        instruments=sample,
        fields=["$open", "$high", "$low", "$close", "$volume"],
        start_time=cal[0],
        end_time=cal[-1],
    )

    # report missingness
    miss = (
        df.reset_index()
          .groupby("instrument")[["$open","$high","$low","$close","$volume"]]
          .apply(lambda d: d.isna().sum())
    )
    print("Missing values per instrument:\n", miss)

if __name__ == "__main__":
    main()