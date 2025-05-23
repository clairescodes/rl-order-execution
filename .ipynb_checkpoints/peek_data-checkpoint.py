import qlib 
from qlib.data import D 

def main():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")

    # First 5 trading dates
    dates = D.calendar(start_time="2005-01-04", end_time="2005-01-15")
    print("Dates:", dates[:5])

    # First 5 tickers (convert to list before slicing)
    tickers = list(D.instruments(market="all"))
    print("Tickers:", tickers[:5])

    # what universes are available
    universes = list(D.instruments())
    print("Universes:", universes)
    
if __name__ == "__main__":
    # On Windows, you’d also do:
    # from multiprocessing import freeze_support
    # freeze_support()
    main()