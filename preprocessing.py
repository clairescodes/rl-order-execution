# preprocessing.py 
# handles data I/O and cleaning 

import pandas as pd

# Forward-then-backward fill missing values for each instrument
def fill_missing(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    # collect each instrument’s filled data in this list 
    filled_parts = []
    for inst in df.index.get_level_values("instrument").unique():
        # cross-section of df for only the subtable for this instrument 
        sub = df.xs(inst, level="instrument")
        # forward fill, then backfill any leading NaNs
        sub_filled = sub.fillna(method=method).fillna(method="bfill")
        # reattach instrument index
        sub_filled.index = pd.MultiIndex.from_product([[inst], sub_filled.index],
                                                      names=["instrument","datetime"])
        filled_parts.append(sub_filled)
    return pd.concat(filled_parts)

def clip_outliers(df: pd.DataFrame, lower_q: float, upper_q: float) -> pd.DataFrame:
    # Clip values outside the [lower_q, upper_q] quantiles for each instrument.
    clipped_parts = []
    for inst in df.index.get_level_values("instrument").unique():
        sub = df.xs(inst, level="instrument")
        # compute per-column bounds
        lower = sub.quantile(lower_q)
        upper = sub.quantile(upper_q)
        sub_clipped = sub.clip(lower=lower, upper=upper, axis=1)
        sub_clipped.index = pd.MultiIndex.from_product([[inst], sub_clipped.index],
                                                       names=["instrument","datetime"])
        clipped_parts.append(sub_clipped)
    return pd.concat(clipped_parts)

def normalize_zscore(df: pd.DataFrame) -> pd.DataFrame:
    # Z-score normalize each feature series per instrument.
    norm_parts = []
    for inst in df.index.get_level_values("instrument").unique():
        sub = df.xs(inst, level="instrument")
        mean = sub.mean()
        std = sub.std() + 1e-8
        sub_norm = (sub - mean) / std
        sub_norm.index = pd.MultiIndex.from_product([[inst], sub_norm.index],
                                                     names=["instrument","datetime"])
        norm_parts.append(sub_norm)
    return pd.concat(norm_parts)

def clean_and_normalize(
    df: pd.DataFrame,
    fill_method: str = "ffill",
    clip_quantile: float = 0.01,
) -> pd.DataFrame:
    # 1) Fill gaps per instrument
    # 2) Clip extreme values at the 1%-99% quantiles
    # 3) Z-score normalize each feature per instrument

    # Step 1: fill
    df_filled = fill_missing(df, method=fill_method)
    
    # Step 2: clip
    lower_q = clip_quantile
    upper_q = 1 - clip_quantile
    df_clipped = clip_outliers(df_filled, lower_q, upper_q)
    
    # Step 3: normalize
    df_normalized = normalize_zscore(df_clipped)
    
    return df_normalized

