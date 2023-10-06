import pandas as pd


def expand_to_daily(
    df: pd.DataFrame,
    start_date_missing,
    start_date,
    end_date,
    ffill: bool = True,
    cut_sample: bool = True,
    last_valid: bool = False,
    ffill_limit=None,
) -> pd.DataFrame:
    """Upscale DataFrame object from mixed/random frequency to filled
    data with `freq='D'` index

    Parameters:
    `df` (pd.DataFrame): DataFrame with datetime-index, and N columns
    `ffill` (bool): Whether to forward fill data (default=True)
    `cut_sample` (bool): Whether to cut sample to sample period (default=True)
    `last_valid` (bool): Whether to forward fill till last observation or extend
                         to end of sample (default=False)
    `ffill_limit` (int or None): Number of observations to retain (forward fill)
                                 the last observation (default=None)

    Returns:
    pd.DataFrame: Returns DataFrame object with manipulated index at day-level
    """

    # create empty dataframe with `freq='D` index
    new_idx = pd.DataFrame(
        index=pd.date_range(start=start_date_missing, end=end_date, freq="D")
    )

    # merge input data on
    df = new_idx.merge(df, right_index=True, left_index=True, how="left")

    # `kwargs` to adapt data-merge
    if ffill:
        if last_valid:
            # only fill until next observation
            df = df.apply(
                lambda series: series.loc[: series.last_valid_index()].ffill()
            )
        else:
            # first only forward fill between observations (i.e. due to ffill_limit being
            # smaller than distance between two observations)
            df = df.apply(
                lambda series: series.loc[: series.last_valid_index()].ffill()
            )

            # Only forward fill the number of observations in `ffill_limit`.
            # Default: None (i.e. end of sample).
            df = df.ffill(limit=ffill_limit)

            # Particularly for ESG ratings:
            # if last observation was recent (list of years), forward fill till end of sample
            if ffill_limit is not None:
                for col in df.columns:

                    max_date = df[col].dropna().index.max()

                    if max_date.year in [2020, 2021, 2022, 2023]:
                        df[col] = df[col].ffill()

    if cut_sample:
        df = df.loc[(df.index >= start_date) & (df.index <= end_date)]

    return df
