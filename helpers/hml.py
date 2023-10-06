import numpy as np
import pandas as pd
from tqdm import tqdm


def in_snp(constituents, ts, series):
    snp = constituents.loc[ts]

    # This can drop some de-listed stocks, that we do not have data on
    series = series.loc[series.index.intersection(snp.ric)]

    return series


def high_minus_low(sorts, snp, slice_method="tiles", q1=0.2, q2=0.8):
    # placeholder for new data
    sort_pf = pd.DataFrame(columns=["date", "hml", "snp_ric"])

    # memory safety
    sort = sorts.copy()

    # different splitting methods
    methods = {
        # IDEA: implement top 10% vs bottom 10%
        "mean": {
            "metric1": np.nanmean,
            "metric2": np.nanmean,
            "interval": [False, True],
        },
        "median": {
            "metric1": np.nanmedian,
            "metric2": np.nanmedian,
            "interval": [False, True],
        },
        "tiles": {
            "metric1": np.nanquantile,
            "metric2": np.nanquantile,
            "interval": [False, np.inf, True],
        },
    }
    # select from parameter
    method = methods[slice_method]

    for timestamp, series in tqdm(sort.iterrows(), total=sort.shape[0]):
        # drop observations with no rating (NaN)
        series = series.dropna()

        # get the sample that is in S&P at timestamp
        try:
            snp_sample = in_snp(snp, timestamp, series)
        except KeyError:
            # break if indices do not match - i.e. we have more returns data than SNP changes
            break

        # get the inverse sample
        snp_inv_sample = series.loc[~series.index.isin(snp_sample.index)]

        # set inverse sample to NaN
        series.loc[snp_inv_sample.index] = np.nan

        # split data into boolean based groups (True is high, False is low)
        # for `tiles` between-values are filled with `np.inf`
        if slice_method == "tiles":

            series_split = pd.cut(
                x=series,
                bins=[
                    -np.inf,
                    method["metric1"](series, q=q1),
                    method["metric2"](series, q=q2),
                    np.inf,
                ],
                labels=method["interval"],
                retbins=True,  # return the bins and save them
                duplicates="drop",
            )

        else:
            series_split = pd.cut(
                x=series,
                bins=[
                    -np.inf,
                    method["metric1"](series),
                    method["metric2"](series),
                    np.inf,
                ],
                labels=method["interval"],
                retbins=True,  # return the bins and save them
                duplicates="drop",
            )

        # extract bins and overwrite mapped bins
        bins = series_split[1]
        series_split = series_split[0]

        series_split = pd.DataFrame(series_split).reset_index()
        series_split["date"] = series_split.columns[-1]
        series_split = series_split.rename(
            columns={"index": "snp_ric", series_split.columns[1]: "hml"}
        )

        # merge ESG scores on `series_split`
        series_split = pd.merge(
            left=series_split, right=series, left_on="snp_ric", right_index=True
        )
        # re-name newly jo
        series_split = series_split.rename(columns={series_split.columns[-1]: "rating"})
        series_split["low_bin"] = bins[1]
        series_split["high_bin"] = bins[2]

        sort_pf = pd.concat([sort_pf, series_split], ignore_index=True)

    return sort_pf


def get_weights(mkt_cp, hml_grouped, timestamp, weighting_scheme):
    # join market-caps: we will do this either way
    caps = mkt_cp.loc[timestamp]
    caps.name = "market_cap"
    hml_grouped = pd.merge(
        left=hml_grouped, right=caps, left_on="snp_ric", right_index=True
    )

    high = hml_grouped.loc[hml_grouped["hml"] == True].copy()
    low = hml_grouped.loc[hml_grouped["hml"] == False].copy()

    if weighting_scheme == "equal":
        high["weight"] = 1 / high.shape[0]
        low["weight"] = 1 / low.shape[0]

        return (high, low)

    if weighting_scheme == "cap":
        high["weight"] = high["market_cap"].apply(
            lambda x: x / np.sum(high["market_cap"])
        )
        low["weight"] = low["market_cap"].apply(lambda x: x / np.sum(low["market_cap"]))

    return (high, low)


def join_ret(ret, rf, hml_grouped, timestamp):
    # join returns
    ret = ret.loc[timestamp]
    ret.name = "returns"
    hml_grouped = pd.merge(
        left=hml_grouped, right=ret, left_on="snp_ric", right_index=True
    )

    # join risk-free rate
    rf = rf.loc[timestamp]
    hml_grouped["rf"] = rf.values[0]

    # we can have some missing observations from delisted stocks
    # simply drop these rows
    hml_grouped = hml_grouped.dropna(axis=0)

    return hml_grouped


def construct_portfolio(ret, rf, mkt_cp, hml, weights="cap", calc_excess=True):
    """
    TODO: maybe write this out...
    weights = {"cap"|"equal"}
    """

    # memory safety
    hml_series = hml.copy()

    # placeholder for resulting data
    hml_df = pd.DataFrame(
        columns=[
            "date",
            "hml",
            "snp_ric",
            "rating",
            "low_bin",
            "high_bin",
            "returns",
            "rf",
            "excess",
            "weight_high",
            "market_cap_high",
            "weight_low",
            "market_cap_low",
            "high_return",
            "low_return",
        ]
    )

    # clean-up: we can safely remove any `NaN` or `np.inf` rows
    # these are either __not__ in S&P or high/low
    hml_series = hml_series.replace(np.inf, np.nan).dropna()

    for timestamp, series in tqdm(
        hml_series.groupby("date"), total=hml_series["date"].unique().shape[0]
    ):
        # join return series and calculate excess returns
        series = join_ret(ret, rf, series, timestamp)
        if calc_excess:
            series["excess"] = series["returns"] - series["rf"]
        else:
            series["excess"] = series["returns"]

        # get market_cap
        high, low = get_weights(mkt_cp, series, timestamp, weighting_scheme=weights)

        # this is a little hack to have everything collected in a single DataFrame
        # we simply join the weights for both groups. The suffixes tells us which
        # subsample the data comes from (high vs. low)
        series = pd.merge(
            how="left",
            left=series,
            right=high[["snp_ric", "weight", "market_cap"]],
            left_on="snp_ric",
            right_on="snp_ric",
            suffixes=["_low", "_high"],
        )
        series = pd.merge(
            how="left",
            left=series,
            right=low[["snp_ric", "weight", "market_cap"]],
            left_on="snp_ric",
            right_on="snp_ric",
            suffixes=["_high", "_low"],
        )

        # there can be some stocks for which we have missing:
        # `returns`, `market_cap`
        # we do a simple `.dropna()` when calculating the portfolio returns
        high = high.dropna(subset=["returns", "market_cap"])
        low = low.dropna(subset=["returns", "market_cap"])

        # vectorized portfolio return
        high_ret = np.dot(high["weight"].to_numpy(), high["excess"].to_numpy())
        low_ret = np.dot(low["weight"].to_numpy(), low["excess"].to_numpy())

        # save values
        series["high_return"] = high_ret
        series["low_return"] = low_ret

        hml_df = pd.concat([hml_df, series], ignore_index=True)

        # break

    # Calculate high-minus-low return
    hml_df['hml_return'] = hml_df['high_return'] - hml_df['low_return']

    return hml_df
