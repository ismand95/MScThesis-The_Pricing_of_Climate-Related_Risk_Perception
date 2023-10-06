from datetime import datetime
import calendar
from dateutil.relativedelta import relativedelta
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm


def construct_factor_space(fac, sent, fac_cols, sent_col, dropna=True):
    # join textual factor and construct X
    fac = fac[fac_cols]  # FF 3
    fac = fac.dropna()

    fac = pd.merge(
        right=fac, left=sent[sent_col], how="left", left_index=True, right_index=True
    )

    if dropna:
        return fac.dropna()

    return fac


def first_of_month(date):
    return datetime(year=date.year, month=date.month, day=1)


def last_of_month(date):
    day = calendar.monthrange(date.year, date.month)[1]
    return datetime(year=date.year, month=date.month, day=day)


def get_date_offset(date, month_offset=3):
    return date - relativedelta(months=month_offset - 1)


def get_first_snp_constituent(snp, date):
    # this function takes a date argument, finds the first date
    # in the month of the date and returns the S&P constituents
    # at this particular first day
    date = first_of_month(date)

    return snp.loc[date]


def estimate_portfolio_sorts(
    fac,
    sent,
    snp,
    ret,
    fac_cols,
    sent_col,
    nw_lags: int,
    look_back:int,
    dropna=True,
    prettify=True,
):
    # here we simply expand the index of the `factors` data frame
    # to ensure that we can apply pandas rolling method
    ix = pd.date_range(
        start=first_of_month(fac.index.min()),
        end=last_of_month(fac.index.max()),
        freq="D",
    )
    # convert `factors` object
    X = construct_factor_space(fac, sent, fac_cols, sent_col, dropna)
    X = X.reindex(ix).copy()

    # placeholder for estimated parameters
    data = {}

    for timestamp, fac_m in tqdm(X.groupby(pd.Grouper(freq="M"))):
        # we look back three months from timestamp (i.e. last of month)
        # and then convert this date to the first of the given (lagged) month
        start_back = first_of_month(get_date_offset(timestamp, month_offset=look_back))

        # This `try/except` is for the first observation
        try:
            fac_m = X.loc[
                pd.date_range(freq="D", start=start_back, end=fac_m.index.max())
            ]
        except KeyError:
            continue

        # drop missing observations from factors
        # we have a lot of NaNs from the expansion of the index
        fac_m = fac_m.dropna()

        # get S&P constituents at beginning of backward-looking sample
        tickers = get_first_snp_constituent(snp, start_back)

        # get return observations for applicable three months
        ret_months = ret[tickers["ric"]].loc[fac_m.index]

        # create placeholder in data
        data.update({timestamp: []})

        for ticker in ret_months.columns:
            Y_m = ret_months[ticker].dropna()
            X_m = sm.add_constant(fac_m)

            # ensure indices are matching if we have i.e. missing observations
            X_m = X_m.reindex(Y_m.index)

            # estimate time-series regression using Newey West errors
            try:
                fit = sm.OLS(endog=Y_m, exog=X_m).fit(
                    cov_type="HAC", cov_kwds={"maxlags": nw_lags}
                )
            except ValueError:
                # here we have some data without observations and simply skip these...
                continue

            data[timestamp].append(
                {
                    "ticker": ticker,
                    "date": timestamp,
                    "params": fit.params,
                    "n": fit.nobs,
                    "std": fit.bse,
                    "tvalues": fit.tvalues,
                }
            )

    if prettify:
        return prettified(data, sent_col, look_back)

    return data


def prettified(data, sent_col, look_back):
    to_pd = []

    for key in data.keys():
        for obs in data[key]:
            to_pd.append(
                {
                    "date": obs["date"],
                    "ticker": obs["ticker"],
                    f"sentiment_beta": obs["params"][sent_col],
                    f"sentiment_std": obs["std"][sent_col],
                    f"sentiment_t": obs["tvalues"][sent_col],
                    "n": obs["n"],
                }
            )
    to_pd = pd.DataFrame(to_pd)

    # drop rows with too few observations
    to_pd = to_pd.loc[
        to_pd["n"] > ((look_back * 20) / 2)
    ]  # 20 business days (month) times look-back divided by two

    return to_pd
