import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.stats.stattools import durbin_watson
from itertools import dropwhile

# Transformation
# Can choose to create either function of log transformation or square root transformation
def transformation(value, type_of):
    if type_of == "log":
        # for using log transformation
        return np.sign(value) * np.log(np.abs(value) + 1)

    if type_of == "sqrt":
        # for using square root transformation (Ardia uses this square root transformation)
        return np.sign(value) * np.sqrt(np.abs(value))


def transform(
    raw_sent_data: pd.DataFrame, transform_type: str, mapping: str, topic: str
):
    raw_sent_data["topic"] = raw_sent_data["topic"].apply(lambda x: x.lower())
    topic = topic.lower()

    if topic != "aggregate":
        raw_sent_data = raw_sent_data.loc[raw_sent_data["topic"] == topic]
    else:
        topic = "aggregate"
        raw_sent_data = raw_sent_data

    df_t = (
        raw_sent_data["sentiment"]
        .resample("D")
        .agg(
            {
                f"{topic}_average": "mean",
                f"{topic}_sum": "sum",
                f"{topic}_count": "count",
            }
        )
    )

    if transform_type == "sum":
        # transforming the measure and filling NaN's
        df_t[f"{topic}_transformed"] = (
            df_t[f"{topic}_sum"].apply(transformation, type_of=mapping).fillna(0)
        )
        # return df_t[f"{topic}_transformed"].to_frame()
        return df_t  # [f"{topic}_transformed"].to_frame()

    if transform_type == "mean_n":
        df_t[f"{topic}_count_transformed"] = df_t[f"{topic}_count"].apply(
            transformation, type_of=mapping
        )

        # transforming the measure and filling NaN's
        df_t[f"{topic}_transformed"] = (
            df_t[f"{topic}_count_transformed"] * df_t[f"{topic}_average"]
        ).fillna(0)

        # return df_t[f"{topic}_transformed"].to_frame()
        return df_t  # [f"{topic}_transformed"].to_frame()


def forecast_residual(
    Y,
    window_size: int = 1000,
    return_estimates: bool = False,
    return_spec_test: bool = False,
    auto_lag: bool = False,
    lag_param: int = 1,
):
    # ensures windows are of fixed size equal to `window_size`
    windows = dropwhile(lambda w: len(w) < window_size, Y.rolling(window_size))

    # placeholder for predictions
    ar1_predictions = pd.DataFrame()
    ar1_estimates = pd.DataFrame()
    spec_stats = pd.DataFrame()

    for rolling in windows:
        # AR model specification
        if auto_lag:
            lag_param = ar_select_order(rolling, maxlag=35).ar_lags

        model = AutoReg(rolling, lags=lag_param).fit()

        # AR (1) forecast - this is date-time robust
        ar1_predictions = pd.concat([ar1_predictions, model.forecast(1)])

        if return_spec_test:
            latest_date = rolling.index[-1]

            # calculate joint f-test for all lag parameters

            # estimate auxiliary model
            aux_model = AutoReg(model.resid, lags=lag_param).fit()
            A = np.identity(len(aux_model.params))
            # remove the intercept
            A = A[1:, :]
            f_test = aux_model.f_test(A)

            spec_stats = pd.concat(
                [
                    spec_stats,
                    pd.DataFrame(
                        {
                            f"{Y.name}_dw_statistic": [durbin_watson(model.resid)],
                            f"{Y.name}_f_statistic": [f_test.fvalue],
                            f"{Y.name}_f_pval": [f_test.pvalue],
                            f"{Y.name}_lags": [lag_param][-1]
                            if lag_param == 1
                            else lag_param[-1],
                        },
                        index=[latest_date],
                    ),
                ]
            )

        if return_estimates:
            latest_date = rolling.index[-1]
            ar1_estimates = pd.concat(
                [
                    ar1_estimates,
                    pd.DataFrame(
                        {
                            f"{Y.name}_AR_1_coef": [model.params[1]],
                            f"{Y.name}_AR_1_pval": [model.pvalues[1]],
                            f"{Y.name}_AR_1_tval": [model.tvalues[1]],
                        },
                        index=[latest_date],
                    ),
                ]
            )

    # calculate residuals
    ar1_predictions = ar1_predictions.rename(columns={0: f"{Y.name}_predictions"})
    ar1_predictions[f"{Y.name}_observed"] = Y
    ar1_predictions[f"{Y.name}_residuals"] = (
        ar1_predictions[f"{Y.name}_observed"] - ar1_predictions[f"{Y.name}_predictions"]
    )

    # join estimates on dataframe
    if return_estimates and return_spec_test:
        ar1_predictions = pd.merge(
            left=ar1_predictions, right=ar1_estimates, left_index=True, right_index=True
        )
        ar1_predictions = pd.merge(
            left=ar1_predictions, right=spec_stats, left_index=True, right_index=True
        )

        return ar1_predictions

    if return_estimates:
        ar1_predictions = pd.merge(
            left=ar1_predictions, right=ar1_estimates, left_index=True, right_index=True
        )

        return ar1_predictions

    if return_spec_test:
        ar1_predictions = pd.merge(
            left=ar1_predictions, right=spec_stats, left_index=True, right_index=True
        )

        return ar1_predictions

    return ar1_predictions[f"{Y.name}_residuals"].dropna()
