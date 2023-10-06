import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
import numpy as np

from typing import Optional
import matplotlib.pyplot as plt


class FamaMacbeth:
    def __init__(
        self,
        gmm_errors: bool,
        n_west_lags: int,
        assets: pd.DataFrame,
        factors: pd.DataFrame,
    ):
        # inputs and validation of inputs
        self.gmm_errors = gmm_errors
        self.n_west_lags = n_west_lags
        self.assets = assets
        self.factors = factors

        self._validate_data()

        # parameters of the model
        # t: number of time-observations
        # n: number of test-assets (i.e. industry portfolios)
        # k: number of factors (minus alpha): i.e. 3 for FF3
        self.t, self.n = self.assets.shape
        self.k = self.factors.shape[1]

        # resulting data
        self.summary: Optional[pd.DataFrame] = None  # fama_macbeth table
        self.first_stage: Optional[pd.DataFrame] = None  # ts estimates
        self.first_stage_residuals: Optional[pd.DataFrame] = None  # residuals from ts
        self.second_stage: Optional[pd.DataFrame] = None  # cs estimates
        self.second_stage_r2: Optional[pd.Series] = None  # cs estimates
        self.gmm_fit: Optional[dict] = None  # cs estimates

    def fit(self):
        self._fit_first_stage()
        self._fit_second_stage()

        if self.gmm_errors:
            self._fm_gmm_fit()

        self._summary()

    def plot(self, ax=None):
        # calculate and plot predicted vs realized returns
        # calculating predicted returns
        intercept = self.second_stage["c"].mean()
        lambdas = self.second_stage.drop(columns="c").mean().to_numpy()
        betas = self.first_stage.drop(columns="const").to_numpy()

        # running prediction
        pred_returns = intercept + np.dot(betas, lambdas.T)

        if not ax:
            # if no kwargs `ax` is provided, create the axis
            fig, ax = plt.subplots()

        ax.scatter(self.assets.mean().values, np.array(pred_returns))

        if not ax:
            ax.set_xlabel("Realized returns")
            ax.set_ylabel("Predicted returns")
            ax.set_title("Fama-Macbeth Regression")

            plt.show()

    def _validate_data(self):
        # do some quick sanity checks
        if self.assets.index.duplicated().any():
            raise IndexError("Duplicated index in assets")

        if self.factors.index.duplicated().any():
            raise IndexError("Duplicated index in factors")

        if not self.assets.index.equals(self.factors.index):
            raise IndexError("Indices do not match")

        if self.n_west_lags < 0:
            raise ValueError("Newey West lags cannot be negative")

    def _summary(self):
        if self.first_stage is None:
            raise ValueError("Need to run first_stage before GMM")
        if self.second_stage is None:
            raise ValueError("Need to run second_stage before GMM")

        # helper function to calculate FM statistics
        def fm_t_stat(cs_estimates_vector):
            mean_loading = np.mean(cs_estimates_vector)

            # calculate sum of squared differences
            var_loading = np.sum(np.power(cs_estimates_vector - mean_loading, 2))
            var_loading = var_loading / (self.t - 1)
            se_loading = np.sqrt(var_loading) / np.sqrt(self.t)
            t_loading = mean_loading / se_loading

            # asymptotically normally distributed
            p_val = self.norm_pval(t_loading)

            return {
                "gammas": cs_estimates_vector.name,
                "gamma": mean_loading,
                "se_fm": se_loading,
                "tstat_fm": t_loading,
                "p_fm": p_val,
            }

        # add to output
        fm_results = []
        for parameter in self.second_stage.columns:
            fm_results.append(fm_t_stat(self.second_stage[parameter]))

        # calculate CS R2
        X = sm.add_constant(self.first_stage.drop(columns="const"))
        Y = np.mean(self.assets, axis=0)
        cs_mean_fit = sm.OLS(Y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 1})

        self.summary = pd.DataFrame(fm_results)

        # estimate GMM erorors and t-stats
        if self.gmm_errors:
            self.summary["se_gmm"] = self.gmm_fit["se_gamma_gmm"]
            self.summary["tstat_gmm"] = self.gmm_fit["t_gamma_gmm"]

            # asymptotically normally distributed
            self.summary["pvalue_gmm"] = [
                self.norm_pval(x) for x in self.gmm_fit["t_gamma_gmm"]
            ]

        self.summary["nobs"] = self.t
        self.summary["mean_cs_r2"] = self.second_stage_r2.mean(axis=0).values[0]
        self.summary["cs_r2"] = cs_mean_fit.rsquared

    def _fit_first_stage(self):
        # A potential extension here is to allow for time-varying betas, however
        # this would also require a re-factor of the second-stage function.

        # First-pass time-series regressions
        self.first_stage = pd.DataFrame()
        self.first_stage_residuals = pd.DataFrame()

        # This function only estimates the first-pass time-series full sample, and not rolling
        # maybe this should be extended in the future
        for asset in self.assets.columns:
            Y = self.assets[asset]
            X = sm.add_constant(self.factors)

            # estimate full-sample time-series regression using Newey West errors
            fit = sm.OLS(endog=Y, exog=X).fit(
                cov_type="HAC", cov_kwds={"maxlags": self.n_west_lags}
            )

            # save betas from regression
            self.first_stage = pd.concat(
                [self.first_stage, pd.Series(fit.params, name=asset)], axis=1
            )

            # save residuals from regression
            self.first_stage_residuals = pd.concat(
                [self.first_stage_residuals, pd.Series(fit.resid, name=asset)], axis=1
            )

        # transpose to have factors (columns) by portfolios (rows)
        self.first_stage = self.first_stage.transpose()

    def _fit_second_stage(self):
        if self.first_stage is None:
            raise ValueError("Need to run first_stage before second_stage")

        # Second-stage regressions
        self.second_stage = pd.DataFrame()
        self.second_stage_r2 = pd.Series(dtype="float64")

        for timestamp, period in self.assets.iterrows():
            X = sm.add_constant(self.first_stage.drop(columns="const"))
            Y = period

            # estimate the second stage - only one NW lag here
            fit = sm.OLS(endog=Y, exog=X).fit(cov_type="HAC", cov_kwds={"maxlags": 1})

            # save betas from regression
            self.second_stage = pd.concat(
                [self.second_stage, pd.Series(fit.params, name=timestamp)], axis=1
            )

            # save cross-sectional R2
            self.second_stage_r2 = pd.concat(
                [self.second_stage_r2, pd.Series(fit.rsquared_adj, name=timestamp)],
                axis=1,
            )

        # transpose to have factors (columns) by time (rows)
        self.second_stage = self.second_stage.transpose()
        # rename columns
        self.second_stage = self.second_stage.rename(columns={"const": "c"})

        # transpose to have r2 by time (rows)
        self.second_stage_r2 = (
            self.second_stage_r2.transpose().dropna().rename(columns={0: "cs_r2"})
        )

    def _fm_gmm_fit(self):
        if self.first_stage is None:
            raise ValueError("Need to run first_stage before GMM")
        if self.second_stage is None:
            raise ValueError("Need to run second_stage before GMM")

        # ,ret, fac, first_stage, risk_prices):
        assets_np = self.assets.to_numpy()
        factors_np = self.factors.to_numpy()
        first_stage_np = self.first_stage.transpose().to_numpy()
        gamma_np = np.mean(self.second_stage, 0).to_numpy()

        # setting up moment conditions
        factors_np_c = np.hstack(
            [np.ones([self.t, 1]), factors_np]
        )  # factor space with intercept
        epsilon = assets_np - np.matmul(factors_np_c, first_stage_np)

        mom_1 = np.kron(epsilon, np.ones([1, (self.k + 1)])) * np.kron(
            np.ones([1, self.n]), factors_np_c
        )
        mom_2 = assets_np - np.matmul(
            np.hstack([np.ones([self.n, 1]), first_stage_np[1:, :].T]),
            gamma_np.T,
        )

        # combine moment conditions in one matrix
        moments = np.hstack([mom_1, mom_2])

        # estimate HAC robust long-run covariance matrix, S
        S = self.long_run_hac(moments)

        # define weigthing matrix, chi (re-produce OLS estimates)
        chi = np.hstack([np.ones([self.n, 1]), first_stage_np[1:, :].T])

        # maybe this is wrong :)
        # Yes - it was wrong. We fixed it now ;)
        e_1 = np.hstack(
            [
                np.identity(self.n * (self.k + 1)),
                np.zeros([self.n * (self.k + 1), self.n]),
            ]
        )
        e_2 = np.hstack([np.zeros([self.k + 1, self.n * (self.k + 1)]), chi.T])

        # define adjustment matrix
        e = np.vstack([e_1, e_2])

        # construct gradient matrix D
        Dupper = np.empty((self.k + 1, self.k + 1))

        Dupper[0, 0] = 1
        Dupper[0, 1:] = np.mean(factors_np, axis=0)
        Dupper[1:, 0] = np.mean(factors_np, axis=0).T
        Dupper[1:, 1:] = (1 / self.t) * np.matmul(factors_np.T, factors_np)

        d_11 = np.kron(np.identity(self.n), Dupper)
        d_12 = np.zeros([self.n * (self.k + 1), self.k + 1])
        d_21 = np.kron(np.identity(self.n), np.hstack([0, gamma_np[1:]]))
        d_22 = chi

        D = -np.vstack([np.hstack([d_11, d_12]), np.hstack([d_21, d_22])])

        # Estimating the covariance matrix for theta (all parameters)
        # we simply split up the calculation of cov(theta)
        e_d = np.linalg.inv(np.matmul(e, D))
        ese = np.matmul(np.matmul(e, S), e.T)

        theta_cov = (1 / self.t) * (np.matmul(np.matmul(e_d, ese), e_d.T))

        # Pick out covariance related to the risk prices (lower corner matrix)
        gamma_cov = theta_cov[self.n * (self.k + 1) :, self.n * (self.k + 1) :]

        # GMM standard errors and t-statistics
        se_gamma_gmm = np.sqrt(np.diag(gamma_cov))
        t_gamma_gmm = gamma_np / se_gamma_gmm

        self.gmm_fit = {"se_gamma_gmm": se_gamma_gmm, "t_gamma_gmm": t_gamma_gmm}

    @staticmethod
    def norm_pval(x):
        # Return p-value of asymptotically normally distributed
        return 1 - stats.norm.cdf(np.abs(x))

    @staticmethod
    def long_run_hac(gt, flag_andrews: bool = True):
        """
        This function computes the long-run covariance matrix S, with a
        HAC correction.

        Parameters:
            gt (np.array): time series of sample moments

        Returns:
            S: (np.array): HAC robust long-run covariance matrix
        """
        if not flag_andrews:
            raise ValueError("Only implemented optimal bandwidth method")

        (t, k) = gt.shape

        # get 0 axis mean of moments
        g_bar = gt.mean(axis=0)

        # compute first s as sample covariance
        # gde_mean = gt - mnp.repmat(g_bar, t, 1)
        # this is a simple edit that eliminates the dependence on `mnp`
        gde_mean = gt - g_bar

        gamma = (1 / t) * np.matmul(gde_mean.T, gde_mean)
        S = gamma

        # placeholders for data
        alphaN = np.zeros(k)
        alphaD = np.zeros(k)

        # Compute optimal bandwidth via Andrews (1991)
        for i in range(k):
            # estimate AR(1) by Andrews (1991). `trend="c"` simply includes an intercept
            ar1_fit = AutoReg(endog=np.matrix(gt[:, i]).T, lags=1, trend="c").fit()

            # save rho (parameter vector)
            rho = ar1_fit.params

            # Calculate model MSE (sum of squared residuals)
            sigma_2 = np.mean(np.power(ar1_fit.resid, 2))

            # note here that `rho[1]` is the AR coefficient of the sample moments,
            # not on the estimated intercept
            alphaN[i] = (
                4 * rho[1] ** 2 * sigma_2**2 / ((1 - rho[1]) ** 6 * (1 + rho[1]) ** 2)
            )
            alphaD[i] = sigma_2**2 / (1 - rho[1]) ** 4

        alpha_params = np.sum(alphaN) / np.sum(alphaD)

        # Estimating the Bartlett specific bandwidth parameter
        n_lags = np.ceil(1.1447 * (alpha_params * t) ** (1 / 3))

        # Add HAC correction (if n_lags > 0)
        if n_lags > 0:
            for j in range(int(n_lags)):
                gamma = (1 / t) * np.matmul(
                    gde_mean[(j + 1) :, :].T, gde_mean[: -(j + 1), :]
                )

                # calculate Barlett type weights
                weight = 1 - (j + 1) / (n_lags + 1)

                S = S + weight * (gamma + gamma.T)

        return S
