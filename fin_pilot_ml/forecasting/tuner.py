import itertools
import logging

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from statsmodels.tsa.statespace.sarimax import (
    SARIMAX,
)

from fin_pilot_ml.forecasting.config import (
    ForecastingConfig,
)

logger = logging.getLogger(__name__)


class SarimaTuner:
    def __init__(
        self,
        config: ForecastingConfig,
    ) -> None:
        self.config = config

    def tune(
        self,
        train_data: pd.Series,
    ) -> tuple[
        tuple[int, int, int],
        tuple[int, int, int, int],
    ]:

        pdq = list(
            itertools.product(
                self.config.p_values,
                self.config.d_values,
                self.config.q_values,
            )
        )

        seasonal_pdq = list(
            itertools.product(
                self.config.P_values,
                self.config.D_values,
                self.config.Q_values,
                [self.config.seasonal_period],
            )
        )

        best_aic = np.inf

        best_order = None

        best_seasonal_order = None

        total_models = len(pdq) * len(seasonal_pdq)

        current_model = 0

        logger.info(
            "Starting SARIMA tuning " "(%s combinations).",
            total_models,
        )

        for order in pdq:
            for seasonal_order in seasonal_pdq:

                current_model += 1

                logger.debug(
                    "Fitting SARIMA " "%s/%s | " "order=%s seasonal=%s",
                    current_model,
                    total_models,
                    order,
                    seasonal_order,
                )

                try:
                    model = SARIMAX(
                        train_data,
                        order=order,
                        seasonal_order=(seasonal_order),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )

                    result = model.fit(
                        disp=False,
                        method="lbfgs",
                        maxiter=200,
                    )

                    if not result.mle_retvals.get(
                        "converged",
                        False,
                    ):
                        continue

                    if np.isnan(result.aic):
                        continue

                    if result.aic < best_aic:
                        best_aic = result.aic

                        best_order = order

                        best_seasonal_order = seasonal_order

                        logger.info(
                            "New best SARIMA | "
                            "AIC=%.2f "
                            "order=%s "
                            "seasonal=%s",
                            best_aic,
                            best_order,
                            best_seasonal_order,
                        )

                except (
                    ValueError,
                    LinAlgError,
                ) as exc:

                    logger.debug(
                        "SARIMA fit failed | "
                        "order=%s "
                        "seasonal=%s "
                        "error=%s",
                        order,
                        seasonal_order,
                        exc,
                    )

                    continue

        if best_order is None or best_seasonal_order is None:
            raise RuntimeError("No valid SARIMA model found.")

        logger.info(
            "SARIMA tuning complete | "
            "Best AIC=%.2f "
            "order=%s "
            "seasonal=%s",
            best_aic,
            best_order,
            best_seasonal_order,
        )

        return (
            best_order,
            best_seasonal_order,
        )
