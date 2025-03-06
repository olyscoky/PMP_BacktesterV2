import pandas as pd
from typing import Callable
import numpy as np
import operator
from typing import Tuple
import inspect


class StrategyFunction:

    def __init__(self, func, name, **params):
        self.func = func
        self.name = name
        self.params = params

    def __call__(self, *args, **kwargs) -> Callable[[pd.DataFrame, pd.Series, pd.Timestamp], pd.Series]:
        return self.func(*args, **kwargs)

    def get_input_params(self):
        return inspect.signature(self.func).parameters

    def to_dict(self):
        return {
            'name': self.name,
            'params': self.params
        }

    @staticmethod
    def from_dict(data):
        name = data['name']
        params = data['params']
        try:
            strategy_method = getattr(Strategy, name)
            return strategy_method(**params)
        except AttributeError:
            raise ValueError(f"Unknown strategy name: {name}")


class Strategy:

    __operators_map = {
        "<=": operator.le,
        "<": operator.lt,
        ">=": operator.ge,
        ">": operator.gt,
        "==": operator.eq,
        "!=": operator.ne
    }

    @staticmethod
    def equally_weighted() -> StrategyFunction:
        def eq_w(assets: pd.DataFrame, **kwargs) -> pd.Series:
            return pd.Series([(1 / len(assets.columns))] * len(assets.columns), index=assets.columns)
        return StrategyFunction(func=eq_w, name="equally_weighted")

    @staticmethod
    def fixed_weights(weights: dict[str: float]) -> StrategyFunction:
        def fi_w(assets: pd.DataFrame, **kwargs) -> pd.Series:
            assert set(weights.keys()) == set(assets.columns), \
                f"mismatch -> weights: {set(weights.keys())}, #assets: {set(assets.columns)}"
            return pd.Series([weights[a] for a in assets.columns], index=assets.columns)
        return StrategyFunction(func=fi_w, name="fixed_weights", weights=weights)

    @staticmethod
    def risk_parity() -> StrategyFunction:
        def rp_w(
                assets: pd.DataFrame,
                assets_alloc_bounds: dict[str, Tuple[float]],
                **kwargs
        ) -> pd.Series:
            inv_vols = 1 / assets.std(ddof=1)
            weights = inv_vols / np.sum(inv_vols)

            overallocations = sorted(
                [(weights[a] - assets_alloc_bounds[a][1], a) for a in assets.columns if assets_alloc_bounds[a][1]],
                key=lambda x: x[0],
                reverse=True
            )[0]

            if overallocations:
                max_overalloc = overallocations[0]
                overalloc_asset = overallocations[1]

                if max_overalloc >= 0:
                    weights[overalloc_asset] = assets_alloc_bounds[overalloc_asset]
                    del assets_alloc_bounds[overalloc_asset]
                    weights = weights.add(
                        rp_w(
                            assets=assets[[a for a in assets.columns if a != overalloc_asset]],
                            assets_alloc_bounds=assets_alloc_bounds,
                        ) * max_overalloc,
                        fill_value=0
                    )

            underallocations = sorted(
                [(weights[a] - assets_alloc_bounds[a][0], a) for a in assets.columns if assets_alloc_bounds[a][0]],
                key=lambda x: x[0],
            )[0]

            if underallocations:
                max_underalloc = underallocations[0]
                underalloc_asset = underallocations[1]

                if max_underalloc <= 0:
                    weights[underalloc_asset] = assets_alloc_bounds[underalloc_asset]
                    del assets_alloc_bounds[underalloc_asset]
                    weights = weights.subtract(
                        rp_w(  # is recursion the optimal solution ?
                            assets=assets[[a for a in assets.columns if a != underalloc_asset]],
                            assets_alloc_bounds=assets_alloc_bounds,
                        ) * max_underalloc,
                        fill_value=0
                    )

            return weights
        return StrategyFunction(func=rp_w, name="risk_parity")

    @staticmethod
    def minimum_variance() -> StrategyFunction:
        def min_v(
                assets: pd.DataFrame,
                assets_alloc_bounds: dict[str, Tuple[float]],
                gap_days: int,
                freq_multiplier: int,
                shorting_allowed: bool,
                **kwargs
        ) -> pd.Series:
            cov_matrix = StrategyHelpers.get_cov(
                assets=assets,
                freq_multiplier=freq_multiplier,
                gap_days=gap_days
            )
            assert np.linalg.det(cov_matrix) != 0, f"Covariance matrix is singular and cannot be inverted"

            inv_cov_matrix = np.linalg.inv(cov_matrix)
            ones = np.ones(len(cov_matrix))
            res = inv_cov_matrix @ ones / np.sum(inv_cov_matrix @ ones)

            assert abs(1 - np.sum(res)) < 1e-4, f"weight vector: {res} does not sum to ~1"

            weights = pd.Series(res, index=assets.columns)

            if not shorting_allowed:
                for asset in assets.columns:
                    if weights[asset] < 0:
                        weights[asset] = 0
                weights = weights / weights.sum()

            overallocations = sorted(
                [(weights[a] - assets_alloc_bounds[a][1], a) for a in assets.columns if assets_alloc_bounds[a][1]],
                key=lambda x: x[0],
                reverse=True
            )[0]

            if overallocations:
                max_overalloc = overallocations[0]
                overalloc_asset = overallocations[1]

                if max_overalloc >= 0:
                    weights[overalloc_asset] = assets_alloc_bounds[overalloc_asset]
                    del assets_alloc_bounds[overalloc_asset]
                    weights = weights.add(
                        min_v(
                            assets=assets[[a for a in assets.columns if a != overalloc_asset]],
                            assets_alloc_bounds=assets_alloc_bounds,
                            gap_days=gap_days,
                            freq_multiplier=freq_multiplier,
                            shorting_allowed=shorting_allowed
                        ) * max_overalloc,
                        fill_value=0
                    )

            underallocations = sorted(
                [(weights[a] - assets_alloc_bounds[a][0], a) for a in assets.columns if assets_alloc_bounds[a][0]],
                key=lambda x: x[0],
            )[0]

            if underallocations:
                max_underalloc = underallocations[0]
                underalloc_asset = underallocations[1]

                if max_underalloc <= 0:
                    weights[underalloc_asset] = assets_alloc_bounds[underalloc_asset]
                    del assets_alloc_bounds[underalloc_asset]
                    weights = weights.subtract(
                        min_v(  # is recursion the optimal solution ?
                            assets=assets[[a for a in assets.columns if a != underalloc_asset]],
                            assets_alloc_bounds=assets_alloc_bounds,
                            gap_days=gap_days,
                            freq_multiplier=freq_multiplier,
                            shorting_allowed=shorting_allowed
                        ) * max_underalloc,
                        fill_value=0
                    )

            return weights

        return StrategyFunction(func=min_v, name="minimum_variance")

    @staticmethod
    def markovitz_mean_variance() -> StrategyFunction:
        def mktz_mv(
                assets: pd.DataFrame,
                assets_alloc_bounds: dict[str, Tuple[float]],
                rf: pd.Series,
                freq_multiplier: int,
                gap_days: int,
                shorting_allowed: bool,
                **kwargs
        ) -> pd.Series:
            if len(assets.columns) == 1:
                return pd.Series(1, index=assets.columns)

            sigma = StrategyHelpers.get_cov(
                assets=assets,
                freq_multiplier=freq_multiplier,
                gap_days=gap_days
            )
            assert np.linalg.det(sigma) != 0, f"Covariance matrix is singular and cannot be inverted"
            inv_sigma = np.linalg.inv(sigma)

            mu = np.expm1(np.log1p(assets.mean()) * freq_multiplier)
            rsk_fr = rf.iloc[-1]

            res = inv_sigma @ (mu - rsk_fr) / np.sum(inv_sigma @ (mu - rsk_fr))
            assert abs(1 - np.sum(res)) < 1e-4, f"weight vector: {res} does not sum to ~1"
            weights = pd.Series(res, index=assets.columns)

            if not shorting_allowed:
                for asset in assets.columns:
                    if weights[asset] < 0:
                        weights[asset] = 0
                weights = weights / weights.sum()

            overallocations = sorted(
                [(weights[a] - assets_alloc_bounds[a][1], a) for a in assets.columns if assets_alloc_bounds[a][1]],
                key=lambda x: x[0],
                reverse=True
            )[0]

            if overallocations:
                max_overalloc = overallocations[0]
                overalloc_asset = overallocations[1]

                if max_overalloc >= 0:
                    weights[overalloc_asset] = assets_alloc_bounds[overalloc_asset]
                    del assets_alloc_bounds[overalloc_asset]
                    weights = weights.add(
                        mktz_mv(    # is recursion the optimal solution ?
                            assets=assets[[a for a in assets.columns if a != overalloc_asset]],
                            assets_alloc_bounds=assets_alloc_bounds,
                            gap_days=gap_days,
                            freq_multiplier=freq_multiplier,
                            rf=rf,
                            shorting_allowed=shorting_allowed
                        ) * max_overalloc,
                        fill_value=0
                    )

            underallocations = sorted(
                [(weights[a] - assets_alloc_bounds[a][0], a) for a in assets.columns if assets_alloc_bounds[a][0]],
                key=lambda x: x[0],
            )[0]

            if underallocations:
                max_underalloc = underallocations[0]
                underalloc_asset = underallocations[1]

                if max_underalloc <= 0:
                    weights[underalloc_asset] = assets_alloc_bounds[underalloc_asset]
                    del assets_alloc_bounds[underalloc_asset]
                    weights = weights.subtract(
                        mktz_mv(    # is recursion the optimal solution ?
                            assets=assets[[a for a in assets.columns if a != underalloc_asset]],
                            assets_alloc_bounds=assets_alloc_bounds,
                            gap_days=gap_days,
                            freq_multiplier=freq_multiplier,
                            rf=rf,
                            shorting_allowed=shorting_allowed
                        ) * max_underalloc,
                        fill_value=0
                    )

            return weights

        return StrategyFunction(func=mktz_mv, name="markovitz_mean_variance")

# SPECIFIC USE BACKTEST FUNCTION ///////////////////////////////////////////////////////////////////////////////////////
# not parametrizablle //////////////////////////////////////////////////////////////////////////////////////////////////

    # SKELETTON --------------------------------------------------------------------------------------------------------
    # Unimplemented skelletion for fast timing strategy developement
    @staticmethod
    def __timing_function() -> StrategyFunction:
        def ts(
                assets: pd.DataFrame,
                global_df: pd.DataFrame,
                timed_asset_allocation: float,
                secondary_strategy: StrategyFunction,
                previous_weights: pd.Series,
                entry_time: pd.Timestamp | None,
                rebalance: bool,
                t: pd.Timestamp,
                **kwargs
        ) -> Tuple[pd.Timestamp, pd.Series]:
            args = dict(locals())
            if rebalance and secondary_strategy is not None:
                weights = secondary_strategy(**args)
            else:
                weights = previous_weights
            return entry_time, weights
        return StrategyFunction(func=ts, name="tbd_name")
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def time_gold_with_vix() -> StrategyFunction:
        def twg(
                assets: pd.DataFrame,
                global_df: pd.DataFrame,
                timed_asset_allocation: float,
                secondary_strategy: StrategyFunction,
                previous_weights: pd.Series,
                entry_time: pd.Timestamp | None,
                rebalance: bool,
                t: pd.Timestamp,
                **kwargs
        ) -> Tuple[pd.Timestamp | None, pd.Series]:

            args = dict(locals())
            args["assets"] = assets[[a for a in assets.columns if a != "RETURN_XAUUSD"]]

            if rebalance and secondary_strategy is not None:
                weights = secondary_strategy(**args)
                if (entry_time is not None) and ((t - entry_time).days <= 30):
                    weights = weights * (1 - previous_weights["RETURN_XAUUSD"])
                    weights["RETURN_XAUUSD"] = previous_weights["RETURN_XAUUSD"]
                else:
                    weights["RETURN_XAUUSD"] = 0
            else:
                weights = previous_weights

            vix_last_val = global_df["PX_LAST_VIX"].loc[t]
            if (vix_last_val > 40) and (entry_time is None or ((t - entry_time).days > 30)):
                weight_gold = 0.90
                weights = weights * (1 - weight_gold)
                weights["RETURN_XAUUSD"] = weight_gold
                entry_time = t

            elif (entry_time is not None) and ((t - entry_time).days > 30):
                weights = secondary_strategy(**args)
                weights["RETURN_XAUUSD"] = 0
                entry_time = None

            return entry_time, weights

        return StrategyFunction(func=twg, name="time_gold_with_vix")

    @staticmethod
    def time_gold_with_cpi() -> StrategyFunction:
        def twg(
                assets: pd.DataFrame,
                global_df: pd.DataFrame,
                timed_asset_allocation: float,
                secondary_strategy: StrategyFunction,
                previous_weights: pd.Series,
                entry_time: pd.Timestamp | None,
                rebalance: bool,
                t: pd.Timestamp,
                **kwargs
        ) -> Tuple[pd.Timestamp | None, pd.Series]:

            args = dict(locals())
            args["assets"] = assets[[a for a in assets.columns if a != "RETURN_XAUUSD"]]

            if rebalance and secondary_strategy is not None:
                weights = secondary_strategy(**args)
                weights["RETURN_XAUUSD"] = 0
                if (entry_time is not None) and ((t - entry_time).days <= 30):
                    weights = weights * (1 - previous_weights["RETURN_XAUUSD"])
                    weights["RETURN_XAUUSD"] = previous_weights["RETURN_XAUUSD"]
                else:
                    weights["RETURN_XAUUSD"] = 0
            else:
                weights = previous_weights

            cpi_change_vs_last_month = global_df["M2M_INLFATION"].loc[t]
            if (cpi_change_vs_last_month > 0.005) and ((entry_time is None) or (t - entry_time).days > 30):
                weight_gold = 0.25
                weights = weights * (1 - weight_gold)
                weights["RETURN_XAUUSD"] = weight_gold
                entry_time = t

            elif (entry_time is not None) and ((t - entry_time).days > 30):
                weights = secondary_strategy(**args)
                weights["RETURN_XAUUSD"] = 0
                entry_time = None

            return entry_time, weights

        return StrategyFunction(func=twg, name="time_gold_with_cpi")


class StrategyHelpers:

    @staticmethod
    def get_cov(
            assets: pd.DataFrame,
            freq_multiplier: int,
            gap_days: int
    ) -> pd.DataFrame:
        return assets.loc[:assets.index[-1] - pd.Timedelta(days=gap_days)].cov() * freq_multiplier
