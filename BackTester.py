import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from scipy.stats import skew, kurtosis
from typing import Callable, Tuple
from tqdm import tqdm

from Paths import Path
from Strategy import StrategyFunction
from BackTestRec import BackTestRec
from Portfolio import InvesmentUniverse
from Utils import convert_return_period


warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_rows', None)


class BackTester:

    # CONFIGURATIONS
    __valid_frequencies = ["D", "W", "M", "Q", "Y"]
    __reindexation_complementary_data = "bfill"     # "nearest", None

    def __init__(
            self,
            investment_universe: 'InvesmentUniverse',
            data_frequency: str,
            trading_costs_bps: int = 0,
            ccy_exchg_costs_bps: int = 0,
            trading_days_count: int = 252,
            fill_nan: float | None = None,
            drop_nan: bool = False,
    ):
        self.__fill_nan = fill_nan
        self.__drop_nan = drop_nan
        self.__investment_universe = investment_universe
        self.__df = investment_universe.get_investable_return_df(fill_nan=fill_nan, drop_nan=drop_nan)
        self.__rf_d = investment_universe.get_daily_rf()
        self.__rf_y = investment_universe.get_yearly_rf()

        self.__start_date = self.__df.index[0]
        self.__end_date = self.__df.index[-1]

        self.__main_ccy = self.__investment_universe.get_main_ccy()
        self.__data_frequency = data_frequency

        self.__tc = trading_costs_bps / 10_000
        self.__ccy_exchg_c = ccy_exchg_costs_bps / 10_000
        self.__trading_days_cnt = trading_days_count

        self.__performed_backtests = dict()

    def get_performed_backtet(self) -> dict[str, 'BackTestRec']:
        return self.__performed_backtests

    def __initialize_plot_colors(self):
        np.random.seed(12)

        blue_cmap = colormaps.get_cmap("Blues")
        blues = blue_cmap(np.linspace(0, 1, 8192))
        np.random.shuffle(blues)
        self.__plot_colors = blues

        self.__red_color = "#ff4747"
        self.__green_color = "#b0e892"
        self.__blue_color = "#01BFFF"

        gray_cmap = colormaps.get_cmap("gray")
        greys = gray_cmap(np.linspace(0, 1, 8192))
        np.random.shuffle(greys)
        self.__grey_scale = greys

    def __get_closest_index(
            self,
            ts: pd.Timestamp,
            approach_from_above: bool = False,
            approach_from_below: bool = False
    ) -> pd.Timestamp:
        if not approach_from_below and not approach_from_above:
            return self.__df.index[(self.__df.index - ts).to_series().abs().argmin()]
        elif approach_from_above:
            lt_i = self.__df.index[self.__df.index <= ts]
            return lt_i[-1] if not lt_i.empty else None
        elif approach_from_below:
            gt_i = self.__df.index[self.__df.index >= ts]
            return gt_i[0] if not gt_i.empty else None
        else:
            raise ValueError("approach_above & approach_below cannot both be True")

    @staticmethod
    def __cumulate_returns(ret: pd.Series) -> pd.Series:
        return np.expm1(np.log1p(ret).cumsum())

    def __compute_metrics(self, returns: pd.Series) -> Tuple:
        freq_multiplier = self.__get_freq_multiplier(freq=self.__data_frequency)

        rf = self.__rf_d.loc[self.__rf_d.index.intersection(returns.index)]
        rf = convert_return_period(
            ret=rf,
            period_d_current=1,
            period_d_new=self.__frequency_to_day(freq=self.__data_frequency)
        )
        # risk-free now logic for frequencies other than daily
        avg_geom_excess_return = np.exp(np.log(1 + (returns - rf)).mean() * freq_multiplier) - 1
        avg_vol = np.std(returns) * np.sqrt(freq_multiplier)
        avg_arithm_excess_return = np.mean(returns - rf)
        neg_excess_rets = returns[returns < avg_arithm_excess_return]
        avg_semi_vol = np.sqrt(np.mean((neg_excess_rets - avg_arithm_excess_return) ** 2)
                               if len(neg_excess_rets) > 0 else 0)
        sharpe_ratio = avg_geom_excess_return / avg_vol     # Ziegler want geom sharpi

        cum_ret = (1 + returns).cumprod()
        dd = (cum_ret / cum_ret.cummax()) - 1
        max_drawdown = abs(dd.min())

        skewness = float(skew(returns.to_numpy(), bias=False))
        kurt = kurtosis(returns, bias=False)

        return avg_geom_excess_return, avg_vol, avg_semi_vol, sharpe_ratio, max_drawdown, skewness, kurt

    @staticmethod
    def __get_freq_multiplier(freq: str | int) -> int:
        if isinstance(freq, int):
            return freq
        if freq == "Y":
            return 1
        elif freq == "Q":
            return 4
        elif freq == "M":
            return 12
        elif freq == "W":
            return 52
        elif freq == "D":
            return 252
        else:
            raise ValueError(f"frequency: {freq} was not recognized")

    @staticmethod
    def __frequency_to_day(freq: str | int) -> int:
        if isinstance(freq, int):
            return freq
        if freq == "Y":
            return 365
        elif freq == "Q":
            return 91
        elif freq == "M":
            return 30
        elif freq == "W":
            return 7
        elif freq == "D":
            return 1
        else:
            raise ValueError(f"frequency: {freq} was not recognized")

    @staticmethod
    def __get_offset(rebalancing_freq: str | int, gap_days: int = 0) -> pd.DateOffset:
        if isinstance(rebalancing_freq, int):
            return pd.DateOffset(days=(rebalancing_freq + gap_days))
        elif rebalancing_freq == "Y":
            return pd.DateOffset(years=1, days=gap_days)
        elif rebalancing_freq == "Q":
            return pd.DateOffset(months=3, days=gap_days)
        elif rebalancing_freq == "M":
            return pd.DateOffset(months=1, days=gap_days)
        else:
            return pd.DateOffset(days=(1 + gap_days))

    @staticmethod
    def __get_days_from_freq(freq: str) -> int:
        dummy_date = pd.to_datetime("2000-01-01")
        return ((dummy_date + pd.tseries.frequencies.to_offset(freq)) - dummy_date).days

    def __make_ccy_hedge(self, ccy_hedge_ratio: float | None) -> pd.DataFrame:
        asset_ccy_hedge_returns = pd.DataFrame(
            data=np.zeros((len(self.__df), len(self.__df.columns))),
            columns=self.__df.columns,
            index=self.__df.index
        )
        ccy_hedge_returns = self.__investment_universe.get_ccy_hedge_return_df(
            fill_nan=self.__fill_nan,
            drop_nan=self.__drop_nan
        )
        ccy_returns = self.__investment_universe.get_ccy_return_df(
            fill_nan=self.__fill_nan,
            drop_nan=self.__drop_nan
        )

        for asset in self.__investment_universe.get_investable_asset_universe():
            if asset.get_ccy() == self.__main_ccy:
                continue

            ccy_pair = None
            for ccy_p in [asset.get_ccy() + self.__main_ccy, self.__main_ccy + asset.get_ccy()]:
                if ccy_p in ccy_hedge_returns.columns:
                    ccy_pair = ccy_p
                    break

            try:
                if ccy_pair[3:] == self.__main_ccy:
                    hedge_return = -ccy_hedge_returns[ccy_pair] * ccy_hedge_ratio

                else:
                    hedge_return = (
                        ((1 + ccy_hedge_returns[ccy_pair]) / (1 + ccy_returns[ccy_pair])) - 1
                    ) * ccy_hedge_ratio

                asset_ccy_hedge_returns[asset] = hedge_return

            except KeyError:
                raise Warning(f"No ccy_hedge_returns could be found for asset {asset.get_name()}")

        assert not asset_ccy_hedge_returns.isna().all(axis=0).any(), f"some asset have undefined hedge returns"
        return asset_ccy_hedge_returns

    @staticmethod
    def make_systematic_weights(
            assets_of_interest: dict[str, [float | None, float | None]],
            step: float,
            max_investment_level: float = 1
    ) -> list[dict[str, float]]:

        def generate_weight_range(min_weight: float, max_weight: float) -> list[float]:
            return [round(min_weight + i * step, 3) for i in range(int((max_weight - min_weight) / step) + 1)]

        def sys_weights_generator(assets: list[str], current_weights: dict[str, float]) -> list[dict[str, float]]:
            if not assets:
                if round(sum(current_weights.values()), 3) == max_investment_level:
                    return [current_weights]
                else:
                    return []

            current_asset = assets[0]
            remaining_assets = assets[1:]
            min_weight, max_weight = assets_of_interest[current_asset]

            weight_range = generate_weight_range(min_weight=min_weight, max_weight=max_weight)

            combinations = []
            for weight in weight_range:
                new_weights = current_weights.copy()
                new_weights[current_asset] = weight
                combinations += sys_weights_generator(assets=remaining_assets, current_weights=new_weights)

            return combinations

        return sys_weights_generator(assets=list(assets_of_interest.keys()), current_weights={})

    def __get_ccy_indepedent_returns(self) -> pd.DataFrame:

        if all(
                ccy == self.__main_ccy for ccy in [
                    asset.get_ccy() for asset in self.__investment_universe.get_investable_asset_universe()
                ]
        ):
            return self.__df

        ccy_returns = self.__investment_universe.get_ccy_return_df(
            fill_nan=self.__fill_nan,
            drop_nan=self.__drop_nan
        )

        for asset in self.__investment_universe.get_investable_asset_universe():
            if asset.get_ccy() != self.__main_ccy:
                try:
                    self.__df.loc[:, asset.get_name()] = (1 + self.__df[asset.get_name()]) * \
                        (1 + ccy_returns[f"{asset.get_ccy()}{self.__main_ccy}"]) - 1
                except KeyError:
                    try:
                        self.__df.loc[:, asset.get_name()] = (1 + self.__df[asset.get_name()]) / \
                            (1 + ccy_returns[f"{self.__main_ccy}{asset.get_ccy()}"]) - 1
                    except KeyError:
                        raise KeyError(f"Could not resolve currency Returns for asset: {asset.get_name()}")

        return self.__df

    def plot_return_evolution(
            self,
            backtests: list[BackTestRec] | None = None,
            show_best_worst_backtest: bool = False,
            best_backtest_name: str = "BEST_BACKTEST_NAME",
            worst_backtest_name: str = "WORST_BACKTEST_NAME",
            backtest_quality_metric: Callable[['BackTestRec'], None] = lambda bt: bt.get_sharpe_ratio(),
            best_quality_metric_decreasing: bool = False,
            start_date: str | pd.Timestamp | None = None,
            end_date: str | pd.Timestamp | None = None,
            excess_returns: bool = False,
            incl_legend: bool = False,
            legend_position: str = "upper left",
            save_plot: bool = False,
            plot_saving_format: str = "png",
            multi_bt_plot_name: str = "NO_NAME_PLOT",
    ):
        self.__initialize_plot_colors()

        sd = (start_date if isinstance(start_date, pd.Timestamp) else pd.to_datetime(start_date)) \
            if start_date is not None else min([bt.get_start_date() for bt in backtests])
        ed = (end_date if isinstance(end_date, pd.Timestamp) else pd.to_datetime(end_date)) \
            if end_date is not None else max(bt.get_end_date() for bt in backtests)

        assert sd < ed, f"start_date: {sd} > end_date: {ed}"
        assert len(backtests) >= 1, "At least 1 backtest is required to generate a plot"

        _, sorted_bt = zip(
            *sorted(
                ((backtest_quality_metric(bt), bt) for bt in backtests),
                reverse=best_quality_metric_decreasing
            )
        )

        tadjust_daily_returns = []
        for bt in backtests:
            if bt.get_strategy_daily_returns().index[0] > sd:
                tadjust_daily_returns.append(
                    pd.concat([pd.Series([0], index=[sd]), bt.get_strategy_daily_returns()])
                )
            else:
                tadjust_daily_returns.append(bt.get_strategy_daily_returns())

        plt.figure(figsize=(10, 6))
        for i, (daily_returns, bt) in enumerate(zip(tadjust_daily_returns, sorted_bt)):
            plt.plot(
                self.__cumulate_returns(daily_returns),
                label=backtest_quality_metric(bt) if not show_best_worst_backtest else None,
                linestyle="-",
                linewidth=min(0.0, max(1.5 / (len(sorted_bt) / 2), 0.3)),
                color=self.__grey_scale[i] if show_best_worst_backtest else self.__plot_colors[i]
            )
        if show_best_worst_backtest and (len(sorted_bt) >= 2):
            plt.plot(
                self.__cumulate_returns(sorted_bt[0].get_strategy_daily_returns()),
                label=best_backtest_name,
                linestyle="-",
                linewidth=0.5,
                color=self.__green_color
            )
            plt.plot(
                self.__cumulate_returns(sorted_bt[-1].get_strategy_daily_returns()),
                label=worst_backtest_name,
                linestyle="-",
                linewidth=0.5,
                color=self.__red_color
            )

        plt.title(
            f"Cumulative {'Excess ' if excess_returns else ''}"
            f"Returns of Investment Strategie{'s' if len(backtests) > 1 else ''}"
        )
        plt.xlabel("Year", fontsize=12)
        plt.ylabel(f"Cumulative {'Excess ' if excess_returns else ''}Returns", fontsize=12)
        plt.grid(True, linestyle="--", linewidth=0.5)
        if incl_legend:
            plt.legend(loc=legend_position, fontsize=10, frameon=True, framealpha=0.9)

        if save_plot:
            plt.savefig(
                os.path.join(
                    Path.PLOT_PATH, f"{backtests[0].get_id()}.{plot_saving_format}"
                    if len(backtests) == 1 else f"{multi_bt_plot_name}.{plot_saving_format}"
                ),
                format=plot_saving_format
            )

        plt.show()

    def backtest(
            self,
            strategy: StrategyFunction,
            secondary_strategy: StrategyFunction | None = None,
            rebalancing_freq: str | int = "D",
            complementary_data: pd.DataFrame | None = None,
            shorting_allowed: bool = False,
            gap_days: int = 0,
            constrained: bool = True,
            rolling: bool = False,
            ccy_hedge_ratio: float | None = None,
            oos_backtest: bool = True,
            bt_start_date: str | pd.Timestamp | None = None,
            bt_end_date: str | pd.Timestamp | None = None,
    ) -> BackTestRec:

        # DATE INITIALIZATION ------------------------------------------------------------------------------------------
        bt_sd = (bt_start_date if isinstance(bt_start_date, pd.Timestamp) else pd.to_datetime(bt_start_date)) \
            if bt_start_date is not None else self.__start_date
        bt_ed = (bt_end_date if isinstance(bt_end_date, pd.Timestamp) else pd.to_datetime(bt_end_date)) \
            if bt_end_date is not None else self.__end_date
        # --------------------------------------------------------------------------------------------------------------

        # VARIABLE INITIALIZATION --------------------------------------------------------------------------------------
        asset_ccy_hedge = self.__make_ccy_hedge(ccy_hedge_ratio=ccy_hedge_ratio)
        assets = self.__get_ccy_indepedent_returns()

        if complementary_data is not None:
            if self.__reindexation_complementary_data is not None:
                complementary_data = complementary_data.reindex(
                    assets.index,
                    method=self.__reindexation_complementary_data
                )
                assert(complementary_data.index.equals(assets.index)), "indices are not equal"
            else:
                if not complementary_data.index.equals(assets.index):
                    raise Warning("index mismatch between complementary_data and assets")

        offset = self.__get_offset(rebalancing_freq=rebalancing_freq, gap_days=gap_days)
        window_start = assets.index.min()   # just to get a fixed anchor date point
        # will vary anyway depending on backtest specification for rolling window.

        strat_ret = pd.Series(name="Strategy_Returns", dtype=np.float64)
        strat_weights = pd.DataFrame(columns=assets.columns, dtype=np.float64)

        asset_ret = pd.Series(name="Asset_Returns", dtype=np.float64)
        hedge_ret = pd.Series(name="Hedge_Returns", dtype=np.float64)

        asset_turnover = pd.Series(name="Asset_Turnover", dtype=np.float64)
        ccy_turnover = pd.Series(name="CCY_Turnover", dtype=np.float64)

        rebalancing_tc = pd.Series(name="Rebalancing_TransCosts", dtype=np.float64)
        hedge_tc = pd.Series(name="Hedge_TransCosts", dtype=np.float64)

        weights_old = pd.Series(0, index=assets.columns, dtype=np.float64)
        freq_multiplier = self.__get_freq_multiplier(self.__data_frequency)

        entry_time = None
        # --------------------------------------------------------------------------------------------------------------

        # CREATING REBALANCING INVESMTENT BLOCKS -----------------------------------------------------------------------
        rebalancings = dict()
        for (t1, assets_ret_t), (t2, hedges_ret_t) in zip(
                assets.groupby(pd.Grouper(freq=rebalancing_freq)),
                asset_ccy_hedge.groupby(pd.Grouper(freq=rebalancing_freq)),
                # assets.groupby(pd.Grouper(freq=rebalancing_freq, label="left")),  # UNCERTAINTY ABOUT LABEL
                # asset_ccy_hedge.groupby(pd.Grouper(freq=rebalancing_freq, label="left")),
        ):
            if (oos_backtest and t1 < bt_sd) or t1 > bt_ed:
                continue
            rebalancings[self.__get_closest_index(ts=t1, approach_from_above=True)] = (assets_ret_t, hedges_ret_t)
        # --------------------------------------------------------------------------------------------------------------

        # BACKTEST FRAMEWORK -------------------------------------------------------------------------------------------
        for t in tqdm(self.__df[min(rebalancings.keys()):max(rebalancings.keys())].index):

            assets_past_ret = assets.loc[window_start:(t - offset if oos_backtest else t)]

            strategy_params = {
                "assets": assets_past_ret,
                "complementary_data": complementary_data,
                "rf": self.__rf_d[window_start:(t - offset if oos_backtest else t)],
                "t": t,
                "secondary_strategy": secondary_strategy,
                "gap_days": gap_days,
                "previous_weights": weights_old,
                "assets_alloc_bounds": {
                    asset_name: (
                        self.__investment_universe.get_asset_from_name(name=asset_name).get_asset_min_allocation(),
                        self.__investment_universe.get_asset_from_name(name=asset_name).get_asset_max_allocation()
                    ) for asset_name in assets.columns
                },
                "freq_multiplier": freq_multiplier,
                "shorting_allowed": shorting_allowed,
                "entry_time": entry_time,
                "global_df": self.__df,
            }

            if t in rebalancings.keys():
                strategy_params["rebalance"] = True
                res = strategy(**strategy_params)
                if isinstance(res, tuple):
                    entry_time = res[0]
                    weights_new = res[1]
                else:
                    weights_new = res

            else:
                strategy_params["rebalance"] = False
                if "rebalance" in strategy.get_input_params():
                    res = strategy(**strategy_params)
                    if isinstance(res, tuple):
                        entry_time = res[0]
                        weights_new = res[1]
                    else:
                        weights_new = res

                else:
                    weights_new = weights_old

            assets_ret_t = assets.loc[t]
            hedges_ret_t = asset_ccy_hedge.loc[t]

            strat_weights.loc[t] = weights_new if not constrained else (weights_new / weights_new.sum())

            asset_turnover_t = np.abs(weights_new - weights_old)
            asset_turnover.loc[t] = asset_turnover_t

            ccy_change_t = 0
            ccy_turnover_t = pd.Series(dtype=np.float64)
            for ccy, indices in weights_new.groupby(
                weights_new.index.map(lambda name: self.__investment_universe.get_asset_from_name(name).get_ccy())
            ).groups.items():
                if ccy != self.__main_ccy:
                    changes = weights_new.loc[indices] - weights_old.loc[indices]
                    ccy_change_t += changes.sum()
                    ccy_turnover_t[ccy] = changes.abs().sum()
            ccy_turnover.loc[t] = ccy_turnover_t

            # hedging cost calculating part is wrong needs to be modified
            rolling_turnover = max(((rebalancing_freq if isinstance(rebalancing_freq, int)
                                     else self.__get_days_from_freq(rebalancing_freq)) / 22), 0)

            hedge_tc_t = np.sum(
                self.__tc * ((2 * rolling_turnover) + ccy_change_t) *
                ccy_hedge_ratio if ccy_hedge_ratio is not None else 0
            )
            rebalancing_tc_t = np.sum(self.__tc * asset_turnover_t)
            rebalancing_tc.loc[t] = rebalancing_tc_t
            hedge_tc.loc[t] = hedge_tc_t

            asset_ret_t_cum = (assets_ret_t * weights_new).sum()
            asset_ret.loc[t] = asset_ret_t_cum
            hedge_ret_t_cum = (hedges_ret_t * weights_new).sum()
            hedge_ret.loc[t] = hedge_ret_t_cum

            strat_ret.loc[t] = asset_ret_t_cum + hedge_ret_t_cum - hedge_tc_t - rebalancing_tc_t - \
                               ccy_turnover_t.sum() * self.__ccy_exchg_c + \
                               (1 - weights_new.sum()) * (((1 + self.__rf_d.loc[t]) ** freq_multiplier) - 1)
            # added risk-free lending / borrowing if totalweights differ from 1
            # changes in compute_metrics have been reverted

            weights_old = weights_new * (assets_ret_t + 1)
            weights_old = (weights_old / weights_old.sum()) * weights_new.sum()
            if rolling:
                window_start += offset

            period_perf = (weights_new * assets_ret_t).sum()
            if period_perf < -0.8:     # there are no margin calls -> we go bankrupt or not
                raise Warning(
                    f"Portfolio lost {period_perf * 100}%\n" if period_perf <= -1
                    else f"margin call occured -> portfolio lost {period_perf * 100}%\n",
                    f"  -> t = {t}\n"
                    f"  -> weights = {weights_new}"
                )

        # --------------------------------------------------------------------------------------------------------------

        # RECORDING BACKTEST -------------------------------------------------------------------------------------------
        backtest_instance = self.__make_backtest_record(
            strategy=strategy,
            rebalancing_freq=rebalancing_freq,
            gap_days=gap_days,
            constrained=constrained,
            rolling=rolling,
            ccy_hedge_ratio=ccy_hedge_ratio,
            strat_ret=strat_ret,
            strat_weights=strat_weights,
            asset_ret=asset_ret,
            hedge_ret=hedge_ret,
            asset_turnover=asset_turnover,
            ccy_turnover=ccy_turnover,
            rebalancing_tc=rebalancing_tc,
            hedge_tc=hedge_tc,
            input_data_days_cnt=(bt_sd - self.__start_date).days,
            bt_start_date=bt_sd,
            bt_end_date=bt_ed,
        )
        # --------------------------------------------------------------------------------------------------------------

        print(strat_weights)
        print(strat_ret)

        return backtest_instance

    def __make_backtest_record(
            self,
            strategy: StrategyFunction,
            rebalancing_freq: str | int,
            gap_days: int,
            constrained: bool,
            rolling: bool,
            ccy_hedge_ratio: float | None,
            strat_ret: pd.Series,
            strat_weights: pd.DataFrame,
            asset_ret: pd.Series,
            hedge_ret: pd.Series,
            asset_turnover: pd.Series,
            ccy_turnover: pd.Series,
            rebalancing_tc: pd.Series,
            hedge_tc: pd.Series,
            input_data_days_cnt: int,
            bt_start_date: pd.Timestamp,
            bt_end_date: pd.Timestamp,
    ) -> 'BackTestRec':

        avg_excess_return, avg_vol, avg_semi_vol, sharpe_ratio, max_drawdown, skewness, kurt = self.__compute_metrics(
            returns=strat_ret
        )

        bt_rec = BackTestRec(
            assets=self.__investment_universe.get_investable_asset_names(),
            main_ccy=self.__main_ccy,
            rebalancing_freq=rebalancing_freq,
            strategy=strategy,
            gap_days=gap_days,
            constrained=constrained,
            rolling=rolling,
            ccy_hedge_ratio=ccy_hedge_ratio,
            strat_ret=strat_ret,
            strat_weights=strat_weights,
            asset_ret=asset_ret,
            hedge_ret=hedge_ret,
            asset_turnover=asset_turnover,
            ccy_turnover=ccy_turnover,
            rebalancing_tc=rebalancing_tc,
            hedge_tc=hedge_tc,
            avg_excess_ret=avg_excess_return,
            avg_vol=avg_vol,
            avg_semi_vol=avg_semi_vol,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            skewness=skewness,
            kurt=kurt,
            trading_costs_bps=int(round(self.__tc * 10_000)),
            ccy_exchg_costs_bps=int(round(self.__ccy_exchg_c * 10_000)),
            input_data_days_cnt=input_data_days_cnt,
            start_date=bt_start_date,
            end_date=bt_end_date
        )

        self.__performed_backtests[bt_rec.get_id()] = bt_rec
        return bt_rec


if __name__ == "__main__":
    pass

# TODO
# - implement additional non tradeable information passing to backtesting.
