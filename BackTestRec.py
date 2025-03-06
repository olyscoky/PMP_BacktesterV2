import pandas as pd
from tabulate import tabulate
import json
import os

from Strategy import StrategyFunction

from Paths import Path


class BackTestRec:

    def __init__(
            self,
            assets: list[str],
            main_ccy: str,
            rebalancing_freq: str,
            strategy: StrategyFunction,
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
            avg_excess_ret: float,
            avg_vol: float,
            avg_semi_vol: float,
            sharpe_ratio: float,
            max_drawdown: float,
            skewness: float,
            kurt: float,
            trading_costs_bps: int,
            ccy_exchg_costs_bps: int,
            input_data_days_cnt: int,
            start_date: pd.Timestamp,
            end_date: pd.Timestamp
    ):
        self.__assets = assets
        self.__main_ccy = main_ccy
        self.__rebalancing_freq = rebalancing_freq
        self.__strategy = strategy
        self.__gap_days = gap_days
        self.__constrained = constrained
        self.__rolling = rolling
        self.__ccy_hedge_ratio = ccy_hedge_ratio

        self.__strat_ret = strat_ret
        self.__strat_weights = strat_weights
        self.__asset_ret = asset_ret
        self.__hedge_ret = hedge_ret
        self.__asset_turnover = asset_turnover
        self.__ccy_turnover = ccy_turnover
        self.__rebalancing_tc = rebalancing_tc
        self.__hedge_tc = hedge_tc

        self.__avg_excess_ret = avg_excess_ret
        self.__avg_vol = avg_vol
        self.__avg_semi_vol = avg_semi_vol
        self.__sharpe_ratio = sharpe_ratio
        self.__max_drawdown = max_drawdown
        self.__skewness = skewness
        self.__kurtosis = kurt

        self.__trading_costs = trading_costs_bps
        self.__ccy_exchg_c_bps = ccy_exchg_costs_bps
        self.__input_data_days_cnt = input_data_days_cnt
        self.__start_date = start_date
        self.__end_date = end_date

        self.__generate_backtest_id()

    def __generate_backtest_id(self):
        self.__id = "_".join([
            "-".join(self.__assets),
            f"sd={self.__start_date.strftime('%Y-%m-%d')}",
            f"ed={self.__end_date.strftime('%Y-%m-%d')}",
            f"tc={self.__trading_costs}"    # put a coma here
            f"{self.__strategy.name}@{str(self.__strategy.params)}",
            str(self.__rebalancing_freq),
            f"gd={self.__gap_days}",
            "cnst" if not self.__constrained else "uncnst",
            "rol" if self.__rolling else "n-rol",
            "n-hdg" if (self.__ccy_hedge_ratio is None) or (abs(self.__ccy_hedge_ratio) < 1e-3)
            else f"hdg@{self.__ccy_hedge_ratio}",
        ])

    def get_id(self) -> str:
        return self.__id

    def get_assets(self) -> list[str]:
        return self.__assets

    def get_main_ccy(self):
        return self.__main_ccy

    def get_rebalancing_freq(self) -> str:
        return self.__rebalancing_freq

    def get_strategy_name(self) -> str:
        return self.__strategy.name

    def get_strategy_params(self) -> dict:
        return self.__strategy.params

    def get_gap_days(self) -> int:
        return self.__gap_days

    def is_constrained(self) -> bool:
        return True if self.__constrained else False

    def is_rolling(self) -> bool:
        return True if self.__rolling else False

    def is_ccy_hedged(self) -> bool:
        return True if self.__ccy_hedge_ratio is not None and abs(self.__ccy_hedge_ratio) > 1e-3 else False

    def get_ccy_hedge_ratio(self) -> float:
        return self.__ccy_hedge_ratio

    def get_strategy_returns(self) -> pd.Series:
        return self.__strat_ret

    def get_strat_weights(self) -> pd.DataFrame:
        return self.__strat_weights

    def get_asset_returns(self) -> pd.Series:
        return self.__asset_ret

    def get_hedge_returns(self) -> pd.Series:
        return self.__hedge_ret

    def get_turnover(self) -> pd.Series:
        return self.__asset_turnover

    def get_ccy_turnover(self) -> pd.Series:
        return self.__ccy_turnover

    def get_rebalancing_trading_costs(self) -> pd.Series:
        return self.__rebalancing_tc

    def get_hedging_trading_costs(self) -> pd.Series:
        return self.__hedge_tc

    def get_avg_excess_returns(self) -> float:
        return self.__avg_excess_ret

    def get_avg_volatility(self) -> float:
        return self.__avg_vol

    def get_avg_semi_volatility(self) -> float:
        return self.__avg_semi_vol

    def get_sharpe_ratio(self) -> float:
        return self.__sharpe_ratio

    def get_max_drawdown(self) -> float:
        return self.__max_drawdown

    def get_skewness(self) -> float:
        return self.__skewness

    def get_kurtosis(self) -> float:
        return self.__kurtosis

    def get_trading_costs(self) -> int:
        return self.__trading_costs

    def get_ccy_exchange_costs(self) -> int:
        return self.__ccy_exchg_c_bps

    def get_input_data_days_cnt(self) -> int:
        return self.__input_data_days_cnt

    def get_start_date(self) -> pd.Timestamp:
        return self.__start_date

    def get_end_date(self) -> pd.Timestamp:
        return self.__end_date

    def summarize(self):
        data = [
            ["Assets", ", ".join(self.__assets)],
            ["Main Currency", self.__main_ccy],
            ["#Days of Input Date", self.__input_data_days_cnt],
            ["Start Date", self.__start_date.strftime("%Y-%m-%d")],
            ["End Date", self.__end_date.strftime("%Y-%m-%d")],
            ["Trading Costs bps", self.__trading_costs],
            ["CCY Exchange Costs bps", self.__ccy_exchg_c_bps],
            ["Rebalancing Frequency", self.__rebalancing_freq],
            ["Strategy", f"{self.__strategy.name} | {self.__strategy.params}"],
            ["Gap Days", self.__gap_days],
            ["Constrained", self.__constrained],
            ["Rolling", self.__rolling],
            ["Currency Hedge Ratio", f"{round(self.__ccy_hedge_ratio * 100)}%"
                if self.__ccy_hedge_ratio is not None else "N/A"],
            ["Average Excess Return", f"{round(self.__avg_excess_ret * 100, 2)}%"],
            ["Average Volatility", f"{round(self.__avg_vol * 100, 2)}%"],
            ["Average Semi-Volatility", f"{round(self.__avg_semi_vol * 100, 2)}%"],
            ["Sharpe Ratio", round(self.__sharpe_ratio, 4)],
            ["Max Drawdown", f"{round(self.__max_drawdown * 100, 2)}%"],
            ["Skewness", round(self.__skewness, 2)],
            ["Kurtosis", round(self.__kurtosis, 2)],
        ]
        table = tabulate(data, headers=["Metric", "Value"], tablefmt="fancy_grid")
        print(table)

    def save_backtest(self, path_extention: str | None = None, name_overwirte: str | None = None):

        def dataframe_to_json(df):
            result = {}
            for column in df:
                result[column] = series_to_json(df[column])
            return result

        def series_to_json(series):
            if isinstance(series, pd.Series):
                if isinstance(series.index, pd.DatetimeIndex):
                    return {date.strftime('%Y-%m-%d'): val if not isinstance(val, pd.Series) else series_to_json(val)
                            for date, val in series.items()}
                else:
                    return {str(index): val if not isinstance(val, pd.Series) else series_to_json(val) for index, val in
                            series.items()}
            else:
                return series

        data = {
            'assets': self.__assets,
            'main_ccy': self.__main_ccy,
            'rebalancing_freq': self.__rebalancing_freq,
            "strategy": self.__strategy.to_dict(),
            'gap_days': self.__gap_days,
            'constrained': self.__constrained,
            'rolling': self.__rolling,
            'ccy_hedge_ratio': self.__ccy_hedge_ratio,
            'strat_ret': series_to_json(self.__strat_ret),
            'strat_weights': dataframe_to_json(self.__strat_weights),
            'asset_ret': series_to_json(self.__asset_ret),
            'hedge_ret': series_to_json(self.__hedge_ret),
            'asset_turnover': series_to_json(self.__asset_turnover),
            'ccy_turnover': series_to_json(self.__ccy_turnover),
            'rebalancing_tc': series_to_json(self.__rebalancing_tc),
            'hedge_tc': series_to_json(self.__hedge_tc),
            'avg_excess_ret': self.__avg_excess_ret,
            'avg_vol': self.__avg_vol,
            "avg_semi_vol": self.__avg_semi_vol,
            'sharpe_ratio': self.__sharpe_ratio,
            'max_drawdown': self.__max_drawdown,
            'skewness': self.__skewness,
            'kurt': self.__kurtosis,
            'input_data_days_cnt': self.__input_data_days_cnt,
            'start_date': self.__start_date.strftime("%Y-%m-%d"),
            'end_date': self.__end_date.strftime("%Y-%m-%d"),
            'trading_costs_bps': self.__trading_costs,
            'ccy_exchg_costs_bps': self.__ccy_exchg_c_bps
        }

        bt_path = Path.BACKTEST_PATH if path_extention is None else os.path.join(Path.BACKTEST_PATH, path_extention)
        with open(
                os.path.join(bt_path, name_overwirte if name_overwirte is not None else f"{self.__id}.json"), "w"
        ) as file:
            json.dump(data, file, indent=4)

    @classmethod
    def load_backtest(cls, path_file: str) -> 'BackTestRec':

        def json_to_dataframe(data):
            df = pd.DataFrame()
            for column, series_data in data.items():
                df[column] = json_to_series(series_data)
            return df

        def json_to_series(data):
            if isinstance(data, dict):
                if all('-' in key for key in data.keys()):
                    index = pd.to_datetime(list(data.keys()))
                    values = [val if not isinstance(val, dict) else json_to_series(val) for val in data.values()]
                else:
                    index = list(data.keys())
                    values = [val if not isinstance(val, dict) else json_to_series(val) for val in data.values()]
                return pd.Series(values, index=index)
            else:
                return pd.Series(data)

        with open(path_file, "r") as file:
            data = json.load(file)

            data['strategy'] = StrategyFunction.from_dict(data['strategy'])
            data['strat_ret'] = json_to_series(data['strat_ret'])
            data['strat_weights'] = json_to_dataframe(data['strat_weights'])
            data['asset_ret'] = json_to_series(data['asset_ret'])
            data['hedge_ret'] = json_to_series(data['hedge_ret'])
            data['asset_turnover'] = json_to_series(data['asset_turnover'])
            data['ccy_turnover'] = json_to_series(data['ccy_turnover'])
            data['rebalancing_tc'] = json_to_series(data['rebalancing_tc'])
            data['hedge_tc'] = json_to_series(data['hedge_tc'])
            data['start_date'] = pd.to_datetime(data['start_date'])
            data['end_date'] = pd.to_datetime(data['end_date'])

        return cls(**data)

    @staticmethod
    def backtests_to_xlsx(backtests: list['BackTestRec'], file_name: str):
        pd.DataFrame(
            [
                {
                    "id": bt.get_id(),
                    "start_date": bt.get_start_date(),
                    "end_date": bt.get_end_date(),
                    "input_data_days_cnt": bt.get_input_data_days_cnt(),
                    "main_ccy": bt.get_main_ccy(),
                    "strategy": bt.get_strategy_name(),
                    "rebalancing_frequency": bt.get_rebalancing_freq(),
                    "trading_costs_bps": bt.get_trading_costs(),
                    "ccy_exchange_costs_bps": bt.get_ccy_exchange_costs(),
                    "gap_days": bt.get_gap_days(),
                    "contrained": bt.is_constrained(),
                    "rolling": bt.is_rolling(),
                    "ccy_hedge_ratio": bt.get_ccy_hedge_ratio(),
                    "avg_excess_return": bt.get_avg_excess_returns(),
                    "avg_vol": bt.get_avg_volatility(),
                    "avg_semi_vol": bt.get_avg_semi_volatility(),
                    "sharpe_ratio": bt.get_sharpe_ratio(),
                    "max_drawdown": bt.get_max_drawdown(),
                    "skewness": bt.get_skewness(),
                    "kurtosis": bt.get_kurtosis()
                }
                for bt in backtests
            ]
        ).to_excel(os.path.join(Path.BACKTEST_PATH, file_name), index=False)

