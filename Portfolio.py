import pandas as pd

from Utils import convert_return_period


class InvesmentUniverse:

    def __init__(self, main_ccy: str):
        self.__main_ccy = main_ccy
        self.__assets = []
        self.__ccys = []
        self.__rf_daily = None
        self.__rf_yearly = None
        self.__index_start = None
        self.__index_end = None

    def add_asset(self, asset: 'Asset'):

        if asset.get_name() in self.get_all_assets_names():
            raise Warning("asset with same name already part of the investment universe")

        self.__adjust_base_indices(asset=asset)

        if asset.get_class() != "ccy":
            self.__assets.append(asset)
        else:
            self.__ccys.append(asset)

    def remove_asset(self, asset_name: str):
        if asset_name not in self.get_all_assets_names():
            raise Warning(f"asset with name: {asset_name} is already part of investment universe")
        del self.__assets[self.get_investable_asset_names().index(asset_name)]

    def set_risk_free_rate(self, rf: 'Asset', period: int):
        self.__adjust_base_indices(asset=rf)
        ret = rf.get_return_serie()
        self.__rf_daily = convert_return_period(ret=ret, period_d_current=period, period_d_new=1)
        self.__rf_yearly = convert_return_period(ret=ret, period_d_current=period, period_d_new=365)
        self.__rf_yearly = self.__rf_yearly.reindex(self.__create_all_asset_df().index, method="nearest")
        self.__rf_daily = self.__rf_daily.reindex(self.__create_all_asset_df().index, method="nearest")

    def __adjust_base_indices(self, asset: 'Asset'):
        min_asset_index = asset.get_return_serie().index[0]
        max_asset_index = asset.get_return_serie().index[-1]

        if self.__index_start is None:
            self.__index_start = min_asset_index
            self.__index_end = max_asset_index
        else:
            if min_asset_index > self.__index_start:
                self.__index_start = min_asset_index
            if max_asset_index < self.__index_end:
                self.__index_end = max_asset_index

    def get_main_ccy(self) -> str:
        return self.__main_ccy

    def get_asset_from_name(self, name: str) -> 'Asset':
        for asset in self.__assets:
            if asset.get_name() == name:
                return asset
        raise Warning(f"asset: {name} is not part of the investment universe")

    def get_daily_rf(self) -> pd.Series:
        return self.__rf_daily[self.__index_start:self.__index_end]

    def get_yearly_rf(self) -> pd.Series:
        return self.__rf_yearly[self.__index_start:self.__index_end]

    def get_all_assets_names(self) -> list[str]:
        return [asset.get_name() for asset in self.__assets]

    def __process_df_nan(self, df: pd.DataFrame, fill_nan: float | None = None, drop_nan: bool = False) -> pd.DataFrame:
        if all([fill_nan is not None, drop_nan is not None]):
            raise Warning(f"fill_nan = {fill_nan} & drop_nan = {drop_nan} provided, please provide only either of")
        if fill_nan is not None:
            return df.fillna(fill_nan, inplace=True)[self.__index_start:self.__index_end]
        elif drop_nan:
            return df.dropna()[self.__index_start:self.__index_end]
        else:
            return df[self.__index_start:self.__index_end]

    def __create_all_asset_df(self):
        return pd.concat(
            {asset.get_name(): asset.get_return_serie() for asset in self.__assets},
            axis=1
        )

    def get_subset_asset_universe(
            self,
            subset_asset_names: list[str],
            fill_nan: float | None = None,
            drop_nan: bool = False
    ) -> pd.DataFrame:
        if all(sa in self.get_all_assets_names() for sa in subset_asset_names):
            df = self.__create_all_asset_df()[subset_asset_names]
            return self.__process_df_nan(df=df, fill_nan=fill_nan, drop_nan=drop_nan)
        else:
            raise Warning(
                "some asset is not part of the investment universe:\n"
                f"  -> invesment universe: {self.get_all_assets_names()}",
                f"  -> subset assets: {subset_asset_names}"
            )

    def get_investable_asset_universe(self) -> list['Asset']:
        return [asset for asset in self.__assets if not asset.is_ccy_hedge()]

    def get_investable_return_df(self, fill_nan: float | None = None, drop_nan: bool = False) -> pd.DataFrame:
        df = self.__create_all_asset_df()[[asset.get_name() for asset in self.__assets if not asset.is_ccy_hedge()]]
        return self.__process_df_nan(df=df, fill_nan=fill_nan, drop_nan=drop_nan)

    def get_investable_asset_names(self) -> list[str]:
        return [asset.get_name() for asset in self.__assets if not asset.is_ccy_hedge()]

    def get_ccy_universe(self):
        return self.__ccys

    def get_ccy_return_df(self, fill_nan: float | None = None, drop_nan: bool = False) -> pd.DataFrame:
        df = self.__create_all_asset_df()[[ccy.get_name() for ccy in self.__ccys]]
        return self.__process_df_nan(df=df, fill_nan=fill_nan, drop_nan=drop_nan)

    def get_ccy_names(self) -> list[str]:
        return list(
            set(
                [ccy.get_name()[3:] for ccy in self.__ccys] +
                [ccy.get_name()[:3] for ccy in self.__ccys] +
                [self.__main_ccy]
            )
        )

    def get_ccy_hedge_universe(self):
        return [asset for asset in self.__assets if asset.is_ccy_hedge()]

    def get_ccy_hedge_return_df(self, fill_nan: float | None = None, drop_nan: bool = False) -> pd.DataFrame:
        df = self.__create_all_asset_df()[[asset.get_name() for asset in self.__assets if asset.is_ccy_hedge()]]
        return self.__process_df_nan(df=df, fill_nan=fill_nan, drop_nan=drop_nan)

    def get_ccy_hedge_names(self) -> list[str]:
        return [asset.get_name() for asset in self.__assets if asset.is_ccy_hedge()]

    @staticmethod
    def __to_pd_timestamp(index: str | pd.Timestamp) -> pd.Timestamp:
        if isinstance(index, pd.Timestamp):
            return index
        else:
            try:
                return pd.Timestamp(index)
            except ValueError:
                raise Warning(f"index: {index} does not have valid format: srt('YYYY-mm-dd')")

    def set_start_index(self, index: str | pd.Timestamp):
        pd_index = self.__to_pd_timestamp(index=index)
        if pd_index > self.__index_start:
            self.__index_start = pd_index

    def set_end_index(self, index: str | pd.Timestamp):
        pd_index = self.__to_pd_timestamp(index=index)
        if pd_index < self.__index_end:
            self.__index_end = pd_index


class Asset:

    def __init__(
            self,
            name: str,
            asset_class: str,
            return_serie: pd.Series,
            ccy: str = None,
            hedge: bool = False,
            annual_expense_ratio_bps: int = 0,
            min_allocation: float | None = None,
            max_allocation: float | None = None
    ):
        self.__name = name
        if asset_class.lower() not in [
            "equity", "bond", "ccy", "future", "forward", "option", "crypto", "interest_rate"
        ]:
            raise Exception(f"Asset type not supported: given type = {asset_class}")
        self.__asset_class = asset_class.lower()
        self.__ccy = ccy if asset_class != "ccy" else None
        self.__hedge = hedge
        self.__annual_expense_ratio_bps = annual_expense_ratio_bps
        self.__return_serie = return_serie - ((self.__annual_expense_ratio_bps / 10_000) / 252)
        self.__return_serie = self.__return_serie.rename(self.__name)
        self.__min_allocation = min_allocation
        self.__max_allocation = max_allocation

    def get_name(self) -> str:
        return self.__name

    def get_class(self) -> str:
        return self.__asset_class

    def get_ccy(self) -> str:
        return self.__ccy

    def get_return_serie(self) -> pd.Series:
        return self.__return_serie

    def is_ccy_hedge(self) -> bool:
        return self.__hedge

    def get_annual_expense_ratio_bps(self) -> int:
        return self.__annual_expense_ratio_bps

    def get_asset_min_allocation(self) -> float | None:
        return self.__max_allocation

    def get_asset_max_allocation(self) -> float | None:
        return self.__max_allocation
