import pandas as pd


class InvesmentUniverse:

    def __init__(self):
        self.__assets = []
        self.__ccys = []
        self.__rf_daily = None
        self.__rf_yearly = None
        self.__index_start = None
        self.__index_end = None

    def add_asset(self, asset: 'Asset'):

        if asset.get_name() in self.get_all_assets_names():
            raise Warning("asset with same name already part of the investment universe")

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

        if asset.get_class() != "ccy":
            self.__assets.append(asset)
        else:
            self.__ccys.append(asset)

    def remove_asset(self, asset_name: str):
        if asset_name not in self.get_all_assets_names():
            raise Warning(f"asset with name: {asset_name} is already part of investment universe")
        del self.__assets[self.get_investable_asset_names().index(asset_name)]

    def set_risk_free_rate(self, rf: 'Asset'):
        if rf.get_return_serie().abs().mean() >= 1e-3:
            self.__rf_yearly = rf.get_return_serie()
            self.__rf_daily = self.__convert_return_period(
                ret=rf.get_return_serie(),
                period_d_current=365,
                period_d_new=1
            )
        else:
            self.__rf_yearly = self.__convert_return_period(
                ret=rf.get_return_serie(),
                period_d_current=1,
                period_d_new=365
            )
            self.__rf_daily = rf.get_return_serie()

    def get_daily_rf(self) -> pd.Series:
        return self.__rf_daily[self.__index_start:self.__index_end]

    def get_yearly_rf(self) -> pd.Series:
        return self.__rf_yearly[self.__index_start:self.__index_end]

    def get_all_assets_names(self) -> list[str]:
        return [asset.get_name() for asset in self.__assets]

    def __process_df_nan(self, df: pd.DataFrame, fill_nan: float | None = None, drop_nan: bool = False) -> pd.DataFrame:
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
        return [ccy.get_name() for ccy in self.__ccys]

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

    @staticmethod
    def __convert_return_period(ret: pd.Series, period_d_current: int, period_d_new: int) -> pd.Series:
        return ((1 + ret) ** (period_d_new / period_d_current)) - 1


class Asset:

    def __init__(self, name: str, asset_class: str, return_serie: pd.Series, ccy: str = None, hedge: bool = False):
        self.__name = name
        if asset_class.lower() not in ["equity", "bond", "ccy", "future", "forward", "option", "crypto"]:
            raise Exception(f"Asset type not supported: given type = {asset_class}")
        self.__asset_class = asset_class.lower()
        self.__ccy = ccy if asset_class != "ccy" else None
        self.__return_serie = return_serie
        self.__hedge = hedge

    def get_name(self) -> str:
        return self.__name

    def get_class(self) -> str:
        return self.__asset_class

    def get_ccy(self) -> str:
        return self.__ccy

    def get_return_serie(self) -> pd.Series:
        return self.__return_serie

    def is_ccy_hedge(self):
        return self.__hedge
