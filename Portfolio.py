import pandas as pd


class InvesmentUniverse:

    def __init__(self):
        self.__assets = []
        self.__ccys = []
        self.__rf_daily = None
        self.__rf_yearly = None
        self.__series_index = None

    def add_asset(self, asset: 'Asset'):

        def check_index_before_appending(lst_tbd: list):
            if not self.__assets and not self.__ccys:
                self.__assets.append(asset)
                self.__series_index = asset.get_return_serie().index
            else:
                new_asset_index = asset.get_return_serie().index
                if self.__series_index == new_asset_index:
                    lst_tbd.append(asset)
                else:
                    print(f"index mismatch: main index: {self.__series_index[0]} - {self.__series_index[-1]} | "
                          f"new asset index: {new_asset_index[0]} - {new_asset_index[-1]}")
                    if input("add asset regardless ? (Y/n): ") == "Y":
                        lst_tbd.append(asset)

        if asset.get_name() in self.get_all_assets_names():
            raise Warning("asset with same name already part of the investment universe")

        if asset.get_class() != "ccy":
            check_index_before_appending(lst_tbd=self.__assets)
        else:
            check_index_before_appending(lst_tbd=self.__ccys)

    def remove_asset(self, asset_name: str):
        if asset_name not in self.get_all_assets_names():
            raise Warning(f"asset with name: {asset_name} is already part of investment universe")
        del self.__assets[self.get_investable_asset_names().index(asset_name)]

    def set_risk_free_rate(self, rf: 'Asset'):
        if rf.get_return_serie().index == self.__series_index:
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
        return self.__rf_daily

    def get_yearly_rf(self) -> pd.Series:
        return self.__rf_yearly

    def get_all_assets_names(self) -> list[str]:
        return [asset.get_name() for asset in self.__assets]

    def get_subset_asset_universe(self, subset_asset_names: list[str]) -> pd.DataFrame:
        if all(sa in self.get_all_assets_names() for sa in subset_asset_names):
            return pd.DataFrame(
                data=[asset.get_return_serie() for asset in self.__assets if asset in subset_asset_names],
                columns=[asset_name for asset_name in subset_asset_names],
                index=self.__series_index
            )
        else:
            raise Warning(
                "some asset is not part of the investment universe:\n"
                f"  -> invesment universe: {self.get_all_assets_names()}",
                f"  -> subset assets: {subset_asset_names}"
            )

    def get_investable_asset_universe(self) -> list['Asset']:
        return [asset for asset in self.__assets if not asset.is_ccy_hedge()]

    def get_investable_return_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            data=[asset.get_return_serie() for asset in self.__assets if not asset.is_ccy_hedge()],
            columns=[asset.get_name() for asset in self.__assets if not asset.is_ccy_hedge()],
            index=self.__series_index
        )

    def get_investable_asset_names(self) -> list[str]:
        return [asset.get_name() for asset in self.__assets if not asset.is_ccy_hedge()]

    def get_ccy_universe(self):
        return self.__ccys

    def get_ccy_return_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            data=[ccy.get_return_serie() for ccy in self.__ccys],
            columns=[ccy.get_name() for ccy in self.__ccys],
            index=self.__series_index
        )

    def get_ccy_names(self) -> list[str]:
        return [ccy.get_name() for ccy in self.__ccys]

    def get_ccy_hedge_universe(self):
        return [asset for asset in self.__assets if asset.is_ccy_hedge()]

    def get_ccy_hedge_return_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            data=[asset.get_return_serie() for asset in self.__assets if asset.is_ccy_hedge()],
            columns=[asset.get_return_serie() for asset in self.__assets if asset.is_ccy_hedge()],
            index=self.__series_index
        )

    def get_ccy_hedge_names(self) -> list[str]:
        return [asset.get_name() for asset in self.__assets if asset.is_ccy_hedge()]

    @staticmethod
    def __convert_return_period(ret: pd.Series, period_d_current: int, period_d_new: int) -> pd.Series:
        return ((1 + ret) ** (period_d_new / period_d_current)) - 1


class Asset:

    def __init__(self, name: str, asset_class: str, return_serie: pd.Series, ccy: str = None, hedge: bool = False):
        self.__name = name
        if asset_class not in ["stock", "bond", "ccy", "future", "forward", "option"]:
            raise Exception(f"Asset type not supported: given type = {asset_class}")
        self.__asset_class = asset_class
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
