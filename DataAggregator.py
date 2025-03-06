import os
import holidays
import pandas as pd
import json
from datetime import date
from typing import Any
from datetime import datetime
import warnings

from Paths import Path


warnings.filterwarnings("ignore", category=FutureWarning)


class DataAggregator:

    @staticmethod
    def get_min_start_date(
            path: str,
            file_lst: list,
            save_json: bool = False,
            json_file_name: str = "min_max_dates.csv"
    ) -> dict:
        min_max_dates = dict()
        for file in file_lst:
            if file == ".DS_Store":
                continue
            df = pd.read_csv(os.path.join(path, file), index_col=0, parse_dates=True, date_format="%d-%m-%Y")
            min_max_dates[file] = {"min_date": df.index.min(), "max_date": df.index.max()}
        if save_json:
            DataAggregator.save_dict_as_json(path=path, dic=min_max_dates, file_output=json_file_name)

        min_dates = list()
        max_dates = list()
        for mm_dates in min_max_dates.values():
            min_dates.append(mm_dates["min_date"])
            max_dates.append(mm_dates["max_date"])

        print("mininum date across all files", sorted(min_dates)[-1])
        print("maximum date accross all files", sorted(max_dates)[0])

        return min_max_dates

    @staticmethod
    def save_dict_as_json(path: str, dic: dict, file_output: str = "min_max_dates.csv"):
        with open(os.path.join(path, file_output), "w") as json_file:
            json.dump(dic, json_file, indent=4)

    @staticmethod
    def get_holidays(
            countries: list[str],
            start_date: str | None = None,
            end_date: str | None = None
    ) -> list[date]:
        if (start_date is not None) and (end_date is not None):
            sd = datetime.strptime(start_date, "%Y-%m-%d").date()
            ed = datetime.strptime(end_date, "%Y-%m-%d").date()
        else:
            sd = datetime.strptime("2000-01-01", "%Y-%m-%d").date()
            ed = datetime.strptime("2001-01-01", "%Y-%m-%d").date()
        h_days = list()
        for country in countries:
            h_days_country = holidays.country_holidays(country=country, years=range(sd.year, ed.year + 1))
            for hd in h_days_country.keys():
                if (hd not in h_days) and (hd.weekday() < 5) and (hd >= sd) and (hd <= ed):
                    h_days.append(hd)
        return sorted(h_days)

    @staticmethod
    def add_return(path: str, file: str, fee_pa_bps: int = 0, holidays_cnt: int = 0, saving_path: str | None = None):
        df = pd.read_csv(
            os.path.join(path, file),
            index_col=0,
            parse_dates=True,
            date_format="%Y-%m-%d"
        )
        daily_etf_fee = (fee_pa_bps / 10_000) / (252 - holidays_cnt)
        returns = list()
        for i, (index, row) in enumerate(df.iterrows()):
            if i != 0 and i != len(df) - 1:
                returns.append(
                    (row["PX_LAST"] / df.iloc[i - 1, df.columns.get_loc("PX_LAST")]) - 1 - daily_etf_fee
                )
            else:
                returns.append(None)

        df["RETURN"] = returns
        df.to_csv(os.path.join(saving_path if not None else path, file))

    @staticmethod
    def add_rfs(path: str, file: str, factor_to_percent: int, saving_path: str | None = None):
        df = pd.read_csv(
            os.path.join(path, file),
            index_col=0,
            parse_dates=True,
            date_format="%Y-%m-%d"
        )
        df["YEARLY_RF"] = df["PX_LAST"] / factor_to_percent
        df["DAILY_RF"] = (
            1 + df["YEARLY_RF"]) ** (1 /
                (252 - len(DataAggregator.get_holidays(
                        countries=["CH", "US"],
                        start_date="2000-01-01",
                        end_date="2001-01-01"
                    )
                ))
        ) - 1
        df.to_csv(os.path.join(saving_path if not None else path, file))

    @staticmethod
    def rename_cols_to_concat(
            path: str,
            file_name: str,
            name_extention: str,
            cols_to_keep: list[str],
            saving_path: str | None = None
    ):
        df = pd.read_csv(os.path.join(path, file_name), index_col=0, parse_dates=True, date_format="%Y-%m-%d")
        df = df[cols_to_keep]
        newcols_dic = dict()
        for col in cols_to_keep:
            newcols_dic[col] = "_".join([col, name_extention])
        df = df.rename(columns=newcols_dic)
        df.to_csv(os.path.join(saving_path if saving_path is not None else path, file_name))

    @staticmethod
    def concat_data_single_file(
            input_directory: str,
            new_file_name: str,
            start_date: str,
            end_date: str,
            hd: list[date] | None = None,
            fill_nan_columns: dict[str, dict[str, Any]] | None = None,
            saving_directory: str | None = None,
            resampling: str | None = None
    ):
        dataframes = list()
        for file in [f for f in os.listdir(input_directory) if f.split(".")[1] == "csv"]:
            df = pd.read_csv(
                os.path.join(input_directory, file),
                index_col=0,
                parse_dates=True,
                date_format="%Y-%m-%d"
            )
            dataframes.append(df)

        merged_df = pd.concat(dataframes, axis=1, join="outer")
        merged_df = merged_df[merged_df.index.dayofweek < 5]
        if hd is not None:
            merged_df = merged_df[~merged_df.index.isin([pd.to_datetime(d) for d in hd])]

        if fill_nan_columns is not None:
            for col, filling_dic in fill_nan_columns.items():
                if "backfill" in filling_dic.keys():
                    merged_df[col] = merged_df[col].ffill()

                if "costfill" in filling_dic.keys():
                    merged_df[col] = merged_df[col].fillna(-((filling_dic["costfill"] / 10_000) / 252))

                if "rffill" in filling_dic.keys():
                    merged_df[col] = merged_df[col] + merged_df[filling_dic["rffill"]]

        merged_df = merged_df[
            ((merged_df.index >= pd.to_datetime(start_date)) & (merged_df.index <= pd.to_datetime(end_date)))
        ]

        if resampling is not None:
            merged_df = merged_df.resample(resampling).last()

        merged_df.to_csv(os.path.join(
            saving_directory if saving_directory is not None else input_directory, new_file_name
        ))


if __name__ == "__main__":

    min_start_date = "1999-01-01"
    max_end_date = "2024-11-29"

    hd = DataAggregator.get_holidays(
        countries=["US"],
        start_date=min_start_date,
        end_date=max_end_date
    )

    # SINGLE FILE PROCEEDINGS //////////////////////////////////////////////////////////////////////////////////////////

    file = "VIX_Index_hist_sd1965-01-01_ed2024-12-01.csv"
    # #
    # DataAggregator.add_return(
    #     file=file,
    #     path=Path.RAW_DATA_PATH,
    #     saving_path=Path.SEMIPREPED_DATA,
    #     holidays_cnt=len(hd),
    #     fee_pa_bps=10
    # )
    #
    # DataAggregator.add_rfs(
    #     path=Path.RAW_DATA_PATH,
    #     saving_path=Path.SEMIPREPED_DATA,
    #     file=file,
    #     factor_to_percent=100
    # )

    # DataAggregator.rename_cols_to_concat(
    #     path=Path.RAW_DATA_PATH,
    #     saving_path=Path.SEMIPREPED_DATA,
    #     file_name=file,
    #     name_extention="VIX",
    #     cols_to_keep=["PX_LAST"]
    # )

    # GET MIN START DATE ///////////////////////////////////////////////////////////////////////////////////////////////
    # d = DataAggregator.get_min_start_date(
    #     path=Path.SEMIPREPED_DATA,
    #     file_lst=[file for file in os.listdir(Path.SEMIPREPED_DATA)]
    # )

    # CONCAT FILES /////////////////////////////////////////////////////////////////////////////////////////////////////

    DataAggregator.concat_data_single_file(
        input_directory=Path.SEMIPREPED_DATA,
        saving_directory=Path.DATA_PATH,
        new_file_name="Dez16.csv",
        start_date=min_start_date,
        end_date=max_end_date,
        hd=hd,
        fill_nan_columns={
            "PX_LAST_XAUUSD": {"backfill": True},
            "RETURN_XAUUSD": {"costfill": 10},
            "PX_LAST_M2WO": {"backfill": True},
            "RETURN_M2WO": {"costfill": 10},
            "PX_LAST_EUR001M": {"backfill": True},
            "DAILY_RF_EUR001M": {"backfill": True},
            "YEARLY_RF_EUR001M": {"backfill": True},
            "PX_LAST_EURUSD": {"backfill": True},
            "RETURN_EURUSD": {"costfill": 0},
            "PX_LAST_VIX": {"backfill": True},
            "M2M_INLFATION": {"costfill": 0},
        },
        resampling=None
    )



