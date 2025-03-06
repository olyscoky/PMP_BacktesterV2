import os
import pandas as pd
import sys

from BackTester import BackTester
from DataAggregator import DataAggregator
from Strategy import Strategy
from BackTestRec import BackTestRec
from Paths import Path
from Portfolio import InvesmentUniverse, Asset
from Return_Analysis import ReturnAnalyser


if __name__ == "__main__":

    # INITIALIZING INVESTMENT UNIVERSE
    # data = pd.read_excel(os.path.join(Path.DATA_PATH, "clean_data.xlsx"), index_col="Dates", parse_dates=["Dates"])

    mstr_data = pd.read_csv(
        os.path.join(Path.RAW_DATA_PATH, "MSTR_US_Equity_hist_sd2020-09-01_ed2025-03-01.csv"),
        parse_dates=["date"],
        index_col="date"
    )
    xbt_data = pd.read_csv(
        os.path.join(Path.RAW_DATA_PATH, "XBTUSD_Curncy_hist_sd2020-09-01_ed2025-03-01.csv"),
        parse_dates=["date"],
        index_col="date"
    )

    invest_univ = InvesmentUniverse()
    invest_univ.add_asset(
        asset=Asset(
            name="MSTR",
            asset_class="equity",
            return_serie=mstr_data["PX_LAST"].pct_change().iloc[1:],
            ccy="USD",
            hedge=False
        )
    )
    invest_univ.add_asset(
        asset=Asset(
            name="XBTUSD",
            asset_class="crypto",
            return_serie=xbt_data["PX_LAST"].pct_change().iloc[1:],
            ccy="USD",
            hedge=False
        )
    )

    ret_an = ReturnAnalyser(investment_universe=invest_univ)
    ret_an.perform_OLS_regression(
        x_asset_names=["MSTR"],
        y_asset_name="XBTUSD",
        show_regression_plot=True,
        save_regression_plot=False,
    )
    ret_an.analyse_distribution(
        asset_names=["MSTR", "XBTUSD"],
        show_distribution_plot=True
    )


