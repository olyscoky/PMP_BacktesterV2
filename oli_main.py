import os
import pandas as pd
import sys
import numpy as np

from BackTester import BackTester
from DataAggregator import DataAggregator
from Strategy import Strategy
from BackTestRec import BackTestRec
from Paths import Path
from Portfolio import InvesmentUniverse, Asset
from Return_Analysis import DataAnalyser

from Utils import concat_df_series_with_nearest_index


if __name__ == "__main__":

    # LOADING DATA #####################################################################################################
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
    bmr1_data = pd.read_csv(
        os.path.join(Path.RAW_DATA_PATH, "BMR1_Curncy_hist_sd2020-09-01_ed2025-03-01.csv"),
        parse_dates=["date"],
        index_col="date"
    )
    bito_data = pd.read_csv(
        os.path.join(Path.RAW_DATA_PATH, "BITO_US_Equity_hist_sd2020-09-01_ed2025-03-01.csv"),
        parse_dates=["date"],
        index_col="date"
    )
    sofr_data = pd.read_csv(
        os.path.join(Path.RAW_DATA_PATH, "SOFRRATE_Index_hist_sd2020-09-01_ed2025-03-01.csv"),
        parse_dates=["date"],
        index_col="date"
    )

    mstr_btc_investment = pd.read_csv(
        os.path.join(Path.DATA_PATH, "btc-holdings-over-time.csv"),
        parse_dates=["DateTime"],
        index_col="DateTime"
    )
    mstr_btc_investment.index = mstr_btc_investment.index.date

    # __________________________________________________________________________________________________________________

    # INITIALIAZING ASSETS #############################################################################################
    mstr_asset = asset=Asset(
        name="MSTR",
        asset_class="equity",
        return_serie=mstr_data["PX_LAST"].pct_change().iloc[1:],
        ccy="USD",
        hedge=False
    )
    xbtusd_asset = Asset(
        name="XBTUSD",
        asset_class="crypto",
        return_serie=xbt_data["PX_LAST"].pct_change().iloc[1:],
        ccy="USD",
        hedge=False
    )
    bmr1_asset = Asset(
        name="BMR1",
        asset_class="future",
        return_serie=bmr1_data["PX_LAST"].pct_change().iloc[1:],
        ccy="USD",
        hedge=False
    )
    bito_asset = Asset(
        name="BITO",
        asset_class="equity",
        return_serie=bito_data["PX_LAST"].pct_change().iloc[1:],
        ccy="USD",
        hedge=False,
        annual_expense_ratio_bps=95
    )
    sofr_rf = Asset(
        name="SOFR",
        asset_class="interest_rate",
        return_serie=(sofr_data["PX_LAST"] / 100),
        ccy=None,
        hedge=False,
    )

    # __________________________________________________________________________________________________________________

    # CREATING COMPLEMENTARY DATA ######################################################################################
    xbt_spot = xbt_data["PX_LAST"].rename("XBT_SPOT")
    bmr1_spot = bmr1_data["PX_LAST"].rename("BMR1_SPOT")
    complementary_data = concat_df_series_with_nearest_index(
        df_lst=[
            xbt_spot,
            bmr1_spot
        ],
    )
    # __________________________________________________________________________________________________________________

    # INITIALIZING INVESTMENT UNIVERSE #################################################################################
    invest_univ = InvesmentUniverse(main_ccy="USD")
    invest_univ.add_asset(asset=mstr_asset)
    invest_univ.add_asset(asset=xbtusd_asset)
    invest_univ.add_asset(asset=bmr1_asset)
    invest_univ.add_asset(asset=bito_asset)
    invest_univ.set_risk_free_rate(rf=sofr_rf, period=365)
    # __________________________________________________________________________________________________________________

    ret_an = DataAnalyser(investment_universe=invest_univ)
    # PERFORMING ANALYSIS ##############################################################################################
    # ret_an.perform_OLS_regression(
    #     x_asset_names=["MSTR"],
    #     y_asset_name="XBTUSD",
    #     show_regression_plot=False,
    #     save_regression_plot=False,
    # )
    # ret_an.perform_OLS_regression(
    #     x_direct=concat_df_series_with_nearest_index(df_lst=[
    #         xbtusd_asset.get_return_serie(),
    #         mstr_btc_investment
    #     ]),
    #     y_asset_name="MSTR",
    #     show_regression_plot=False
    # )
    #
    # ret_an.single_plot(
    #     x_series=mstr_btc_investment.index,
    #     y_series=mstr_btc_investment,
    #     line_plot=True,
    #     save_plot=False
    # )
    #
    # ret_an.analyse_return_distribution(
    #     asset_names=["MSTR", "XBTUSD"],
    #     show_distribution_plot=True
    # )
    # __________________________________________________________________________________________________________________
    # ret_an.single_plot(
    #     x_series=bmr1_asset.get_return_serie().index,
    #     y_series=np.expm1(np.log1p(bito_asset.get_return_serie()).cumsum()),
    #     line_plot=True,
    #     save_plot=False
    # )
    # sys.exit()

    if True:
        bt = BackTester(
            investment_universe=invest_univ,
            data_frequency="D",
            trading_costs_bps=5,
            ccy_exchg_costs_bps=10,
            trading_days_count=252,
            drop_nan=True
        )

        backtest = bt.backtest(
            strategy=Strategy.xbt_carry_trade(),
            complementary_data=complementary_data,
            shorting_allowed=True,
            constrained=False,
            bt_end_date="2024-10-01"
        )
        backtest.summarize()
