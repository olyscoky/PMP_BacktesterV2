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

from Utils import concat_df_series_with_nearest_index, convert_return_period

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


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
    ibit_data = pd.read_csv(
        os.path.join(Path.RAW_DATA_PATH, "IBIT_US_Equity_hist_sd2020-09-01_ed2025-03-01.csv"),
        parse_dates=["date"],
        index_col="date"
    )

    mstr_btc_investment = pd.read_csv(
        os.path.join(Path.DATA_PATH, "btc-holdings-over-time.csv"),
        parse_dates=["DateTime"],
        index_col="DateTime"
    )
    mstr_btc_investment.index = mstr_btc_investment.index.normalize()
    mstr_mkt_cap = pd.read_csv(
        os.path.join(Path.DATA_PATH, "MSTR_US_Equity++CUR_MKT_CAP++hist_sd2020-09-01_ed2025-03-01.csv"),
        parse_dates=["date"],
        index_col=["date"]
    )
    brrny_data = pd.read_csv(
        os.path.join(Path.RAW_DATA_PATH, "BRRNY_Index++PX_LAST++hist_sd2020-09-01_ed2025-03-01.csv"),
        parse_dates=["date"],
        index_col="date"
    )

    # __________________________________________________________________________________________________________________

    # INITIALIAZING ASSETS #############################################################################################
    mstr_asset = Asset(
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
    rf_ret = convert_return_period(sofr_data["PX_LAST"] / 100, period_d_current=365, period_d_new=1)
    bmr1_fut_ret = bmr1_data["PX_LAST"].pct_change().iloc[1:]
    rf_ret = rf_ret.reindex(bmr1_fut_ret.index, method="nearest")
    bmr1_asset = Asset(
        name="S_BMR1",
        asset_class="future",
        return_serie=bmr1_fut_ret + rf_ret,
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
    ibit_asset = Asset(
        name="IBIT",
        asset_class="equity",
        return_serie=ibit_data["PX_LAST"].pct_change().iloc[1:],
        ccy="USD",
        hedge=False,
        annual_expense_ratio_bps=25
    )
    sofr_rf = Asset(
        name="SOFR",
        asset_class="interest_rate",
        return_serie=(sofr_data["PX_LAST"] / 100),
        ccy=None,
        hedge=False,
    )
    brrny_asset = Asset(
        name="BRRNY",
        asset_class="crypto",
        return_serie=brrny_data["PX_LAST"].pct_change().iloc[1:],
        hedge=False,
    )

    # __________________________________________________________________________________________________________________

    # CREATING COMPLEMENTARY DATA ######################################################################################
    xbt_spot = xbt_data["PX_LAST"].rename("XBT_SPOT")
    bmr1_spot = bmr1_data["PX_LAST"].rename("BMR1_SPOT")
    bmr1_spot = bmr1_spot * 0.882431481
    complementary_data = concat_df_series_with_nearest_index(
        df_lst=[
            xbt_spot,
            bmr1_spot
        ],
    )
    brrny_spot = brrny_data["PX_LAST"].rename("BRRNY_SPOT")
    # __________________________________________________________________________________________________________________

    # INITIALIZING INVESTMENT UNIVERSE #################################################################################
    invest_univ = InvesmentUniverse(main_ccy="USD")
    invest_univ.add_asset(asset=mstr_asset)
    invest_univ.add_asset(asset=brrny_asset)
    # invest_univ.add_asset(asset=xbtusd_asset)
    # invest_univ.add_asset(asset=bmr1_asset)
    # invest_univ.add_asset(asset=ibit_asset)
    # invest_univ.add_asset(asset=bito_asset)
    invest_univ.set_risk_free_rate(rf=sofr_rf, period=365)

    # __________________________________________________________________________________________________________________

    ret_an = DataAnalyser(investment_universe=invest_univ)
    # PERFORMING ANALYSIS ##############################################################################################
    x_direct = concat_df_series_with_nearest_index(df_lst=[
        mstr_btc_investment,
        mstr_mkt_cap,
        brrny_spot,
        brrny_asset.get_return_serie(),
        mstr_asset.get_return_serie()
    ])
    x_direct["relative_BTC_to_MKT"] = (x_direct["BTC holdings"] * x_direct["BRRNY_SPOT"]) / \
                                      (x_direct["CUR_MKT_CAP"] * 1_000_000)
    x_direct["1M_trailing_corr"] = x_direct["MSTR"].rolling(window=30).corr(x_direct["BRRNY"])
    x_direct["BTC_lag_1d"] = x_direct["BRRNY"].abs().shift(1)
    x_direct["MSTR_lag_1d"] = x_direct["MSTR"].shift(1)
    x_direct["MSTR_1M_mean_ret"] = x_direct["MSTR"].rolling(window=30).mean()
    x_direct["BTC_1M_mean_ret"] = x_direct["BRRNY"].rolling(window=30).mean()

    x_direct = x_direct.dropna()

    # x_direct = x_direct[pd.Timestamp("2024-03-01"):pd.Timestamp("2025-03-01")]

    # vol_mstr = x_direct["MSTR"].std()
    # vol_btc = x_direct["BRNNY"].std()
    # w_mstr = 1
    # w_btc = -1 / 1.5
    # correlation = 0.45
    #
    # pf_vol = vol_mstr**2 * w_mstr**2 + vol_btc**2 * w_btc**2 + \
    #          2 * vol_btc * vol_mstr * correlation * w_btc * w_mstr
    # print(np.sqrt(pf_vol) * np.sqrt(252))

    print(0.08 * np.sqrt(252))

    ret_an.single_plot(
        y_series=mstr_asset.get_return_serie().rolling(window=30).std().dropna(),
        x_series=x_direct[["relative_BTC_to_MKT"]],
        scatter_plot=True
    )
    sys.exit()

    ret_an.perform_OLS_regression(
         x_direct=x_direct[["relative_BTC_to_MKT"]],
         y_asset_name="MSTR",
         show_regression_plot=True,
         save_regression_plot=True,
         drop_nan=True
    )

    # ret_an.single_plot(
    #     y_series=x_direct[["1M_trailing_corr"]],
    #     line_plot=True,
    #     save_plot=False
    # )
    #
    ret_an.perform_OLS_regression(
         x_direct=x_direct[["relative_BTC_to_MKT"]],
         y_direct=mstr_asset.get_return_serie().rolling(window=30).std().dropna(),
         constant_term=False,
         show_regression_plot=True,
         save_regression_plot=False,
         drop_nan=True
    )
    #
    #
    #
    # ret_an.single_plot(
    #     y_series=x_direct[["relative_BTC_to_MKT"]],
    #     line_plot=True,
    #     save_plot=False
    # )


    # ret_an.perform_OLS_regression(
    #     x_direct=x_direct[["1M_trailing_corr"]],
    #     y_asset_name="MSTR",
    #     show_regression_plot=True,
    #     drop_nan=True
    # )
    #ret_an.perform_OLS_regression(
    #   x_direct=x_direct[["BRRNY", "BTC_lag_1d", "MSTR_lag_1d", "MSTR_1M_mean_ret", "BTC_1M_mean_ret"]],
    #    y_asset_name="MSTR",
    #    show_regression_plot=False,
    #    save_regression_plot=True,
    #    constant_term=False,
    #    drop_nan=True
    #)
    #sys.exit()

    # ret_an.analyse_return_distribution(
    #     asset_names=["MSTR", "BRRNY"],
    #     show_distribution_plot=True,
    #     save_distribution_plot=True
    # )
    # sys.exit()

    # ret_an.single_plot(
    #     y_series=mstr_btc_investment,
    #     line_plot=True,
    #     save_plot=True
    # )
    # sys.exit()
    # __________________________________________________________________________________________________________________

    # if True:
    #     bt = BackTester(
    #         investment_universe=invest_univ,
    #         data_frequency="D",
    #         trading_costs_bps=5,
    #         ccy_exchg_costs_bps=10,
    #         trading_days_count=252,
    #         drop_nan=True
    #     )
    #
    #     backtest = bt.backtest(
    #         strategy=Strategy.xbt_carry_trade(),
    #         complementary_data=complementary_data,
    #         shorting_allowed=True,
    #         constrained=False,
    #         bt_end_date="2024-10-01",
    #         long_short_strat=True
    #     )
    #
    #     backtest = bt.backtest(
    #         strategy=Strategy.xbt_carry_trade(),
    #         complementary_data=complementary_data,
    #         shorting_allowed=False,
    #         constrained=False,
    #         bt_end_date="2024-10-01",
    #         long_short_strat=False
    #     )
    #     backtest.summarize()
    #     print(backtest.get_strat_weights())
