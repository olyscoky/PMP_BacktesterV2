import os
import pandas as pd

from BackTester import BackTester
from DataAggregator import DataAggregator
from Strategy import Strategy
from BackTestRec import BackTestRec
from Paths import Path
from Portfolio import InvesmentUniverse, Asset


if __name__ == "__main__":

    # INITIALIZING INVESTMENT UNIVERSE
    data = pd.read_excel(os.path.join(Path.DATA_PATH, "clean_data.xlsx"), index_col="Dates", parse_dates=["Dates"])

    invest_univ = InvesmentUniverse()
    invest_univ.add_asset(
        name="akjsdhf",
        asset_class="aksjdfs",
        return_serie="aksjdfs",
        ccy="askjhdfsf",
        hedge=False
    )



