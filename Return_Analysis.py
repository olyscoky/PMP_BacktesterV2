import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
import seaborn as sns
import statsmodels.api as sm

from Portfolio import InvesmentUniverse


class ReturnAnalyser:

    def __init__(
            self,
            investment_universe: 'InvesmentUniverse'
    ):
        self.__investment_universe = investment_universe

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

    def perform_OLS_regression(self, x_asset_names: list[str], y_asset_name: str):
        X = self.__investment_universe.get_subset_asset_universe(subset_asset_names=x_asset_names)
        X = sm.add_constant(X)
        y = self.__investment_universe.get_subset_asset_universe(subset_asset_names=[y_asset_name])
        model = sm.OLS(y, X).fit()
        print(model.summary())

    def plot_return_distribution(self, assets: list[str]):
        df = self.__investment_universe.get_subset_asset_universe(subset_asset_names=assets)
        for i, column in enumerate(df.columns):
            sns.kdeplot(df[column], color=self.__plot_colors[i], linewidht=2)
        plt.xlabel("Returns")
        plt.ylabel("Density")
        plt.title(f"Distribution of {', '.join(assets)} Returns")
        plt.show()
