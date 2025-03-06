import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.figure import Figure
import seaborn as sns
import statsmodels.api as sm
import os
from datetime import datetime
from tabulate import tabulate
from scipy.stats import skew, kurtosis, norm
from typing import Callable

from Portfolio import InvesmentUniverse
from Paths import Path


class ReturnAnalyser:

    def __init__(
            self,
            investment_universe: 'InvesmentUniverse'
    ):
        self.__investment_universe = investment_universe
        self.__initialize_plot_colors()

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

    def perform_OLS_regression(
            self,
            x_asset_names: list[str],
            y_asset_name: str,
            x_operation: Callable[[pd.DataFrame], pd.DataFrame] = lambda df: df,
            y_operation: Callable[[pd.Series], pd.Series] = lambda s: s,
            show_regression_plot: bool = False,
            save_regression_plot: bool = False
    ):
        X = x_operation(
            self.__investment_universe.get_subset_asset_universe(subset_asset_names=x_asset_names)
        )
        X = sm.add_constant(X)
        y = y_operation(
            self.__investment_universe.get_subset_asset_universe(subset_asset_names=[y_asset_name])
            .squeeze()
        )
        model = sm.OLS(y, X).fit()
        print("\n")
        print(model.summary())
        print("\n")

        if show_regression_plot or save_regression_plot:
            num_vars = X.shape[1] - 1
            fig, axes = plt.subplots(nrows=num_vars, ncols=1, figsize=(6, 4 * num_vars))

            if num_vars == 1:
                axes = [axes]

            for i, ax, in enumerate(axes):
                X_var = X.iloc[:, i + 1]
                y_pred = model.predict(X)
                ax.scatter(X_var, y, alpha=0.5, label="Actual Data")
                ax.plot(X_var, y_pred, color='red', label="Regression")
                ax.set_xlabel(f"X_{i}: {x_asset_names[i]}")
                ax.set_ylabel(f"y: {y_asset_name}")
                ax.legend()

            plt.tight_layout()

            if show_regression_plot:
                plt.show()

            if save_regression_plot:
                plt.savefig(
                    os.path.join(
                        Path.PLOT_PATH,
                        f"REG-{y_asset_name}<>{'_'.join(x_asset_names)}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}.png"
                    ),
                    dpi=300,
                    bbox_inches="tight"
                )

    def analyse_returns(
            self,
            asset_names: list[str]
    ):
        df = operation(
            self.__investment_universe.get_subset_asset_universe(subset_asset_names=asset_names)
        )

        stats_data = []
        for asset, column in zip(asset_names, df.columns):
            mean = df[column].mean()
            std_dev = df[column].std()
            neg_rets = df[column][df[column] < mean]
            semi_std_dev = np.sqrt(np.mean((neg_rets - mean) ** 2) if len(neg_rets) > 0 else 0)
            skewness = skew(df[column], nan_policy="omit")
            kurt = kurtosis(df[column], nan_policy="omit")
            geom_mean = np.exp(np.log1p(df[column]).mean()) - 1
            var_95 = mean + norm.ppf(0.05) * std_dev
            stats_data.append([asset, mean, geom_mean, std_dev, semi_std_dev, skewness, kurt, var_95])

        headers = ["Asset", "Mean", "Geom-Mean", "Std Dev", "Semi Std Dev", "Skewness", "Kurtosis", "95%-VaR"]
        print("\n")
        print("Return Metrics")
        print(tabulate(stats_data, headers=headers, tablefmt="fancy_grid", floatfmt=".4f"))
        print("\n")

    def analyse_return_distribution(
            self,
            asset_names: list[str],
            show_distribution_plot: bool = False,
            save_distribution_plot: bool = False
    ):
        df = self.__investment_universe.get_subset_asset_universe(subset_asset_names=asset_names)

        stats_data = []
        for asset, column in zip(asset_names, df.columns):
            mean = df[column].mean()
            std_dev = df[column].std()
            skewness = skew(df[column], nan_policy="omit")
            kurt = kurtosis(df[column], nan_policy="omit")
            geom_mean = np.exp(np.log1p(df[column]).mean()) - 1
            stats_data.append([asset, mean, geom_mean, std_dev, skewness, kurt])

        headers = ["Asset", "Mean", "Geom-Mean", "Std Dev", "Semi Std Dev", "Skewness", "Kurtosis", "95%-VaR"]
        print("\n")
        print("Return Metrics")
        print(tabulate(stats_data, headers=headers, tablefmt="fancy_grid", floatfmt=".4f"))
        print("\n")

        if show_distribution_plot or save_distribution_plot:
            for i, column in enumerate(df.columns):
                sns.kdeplot(
                    df[column],
                    color=self.__plot_colors[i],
                    linewidth=2,
                    label=asset_names[i],
                    common_norm=True
                )
            plt.xlabel("Returns")
            plt.ylabel("Density")
            plt.legend()
            plt.title(f"Distribution of Returns")

            if show_distribution_plot:
                plt.show()

            if save_distribution_plot:
                plt.savefig(
                    os.path.join(
                        Path.PLOT_PATH,
                        f"DIST-{'_'.join(asset_names)}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}.png",
                    ),
                    dpi=300,
                    bbox_inches="tight"
                )
