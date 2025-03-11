import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
import seaborn as sns
import statsmodels.api as sm
import os
from datetime import datetime
from tabulate import tabulate
from scipy.stats import skew, kurtosis, norm
from typing import Callable

from Portfolio import InvesmentUniverse
from Paths import Path
from Utils import concat_df_series_with_nearest_index


class DataAnalyser:

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
            x_asset_names: list[str] | None = None,
            x_direct: pd.DataFrame | None = None,
            y_asset_name: str | None = None,
            y_direct: pd.Series | None = None,
            x_operation: Callable[[pd.DataFrame], pd.DataFrame] = lambda df: df,
            y_operation: Callable[[pd.Series], pd.Series] = lambda s: s,
            constant_term: bool = True,
            show_regression_plot: bool = False,
            save_regression_plot: bool = False,
            fill_nan: float | None = None,
            drop_nan: bool = False
    ):
        if all([x_asset_names is None, x_direct is None]) or not any([x_asset_names is None, x_direct is None]):
            raise Warning(
                f"Only provide either x_asset_names or x_direct: "
                f"{'Both provided' if all([x_asset_names, x_direct]) else 'None provided'}"
            )
        elif x_direct is not None:
            X = x_direct
        else:
            X = x_operation(
                self.__investment_universe.get_subset_asset_universe(
                    subset_asset_names=x_asset_names,
                    fill_nan=fill_nan,
                    drop_nan=drop_nan
                )
            )

        if all([y_asset_name is None, y_direct is None]) or not any([y_asset_name is None, y_direct is None]):
            raise Warning(
                f"Only provide either y_asset_name or y_direct: "
                f"{'Both provided' if all([y_asset_name, y_direct]) else 'None provided'}"
            )
        elif y_direct is not None:
            y = y_direct
        else:
            y = y_operation(
                self.__investment_universe.get_subset_asset_universe(
                    subset_asset_names=[y_asset_name],
                    fill_nan=fill_nan,
                    drop_nan=drop_nan
                )
                .squeeze()
            )

        X_freq = X.index.to_series().diff().median()
        y_freq = y.index.to_series().diff().median()
        index_start = max(X.index[0], y.index[0])
        index_end = min(X.index[-1], y.index[-1])
        if X_freq < y_freq:
            y = y.reindex(X.index, method="nearest")
        else:
            X = X.reindex(y.index, method="nearest")

        y = y[index_start:index_end]
        X = X.loc[index_start:index_end]

        print(X)

        if constant_term:
            X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()
        print("\n")
        print(model.summary())
        print("\n")

        if show_regression_plot or save_regression_plot:
            num_vars = X.shape[1] - (1 if constant_term else 0)
            fig, axes = plt.subplots(nrows=num_vars, ncols=1, figsize=(6, 4 * num_vars))

            if num_vars == 1:
                axes = [axes]

            for i, ax, in enumerate(axes):
                X_var = X.iloc[:, i + 1] if constant_term else X.iloc[:, i]
                X_range = pd.Series(np.linspace(X_var.min(), X_var.max(), 100))
                if constant_term:
                    beta_0 = model.params[0]
                    beta_i = model.params[i + 1]
                else:
                    beta_0 = 0
                    beta_i = model.params[i]
                y_regression = beta_0 + beta_i * X_range

                ax.scatter(X_var, y, alpha=0.5, label="Actual Data")
                ax.plot(X_range, y_regression, color='red', label="Regression", linewidth=2)
                ax.set_xlabel(f"X_{i}: {x_asset_names[i] if x_asset_names is not None else x_direct.columns[i]}")
                ax.set_ylabel(f"y: {y_asset_name}")
                ax.legend()

            plt.tight_layout()

            if save_regression_plot:
                plt.savefig(
                    os.path.join(
                        Path.PLOT_PATH,
                        f"REG-{y_asset_name}<>"
                        f"{'_'.join(x_asset_names if x_asset_names else x_direct.columns)}-"
                        f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}.png"
                    ),
                    dpi=300,
                    bbox_inches="tight"
                )

            if show_regression_plot:
                plt.show()

    def analyse_return_distribution(
            self,
            asset_names: list[str],
            show_distribution_plot: bool = False,
            save_distribution_plot: bool = False,
            fill_nan: float | None = None,
            drop_nan: bool = False
    ):
        df = self.__investment_universe.get_subset_asset_universe(
            subset_asset_names=asset_names,
            fill_nan=fill_nan,
            drop_nan=drop_nan
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

        plt.figure(figsize=(6, 4))

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

            if save_distribution_plot:
                plt.savefig(
                    os.path.join(
                        Path.PLOT_PATH,
                        f"DIST-{'_'.join(asset_names)}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}.png",
                    ),
                    dpi=300,
                    bbox_inches="tight"
                )

            if show_distribution_plot:
                plt.show()

    def single_plot(
            self,
            y_series: pd.Series,
            x_series: pd.Series | None = None,
            scatter_plot: bool = False,
            line_plot: bool = False,
            save_plot: bool = False
    ):
        y_series = y_series.squeeze()
        if x_series is not None:
            x_series = x_series.squeeze()
            df = concat_df_series_with_nearest_index(df_lst=[x_series, y_series])
            x_name = str(x_series.columns[0]) if isinstance(x_series, pd.DataFrame) else str(x_series.name)
        y_name = str(y_series.columns[0]) if isinstance(y_series, pd.DataFrame) else str(y_series.name)
        plt.figure(figsize=(6, 4))

        if scatter_plot and line_plot:
            raise Warning("Plot must be either of type scatter or line not both")
        elif scatter_plot:
            plt.scatter(
                df[x_name] if x_series is not None else y_series.index,
                df[y_name] if x_series is not None else y_series,
                label="Data",
                alpha=0.7
            )
        elif line_plot:
            plt.plot(
                df[x_name] if x_series is not None else y_series.index,
                df[y_name] if x_series is not None else y_series,
                color=self.__plot_colors[1],
            )
        else:
            raise Warning("Please provide a plot type: plot | line")

        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title(
            f"{x_name if x_series is not None else 'Date'} vs "
            f"{y_name}"
        )
        plt.legend()

        if save_plot:
            plt.savefig(
                os.path.join(
                    Path.PLOT_PATH,
                    f"SIMPLE_PLOT-{y_series.name}<>{x_series.name if x_series is not None else 'Date'}-"
                    f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}.png",
                ),
                dpi=300,
                bbox_inches="tight"
            )

        plt.show()
