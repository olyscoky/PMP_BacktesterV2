import os


class Path:

    __BASE_PATHS = [
        "/Users/olivierscokaert/Desktop/BackTesterV2",
    ]

    @staticmethod
    def define_valid_base_path(base_paths):
        for path in base_paths:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"No valid base path found for user: {os.getlogin()}")

    __PATH = define_valid_base_path.__func__(__BASE_PATHS)
    DATA_PATH = os.path.join(__PATH, "Data")
    RAW_DATA_PATH = os.path.join(DATA_PATH, "Raw_Data")
    SEMIPREPED_DATA = os.path.join(DATA_PATH, "SemiPreped_Data")
    PLOT_PATH = os.path.join(__PATH, "Plot")
    BACKTEST_PATH = os.path.join(__PATH, "Backtest")
