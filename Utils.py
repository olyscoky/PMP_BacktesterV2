import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


def concat_df_series_with_nearest_index(df_lst: list[pd.DataFrame | pd.Series]) -> pd.DataFrame:
    if not df_lst:
        raise ValueError("df_lst must contain at least one Series or DataFrame.")

    data_list = [df if isinstance(df, pd.DataFrame) else df.to_frame() for df in df_lst]

    start_idx = max(df.index[0] for df in data_list)
    end_idx = min(df.index[-1] for df in data_list)

    common_index = sorted(set().union(*[df.index for df in data_list]))
    common_index = pd.Index([idx for idx in common_index if start_idx <= idx <= end_idx])

    aligned_data = [df.reindex(common_index, method='nearest') for df in data_list]

    result_df = pd.concat(aligned_data, axis=1)

    return result_df


def get_third_fridays(start_date: pd.Timestamp, end_date: pd.Timestamp) -> list[pd.Timestamp]:
    months = pd.date_range(start=start_date, end=end_date, freq='MS')
    third_fridays = []

    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=start_date, end=end_date)

    for month in months:
        fridays = pd.date_range(start=month, periods=5, freq='W-FRI')
        third_friday = fridays[2]

        while third_friday in holidays:
            third_friday -= pd.Timedelta(days=1)
            while third_friday in holidays or third_friday.weekday() >= 5:
                third_friday -= pd.Timedelta(days=1)

        if start_date <= third_friday <= end_date:
            third_fridays.append(third_friday)

    return third_fridays


def convert_return_period(ret: pd.Series, period_d_current: int, period_d_new: int) -> pd.Series:
    return ((1 + ret) ** (period_d_new / period_d_current)) - 1
