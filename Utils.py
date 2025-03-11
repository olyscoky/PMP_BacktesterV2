import datetime
import pandas as pd
import calendar
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


def get_monthly_settlement_dates(start_date: pd.Timestamp, end_date: pd.Timestamp) -> list[pd.Timestamp]:
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=start_date, end=end_date)
    holidays_set = {holiday.date() for holiday in holidays}

    settlement_dates = []
    year, month = start_date.year, start_date.month

    while (year < end_date.year) or (year == end_date.year and month <= end_date.month):
        first_day = pd.Timestamp(year, month, 1).date()
        days_until_friday = (4 - first_day.weekday() + 7) % 7
        first_friday = first_day + datetime.timedelta(days=days_until_friday)
        third_friday = first_friday + datetime.timedelta(days=14)

        settlement = third_friday
        while settlement in holidays_set:
            settlement -= datetime.timedelta(days=1)

        settlement_ts = pd.Timestamp(settlement)
        if start_date <= settlement_ts <= end_date:
            settlement_dates.append(settlement_ts)

        month += 1
        if month > 12:
            month = 1
            year += 1

    return settlement_dates


def get_last_friday_month(start_date: pd.Timestamp, end_date: pd.Timestamp) -> list[pd.Timestamp]:
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=start_date, end=end_date)
    holidays_set = {holiday.date() for holiday in holidays}

    settlement_dates = []
    year, month = start_date.year, start_date.month

    while (year < end_date.year) or (year == end_date.year and month <= end_date.month):
        last_day = pd.Timestamp(year, month, calendar.monthrange(year, month)[1]).date()
        last_friday = last_day - datetime.timedelta(days=(last_day.weekday() - 4) % 7)

        settlement = last_friday
        while settlement in holidays_set:
            settlement -= datetime.timedelta(days=1)

        settlement_ts = pd.Timestamp(settlement)
        if start_date <= settlement_ts <= end_date:
            settlement_dates.append(settlement_ts)

        month += 1
        if month > 12:
            month = 1
            year += 1

    return settlement_dates


def convert_return_period(ret: pd.Series, period_d_current: int, period_d_new: int) -> pd.Series:
    return ((1 + ret) ** (period_d_new / period_d_current)) - 1

#
# if __name__ == "__main__":
#     start_date = pd.Timestamp("2025-01-17")
#     end_date = pd.Timestamp("2025-03-21")
#     print(get_monthly_settlement_dates(start_date=start_date, end_date=end_date))
