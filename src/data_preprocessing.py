import pandas as pd
import numpy as np
import datetime as dt
from datetime import time


def index_is_valid(idx:pd.Index):
    if not idx.is_monotonic_increasing:
        return False
    if not idx[0] == 0:
        return False
    if not idx[len(idx) - 1] == len(idx) - 1:
        return False
    if idx.has_duplicates:
        return False
    if any(pd.isna(idx)):
        return False
    return True

def change_dst(dati, dst:bool=True):
    """
    Add's or removes daylight saving time (dst) from a datetime if the timezone has this property.
    
    :param datetime dati: datetime object to be edited
    :param bool dst:      True if dst should be added (one hour earlier), False if no dst (winter time)
    
    Returns new datetime object with same properties but evtl. other dst.
    """
    tzinf = dati.tz
    #new_dt = dati.replace(tzinfo=None) # not working
    new_dt = dt.datetime(dati.year, dati.month, dati.day, dati.hour, dati.minute, dati.second, dati.microsecond)
    return tzinf.localize(new_dt, is_dst=dst)

def correct_double_timestamps(stamps, reverse_order:bool=True) -> pd.DataFrame:
    """
    Search all doubled timestamps and add daylight saving time to the first/second one.
    For more then two same stamps,
        if reverse_order: only on the last dst is added (run again to have dst on two stamps)
        else: only on the first dst is added
    
    :param [pd.Timestamp] stamps: Iterable of timestamps to check for duplicates
    :param bool reverse_order:    True if not the first timestamp but the last should have dst add because it happend before
    
    Returns [pd.Timestamp] whit edited timestamps at dst.
    """
    stamps_series = pd.Series(stamps)
    dup_stamps = stamps_series.duplicated(keep=False)
    stamps_subset = stamps_series[dup_stamps]
    dup_stamps_subset = stamps_subset.duplicated(keep='first')
    
    for i, is_orig in enumerate(dup_stamps_subset):
        if is_orig:
            double_stamps = stamps_subset[stamps_subset == stamps_subset.iloc[i]]
            double_stamps = double_stamps.apply(lambda x: change_dst(x, False))
            _dst_idx = -1 if reverse_order else 0
            double_stamps.iloc[_dst_idx] = change_dst(double_stamps.iloc[_dst_idx], True)
            stamps_subset.loc[double_stamps.index] = double_stamps
    
    try:
        new_stamps = stamps.copy()
    except:
        new_stamps = stamps
    if type(new_stamps) == list:
        new_stamps = [stamps_subset.loc[i] if dup_stamps[i] else new_stamps[i] for i in range(len(new_stamps))]
    else:
        new_stamps[stamps_subset.index] = stamps_subset
    return new_stamps

def refactor_daily_gas_price(data:pd.DataFrame, spot_gas_col_name:str, hod:int=0, replacer=np.nan, date_col:str="Date"):
    """
    Takes duplicated timestamps and distributes the different spot gas prices over the next days at hod (hour of day).
    
    :param pd.DataFrame data:     dataset with timestamp variable "Date" and with gas
    :param str spot_gas_col_name: name of the column with the spot gas price
    :param int hod:               hour of day at which the spot gas price is always changed
    :param float replacer:        value that should be entered as gas price in the duplicated timestamps. If None, don't replace.
    :param str date_col:          name of column with the date values
    
    Returns pd.DataFrame data with the changed spot gas price column
    """
    if date_col not in data.columns:
        raise ValueError(f"<data> does not have the column {date_col}. It needs to have a correct <date_col> with Timestamps.")
    if spot_gas_col_name not in data.columns:
        raise ValueError(f"No column named '{spot_gas_col_name}' found in <data>. Please check input for <spot_gas_col_name>.")
    if (hod < 0) | (hod > 23):
        raise ValueError("<hod> (hour of day) must be between 0 and 23.")
    
    date = data[date_col]
    date_gas = data.loc[date.duplicated(), [date_col, spot_gas_col_name]]
    grouped = date_gas.groupby([date_col])
    data = data.copy()
    for dateti, frame in grouped:
        prices = frame[spot_gas_col_name].tolist()
        index = frame.index
        day = dateti[0].date()
        for i in range(len(prices)):
            if all(pd.isna(data.loc[(date.dt.date == day + dt.timedelta(days=i+1)) & 
                                    (date.dt.hour == hod) & (date.dt.minute == 0), 
                                    spot_gas_col_name])):
                data.loc[(date.dt.date == day + dt.timedelta(days=i+1)) & (date.dt.hour == hod) & (date.dt.minute == 0),
                         spot_gas_col_name] = prices[i]
                if replacer is not None:
                    data.loc[index[i], spot_gas_col_name] = replacer
    
    return data

def remove_duplicates(data:pd.DataFrame, keep:str|None="first", ignored_cols:list[str]=[], reset_index:bool=None):
    """
    Removes duplicated rows in pd.DataFrame. Consider only a selection of columns to see if is duplicated.
    
    :param pd.DataFrame data: dataset with duplicated rows
    :param str keep:          which row to keep in ["first", "last", None]. For None, none of the duplicated rows is kept.
    :param list ignored_cols: columns to ignore while checking for duplicated rows
    :param bool reset_index:  True if the index should be resetted afterwards, False if index stays the same.
                              If None, the index will only be reset if it was also a valid index before. 
    
    Returns pd.DataFrame without dupliated rows
    """
    
    if keep not in ["first", "last", None]:
        raise ValueError(f"<keep> must be either 'first', 'last' or None. Given: {keep}")
    if any([col not in data.columns for col in ignored_cols]):
        raise ValueError("All strings from the <ignored_cols> list must be column names from <data>, but found missmatch.")
    
    cols = data.columns
    cols = [col for col in cols if col not in ignored_cols]
    
    if reset_index is None:
        reset_index = index_is_valid(data.index)
    
    if keep is None:
        keep = False
    dups = data.loc[:, cols].duplicated(keep = keep)
    
    data = data.drop(dups[dups].index, axis=0, inplace=False)
    
    if reset_index:
        data.reset_index(inplace=True, drop=True)
    
    return data

def get_steps_from_resolution(resolution: str):
    match resolution:
        case "hour":
            _steps = 3  # 4-1
        case "day":
            _steps = 95  # 4*24 - 1
        case "week":
            _steps = 671  # 4*24*7 -1
        case "month":
            _steps = 2975  # 4*24*31 - 1
        case "year":
            _steps = 35135  # 4*24*366 - 1
        case "all":
            _steps = None
        case _:
            raise ValueError(f"Unexpected resolution value, got: {resolution}, expected one of ['hour', 'day', 'week', 'month', 'year', 'all']")

    return _steps

def missing_vals_reg_frequency(data: pd.DataFrame, col: str, resolution: str):
    _cnt = pd.Series(data[col].isna(), index=data[col].index, dtype=int)
    _consecutive = _cnt.groupby((_cnt != _cnt.shift()).cumsum()).transform('size')

    _steps = get_steps_from_resolution(resolution=resolution) or 1

    return data.loc[_consecutive > _steps].index

def refill_data(data:pd.DataFrame, columns:list, duration_units_map:dict, inplace:bool=True, date_col="Date", verbose: bool=True):
    """
    refills multiple columns in a data frame by replacing the NAs with the value of the current duration_unit.
    This function assumes ordered data!
    
    :param pd.DataFrame data:   ordered dataset with NAs that is to be filled
    :param list columns:        columns that should be filled
    :param dict duration_units_map: all maximal filling times. Can be for "hour", "day", "week", "month", "year" or "all" for all following.
    :param bool inplace:        True if the filling should happen in the given data frame, False if should return a copy with the fillling
    :param str date_col:        name of the column where the datetime values are
    :param bool verbose:        Whether to calculate the expected number of remaining NaNs and print details
    
    Returns a pd.DataFrame where the NAs of all <columns> are filled with the non-NA values for each duration_unit.
    """
    
    if not isinstance(data, pd.DataFrame):
        raise ValueError(f"<data> must be a pandas.DataFrame. Given: {type(data)}.")
    
    if not inplace:
        data = data.copy()
    
    for col in columns:
        duration = duration_units_map[col]
        if verbose:
            _col_missing_idx = missing_vals_reg_frequency(data, col, duration)
            _num_non_explainable = len(_col_missing_idx)
            _num_before = data.loc[:, col].isna().sum()
        data.loc[:, col] = data.loc[:, col].ffill(axis=0, inplace=False, limit=get_steps_from_resolution(duration))

        if verbose:
            _num_remaining = data.loc[:, col].isna().sum()
            _explanation = "The unaccounted for are likely at the beginning of the series." if _num_non_explainable < _num_remaining else ""
            print(f"Reduced missing values in col '{col}' from {_num_before}/{len(data)} to {_num_remaining}, expected at most {_num_non_explainable} remaining. {_explanation}")
            if _num_remaining:
                print(f"{data.loc[data.loc[data.loc[:, col].isna(), col].index, date_col]}")
    if verbose:
        print("Done all filling.")
    return data

def trimmed_func(x, function, keep:float=0.9):
    """
    applies a function on a subset of a vector
    """
    if keep <= 0:
        raise ValueError("It must keep at least one datapoint")

    q = 1 - keep
    q_upper = 1 - (q / 2)
    q_lower = q / 2

    quantile_upper = np.quantile(x, q_upper)
    quantile_lower = np.quantile(x, q_lower)

    x_trimmed = [y for y in x if (y < quantile_upper) & (y >= quantile_lower)]

    return function(x_trimmed)


##### Interpolation for missing data #####

def interpolate_col_flat(data:pd.DataFrame, col, pre:bool=False, pos:bool=True, inplace:bool=True):
    """
    Interpolate the first or the last not-NA values to all values before or after.
    
    :param pd.DataFrame data: dataset
    :param str|int col:       column to interpolate
    :param bool pre:          True if the beginning of column should be filled, else False
    :param bool pos:          True if the end of the columnn should be filled, else False
    :param bool inplace:      True if the given dataset should be interpolated, 
                              False to make changes on (and return a) copy
    
    Returns the data or a copy with the interpolated column.
    """
    if not any([pre, pos]):
        raise ValueError("Any of <pre> or <pos> must be True, else nothing will be interpolated.")
        
    if isinstance(col, int):
        col = data.columns[col]
    if all(data[col].isna()):
        print("Nothing to interpolate as all values in <col> are NA.")
        return data
    if not data[col].hasnans:
        print("Nothing to interpolate because there are no NAs in <col>.")
        return data
    
    if pre:
        ret = data.loc[:, col].bfill(axis="index", limit_area="outside", inplace=inplace)
        if not inplace:
            data = ret
    if pos:
        ret = data.loc[:, col].ffill(axis="index", limit_area="outside", inplace=inplace)
    
    if not inplace:
        return ret
    return data

def interpolate_series(data:pd.DataFrame, col, method: str, pre:bool=False, pos:bool=True, inplace:bool=True):
    """
    Interpolate the missing values with a linear extension of the last not-NA values.
    
    :param pd.DataFrame data: dataset
    :param str|int col:       column to interpolate
    :param str method:        Choose any method of pandas.Series.interpolate
    :param bool pre:          True if the beginning of column should be filled, else False
    :param bool pos:          True if the end of the columnn should be filled, else False
    :param bool inplace:      True if the given dataset should be interpolated, 
                              False to make changes on (and return a) copy
    
    Returns the data or a copy with the interpolated column.
    """

    if pre and pos:
        limit_dir = "both"
    elif pre and not pos:
        limit_dir = "backward"
    elif not pre and pos:
        limit_dir = "forward"
    else:
        raise ValueError(f"pre and pos cannot both be false!")

    if isinstance(col, int):
        col = data.columns[col]

    return data[col].interpolate(method=method, axis="index", limit_direction=limit_dir, inplace=inplace)

def interpolate_col(data:pd.DataFrame, col, method:str="flat", inplace:bool=True, **kwargs):
    """
    Interpolate the first or the last not-NA values with the method to all values before or after.
    
    :param pd.DataFrame data: dataset
    :param str|int col:       column to interpolate
    :param str method:        interpolation method to be used, currently possible: "flat", "linear"
    :param bool inplace:      True if the given dataset should be interpolated, 
                              False to make changes on (and return a) copy
    :param **kwargs:          arguments to pass to the interpolation method such as pre and pos for ex.
    
    Returns the data or a copy with the interpolated column.
    """
    
    match method:
        case "flat":
            data = interpolate_col_flat(data, col, inplace=inplace, **kwargs)
        case _:
            data = interpolate_series(data, col, method, inplace=inplace, **kwargs)

    return data


def list_to_time(time_vals:list[int]) -> time:
    """
    Transforms a list of integers into a datetime.time object.
    
    :param list time_vals:  list of values for hour, minute, second and microsecond. 
                            Must respect the range of each unit.
    """
    n = len(time_vals)
    if n < 4:
        time_vals2 = time_vals + [0] * (4 - n)
    else:
        time_vals2 = time_vals[0:4]
    names = ["hour", "minute", "second", "microsecond"]
    vals = dict(zip(names, time_vals2))
    return time(**vals)


def preprocess_ts_data(data: pd.DataFrame,
                       input_cols: list[str],
                       datetime_col:str="Date",
                       data_interpolation_method:str="flat",
                       resolution_map:dict={},
                       aggregation_methods_map:dict={},
                       ):
    """Pre-processes time series data.
    
    :param list[str] input_cols: the list of columns to use for the input
    :param str datetime_col: str indicating the column with the datetime values
    :param str data_interpolation_method: method to use to fill the last or first NAs of a column. In ["flat", "linear"]
    :param dict[str] resolution_map: frequency of each input_cols where a new value is expected. 
                                     A dictionary with key=col_name, value one of ["hour", "day", "week", "month", "year", NA].
    :param dict[str] aggregation_methods_map: The aggregation method for each col in input_cols to compress it to hourly data.
    """

    # correct errors
    data[datetime_col] = correct_double_timestamps(data[datetime_col])
    if "Spot Gas EEX THE Day Ahead" in input_cols:
        data = refactor_daily_gas_price(data, "Spot Gas EEX THE Day Ahead", 0)
        data = remove_duplicates(data, ignored_cols=["Spot Gas EEX THE Day Ahead"])
    else:
        data = remove_duplicates(data)
    data.sort_values(by=datetime_col, axis=0, inplace=True)
    data.reset_index(inplace=True, drop=True)

    # check for missing or duplicated timesteps
    all_dates = data[datetime_col]
    time_step_diff = (all_dates.values[1:] - all_dates.values[:-1])
    time_step_diff = pd.Series(time_step_diff)
    time_step_diff = time_step_diff.dt.total_seconds() / 60
    if len(set(time_step_diff)) != 1:
        print("There might be some timesteps missing or remaining duplicates.\n" +
                        "These are the found distances in minutes between timesteps " +
                        f"ordered by frequency decreasing: {list(time_step_diff.value_counts().index)}.")

    # fill less frequent data
    print("Start to fill NAs in data.")
    na_columns = [col for col in input_cols if data.loc[:, col].hasnans]
    refill_data(data,
                columns=na_columns, 
                duration_units_map=resolution_map, 
                inplace=True,
                date_col=datetime_col,
                verbose=True)
    
    # data imputation:
    for col in input_cols:
        if data[col].hasnans:
            data[col] = interpolate_col(data, col, method=data_interpolation_method, inplace=False, pre=True, pos=True)
    if data.loc[:, input_cols].isna().values.any().any():
        na_columns = [col for col in input_cols if data.loc[:, col].hasnans]
        raise  ValueError("Not all NAs could be filled in the data.\n" +
                            "Check if there are NAs in the middle that don't get interpolated. " +
                            "Maybe correct the resolution in the config file of your data.\n" +
                            f"Columns with NAs: {na_columns}.")
    else:
        print("All NAs filled successfully.")
    
    # aggregate to hourly data
    data[datetime_col] = data[datetime_col].apply(lambda d: d.replace(minute=0, second=0, microsecond=0))
    _agg_methods = {col: aggregation_methods_map[col] for col in input_cols}
    data = data.groupby(datetime_col).aggregate(_agg_methods)
    data.reset_index(inplace=True, drop=False)

    return data
