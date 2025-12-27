import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime
from warnings import warn
from sklearn.preprocessing import StandardScaler
from src.utils import EnergyDataset
import src.data_preprocessing as datap
from src.submission import get_time_features, add_my_derived_features_, update_my_data_cols


def load_from_parquet(path: str, input_cols: list[str], output_cols: Optional[list[str]]=None, additional_cols:Optional[list[str]]=None):
    """Loads a dataset from a parquet file.
    
    :param str path: the path to the dataset
    :param list[str] input_cols: the list of columns to use for the input
    :param list[str] additional_cols: list of columns that will be used intermediately
    """
    _load_cols = additional_cols + input_cols if additional_cols is not None else input_cols
    if output_cols is not None:
        _load_cols = _load_cols + [_oc for _oc in output_cols if _oc not in _load_cols]
    
    return pd.read_parquet(path=path, columns=_load_cols)


def get_normalized_data(data: pd.DataFrame, features: list[str], targets: list[str], datetime_col: str, valid_cutoff_datetime: str):
    data_X = data[features].to_numpy(dtype=np.float32)
    data_y = data[targets].to_numpy(dtype=np.float32)

    # calculate the index map
    valid_dt = datetime.fromisoformat(valid_cutoff_datetime)
    timezone = data[datetime_col][0].tz
    if timezone is not None:
        valid_dt_copy = valid_dt
        if valid_dt_copy.tzinfo is None:
            train_test_split_local = timezone.localize(valid_dt_copy)
        else:
            train_test_split_local = valid_dt_copy
        _train_map = data[datetime_col] < train_test_split_local
    else:
        _train_map = data[datetime_col] < valid_dt

    # assuming the dataset is ordered, we can calculate the index
    _train_end = int(_train_map.sum()) + 1
    _val_end = None

    scaler_X = StandardScaler().fit(data_X[:_train_end])
    _means_X = data.loc[_train_map, features].apply(datap.trimmed_func, axis=0, function = np.mean, keep=0.95)
    _stds_X = data.loc[_train_map, features].apply(datap.trimmed_func, axis=0, function = np.std, keep=0.95)
    scaler_X.mean_ = _means_X.to_numpy(dtype=np.float64)
    scaler_X.scale_ = _stds_X.to_numpy(dtype=np.float64)

    scaler_y = StandardScaler().fit(data_y[:_train_end])
    _means_y = data.loc[_train_map, targets].apply(datap.trimmed_func, axis=0, function = np.mean, keep=0.95)
    _stds_y = data.loc[_train_map, targets].apply(datap.trimmed_func, axis=0, function = np.std, keep=0.95)
    scaler_y.mean_ = _means_y.to_numpy(dtype=np.float64)
    scaler_y.scale_ = _stds_y.to_numpy(dtype=np.float64)

    X_scaled = scaler_X.transform(data_X)
    y_scaled = scaler_y.transform(data_y)

    return X_scaled, y_scaled, _train_end, _val_end


def get_preprocessed_ts_data(path: str, features: list[str], targets: list[str], window_size: int, forecast_horizon: int, stride: int, preprocess_kwargs: dict):
    _targets = [targets] if not isinstance(targets, list) else targets
    _datetime_col = preprocess_kwargs.get("datetime_col", "Date")
    df = load_from_parquet(path=path, input_cols=features, output_cols=_targets, additional_cols=[_datetime_col])

    _valid_cutoff_datetime = preprocess_kwargs.pop("valid_cutoff_datetime") if preprocess_kwargs.get("valid_cutoff_datetime", None) else None

    _data_cols = features + [_c for _c in _targets if _c not in features]  # all data cols must be normalized
    df = datap.preprocess_ts_data(data=df,
                                  input_cols=_data_cols,
                                  **preprocess_kwargs
                                  )
    
    add_my_derived_features_(df)
    _features, _targets = update_my_data_cols(features, _targets)
    print(f"Using features: {_features}, using targets: {_targets}")

    X_scaled, y_scaled, _train_end, _val_end = get_normalized_data(data=df,
                                                                   features=_features,
                                                                   targets=_targets,
                                                                   datetime_col=_datetime_col,
                                                                   valid_cutoff_datetime=_valid_cutoff_datetime)

    _time_embeddings_feats, _time_embeddings_tgts = get_time_features(df.loc[:, _datetime_col], dtype=np.float32, valid_cutoff_datetime=_valid_cutoff_datetime)
    if _time_embeddings_feats is not None: X_scaled = np.c_[X_scaled, _time_embeddings_feats]
    if _time_embeddings_tgts is not None: y_scaled = np.c_[y_scaled, _time_embeddings_tgts]
    if _time_embeddings_feats is None and _time_embeddings_tgts is None:
        # will be a ValueError in our evaluation script!
        warn(f"Time embeddings are undefined!", RuntimeWarning)

    # TODO check the target features set on the model! should be ["Spot", "wind_mean", "solar_mean"]

    train_ds = EnergyDataset(X_scaled[:_train_end], y_scaled[:_train_end], window_size,
                             forecast_horizon, stride)
    valid_ds = EnergyDataset(X_scaled[_train_end:_val_end], y_scaled[_train_end:_val_end], window_size,
                             forecast_horizon, stride)
    
    return train_ds, valid_ds
