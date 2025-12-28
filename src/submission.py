import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Callable
from functools import partial
from src.utils import init_weights_normal, flatten_weather_batch_transform


def get_train_config() -> dict:
    n_orig_features = 15
    n_derived = 3  
    n_time_feats = 7
    n_hist_features = n_orig_features + n_derived + n_time_feats  # 25

    forecast_history = 192
    forecast_horizon = 48
    n_weather_future = 2

    input_dim = n_hist_features * forecast_history + n_weather_future * forecast_horizon
    output_dim = forecast_horizon

    return {
        "targets": ["Spot", "wind_mean", "solar_mean"],
        "features": [
            "pumped hydro conso", "pumped hydro production",
            "cross-border trade", "brown coal", "black coal", "natural gas",
            "onshore wind", "solar", "solar capacity", "onshore wind capacity",
            "Spot", "conso", "Spot coal", "Spot Gas EEX THE Day Ahead", "CO2 price GER"
        ],
        "stride": 1,
        "forecast_history": forecast_history,
        "forecast_horizon": forecast_horizon,
        "batch_size": 32,
        "net_params": {
            "input_dim": input_dim,
            "hidden_dim": 256,
            "output_dim": output_dim,
            "dropout_p": 0.3,
            "n_hist_features": n_hist_features,
            "forecast_history": forecast_history,
            "n_weather_future": n_weather_future,
            "forecast_horizon": forecast_horizon,
            "num_layers": 2,
            "bidirectional": False
        },
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "epochs": 50,
    }


def get_time_features(datetime_series: pd.Series, valid_cutoff_datetime: str, dtype: type=np.float32) -> tuple[np.ndarray, np.ndarray]:
    ds = pd.to_datetime(datetime_series)
    hour = ds.dt.hour.to_numpy().astype(np.float32)
    days = ds.dt.dayofweek.to_numpy().astype(np.float32)
    month = ds.dt.month.to_numpy().astype(np.float32)

    hour_sin = np.sin(2 * np.pi * hour / 24).astype(dtype)
    hour_cos = np.cos(2 * np.pi * hour / 24).astype(dtype)
    dow_sin = np.sin(2 * np.pi * days / 7).astype(dtype)
    dow_cos = np.cos(2 * np.pi * days / 7).astype(dtype)
    month_sin = np.sin(2 * np.pi * (month - 1) / 12).astype(dtype)
    month_cos = np.cos(2 * np.pi * (month - 1) / 12).astype(dtype)
    weekend = ((ds.dt.dayofweek >= 5).astype(np.float32)).astype(dtype)

    time_feats = np.stack([hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos, weekend], axis=1).astype(dtype)
    return time_feats, None


def get_my_batch_tsfm() -> Callable:
    return partial(flatten_weather_batch_transform, num_targets=1)


def add_my_derived_features_(df: pd.DataFrame) -> None:
    if "Spot" in df.columns:
        df.loc[:, "Spot_roll_24"] = df["Spot"].rolling(window=24, min_periods=1).mean()
        df.loc[:, "Spot_roll_168"] = df["Spot"].rolling(window=168, min_periods=1).mean()
    if "solar" in df.columns and "solar capacity" in df.columns:
        df.loc[:, "solar_util"] = df["solar"] / (df["solar capacity"] + 1e-6)
    return None

def update_my_data_cols(feature_cols: list, target_cols: list) -> tuple[list, list]:
    derived = ["Spot_roll_24", "Spot_roll_168", "solar_util"]
    for col in derived:
        if col not in feature_cols:
            feature_cols.append(col)
    return feature_cols, target_cols

# ------------------------------
# LSTM Model
# ------------------------------
class EnergyLSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_p: float,
                 n_hist_features, forecast_history, n_weather_future, forecast_horizon,
                 num_layers=2, bidirectional=False, rng: torch.Generator = None):
        super().__init__()
        self.n_hist_features = n_hist_features
        self.forecast_history = forecast_history
        self.n_weather_future = n_weather_future
        self.forecast_horizon = forecast_horizon
        self.hidden_dim = hidden_dim

        # LSTM verarbeitet nur die Historie
        self.lstm = nn.LSTM(
            input_size=n_hist_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        # WICHTIG: Dimension des Regressors anpassen
        # Input für Linear Layer ist jetzt: LSTM-Output + Zukunfts-Wetter
        # LSTM Output size: hidden_dim (mal 2 wenn bidirectional)
        # Future Weather size: n_weather_future * forecast_horizon
        lstm_out_dim = (2 if bidirectional else 1) * hidden_dim
        future_weather_flat_dim = n_weather_future * forecast_horizon
        
        self.reg_layer = nn.Linear(lstm_out_dim + future_weather_flat_dim, output_dim)

        if rng is not None:
            self.apply(partial(init_weights_normal, generator=rng))

    def forward(self, x):
        # x shape: (batch, total_input_dim)
        batch_size = x.size(0)

        # 1. Split: Historie vs. Zukunft
        # Wir müssen genau berechnen, wo der Schnitt ist
        hist_feats_len = self.n_hist_features * self.forecast_history
        
        # Historische Daten für LSTM
        hist_feats_flat = x[:, :hist_feats_len]
        hist_feats = hist_feats_flat.view(batch_size, self.forecast_history, self.n_hist_features)

        # Zukünftige Wetterdaten (die vorher ignoriert wurden!)
        future_weather_flat = x[:, hist_feats_len:] 
        
        # 2. LSTM Pass
        lstm_out, _ = self.lstm(hist_feats)
        # Wir nehmen den letzten Hidden State
        last_out = lstm_out[:, -1, :]
        
        # 3. Concatenate: LSTM Wissen + Zukunfts-Wetter Wissen
        combined_input = torch.cat([last_out, future_weather_flat], dim=1)
        
        # 4. Final Prediction
        out = self.reg_layer(combined_input)
        return out


def get_my_model(net_kwargs: dict, rng: torch.Generator) -> nn.Module:
    """Returns an instance of the model."""
    return EnergyLSTMNet(**net_kwargs, rng=rng)

def get_my_model_name() -> str:
    return "my_model_lstm_v1"

def get_group_number() -> str:
    _group_number = 125
    return f"group_{_group_number}"

def get_group_names() -> str:
    return "Tjard_Kleine"
