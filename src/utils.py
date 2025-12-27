import numpy as np
import torch
import random
from warnings import warn
from pathlib import Path
from torch.utils.data import Dataset
from torch.optim import Optimizer
import torch.nn as nn
import pandas as pd
import plotly.express as px
from typing import Callable


def seed_torch(seed, reproducible:bool = False):
    np.random.seed(seed % (2*64-1))  # numpy global rng 
    torch.manual_seed(seed % (2**64-1))  # torch global rng (CPU & CUDA)
    random.seed(seed % (2**64-1))  # python global rng (MT)

    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init_weights_normal(m: nn.Module, generator: torch.Generator):
    if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
        torch.nn.init.kaiming_normal_(m.weight, generator=generator)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class EnergyDataset(Dataset):

    def __init__(self, features, target, window, horizon, stride):
        self.features = torch.tensor(features, dtype=torch.float32) if not torch.is_tensor(features) else features
        self.target = torch.tensor(target, dtype=torch.float32) if not torch.is_tensor(target) else target
        if len(self.target) != len(self.features):
            raise ValueError(f"Expected features and targets to have equal length and the same indices!")
        self.window = window
        self.horizon = horizon
        self.stride = stride

        self.set_usable_idxs()

    def set_usable_idxs(self):
        """The window returned for an index starts at that index, then continues for self.window many steps for the 
        input and ends with another self.horizon steps for the target features."""
        self.usable_idxs = np.arange(stop=len(self.features) - self.window - self.horizon + 1, step=self.stride, dtype=np.int32)

    def __len__(self):
        return len(self.usable_idxs)

    def __getitem__(self, idx):
        _idx = self.usable_idxs[idx]
        x = self.features[_idx: _idx + self.window]
        y = self.target[_idx + self.window: _idx + self.window + self.horizon]
        return x, y

def default_batch_transform(x: torch.Tensor, y: torch.Tensor, device: torch.device, train: bool):
    x, y = x.to(device), y.to(device)
    if y.shape[-1] == 1: y = y.squeeze(-1)  # y has shape (batch_size, 1)
    return x, y

def flatten_weather_batch_transform(x: torch.Tensor, y: torch.Tensor, device: torch.device, train: bool, num_targets: int):
    """this function behaves the same during training and evaluation: It will only re-activate gradient tracking for x and y in training mode.
    counter example: data augmentation would only be applied during training, never during evaluation"""

    with torch.no_grad():
        _flat_x = torch.reshape(x, (x.shape[0], -1))
        _flat_w = torch.reshape(y[:, :, num_targets:], (x.shape[0], -1))
        _x = torch.cat((_flat_x, _flat_w), dim=1)
        _y = y[:, :, :num_targets]
        if _y.shape[-1] == 1: _y = _y.squeeze(-1)
    _x, _y = _x.requires_grad_(train).to(device), _y.requires_grad_(train).to(device)
    return _x, _y

def load_model(model_path: Path, model: nn.Module, optimizer: Optimizer, device: torch.device, with_opt:bool = True, strict: bool = True):
    state = torch.load(model_path, map_location=device, weights_only=True)
    hasopt = set(state)=={'model', 'opt'}
    model_state = state['model'] if hasopt else state
    model.load_state_dict(model_state, strict=strict)
    if hasopt and with_opt:
        try: optimizer.load_state_dict(state['opt'])
        except:
            if with_opt: warn("Could not load the optimizer state.")
    elif with_opt: warn("Saved file doesn't contain an optimizer state.")

def predict_one_sample(model, features, targets, device, transform_batch_fn:Callable=default_batch_transform):
    x, y = transform_batch_fn(features.unsqueeze(0), targets.unsqueeze(0), device, train=False)
    if isinstance(x, tuple):
        x = tuple(_x if _x.shape[0] == 1 else _x.unsqueeze(0) for _x in x)
    pred = model(x)
    return pred, y

def plot_one_prediction(pred, y, ):
    df = pd.DataFrame(data={"pred": pred.cpu().detach().squeeze(0).numpy(), "ground truth": y.cpu().detach().squeeze(0).numpy()}, index=pd.Series(torch.arange(start=1, end=y.shape[1]+1, device="cpu"), name="hours"))
    fig = px.line(df)
    return fig
