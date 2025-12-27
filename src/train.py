import sys
sys.path.append("../")

import torch
import secrets
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR
from typing import Callable, Optional
from pathlib import Path

from src.utils import seed_torch, default_batch_transform, load_model
from src.datasets import get_preprocessed_ts_data
from src.submission import get_train_config, get_my_model, get_my_batch_tsfm, get_my_model_name


def fit(model, train_loader, val_loader, optimizer, criterion, epochs, writer, device, transform_batch_fn:Callable=default_batch_transform,
        save_model_path: Optional[Path]=None, scheduler: Optional[LRScheduler]=None):
    best_val_loss = torch.inf
    eps = 1e-4

    for epoch in range(epochs):
        model.train()
        train_loss_acc = 0.0
        train_batch_num_acc = 0.0

        for x, y in train_loader:
            x, y = transform_batch_fn(x, y, device, train=True)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_loss_acc += loss.item() * len(x)
            train_batch_num_acc += len(x)

        avg_train_loss = train_loss_acc / train_batch_num_acc

        model.eval()
        val_loss_acc = 0.0
        val_batch_num_acc = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = transform_batch_fn(x, y, device, train=False)

                pred = model(x)
                loss = criterion(pred, y)

                val_loss_acc += loss.item() * len(x)
                val_batch_num_acc += len(x)

        avg_val_loss = val_loss_acc / val_batch_num_acc
        if avg_val_loss + eps < best_val_loss:
            print(f"+++ Found better validation loss at epoch {epoch+1}: {avg_val_loss:.4f}")
            best_val_loss = avg_val_loss
            if save_model_path is not None:
                state = {'model': model.state_dict(), 'opt': optimizer.state_dict()}
                torch.save(state, save_model_path)

        print(f"Epoch {epoch + 1}/{epochs} | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}")
        if writer is not None:
            writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
            if scheduler is not None: writer.add_scalar("Hparams/lr", scheduler.get_last_lr()[0])

        # Adjust learning rate
        if scheduler is not None:
            scheduler.step()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on device: {device}")

    # Draw random seeds using Python's cryptographic library and cut them down to size
    _global_seed = secrets.randbits(128) % (2**64-1)
    _local_seed = secrets.randbits(128) % (2**64-1)

    # If you want to reproduce a previous experiment (and only then) set the seeds to that previously recorded value
    # and set `reproducible=True`. N.B.: using `42` is never a good idea!
    seed_torch(_global_seed, reproducible=False)
    _rng_cpu = torch.Generator()
    _rng_cpu.manual_seed(_local_seed)

    writer = SummaryWriter("../logs/holiday_energy_experiment")
    writer.add_scalar("hparams/global_seed", _global_seed)
    writer.add_scalar("hparams/local_seed", _local_seed)

    config = get_train_config()
    config["targets"] = ["Spot", "wind_mean", "solar_mean"]
    preprocess_kwargs = {
        "datetime_col": "Date",
        "data_interpolation_method": "linear",
        "valid_cutoff_datetime": "2025-01-01 00:00:00",
        "resolution_map": {
            'Spot': 'hour',
            'pumped hydro conso': 'hour',
            'cross-border trade': 'hour',
            'nuclear': 'hour',
            'run-of-the-river production': 'hour',
            'biomass': 'hour',
            'brown coal': 'hour',
            'black coal': 'hour',
            'oil': 'hour', 
            'natural gas': 'hour',
            'geothermal': 'hour',
            'stored hydro production': 'hour',
            'pumped hydro production': 'hour',
            'other': 'hour',
            'waste incineration': 'hour',
            'offshore wind': 'hour',
            'onshore wind': 'hour',
            'solar': 'hour',
            'conso': 'skip',
            'coal gas': 'hour',
            'CO2 price GER': 'week',
            'CO2 price EU': 'week',
            'biomass capacity': 'month',
            'offshore wind capacity': 'month',
            'onshore wind capacity': 'month',
            'solar capacity': 'month',
            'battery storage (nominal power)': 'month',
            'pumped hydro capacity': 'year',
            'brown coal capacity': 'year',
            'black coal capacity': 'year',
            'oil capacity': 'year',
            'natural gas capacity': 'year',
            'other (non-renewable) capacity': 'year',
            'Spot Gas EEX THE Day Ahead': 'week',
            'Spot coal': 'week',
            'wind_mean': 'hour',
            'solar_mean': 'hour',
        },
        "aggregation_methods_map": {
            'Spot': 'sum',
            'pumped hydro conso': 'sum',
            'cross-border trade': 'sum',
            'nuclear': 'sum',
            'run-of-the-river production': 'sum',
            'biomass': 'sum',
            'brown coal': 'sum',
            'black coal': 'sum',
            'oil': 'sum',
            'natural gas': 'sum',
            'geothermal': 'sum',
            'stored hydro production': 'sum',
            'pumped hydro production': 'sum',
            'other': 'sum',
            'waste incineration': 'sum',
            'offshore wind': 'sum',
            'onshore wind': 'sum',
            'solar': 'sum',
            'conso': 'mean',
            'coal gas': 'sum',
            'CO2 price GER': 'mean',
            'CO2 price EU': 'mean',
            'biomass capacity': 'mean',
            'offshore wind capacity': 'mean',
            'onshore wind capacity': 'mean',
            'solar capacity': 'mean',
            'battery storage (nominal power)': 'mean',
            'pumped hydro capacity': 'mean',
            'brown coal capacity': 'mean',
            'black coal capacity': 'mean',
            'oil capacity': 'mean',
            'natural gas capacity': 'mean',
            'other (non-renewable) capacity': 'mean',
            'Spot Gas EEX THE Day Ahead': 'mean',
            'Spot coal': 'mean',
            'wind_mean': 'mean',
            'solar_mean': 'mean',
        }
    }

    train_ds, valid_ds = get_preprocessed_ts_data(
        path="../data/ger_elec_price/energy_ger.parquet",
        features=config['features'],
        targets=config['targets'],
        window_size=config['forecast_history'],
        forecast_horizon=config['forecast_horizon'],
        preprocess_kwargs=preprocess_kwargs,
        stride=config["stride"]
    )

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, pin_memory=torch.cuda.is_available(), generator=_rng_cpu)
    val_loader = DataLoader(valid_ds, batch_size=config['batch_size'], shuffle=False, pin_memory=torch.cuda.is_available(), generator=_rng_cpu)

    model = get_my_model(net_kwargs=config["net_params"], rng=_rng_cpu).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    criterion = nn.MSELoss(reduction="mean")
    batch_tsfm_fn = get_my_batch_tsfm()
    save_model_path = Path(f"../models/{get_my_model_name()}.pth")

    fit(model, train_loader, val_loader, optimizer, criterion, config['epochs'], writer, device, transform_batch_fn=batch_tsfm_fn,
        save_model_path=save_model_path, scheduler=scheduler)
    
    print(f"Loading best model...\n")
    load_model(model_path=save_model_path, model=model, optimizer=optimizer, device=device)

    writer.close()


if __name__ == '__main__':
    main()
