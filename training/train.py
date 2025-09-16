from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn 
import torch
import torch.optim as optim
import os
import re
import csv
from typing import Optional 

DEFAULT_MODELS_PATH = '/content/drive/MyDrive/superresolution_3d_data/models'

def train(
    dataset, 
    model, 
    validation_dataset=None, 
    learning_rate=1e-4, 
    epochs=10, 
    batch_size=4, 
    optimizer=None, 
    dataloader_num_workers=2,
    loss_criterion = nn.L1Loss(), 
    train_from_last_checkpoint=False, 
    model_store_name=None,
    epochs_per_checkpoint: int = 1, 
    models_path=DEFAULT_MODELS_PATH,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
):
    '''
    This method creates a sub folder in the models_path directory for the model. 
    It also creates a sub folder for the ckeckpoints, where they are stored. 
    If train_from_last_checkpoint=True it 
    
    '''
    validation_dataloader = None
    if validation_dataset is not None: 
        validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=1, shuffle=False, num_workers=dataloader_num_workers, persistent_workers=True)
    # Prepare DATALOADER, MODEL and OPTIMIZER
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=dataloader_num_workers, persistent_workers=True)
    # TODO: Wie kann ich die Modellparameter bei train_from_last_checkpoint=True laden
    model = model.to(device)
    if optimizer is None: 
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare PATHS and FOLDERS
    if model_store_name is None: 
        model_store_name = type(model).__name__
    # Build paths
    model_path = os.path.join(models_path, model_store_name)
    checkpoints_path = os.path.join(model_path, 'checkpoints')
    # Create paths
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)
    # Build training history path 
    train_history_path = os.path.join(model_path, 'training_history.csv')

    start_epoch = 1 
    # LOAD CHECKPOINT
    if train_from_last_checkpoint: 
        last_checkpoint_path= _find_latest_ckeckpoint_dir(checkpoints_path)
        if last_checkpoint_path is not None:
            checkpoint = torch.load(last_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint: 
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_index = _get_epoch_index(last_checkpoint_path)
            if epoch_index is not None: 
                start_epoch = start_epoch + 1 # We start one epoch further than the last 
        else: 
            print("KEIN CHECKPOINT GEFUNDEN!")
        
    print(f'STARTING TO TRAIN {model_store_name} ON {device}')
    # Iterate through epochs 
    scaler = model.to(device).train()
    for epoch in range(start_epoch, start_epoch + epochs ):
        training_visualizer = tqdm(dataloader, leave=True)
        average_loss = 0.0
        # Iterate trough minibatches
        for lr_image, hr_image in dataloader: 
            # Inside this loop entire batches are handled, not just images
            # Moves data to GPU if available 
            lr_image = lr_image.to(device, non_blocking=True)
            hr_image = hr_image.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(dtyoe=torch.float16)
                sr_image = model(lr_image)                   # Make prediction 
                loss = loss_criterion(sr_image, hr_image)    # Calculate loss 
            # Delete old gradient 
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            average_loss += loss.item()
            
            # AFTER MINIBATCH
        # AFTER EPOCH 
        # TODO: Wie kann ich hier falls vorhanden validieren und werte speichern
        average_loss = average_loss / len(dataloader)
        validation_loss = _evaluate(model, validation_dataloader, loss_criterion, device)
        _append_history_row(train_history_path, epoch, train_loss=average_loss, val_loss=validation_loss)
        print(f'AVERAGE LOSS OF EPOCH {epoch} : {average_loss}')
        
    # AFTER TRAINING
    # TODO: Wie kann ich hier eine trainingsgrafik erstellen und im modelordner sichern?
    print("TRAINING IS COMPLETED")
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        if os.path.exists(train_history_path):
            df = pd.read_csv(train_history_path)
            plt.figure()
            plt.plot(df['epoch'], df['train_loss'], label='train')
            if 'val_loss' in df and df['val_loss'].notna().any():
                plt.plot(df['epoch'], df['val_loss'], label='val')
            plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(model_path, 'loss_curve.png'), dpi=200)
            plt.close()
    except Exception as e:
        print(f"Could not create loss plot: {e}")





def _get_epoch_index(checkpoint_path: str) -> Optional[int]: 
    '''
    /Models/Mymodel2025/checkpoints/epoch219.pt -> 219
    '''
    filename = os.path.basename(checkpoint_path)
    matching = re.search(r"\d+", filename)

    if matching: 
        return matching.group(0)
    else: 
        return None

def _find_latest_ckeckpoint_dir(checkpoints_path: str) -> Optional[str]: 
    checkpoints = os.listdir(checkpoints_path)
    if len(checkpoints) == 0: 
        return None
    else: 
        # Find highest checkpoint 
        checkpoints.sort(key=lambda checkpoint: int(re.findall(r"\d+", checkpoint)))
        return os.path.join(checkpoints_path, checkpoints[-1])

def _append_history_row(csv_path: str, epoch: int, train_loss: float, val_loss: Optional[float]):
    header_needed = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        if header_needed:
            w.writerow(['epoch', 'train_loss', 'val_loss'])
        w.writerow([epoch, train_loss, ("" if val_loss is None else val_loss)])

@torch.no_grad()
def _evaluate(model, dataloader, loss_criterion, device) -> Optional[float]:
    if dataloader is None:
        return None
    model.eval()    # Go in eval mode
    total = 0.0
    for lr_image, hr_image in dataloader:
        lr_image = lr_image.to(device)
        hr_image = hr_image.to(device)
        sr_image = model(lr_image)
        total += loss_criterion(sr_image, hr_image).item()
    model.train()   # Go in training mode 
    return total / max(1, len(dataloader))

def _save_checkpoint(ckpt_dir: str, epoch: int, model, optimizer, best_val_loss: float, is_best: bool):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }
    ep_path = os.path.join(ckpt_dir, f'epoch_{epoch:04d}.pt')  # Endung .pt ist üblich
    torch.save(state, ep_path)
    torch.save(state, os.path.join(ckpt_dir, 'last.pt'))
    if is_best:
        torch.save(state, os.path.join(ckpt_dir, 'best.pt'))