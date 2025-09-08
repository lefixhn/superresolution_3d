from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn 
import torch
import torch.optim as optim
import os
import re
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

    if train_from_last_checkpoint: 
        last_checkpoint_path= _find_latest_ckeckpoint_dir(checkpoints_path)
        checkpoint = torch.load(last_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint: 
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        

    print(f'STARTING TO TRAIN {model_store_name} ON {device}')
    # Iterate through epochs 
    for epoch in range(1, epochs + 1):
        training_visualizer = tqdm(dataloader, leave=True)
        average_loss = 0.0
        # Iterate trough minibatches
        for lr_image, hr_image in dataloader: 
            # Inside this loop entire batches are handled, not just images
            # Moves data to GPU if available 
            lr_image = lr_image.to(device)
            hr_image = hr_image.to(device)
            
            result_image = model(lr_image)
            loss = loss_criterion(result_image, hr_image)

            # Delete old gradient 
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            average_loss += loss.item()
            if epoch % 40 == 0: 
                training_visualizer.set_description(f'EPOCH {epoch}/{epochs+1}')
                training_visualizer.set_postfix(loss=loss.item())
            # AFTER MINIBATCH
        # AFTER EPOCH 
        # TODO: Wie kann ich hier falls vorhanden validieren und werte speichern
        average_loss = average_loss / len(dataloader)
        print(f'AVERAGE LOSS OF EPOCH {epoch} : {average_loss}')
        # Safe the latest parameter settings 
        # TODO: Wie muss hier die dateiendung sein? 
        torch.save(model.state_dict(), f'{checkpoints_path}.')
    # AFTER TRAINING
    # TODO: Wie kann ich hier eine trainingsgrafik erstellen und im modelordner sichern?
    
    print("TRAINING IS COMPLETED")

def _find_latest_ckeckpoint_dir(checkpoints_path: str) -> Optional[str]: 
    checkpoints = os.listdir(checkpoints_path)
    if len(checkpoints) == 0: 
        return None
    else: 
        # Find highest checkpoint 
        checkpoints.sort(key=lambda checkpoint: int(re.findall(r"\d+", checkpoint)))
        return os.path.join(checkpoints_path, checkpoints[-1])

