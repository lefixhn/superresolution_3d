from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn 
import torch
import torch.optim as optim
import os

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
    model_store_name=None,
    models_path=DEFAULT_MODELS_PATH,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
):
    # Prepare DATALOADER, MODEL and OPTIMIZER
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=dataloader_num_workers, persistent_workers=True)
    model = model.to(device)
    if optimizer is None: 
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare PATHS and FOLDERS
    if model_store_name is None: 
        model_store_name = type(model).__name__
    # Create folder for model 
    model_path = os.path.join(models_path, model_store_name)
    os.makedirs(checkpoints_path, exist_ok=True)
    # Create sub folder in model folder for checkpoints 
    checkpoints_path = os.path.join(model_path, 'checkpoints')
    ch = os.path.join()
    os.makedirs(checkpoints_path, exist_ok=True)

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
        average_loss = average_loss / len(dataloader)
        print(f'AVERAGE LOSS OF EPOCH {epoch} : {average_loss}')
        # Safe the latest parameter settings 
        torch.save(model.state_dict(), f'{SAFE_CHECKPOINT_PATH}/{model_store_name}')
    # AFTER TRAINING
    print("TRAINING IS COMPLETED")

if __name__ == "__main__":
    train()
