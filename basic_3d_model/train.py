from dataset import Dataset3DMri
from model import FirstRes3DModel
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn 
import torch
import torch.optim as optim
import os

SAFE_CHECKPOINT_PATH = '/content/drive/MyDrive/superresolution_3d_data/models'

def train(
    model_store_name='MyModel',
    learning_rate=1e-4, 
    epochs=10, 
    batch_size=4, 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
):
    dataset = Dataset3DMri()
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    
    model = FirstRes3DModel()
    model = model.to(device)

    loss_criterion = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    os.makedirs(SAFE_CHECKPOINT_PATH, exist_ok=True)

    print(f'STARTING TO TRAIN ON {device}')
    # Iterate the epochs 
    for epoch in range(1, epochs + 1):
        training_visualizer = tqdm(dataloader, leave=True)
        average_loss = 0.0
        # Iterate though training dataset 
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

        average_loss = average_loss / len(dataloader)
        print(f'AVERAGE LOSS OF EPOCH {epoch} : {average_loss}')
        # Safe the latest parameter settings 
        torch.save(model.state_dict(), f'{SAFE_CHECKPOINT_PATH}/{model_store_name}')
    
    print("TRAINING IS COMPLETED")

if __name__ == "__main__":
    train()


