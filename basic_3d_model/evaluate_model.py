import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from model import FirstRes3DModel
import dataset as ds

def make_shape_5d(image): 
    while image.ndim < 5: 
        image = image.unsqueeze(0)
    return image

@torch.no_grad()
def evaluate_model(model, validation_dataset: Dataset, name='MyModel', 
device= 'cuda' if torch.cuda.is_available() else 'cpu'
):
    
    model = model.to(device).eval()

    dataloader = DataLoader(dataset=validation_dataset)
    
    sample_count = 0
    mse = 0
    mae = 0
    ssim = 0
    lpips = 0

    mse_metric = nn.MSELoss()
    mae_metric = nn.L1Loss()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
    

    for lr_image, hr_image in dataloader: 
        # Ensure the images have a 5D shape (Batch, Channel, Depth, Height, Width)
        lr_image = make_shape_5d(lr_image)
        hr_image = make_shape_5d(hr_image)
        # Move to device 
        lr_image = lr_image.to(device)
        hr_image = hr_image.to(device)

        # Calculate SR Result
        sr_image = model(lr_image)

        mse += mse_metric(sr_image, hr_image).item()
        mae += mae_metric(sr_image, hr_image).item()
        ssim += ssim_metric(sr_image, hr_image).item()
        
        sample_count += 1
    
    # Calculate average 
    mse = mse / sample_count
    mae = mae / sample_count
    ssim = ssim / sample_count
    # Print results 
    print(f'####### {name} VALIDATION RESULTS')
    print(f'MSE: {mse}')
    print(f'MAE: {mae}')
    print(f'SSIM: {ssim}')

if __name__ == '__main__': 
    dataset = ds.Dataset3DMri(
        paths=ds.generate_paths(start_index=3, end_index=4)
    )

    model = FirstRes3DModel()
    checkpoint = torch.load('/content/drive/MyDrive/superresolution_3d_data/models/MyModel')
    model.load_state_dict(checkpoint)
    evaluate_model(model=model, validation_dataset=dataset)

        

        
        

