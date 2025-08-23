import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure


def reduce_shape_3d(image): 
    while image.ndim > 3: 
        image.shqueeze(0)
    return image

def evaluate_model(model, validation_dataset: Dataset, name='MyModel'):

    dataloader = DataLoader(dataset=dataset)
    
    sample_count = 0
    mse = 0
    mae = 0
    ssim = 0
    lpips = 0

    mse_metric = nn.MSELoss()
    mae_metric = nn.L1Loss()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
    

    for lr_image, hr_image in dataloader: 
        # Reduce to 3D Shape 
        lr_image = reduce_shape_3d(lr_image)
        hr_image = reduce_shape_3d(hr_image)

        # Calculate SR Result
        sr_image = model(lr_image)

        mse += mse_metric(sr_image, hr_image)
        mae += mae_metric(sr_image, hr_image)
        ssim += ssim_metric(sr_image, hr_image)
        
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



        

        
        

