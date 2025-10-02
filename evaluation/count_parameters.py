import sys
sys.path.append("/content/superresolution_3d/models/dense_net")
from basic_efficient_dense_net import BasicEfficientDenseNet
from basic_efficient_dense_net_2d import BasicEfficientDenseNet2d

model_2d = BasicEfficientDenseNet2d(num_dense_blocks=8, num_units_per_dense_block=8)
model_3d = BasicEfficientDenseNet(num_dense_blocks=8, num_units_per_dense_block=8)

num_params_2d = sum(p.numel() for p in model_2d.parameters() if p.requires_grad)
num_params_3d = sum(p.numel() for p in model_3d.parameters() if p.requires_grad)

num_all_params_2d = sum(p.numel() for p in model_2d.parameters())
num_all_params_3d = sum(p.numel() for p in model_3d.parameters())

print(f"TRAINIERBARE PARAMETER 2D-MODELL: {num_params_2d}")
print(f"GESAMTZAHL PARAMETER 2D-MODELL: {num_all_params_2d}")
print(f"TRAINIERBARE PARAMETER 3D-MODELL: {num_params_3d}")
print(f"GESAMTZAHL PARAMETER 3D-MODELL: {num_all_params_3d}")


