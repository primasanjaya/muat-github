import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

torch.cuda.current_device()

torch.cuda.device(0)
torch.cuda.device_count()

print(torch.cuda.get_device_name(0))

print('cuda torch available=' + str(torch.cuda.is_available()))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')


print(torch.rand(10).to(device))
print(torch.rand(10, device=device))