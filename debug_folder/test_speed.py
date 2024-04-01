import torch
import time
from tqdm import tqdm
# Size of the tensor (e.g., 1 million floats)
N = 1

tensor = torch.randn(512, 256, 256, 3)
tensor = tensor.cuda()

# Create a tensor with N elements
tensor = torch.randn(256, 256, 256, 3)

# Start the timer
start_time = time.time()
with torch.no_grad():
    for i in tqdm(range(1)):
# Move the tensor to GPU
        tensor = tensor.cuda()
        tensor = tensor.cpu()
        
    

# Measure the time taken
elapsed_time = (time.time() - start_time)/1

print(f"Time to transfer {N} elements to GPU: {elapsed_time:.6f} seconds")

# Calculate the transfer speed in elements per second
speed = N / elapsed_time
print(f"Transfer speed: {speed:.2f} elements/second")