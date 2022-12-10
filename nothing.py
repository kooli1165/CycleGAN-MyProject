import torch
import numpy as np

a = torch.tensor(0.8).max(0).min(0.6)
print(a)