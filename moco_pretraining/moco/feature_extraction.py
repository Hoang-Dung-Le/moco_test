import torch
import torch.nn as nn
import torch.optim as optim

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])