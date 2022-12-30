import torch
import torch.nn.functional as F

def convert_multi_step_actions(x: torch.tensor, predicted_steps: int, num_actions: int=6):
    res = torch.ones(predicted_steps, *x.shape) * (-100) # ignore index: -100

    res[0] = x.clone()
    for i in range(1, predicted_steps):
        res[i][:-i] = x.clone()[i:]
    
    return res.permute(1, 2, 0, 3).to(x.device)