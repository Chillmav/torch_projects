import torch
import torch.nn as nn

class SoftmaxClassificationModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 3)
        
    def softmax_layer(self, inputs: torch.Tensor):

        exp = inputs.exp()

        return exp / (exp.sum(dim=1, keepdim=True) + 1e-6)


    def forward(self, x):

        out = self.lin(x)
        logits = self.softmax_layer(out)
        return logits






