import torch


class CrossLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()


    def loss_fn(self, y, pred):

        return - (y * pred.log()).sum()

    def forward(self, y, pred):

        return self.loss_fn(y, pred)

