from enum import  Enum

from dmg.models.neural_networks import AnnModel, LstmModel


class Test11(Enum):
    TT = 'tt'

import torch

model1 = AnnModel(nx=34, ny=16*24, hidden_size=512)
num_parameters1 = sum(p.numel() for p in model1.parameters())

model2 = LstmModel(nx=37, ny=13*16, hidden_size=256)
num_parameters2 = sum(p.numel() for p in model2.parameters())