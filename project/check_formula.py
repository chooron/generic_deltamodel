import torch

from dmg.models.phy_models.blendv2c import *

import matplotlib.pyplot as plt
import numpy as np

soil = torch.tensor(50.0)
param_dict = {
    'lamb': torch.tensor(8.0),
    'quickflow_k': torch.tensor(-0.8),
    'mmax': torch.tensor(50.0),
    'quickflow_n': torch.tensor(2.0),
    'vadose_max_level': torch.tensor(200.0),
}
qf1 = quickflow_linear(soil, param_dict)
qf2 = quickflow_topmodel(soil, soil / 200.0, param_dict)
qf3 = quickflow_vic(soil, soil / 200.0, param_dict)
