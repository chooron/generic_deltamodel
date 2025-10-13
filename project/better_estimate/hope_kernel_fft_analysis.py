import sys

from matplotlib.colors import TwoSlopeNorm

sys.path.append(r'E:\pycode\generic_deltamodel')
from dmg import ModelHandler
from dmg.core.utils import import_data_loader, import_trainer, set_randomseed
from dmg.models.neural_networks.hope_mlp_v1 import HopeMlpV1
from project.better_estimate import load_config
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
import torch

CONFIG_PATH = r'conf/config_dhbv_hopev1.yaml'
config = load_config(CONFIG_PATH)
config['mode'] = 'test'
config['test']['test_epoch'] = 100
set_randomseed(config['random_seed'])
model = ModelHandler(config, verbose=True)
hope_mlp_model = model.model_dict['Hbv_2'].nn_model
hope_layer = hope_mlp_model.hope_layer

s4_layers = hope_layer.s4_layers
s4_layer_kernel = s4_layers[3].kernel

dt = torch.exp(s4_layer_kernel.log_dt).detach().cpu()
plt.hist(dt, bins=30)
plt.title("Distribution of learned time constants (dt)")
plt.xlabel("dt")
plt.ylabel("count")
plt.show()

A_real = -torch.exp(s4_layer_kernel.log_A_real).detach().cpu()
A_imag = s4_layer_kernel.A_imag.detach().cpu()

plt.scatter(A_real.flatten(), A_imag.flatten(), alpha=0.6)
plt.xlabel("Re(A) (decay rate)")
plt.ylabel("Im(A) (frequency)")
plt.title("Learned Eigenvalues of State Dynamics (A)")
plt.show()

K = s4_layers[0].kernel(L=365).detach().cpu()
plt.plot(K[0].numpy())
plt.title("Impulse Response of S4D Channel 0")
plt.xlabel("Time step")
plt.ylabel("Amplitude")
plt.show()

K_f = torch.fft.rfft(K, dim=-1)
freq = torch.fft.rfftfreq(K.size(-1))
plt.plot(freq, K_f.abs().mean(0))
plt.title("Average Frequency Response of S4D Kernel")
plt.xlabel("Frequency")
plt.ylabel("Gain (|K(f)|)")
plt.show()

A_real = -torch.exp(s4_layer_kernel.log_A_real).detach().cpu()
A_imag = s4_layer_kernel.A_imag.detach().cpu()
plt.scatter(A_real.flatten(), A_imag.flatten(), alpha=0.6)
plt.title("Eigenvalues of State Matrix A (Complex Plane)")
plt.xlabel("Re(A)")
plt.ylabel("Im(A)")
plt.axvline(0, color='gray', linestyle='--')
plt.show()

L = 256
K = s4_layer_kernel(L=L).detach().cpu()  # (H, L)
plt.plot(K[0].numpy())
plt.title("Impulse response (kernel) of first channel")
plt.xlabel("time step")
plt.ylabel("amplitude")
plt.show()

K_f = torch.fft.rfft(K, dim=-1)
freq = torch.fft.rfftfreq(K.size(-1), d=1.0)
plt.plot(freq, K_f.abs().mean(dim=0).numpy())
plt.title("Average frequency response of S4D kernel")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()