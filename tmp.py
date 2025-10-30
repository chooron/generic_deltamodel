import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ---------- 高斯平滑层 ----------
class GaussianSmoother(nn.Module):
    def __init__(self, channels, kernel_size=15, sigma=3.0):
        super().__init__()
        half = kernel_size // 2
        x = torch.arange(-half, half + 1).float()
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size).repeat(channels, 1, 1)
        self.register_buffer("kernel", kernel)

    def forward(self, x):
        pad = self.kernel.size(-1) // 2
        return F.conv1d(x, self.kernel, padding=pad, groups=x.size(1))

# ---------- 构造高频抖动信号 ----------
torch.manual_seed(0)
L = 365
t = torch.linspace(0, 8 * torch.pi, L)
trend = 0.5 + 0.3 * torch.sin(t)                       # 平滑主趋势
noise = 0.5 * torch.randn(L)                           # 高频强扰动
x = torch.clamp(trend + noise, 0.0, 1.0)               # 模拟参数区间 [0,1]
x = x.unsqueeze(0).unsqueeze(0)                        # (1, 1, L)

# ---------- 应用高斯平滑 ----------
smoother = GaussianSmoother(channels=1, kernel_size=15, sigma=3.0)
y = smoother(x)

# ---------- 可视化 ----------
plt.figure(figsize=(10,3))
plt.plot(x.squeeze().numpy(), color='gray', alpha=0.5, label="原始高频信号")
plt.plot(y.squeeze().detach().numpy(), color='red', linewidth=2, label="高斯平滑后")
plt.legend()
plt.xlabel("时间步"); plt.ylabel("参数值"); plt.title("高频水文参数平滑示例")
plt.tight_layout(); plt.show()
