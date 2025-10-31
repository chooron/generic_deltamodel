import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_MDN(nn.Module):
    def __init__(self, input_size, hidden_size=128, n_components=5, n_params=3, n_samples=16):
        super().__init__()
        self.n_components = n_components
        self.n_params = n_params
        self.n_samples = n_samples

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        gmm_out_size = n_components * 3
        self.gmm_heads = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, gmm_out_size)
        ) for _ in range(n_params)])

    def forward(self, x, tau=1.0):  # tau为超参，可退火
        B, T, _ = x.shape
        lstm_out, _ = self.lstm(x)

        params_samples = []
        for i in range(self.n_params):
            flat_out = lstm_out.reshape(-1, lstm_out.shape[-1])
            gmm_raw = self.gmm_heads[i](flat_out)

            mu_raw, sigma_raw, pi_raw = gmm_raw.chunk(3, dim=-1)
            mu = mu_raw.reshape(B, T, self.n_components)
            sigma = F.softplus(sigma_raw.reshape(B, T, self.n_components)) + 1e-5
            pi = F.softmax(pi_raw.reshape(B, T, self.n_components), dim=-1)

            # Gumbel-Softmax采样（vectorized）
            pi_exp = pi.unsqueeze(2).expand(-1, -1, self.n_samples, -1)  # (B, T, 16, K)
            logits = torch.log(pi_exp + 1e-10)
            one_hot = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)

            mu_exp = mu.unsqueeze(2).expand(-1, -1, self.n_samples, -1)
            sigma_exp = sigma.unsqueeze(2).expand(-1, -1, self.n_samples, -1)

            mu_k = torch.sum(one_hot * mu_exp, dim=-1)  # (B, T, 16)
            sigma_k = torch.sum(one_hot * sigma_exp, dim=-1)

            eps = torch.randn_like(mu_k)
            samples = mu_k + sigma_k * eps

            params_samples.append(samples.unsqueeze(-1))

        all_samples = torch.cat(params_samples, dim=-1)  # (B, T, 16, 3)
        return all_samples
    
# 测试代码：检查梯度传播
def test_gradient():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型
    input_size = 10  # 示例输入维度
    model = LSTM_MDN(input_size=input_size).to(device)

    # 生成随机输入
    batch_size = 4
    time_steps = 5
    x = torch.randn(batch_size, time_steps, input_size).to(device)
    x.requires_grad_(True)  # 可选：如果需要检查输入梯度

    # 前向传播
    tau = 1.0  # 示例温度
    all_samples = model(x, tau=tau)  # 输出 (4, 5, 16, 3)

    # 模拟损失（对输出求和作为伪损失，或用MSE）
    loss = all_samples.sum()  # 简单伪损失，确保梯度流

    # 反向传播
    loss.backward()

    # 检查梯度是否计算出（非None且有值）
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and torch.any(param.grad != 0):
            print(f"参数 {name} 有梯度: {param.grad.norm().item()}")
            has_grad = True
        else:
            print(f"参数 {name} 无梯度或为零")

    if has_grad:
        print("梯度计算成功！模型可端到端训练。")
    else:
        print("梯度计算失败，请检查模型。")

# 运行测试
if __name__ == "__main__":
    test_gradient()