import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


# -----------------------------------------------------------------------------
# 1. GMM 模块 (与之前讨论的一致)
# -----------------------------------------------------------------------------
class GMM(nn.Module):
    """Gaussian Mixture Density Network Head"""

    def __init__(self, n_in: int, n_out: int, n_hidden: int = 100):
        super(GMM, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass returns the raw, flattened output for the mixture components.
        Reshaping and activation will be handled by the main model.
        """
        h = torch.relu(self.fc1(x))
        h = self.fc2(h)
        return h


# -----------------------------------------------------------------------------
# 2. LSTM + GMM 组合模型
# -----------------------------------------------------------------------------
class LSTM_MDN(nn.Module):
    """
    A complete LSTM-MDN model that combines an LSTM feature extractor
    with a GMM head for probabilistic forecasting.

    Parameters
    ----------
    input_size : int
        Number of features in the input sequence (e.g., meteorological drivers).
    hidden_size : int
        Number of features in the LSTM hidden state.
    n_params : int
        The dimensionality of the target variable we are predicting.
        (e.g., if you predict 4 HBV parameters, n_params=4).
    n_components : int
        The number of Gaussian components to mix for the distribution.
    """

    def __init__(self, input_size: int, hidden_size: int, n_params: int, n_components: int):
        super(LSTM_MDN, self).__init__()
        self.n_params = n_params
        self.n_components = n_components

        # LSTM Layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)

        # Calculate the required output size for the GMM head.
        # For each component, we need n_params for mu, n_params for sigma, and 1 for pi.
        gmm_out_size = self.n_components * (self.n_params * 2 + 1)

        # GMM Head Layer
        self.gmm_head = GMM(n_in=hidden_size, n_out=gmm_out_size)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Performs the forward pass to get the GMM distribution parameters.
        The output is a dictionary containing the reshaped mu, sigma, and pi tensors.
        """
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)

        # gmm_out shape: (batch_size, seq_len, gmm_out_size)
        gmm_out = self.gmm_head(lstm_out)

        # Split the output and reshape into meaningful components
        mu, sigma, pi = gmm_out.split([self.n_components * self.n_params,
                                       self.n_components * self.n_params,
                                       self.n_components], dim=-1)

        batch_size, seq_len, _ = mu.shape

        # Reshape to (batch, seq, n_components, n_params)
        mu = mu.view(batch_size, seq_len, self.n_components, self.n_params)
        # Apply activation to sigma and reshape
        sigma = torch.exp(sigma.view(batch_size, seq_len, self.n_components, self.n_params))
        # Apply softmax to pi and reshape
        pi = torch.softmax(pi.view(batch_size, seq_len, self.n_components), dim=-1)

        return {'mu': mu, 'sigma': sigma, 'pi': pi}

    def sample(self, dist_params: Dict[str, torch.Tensor], constrain: bool = False) -> torch.Tensor:
        """
        Performs a single differentiable sampling from the GMM distribution.
        """
        mu = dist_params['mu']
        sigma = dist_params['sigma']
        pi = dist_params['pi']

        # Differentiably choose a component using Gumbel-Softmax
        eps = 1e-10
        logits = torch.log(pi + eps)
        idx_one_hot = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)

        # Select the mu and sigma of the chosen component
        idx_expanded = idx_one_hot.unsqueeze(-1)
        mu_selected = torch.sum(mu * idx_expanded, dim=-2)
        sigma_selected = torch.sum(sigma * idx_expanded, dim=-2)

        # Apply the Reparameterization Trick
        epsilon = torch.randn_like(sigma_selected)
        sample = mu_selected + epsilon * sigma_selected

        if constrain:
            sample = torch.sigmoid(sample)

        return sample

    def sample_n(self, dist_params: Dict[str, torch.Tensor], n_samples: int, constrain: bool = False) -> torch.Tensor:
        """
        Generates N samples from the GMM distribution by calling .sample() n times.

        Parameters
        ----------
        dist_params : Dict[str, torch.Tensor]
            Dictionary of distribution parameters from the forward pass.
        n_samples : int
            The number of samples to generate (e.g., 16).
        constrain : bool
            Whether to apply the sigmoid constraint to the samples.

        Returns
        -------
        torch.Tensor
            A tensor of shape (n_samples, batch_size, seq_len, n_params).
        """
        samples_list = [self.sample(dist_params, constrain) for _ in range(n_samples)]

        # Stack the samples along a new dimension (the first one)
        return torch.stack(samples_list, dim=0)


# -----------------------------------------------------------------------------
# 3. 示例：如何使用模型生成16个采样结果
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # --- 模型超参数 ---
    BATCH_SIZE = 4  # 批处理大小 (例如4个不同的流域或数据段)
    SEQ_LEN = 100  # 输入序列长度 (例如100天)
    INPUT_FEATS = 10  # 输入特征数量 (例如10个气象变量)
    LSTM_HIDDEN = 64  # LSTM隐藏层大小
    N_HBV_PARAMS = 4  # 要预测的HBV参数数量
    N_COMPONENTS = 5  # GMM中的高斯混合 komponent 数量
    N_SAMPLES = 16  # **我们需要的采样数量**

    # --- 1. 实例化模型 ---
    print("Initializing model...")
    model = LSTM_MDN(
        input_size=INPUT_FEATS,
        hidden_size=LSTM_HIDDEN,
        n_params=N_HBV_PARAMS,
        n_components=N_COMPONENTS
    )
    print(model)

    # --- 2. 创建模拟输入数据 ---
    # 形状: (batch_size, sequence_length, input_features)
    input_data = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_FEATS)
    print(f"\nShape of input data: {input_data.shape}")

    # --- 3. 模型推理与采样 ---
    print("\nPerforming inference and sampling...")
    # 在推理模式下，我们不需要计算梯度
    model.eval()
    with torch.no_grad():
        # Step 3a: 通过模型前向传播，得到描述参数分布的字典
        distribution_parameters = model(input_data)

        # Step 3b: 调用 sample_n 方法，从分布中采样 N_SAMPLES 次
        # 我们将 constrain 设置为 True, 确保输出在 (0, 1) 之间
        final_samples = model.sample_n(
            distribution_parameters,
            n_samples=N_SAMPLES,
            constrain=True
        )

    # --- 4. 验证输出结果 ---
    print("\n--- Verification ---")
    print(f"Final output shape: {final_samples.shape}")
    print(f"Expected shape: ({N_SAMPLES}, {BATCH_SIZE}, {SEQ_LEN}, {N_HBV_PARAMS})")

    # 检查数值范围
    min_val = final_samples.min()
    max_val = final_samples.max()
    print(f"Sampled values range: [{min_val:.4f}, {max_val:.4f}] (constrained by sigmoid)")

    print("\n--- Interpretation ---")
    print(f"The final tensor `final_samples` now holds {N_SAMPLES} different potential parameter sets")
    print(f"for each of the {SEQ_LEN} time steps, for each of the {BATCH_SIZE} items in the batch.")
    print("This tensor is ready to be fed into your HBV model ensemble.")