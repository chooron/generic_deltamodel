import torch
import pickle
import matplotlib.pyplot as plt
from dmg.models.phy_models.hmets import hmets
from hydrodl2.core.calc import change_param_range


def inv_change_param_range(phys_param: torch.Tensor, bounds: list) -> torch.Tensor:
    """change_param_range 的逆运算：将物理参数从其范围缩放回 [0, 1]."""
    return (phys_param - bounds[0]) / (bounds[1] - bounds[0])


# 注意：logit(x) = log(x / (1 - x))。为避免 x=0或x=1 时出现无穷大，进行截断。
def safe_logit(x: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """torch.logit 的安全版本，避免边界值问题。"""
    x = torch.clamp(x, min=eps, max=1.0 - eps)
    return torch.logit(x)


# --- 0. 准备输入数据 (x_dict) ---
# 所有气象数据现在被整合成一个张量 x_phy，形状为 [time, grids, vars]
with open(r"E:\PaperCode\generic_deltamodel\data\camels_data\camels_dataset", 'rb') as f:
    forcings, target, attributes = pickle.load(f)

real_flow = (10 ** 3) * target[0, 365:365 * 3] * 0.0283168 * 3600 * 24 / (attributes[0, 11] * (10 ** 6))

Prcp = torch.from_numpy(forcings[0, :365 * 3, 0])
Tmean = torch.from_numpy(forcings[0, :365 * 3, 1])
Pet = torch.from_numpy(forcings[0, :365 * 3, 2])

# --- 1. 基本配置 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_steps = Prcp.shape[0]  # 模拟总时长 (天)
n_grid = 1  # 模拟的流域/网格数量 (可以设为多个)
warm_up_days = 365  # 设置预热期 (例如，第一年)

print(f"--- Running HMETS model on device: {device} ---")
print(f"Simulation for {n_grid} grid(s) over {n_steps} days, with {warm_up_days} warm-up days.")

# --- 1. 实例化模型 ---
# 新的初始化方式：使用配置字典
model_config = {
    'warm_up': warm_up_days,
    "nmul": n_grid,
    # 'dynamic_params': {'HMETS': ['ET_efficiency']} # (可选) 演示如何设置动态参数
}
hmet_model = hmets(config=model_config, device=device).to(device)

# b. 按 hmet_model.variables 定义的顺序堆叠变量
# hmet_model.variables -> ['Tmin', 'Tmax', 'Prcp', 'Pet']
x_phy_list = [Prcp, Tmean, Pet]
x_phy = torch.stack(x_phy_list, dim=1).to(device)  # Shape: [time, vars]

# c. 扩展以匹配 [time, grids, vars] 的形状
x_phy = x_phy.unsqueeze(1).repeat(1, n_grid, 1)

# e. 创建最终的输入字典 x_dict
x_dict = {'x_phy': x_phy}
print(f"\nInput data 'x_phy' prepared with shape: {x_phy.shape}")

# --- 3. 准备参数张量 (parameters) ---
# 这是关键区别：我们需要创建一个模拟NN输出的、未经处理的原始参数张量
# 步骤: 物理值 -> [0,1] 归一化 -> logit变换 -> 组合成一个张量

base_phy_values = torch.tensor([
    # Snow(11), ET(1), Subsurface(6) - 总共18个物理参数
    1.5, 3.0, 0.5, 0.002, 0.03, 0.07, 0.001, -0.5, 1.5, 1.2,  # Snow
    0.9,  # ET
    0.1, 0.05, 0.02, 0.01, 500.0, 1000.0  # Subsurface
], device=device)

phy_values_per_member = []
for i in range(hmet_model.nmul):
    # 创建轻微不同的参数集
    perturbation = torch.randn(base_phy_values.shape[0], device=device) * 0.02 * (i + 1)
    phy_values_per_member.append(base_phy_values + perturbation)

# 路由参数是共享的，只需一套
rout_values = torch.tensor([1.5, 0.2, 2.5, 0.05], device=device)

# ✅ b. 将每个成员的物理值转换为 logit 空间，并进行重组成“扁平”向量
logit_phy_per_member = []
for i in range(hmet_model.nmul):
    norm_phy = torch.zeros_like(base_phy_values)
    for p_idx, name in enumerate(hmet_model.phy_param_names):
        bounds = hmet_model.parameter_bounds[name]
        norm_phy[p_idx] = inv_change_param_range(phy_values_per_member[i][p_idx], bounds)
    logit_phy_per_member.append(safe_logit(norm_phy))

# 将多个成员的 logit 参数堆叠起来
# Shape: [phy_param_count, nmul]
stacked_logit_phy = torch.stack(logit_phy_per_member, dim=1)

# 转置并拉平，得到 [p1_m1, p1_m2, p1_m3, p2_m1, p2_m2, p2_m3, ...] 的结构
# Shape: [phy_param_count * nmul]
flat_logit_phy = stacked_logit_phy.T.flatten()

# ✅ c. 转换路由参数 (只需一次)
norm_rout_params = torch.zeros_like(rout_values)
for i, name in enumerate(hmet_model.routing_param_names):
    bounds = hmet_model.routing_parameter_bounds[name]
    norm_rout_params[i] = inv_change_param_range(rout_values[i], bounds)
logit_rout = safe_logit(norm_rout_params)

# ✅ d. 组合并扩展为最终形状
all_logit_params = torch.cat([flat_logit_phy, logit_rout])
parameters = all_logit_params.unsqueeze(0).unsqueeze(0).repeat(n_steps, n_grid, 1)

expected_len = (len(hmet_model.phy_param_names) * hmet_model.nmul) + len(hmet_model.routing_param_names)
print(f"Total length of parameter vector per grid: {parameters.shape[-1]} (Expected: {expected_len})")
assert parameters.shape[-1] == expected_len, "Parameter length mismatch!"
print(f"Raw 'parameters' tensor prepared with shape: {parameters.shape}")

# --- 4. 运行模型 ---
print("\nRunning HMETS forward pass...")
out_dict = hmet_model(x_dict, parameters)
print("Forward pass completed.")

# --- 5. 解析并展示结果 ---
# 返回的是一个字典，包含多个输出变量
# 输出形状: [time_after_warmup, grids, 1]
simulated_flow = out_dict['streamflow']
actual_et = out_dict['AET_hydro']
vadose_storage = out_dict['vadose_storage']

print(f"\nOutput streamflow tensor shape: {simulated_flow.shape}")
print("Simulated streamflow for first 10 days after warm-up (m^3/s):")
# 我们只看第一个流域的结果 (grid index 0)
print(simulated_flow[:10, 0, 0])

# 绘图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# 提取第一个流域的数据用于绘图
flow_to_plot = simulated_flow[:, 0, 0].detach().cpu().numpy()
aet_to_plot = actual_et[:, 0, 0].detach().cpu().numpy()
storage_to_plot = vadose_storage[:, 0, 0].detach().cpu().numpy()

# 图1: 流量
ax1.plot(flow_to_plot, label='Simulated Flow ($m^3/s$)', color='b')
ax1.plot(real_flow, label='Real Flow ($m^3/s$)', color='r')
ax1.set_title('HMETS Simulated Streamflow')
ax1.set_ylabel('Streamflow ($m^3/s$)')
ax1.legend()
ax1.grid(True)

# 图2: 其他变量
ax2.plot(aet_to_plot, label='Actual ET (mm)', color='g', linestyle='--')
ax2_twin = ax2.twinx()
ax2_twin.plot(storage_to_plot, label='Vadose Storage (mm)', color='r', linestyle=':')
ax2.set_title('Other Simulated Variables')
ax2.set_xlabel(f'Days after {warm_up_days}-day warm-up')
ax2.set_ylabel('Evapotranspiration (mm)')
ax2_twin.set_ylabel('Storage (mm)')

# 合并图例
lines, labels = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2_twin.legend(lines + lines2, labels + labels2, loc=0)
ax2.grid(True)

plt.tight_layout()
plt.show()
