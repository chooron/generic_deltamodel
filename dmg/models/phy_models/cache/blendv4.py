# blendv3利用最新的算法，
# 其中动态参数为每个公式的权重，静态参数为其余的参数值, 共计35个参数值
# 模型的权重值是通过softmax进行处理
from functools import reduce
from typing import Any, Optional, Union

from torch import clamp
import torch
import torch.nn as nn
from hydrodl2.core.calc import change_param_range, uh_conv, uh_gamma
from dmg.models.phy_models.fluxes import *


class RandomSelectorAndMean(nn.Module):
    """
    一个 PyTorch 模块，用于对输入的二维张量的每一行进行随机选择，然后计算被选中元素的平均值。

    对于输入的 (N, C) 张量中的每一行，它会随机选择 k 个元素，
    其中 k 的值在 [min_k, max_k] 之间随机确定。
    最后，它会返回一个 (N, 1) 的张量，其中每个元素是对应行中被选中元素的平均值。
    """

    def __init__(self, min_k: int, max_k: int):
        """
        初始化模块。

        Args:
            min_k (int): 每行最少选择的元素数量。
            max_k (int): 每行最多选择的元素数量。
        """
        super().__init__()
        if not (isinstance(min_k, int) and isinstance(max_k, int) and min_k > 0 and min_k <= max_k):
            raise ValueError("min_k 和 max_k 必须是正整数，并且 min_k <= max_k")

        self.min_k = min_k
        self.max_k = max_k
        self.num_choices = self.max_k - self.min_k + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        模块的前向传播逻辑。

        Args:
            x (torch.Tensor): 输入的二维张量，形状为 (batch_size, num_features)。

        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, 1)，代表每行的平均值。
        """
        if x.dim() != 2:
            raise ValueError(f"输入张量必须是二维的, 但收到了 {x.dim()} 维")

        batch_size, num_features = x.shape

        if self.max_k > num_features:
            raise ValueError(f"max_k ({self.max_k}) 不能大于特征数量 ({num_features})")

        # --- 以下步骤与之前相同，用于创建掩码 ---
        k_per_row = torch.randint(low=0, high=self.num_choices, size=(batch_size, 1), device=x.device) + self.min_k
        scores = torch.rand(x.shape, device=x.device, dtype=x.dtype)
        sorted_scores, _ = torch.sort(scores, dim=1, descending=True)
        thresholds = torch.gather(sorted_scores, dim=1, index=k_per_row - 1)
        mask = scores >= thresholds

        # --- 新增的求平均步骤 ---

        # 5. 应用掩码，将被mask掉的元素置为0
        selected_values = x * mask

        # 6. 计算每行被选中元素的总和
        sum_per_row = selected_values.sum(dim=1)

        # 7. 计算每行被选中元素的数量
        # 由于mask是布尔型，直接求和即可得到True的数量
        # 这个结果等价于 k_per_row.squeeze()
        count_per_row = mask.sum(dim=1)

        # 8. 计算平均值
        # 因为我们保证了 min_k > 0，所以 count_per_row 不会为0，
        # 但为安全起见，加上一个 epsilon 是很好的习惯。
        epsilon = 1e-8
        mean_per_row = sum_per_row / (count_per_row.float() + epsilon)

        # 9. 将结果形状从 (batch_size,) 调整为 (batch_size, 1) 并返回
        return mean_per_row.unsqueeze(1)

    def __repr__(self):
        return f"{self.__class__.__name__}(min_k={self.min_k}, max_k={self.max_k})"


class DropoutMean(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        dropout_result = self.dropout(x)
        sum_result = torch.sum(dropout_result, dim=1)
        nonzero_counts = (dropout_result != 0).sum(dim=1)
        correct_mean = sum_result / (nonzero_counts + 1e-8)
        return correct_mean.unsqueeze(1)


class blendv4(torch.nn.Module):
    """
    blend 模型 based on (HMETS, HBV and VIC)

    这是一个功能完整的、可微分的 PyTorch blend 模型，支持状态预热和动态参数。

    Parameters
    ----------
    config : dict, optional
        配置字典，用于覆盖默认设置。
    device : torch.device, optional
        模型运行的设备（'cpu' 或 'cuda'）。
    """

    def __init__(
            self,
            config: Optional[dict[str, Any]] = None,
            device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.name = 'Blend'
        self.config = config
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # --- 从配置或默认值设置模型属性 ---
        self.warm_up = 0
        self.warm_up_states = True
        self.pred_cutoff = 0
        self.dynamic_params = []
        self.dy_drop = 0.0
        # Blend 需要的输入变量
        self.variables = ['Prcp', 'Tmean', 'Pet']
        self.routing = True
        self.initialize = False
        self.nmul = 1
        self.nearzero = 1e-6
        self.infil_dropout = DropoutMean(p=0.5)
        self.bf1_dropout = DropoutMean(p=0.5)
        self.bf2_dropout = DropoutMean(p=0.5)

        self.parameter_bounds = {
            # Infiltration (入渗)
            'inf_pc': [0.0, 1.0],  # [Inf-PC] Partitioning Coefficient
            'inf_hbv_beta': [0.5, 3.0],  # [Inf-HBV] HBV Method beta
            'inf_vic_bexp': [0.001, 3.0],  # [Inf-VICARNO] VIC-ARNO Method bexp
            'inf_hmets_alpha': [0.3, 1.0],  # [Inf-HMETS] HMETS Method alpha

            # Baseflow Soil 1 (土壤层1基流)
            'bf1_gr4j_x3': [20.0, 300.0],  # [BF1-GR4J] GR4J Method X3-1
            'bf1_bfc': [-3.0, -0.0],  # [BF1-PL] Power Law Baseflow BFc1
            'bf1_bfn': [1.0, 5.0],  # [BF1-PL] Power Law Baseflow BFn1
            'bf1_bfmax': [0.1, 200.0],  # [BF1-VIC] VIC Method BFmax1
            'bf1_lam': [1.0, 20.0],  # [BF1-TOPMOD] TOPMODEL Method lam1
            'bf1_thresh': [0.0001, 0.9999],  # [BF1-THRESH] Threshold Baseflow BFThresh1

            # Percolation (渗漏)
            'max_perc': [1.0, 50.0],  # [Perc] GAWSER Method MaxPerc
            'perc_sfc': [0.0001, 0.9999],  # [Perc] saturation at field capacity

            # Capillary Rise (毛管上升)
            'crise_hbv': [0.1, 50.0],  # [CRise] HBV Method CRise

            # Baseflow Soil 2 (土壤层2基流)
            'bf2_bfmax': [0.1, 100.0],  # [BF2-CON] Constant Rate BFMax2
            'bf2_gr4j_x3': [20.0, 300.0],  # [BF2-GR4J] GR4J Method X3-2
            'bf2_bfc': [-3.0, -0.0],  # [BF2-LIN] Linear Baseflow BFc2
            'bf2_bfn': [1.0, 5.0],  # [BF2-PL] Power Law Baseflow BFn2

            # other fluxes
            'Tbf': [-5.0, 2.0], 'Kf': [0.0, 5.0],  # Refreeze Hbv
            'ddf_min': [1.5, 3.0], 'ddf_plus': [0.0, 5.0], 'Kcum': [0.01, 0.2], 'Tbm': [-1.0, 1.0],  # snowmelt Hbv
            'swi': [0.0, 0.4],  # overflow Hbv
            'forest_cover': [0.0, 1.0], 'forest_sparseness': [0.0, 1.0],
            'max_soilwater1': [50.0, 500.0], 'max_soilwater2': [50.0, 500.0],
            # routing
            'alpha1': [0.3, 20.0], 'beta1': [0.01, 5.0],
            'alpha2': [0.5, 13.0], 'beta2': [0.15, 1.5],
        }

        if config is not None:
            self.warm_up = config.get('warm_up', self.warm_up)
            self.warm_up_states = config.get('warm_up_states', self.warm_up_states)
            self.dy_drop = config.get('dy_drop', self.dy_drop)
            self.dynamic_params = config.get('dynamic_params', {}).get(self.__class__.__name__, self.dynamic_params)
            self.variables = config.get('variables', self.variables)
            self.nearzero = config.get('nearzero', self.nearzero)

        self.phy_param_names = list(self.parameter_bounds.keys())
        self.learnable_param_count = len(self.phy_param_names) * self.nmul

    def unpack_parameters(self, parameters: torch.Tensor) -> torch.Tensor:
        # 从神经网络输出中提取物理参数
        phy_param_count = len(self.parameter_bounds)
        # Physical parameters
        phy_params = torch.sigmoid(parameters.view(
            parameters.shape[0], phy_param_count, self.nmul,
        ))
        return phy_params

    def descale_phy_parameters(self, phy_params: torch.Tensor) -> dict:
        """将归一化的物理参数去归一化到其物理范围。"""
        phy_param_dict = {}
        for i, name in enumerate(self.phy_param_names):
            phy_param_dict[name] = change_param_range(phy_params[:, i, :], self.parameter_bounds[name])
            if name in ['alpha1', 'beta1', 'alpha2', 'beta2']:
                phy_param_dict[name] = phy_param_dict[name].unsqueeze(2)
        return phy_param_dict

    def forward(
            self,
            x_dict: dict[str, torch.Tensor],
            parameters: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """HMETS 的前向传播。"""
        # 解包输入数据
        x = x_dict['x_phy']  # shape: [time, grids, vars]

        # 解包参数
        phy_params = self.unpack_parameters(parameters)
        phy_params_dict = self.descale_phy_parameters(phy_params)

        # --- 状态预热 ---
        if self.warm_up_states:
            warm_up = self.warm_up
        else:
            self.pred_cutoff = self.warm_up
            warm_up = 0

        # 初始化模型状态 snowpack_, liquidwater_, soilwater1_, soilwater2_
        n_grid = x.size(1)
        snowpack_ = torch.zeros([n_grid, self.nmul], dtype=torch.float32, device=self.device) + self.nearzero
        liquidwater_ = torch.zeros([n_grid, self.nmul], dtype=torch.float32, device=self.device) + self.nearzero
        cummelt_ = torch.zeros([n_grid, self.nmul], dtype=torch.float32, device=self.device) + self.nearzero
        soilwater1_ = torch.zeros([n_grid, self.nmul], dtype=torch.float32, device=self.device) + self.nearzero
        soilwater2_ = torch.zeros([n_grid, self.nmul], dtype=torch.float32, device=self.device) + self.nearzero

        if warm_up > 0:
            with torch.no_grad():
                initial_states = [snowpack_, liquidwater_, cummelt_, soilwater1_, soilwater2_]
                # Save current model settings.
                initialize = self.initialize
                routing = self.routing

                # Set model settings for warm-up.
                self.initialize = True
                self.routing = False

                # 运行预热并获取最终状态
                final_states = self.PBM(
                    forcing=x[:warm_up, :, :],
                    initial_states=initial_states,
                    full_param_dict=phy_params_dict,
                    is_warmup=True
                )
                snowpack_, liquidwater_, cummelt_, soilwater1_, soilwater2_ = final_states

                # Restore model settings.
                self.initialize = initialize
                self.routing = routing

        out_dict = self.PBM(
            forcing=x[warm_up:, :, :],
            initial_states=[snowpack_, liquidwater_, cummelt_, soilwater1_, soilwater2_],
            full_param_dict=phy_params_dict,
            is_warmup=False,
        )

        if not self.warm_up_states and self.pred_cutoff > 0:
            for key in out_dict.keys():
                if out_dict[key] is not None and out_dict[key].ndim > 1:
                    out_dict[key] = out_dict[key][self.pred_cutoff:, :]

        return out_dict

    def PBM(
            self,
            forcing: torch.Tensor,
            initial_states: list,
            full_param_dict: dict,
            is_warmup: bool,
    ) -> Union[dict, list]:
        snowpack_, liquidwater_, cumulmelt_, soilwater1_, soilwater2_ = initial_states

        Pm = forcing[:, :, self.variables.index('Prcp')].unsqueeze(2)
        Tm = forcing[:, :, self.variables.index('Tmean')].unsqueeze(2)
        Petm = forcing[:, :, self.variables.index('Pet')].unsqueeze(2)
        n_steps, n_grid, _ = Pm.size()

        snowpack_placeholder = torch.zeros(n_steps, n_grid, self.nmul, device=self.device,
                                           dtype=Pm.dtype) + self.nearzero
        liquidwater_placeholder = torch.zeros(n_steps, n_grid, self.nmul, device=self.device,
                                              dtype=Pm.dtype) + self.nearzero
        soilwater1_placeholder = torch.zeros(n_steps, n_grid, self.nmul, device=self.device,
                                             dtype=Pm.dtype) + self.nearzero
        soilwater2_placeholder = torch.zeros(n_steps, n_grid, self.nmul, device=self.device,
                                             dtype=Pm.dtype) + self.nearzero

        surfaceflow_placeholder = torch.zeros(n_steps, n_grid, self.nmul, device=self.device, dtype=Pm.dtype)
        baseflow1_placeholder = torch.zeros(n_steps, n_grid, self.nmul, device=self.device, dtype=Pm.dtype)
        baseflow2_placeholder = torch.zeros(n_steps, n_grid, self.nmul, device=self.device, dtype=Pm.dtype)

        all_infil_placeholder = torch.zeros(n_steps, n_grid, self.nmul, 5, device=self.device, dtype=Pm.dtype)
        all_baseflow1_placeholder = torch.zeros(n_steps, n_grid, self.nmul, 5, device=self.device, dtype=Pm.dtype)
        all_baseflow2_placeholder = torch.zeros(n_steps, n_grid, self.nmul, 5, device=self.device, dtype=Pm.dtype)

        param_dict = full_param_dict
        near_zeros = torch.zeros(n_grid, self.nmul, device=self.device, dtype=Pm.dtype) + self.nearzero

        for t in range(n_steps):
            # -- 雨雪划分 --
            prcp_, temp_, pet_ = Pm[t], Tm[t], Petm[t]
            rainfall_ = torch.mul(prcp_, (temp_ >= 0.0).float())
            snowfall_ = torch.mul(prcp_, (temp_ < 0.0).float())

            # -- 截留 --
            canopy_ = pet_ * param_dict['forest_cover'] * (1 - param_dict['forest_sparseness'])
            rainfall_ = clamp(rainfall_ - canopy_, min=0.0)
            snowfall_ = clamp(snowfall_ - canopy_, min=0.0)

            # -- 融雪模块 --
            refreeze_ = refreeze_hbv(temp_, liquidwater_, param_dict['Tbf'], param_dict['Kf'])
            snowpack_ = snowpack_ + snowfall_ + refreeze_
            liquidwater_ = liquidwater_ - refreeze_
            snowmelt_ = snowmelt_hbv(temp_, snowpack_, cumulmelt_,
                                     param_dict['ddf_min'], param_dict['ddf_plus'],
                                     param_dict['Kcum'], param_dict['Tbm'])
            cumulmelt_ = (cumulmelt_ + snowmelt_) * (snowpack_ > self.nearzero).float()
            snowpack_ = snowpack_ - snowmelt_
            tmp_water_snowpack_ = liquidwater_ + rainfall_ + snowmelt_
            overflow_ = overflow_hbv(snowpack_, tmp_water_snowpack_, param_dict['swi'])
            liquidwater_ = torch.where(overflow_ > self.nearzero, param_dict['swi'] * snowpack_, tmp_water_snowpack_)

            # -- 下渗 --
            inf_flush_ = infiltration_flush(overflow_)
            inf_gr4j_ = infiltration_gr4j(soilwater1_, overflow_, param_dict['max_soilwater1'])
            inf_pc_ = infiltration_partitioning_coefficient(overflow_, param_dict['inf_pc'])
            inf_hbv_ = infiltration_hbv(overflow_, soilwater1_, param_dict['max_soilwater1'],
                                        param_dict['inf_hbv_beta'])
            inf_vicarno_ = infiltration_vic_arno(overflow_, soilwater1_,
                                                 param_dict['max_soilwater1'], param_dict['inf_vic_bexp'])
            inf_hmets_ = infiltration_hmets(overflow_, soilwater1_,
                                            param_dict['max_soilwater1'], param_dict['inf_hmets_alpha'])

            infil_arr = torch.concatenate([inf_pc_, inf_gr4j_, inf_hbv_, inf_vicarno_, inf_hmets_], dim=1)
            # infil_ = self.infil_dropout(infil_arr)
            infil_ = torch.sum(infil_arr, dim=1, keepdim=True)

            surface_flow_ = torch.clamp(overflow_ - infil_, min=0.0)
            soilevap_ = soil_evaporation_hbv(pet_, soilwater1_, param_dict['max_soilwater1'])
            soilwater1_ = torch.clamp(soilwater1_ + infil_ - soilevap_,
                                      min=near_zeros, max=param_dict['max_soilwater1'])
            # -- 基流 --
            baseflow1_gr4j_ = baseflow_gr4j_exchange(soilwater1_, param_dict['bf1_gr4j_x3'], param_dict['bf1_bfmax'])
            baseflow1_power_ = baseflow_power_law(soilwater1_, param_dict['bf1_bfc'], param_dict['bf1_bfn'],
                                                  param_dict['bf1_bfmax'])
            baseflow1_vic_ = baseflow_vic(soilwater1_, param_dict['max_soilwater1'],
                                          param_dict['bf1_bfmax'], param_dict['bf1_bfn'])
            baseflow1_topmodel_ = baseflow_topmodel(soilwater1_,
                                                    param_dict['max_soilwater1'], param_dict['bf1_bfmax'],
                                                    param_dict['bf1_lam'], param_dict['bf1_bfn'])
            baseflow1_threshold_ = baseflow_threshold(soilwater1_,
                                                      param_dict['max_soilwater1'], param_dict['bf1_bfmax'],
                                                      param_dict['bf1_bfn'], param_dict['bf1_thresh'])
            baseflow1_arr = torch.concatenate([baseflow1_power_, baseflow1_gr4j_, baseflow1_vic_,
                                               baseflow1_topmodel_, baseflow1_threshold_], dim=1)
            # baseflow1_ = self.bf1_dropout(baseflow1_arr)
            baseflow1_ = torch.sum(baseflow1_arr, dim=1, keepdim=True)

            soilwater1_ = torch.clamp(soilwater1_ - baseflow1_, min=near_zeros)
            percolation_ = percolation_gawser(soilwater1_, param_dict['max_perc'],
                                              param_dict['max_soilwater1'], param_dict['perc_sfc'])
            capillary_ = capillary_rise_hbv(soilwater2_, param_dict['max_soilwater2'], param_dict['crise_hbv'])
            soilwater1_ = torch.clamp(soilwater1_ - percolation_ + capillary_,
                                      min=near_zeros, max=param_dict['max_soilwater1'] - 1e-6)

            # soilwater 2
            soilwater2_ = torch.clamp(soilwater2_ - capillary_ + percolation_,
                                      min=near_zeros, max=param_dict['max_soilwater2'] - 1e-6)

            baseflow2_noflow_ = torch.zeros_like(soilwater2_)
            baseflow2_const_ = baseflow_constant_rate(soilwater2_, param_dict['bf2_bfmax'])
            baseflow2_gr4j_ = baseflow_gr4j_exchange(soilwater2_, param_dict['bf2_gr4j_x3'], param_dict['bf2_bfmax'])
            baseflow2_lin_ = baseflow_linear(soilwater2_, param_dict['bf2_bfc'], param_dict['bf2_bfmax'])
            baseflow2_pl_ = baseflow_power_law(soilwater2_, param_dict['bf2_bfc'], param_dict['bf2_bfn'],
                                               param_dict['bf2_bfmax'])
            baseflow2_vic_ = baseflow_vic(soilwater2_, param_dict['max_soilwater2'],
                                          param_dict['bf2_bfmax'], param_dict['bf2_bfn'])
            baseflow2_arr = torch.concatenate([baseflow2_const_, baseflow2_gr4j_, baseflow2_lin_,
                                               baseflow2_pl_, baseflow2_vic_], dim=1)
            # baseflow2_ = self.bf2_dropout(baseflow2_arr)
            baseflow2_ = torch.sum(baseflow2_arr, dim=1, keepdim=True)

            soilwater2_ = torch.clamp(soilwater2_ - baseflow2_, min=near_zeros)

            # save various flow
            surfaceflow_placeholder[t, :, :] = surface_flow_
            baseflow1_placeholder[t, :, :] = baseflow1_
            baseflow2_placeholder[t, :, :] = baseflow2_

            names = ['canopy', 'rainfall', 'snowfall', 'pet', 'snowpack', 'liquidwater', 'soilwater1', 'soilwater2', 'surfaceflow', 'baseflow1', 'baseflow2']
            for n, i in zip(names, [canopy_, rainfall_, snowfall_, pet_, snowpack_, liquidwater_, soilwater1_, soilwater2_, surface_flow_, baseflow1_, baseflow2_]):
                if torch.isnan(i).any():
                    raise ValueError(f'nan in {n}')

            # save states
            snowpack_placeholder[t, :, :] = snowpack_
            liquidwater_placeholder[t, :, :] = liquidwater_
            soilwater1_placeholder[t, :, :] = soilwater1_
            soilwater2_placeholder[t, :, :] = soilwater2_

            for i, infil in enumerate([inf_flush_, inf_pc_,  # inf_gr4j_,
                                       inf_hbv_, inf_vicarno_, inf_hmets_]):
                all_infil_placeholder[t, :, :, i] = infil

            for i, baseflow1 in enumerate([baseflow1_gr4j_,  # baseflow1_gr4j_,
                                           baseflow1_power_, baseflow1_vic_,
                                           baseflow1_topmodel_, baseflow1_threshold_]):
                all_baseflow1_placeholder[t, :, :, i] = baseflow1

            for i, baseflow2 in enumerate([baseflow2_noflow_, baseflow2_const_,
                                           baseflow2_lin_,  # baseflow2_gr4j_,
                                           baseflow2_pl_, baseflow2_vic_]):
                all_baseflow2_placeholder[t, :, :, i] = baseflow2

        if is_warmup:
            return [snowpack_, liquidwater_, soilwater1_, soilwater2_]

        baseflow1_mean = baseflow1_placeholder.mean(dim=2)
        baseflow2_mean = baseflow2_placeholder.mean(dim=2)
        surfaceflow_mean = surfaceflow_placeholder.mean(dim=2)

        UH1 = uh_gamma(
            param_dict['alpha1'].squeeze().repeat(n_steps, 1).unsqueeze(-1),  # Shape: [time, n_grid]
            param_dict['beta1'].squeeze().repeat(n_steps, 1).unsqueeze(-1),
            lenF=50
        ).to(baseflow1_mean.dtype)
        bf1_delay = baseflow1_mean.permute(1, 0).unsqueeze(1)  # Shape: [n_grid, 1, time]
        UH1_permuted = UH1.permute(1, 2, 0)  # Shape: [n_grid, 1, time]
        bf1_out = uh_conv(bf1_delay, UH1_permuted).permute(2, 0, 1)

        UH2 = uh_gamma(
            param_dict['alpha2'].squeeze().repeat(n_steps, 1).unsqueeze(-1),  # Shape: [time, n_grid]
            param_dict['beta2'].squeeze().repeat(n_steps, 1).unsqueeze(-1),
            lenF=50
        ).to(baseflow2_mean.dtype)
        bf2_delay = baseflow2_mean.permute(1, 0).unsqueeze(1)  # Shape: [n_grid, 1, time]
        UH2_permuted = UH2.permute(1, 2, 0)  # Shape: [n_grid, 1, time]
        bf2_out = uh_conv(bf2_delay, UH2_permuted).permute(2, 0, 1)

        total_flow = bf1_out.squeeze() + bf2_out.squeeze() + surfaceflow_mean

        out_dict = {
            'streamflow': total_flow,
            'baseflow1': bf1_out.squeeze(),
            'baseflow2': bf2_out.squeeze(),
            'surfaceflow': surfaceflow_mean,
            'snowpack': snowpack_placeholder,
            'liquidwater': liquidwater_placeholder,
            'soilwater1': soilwater1_placeholder,
            'soilwater2': soilwater2_placeholder,
            'all_infil': all_infil_placeholder,
            'all_baseflow1': all_baseflow1_placeholder,
            'all_baseflow2': all_baseflow2_placeholder,
        }
        return out_dict
