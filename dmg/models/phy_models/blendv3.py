# blendv2代表使用LSTM对模型中动态参数进行预测，然后使用MLP模型对静态参数进行预测
# 其中动态参数为每个公式的权重，静态参数为其余的参数值, 共计35个参数值
# 模型的权重值是通过softmax进行处理

from typing import Any, Optional, Union

import torch
from torch import clamp
from hydrodl2.core.calc import change_param_range, uh_conv, uh_gamma
from fluxes import *


def snobal_hbv(tmean, snowfall, rainfall, cumulmelt, snowpack, liquidwater, nearzero, param_dict):
    ddf = torch.min(param_dict['ddf_min'] + param_dict['ddf_plus'], param_dict['ddf_min'] \
                    * (1 + param_dict['Kcum'] * cumulmelt))
    potenmelt = clamp(ddf * (tmean - param_dict['Tbm']), min=0)
    refreeze = torch.min(param_dict['Kf_hbv'] * clamp(param_dict['Tbf'] - tmean, min=nearzero), liquidwater)
    snowpack = snowpack + snowfall + refreeze
    snowmelt = torch.min(potenmelt, snowpack + snowfall + refreeze)
    cumulmelt = (cumulmelt + snowmelt) * (snowpack > nearzero).float()
    snowpack = clamp(snowpack - snowmelt, min=nearzero)
    water_retention = param_dict['SWI'] * snowpack
    water_in_snowpack_temp = liquidwater + snowmelt + rainfall
    overflow = clamp(water_in_snowpack_temp - water_retention, min=0)
    liquidwater = torch.where(overflow > 0, water_retention, water_in_snowpack_temp)
    return snowpack, liquidwater, cumulmelt, snowmelt, refreeze, overflow


class blendv2c(torch.nn.Module):
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
        self.nearzero = 1e-5
        self.nmul = 1

        self.parameter_bounds = {
            # Infiltration (入渗)
            'inf_pc': [0.0, 1.0],  # [Inf-PC] Partitioning Coefficient
            'inf_hbv_beta': [0.5, 3.0],  # [Inf-HBV] HBV Method beta
            'inf_vic_bexp': [0.001, 3.0],  # [Inf-VICARNO] VIC-ARNO Method bexp
            'inf_hmets_alpha': [0.3, 1.0],  # [Inf-HMETS] HMETS Method alpha

            # Surface Runoff (地表径流)
            'sr_soil_depth_1': [0.05, 0.5],  # Infiltration Excess fixed soil depth for soil 1

            # Baseflow Soil 1 (土壤层1基流)
            'bf1_gr4j_x3': [20.0, 300.0],  # [BF1-GR4J] GR4J Method X3-1
            'bf1_pl_bfc': [0.001, 0.5],  # [BF1-PL] Power Law Baseflow BFc1
            'bf1_pl_bfn': [1.0, 5.0],  # [BF1-PL] Power Law Baseflow BFn1
            'bf1_vic_bfmax': [0.1, 50.0],  # [BF1-VIC] VIC Method BFmax1
            'bf1_vic_bfn': [1.0, 5.0],  # [BF1-VIC] VIC Method BFn1
            'bf1_top_bfmax': [0.1, 50.0],  # [BF1-TOPMOD] TOPMODEL Method BFmax1
            'bf1_top_bfn': [1.0, 5.0],  # [BF1-TOPMOD] TOPMODEL Method BFn1
            'bf1_top_lam': [5.0, 10.0],  # [BF1-TOPMOD] TOPMODEL Method lam1
            'bf1_thresh_bfmax': [0.1, 50.0],  # [BF1-THRESH] Threshold Baseflow BFmax1
            'bf1_thresh_bfn': [1.0, 5.0],  # [BF1-THRESH] Threshold Baseflow BFn1
            'bf1_thresh_bfthresh': [0.0, 1.0],  # [BF1-THRESH] Threshold Baseflow BFThresh1

            # Percolation (渗漏)
            'perc_gawser_maxperc': [1.0, 50.0],  # [Perc] GAWSER Method MaxPerc

            # Capillary Rise (毛管上升)
            'crise_hbv_crise': [0.1, 50.0],  # [CRise] HBV Method CRise

            # Baseflow Soil 2 (土壤层2基流)
            'bf2_con_bfmax': [0.1, 50.0],  # [BF2-CON] Constant Rate BFMax2
            'bf2_gr4j_x3': [20.0, 300.0],  # [BF2-GR4J] GR4J Method X3-2
            'bf2_lin_bfc': [0.0001, 0.1],  # [BF2-LIN] Linear Baseflow BFc2
            'bf2_pl_bfc': [0.0001, 0.1],  # [BF2-PL] Power Law Baseflow BFc2
            'bf2_pl_bfn': [1.0, 5.0],  # [BF2-PL] Power Law Baseflow BFn2
            'bf2_vic_bfmax': [0.1, 50.0],  # [BF2-VIC] VIC Method BFmax2
            'bf2_vic_bfn': [1.0, 5.0],  # [BF2-VIC] VIC Method BFn2
        }
        self.weights_name = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9']
        self.weights_group = [['w1', 'w2', 'w3'], ['w4', 'w5', 'w6'], ['w7', 'w8', 'w9']]

        self.routing_parameter_bounds = {
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
            self.nmul = config.get('nmul', self.nmul)

        self.set_parameters()

    def set_parameters(self) -> None:
        """设置物理参数名称和数量。"""
        self.phy_param_names = list(self.parameter_bounds.keys())
        if self.routing:
            self.routing_param_names = self.routing_parameter_bounds.keys()
        else:
            self.routing_param_names = []
        self.learnable_param_count1 = len(self.weights_name) * self.nmul
        self.learnable_param_count2 = len(self.phy_param_names) * self.nmul \
                                      + len(self.routing_param_names)
        self.learnable_param_count = self.learnable_param_count1 + self.learnable_param_count2

    def unpack_parameters(self, parameters: tuple[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, None]:
        weights_params, hybrid_params = parameters[0], parameters[1]
        """从神经网络输出中提取物理参数。"""
        phy_param_count = len(self.parameter_bounds)
        # Physical parameters
        phy_params = torch.sigmoid(hybrid_params[:, :phy_param_count * self.nmul].view(
            hybrid_params.shape[0],
            phy_param_count,
            self.nmul,
        ))
        weights_params = weights_params.view(
            weights_params.shape[0],
            weights_params.shape[1],
            len(self.weights_name),
            self.nmul
        )
        # 获取所有weights的index，然后通过softmax将各组参数投射到0-1
        for weights in self.weights_group:
            mask = torch.zeros(weights_params.shape[2], dtype=torch.bool)
            weights_index = [self.weights_name.index(weight) for weight in weights]
            mask[weights_index] = True
            tmp_subset_params = weights_params[:, :, mask, :]
            weights_params[:, :, mask, :] = torch.softmax(tmp_subset_params, dim=2)
        # Routing parameters
        routing_params = None
        if self.routing:
            routing_params = torch.sigmoid(
                hybrid_params[:, phy_param_count * self.nmul:],
            )
        return phy_params, weights_params, routing_params

    def descale_phy_parameters(self, phy_params: torch.Tensor) -> dict:
        """将归一化的物理参数去归一化到其物理范围。"""
        phy_param_dict = {}
        for i, name in enumerate(self.phy_param_names):
            phy_param_dict[name] = change_param_range(phy_params[:, i, :], self.parameter_bounds[name])
        return phy_param_dict

    def descale_weight_parameters(self, weight_params: torch.Tensor) -> dict:
        weight_param_dict = {}
        for i, name in enumerate(self.weights_name):
            weight_param_dict[name] = weight_params[:, :, i, :]
        return weight_param_dict

    def descale_rout_parameters(self, routing_params: torch.Tensor) -> dict:
        """将归一化的路由参数去归一化到其物理范围。"""
        parameter_dict = {}
        for i, name in enumerate(self.routing_parameter_bounds.keys()):
            param = routing_params[:, i]
            parameter_dict[name] = change_param_range(
                param=param,
                bounds=self.routing_parameter_bounds[name],
            )
        return parameter_dict

    def forward(
            self,
            x_dict: dict[str, torch.Tensor],
            parameters: tuple[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """HMETS 的前向传播。"""
        # 解包输入数据
        x = x_dict['x_phy']  # shape: [time, grids, vars]

        # 解包参数
        phy_params, weights_params, routing_params = self.unpack_parameters(parameters)
        phy_params_dict = self.descale_phy_parameters(phy_params)

        if self.routing:
            self.routing_param_dict = self.descale_rout_parameters(routing_params)

        # --- 状态预热 ---
        if self.warm_up_states:
            warm_up = self.warm_up
        else:
            self.pred_cutoff = self.warm_up
            warm_up = 0

        n_grid = x.size(1)

        # 初始化模型状态
        SNOW_ON_GROUND = torch.zeros([n_grid, self.nmul, 3], dtype=torch.float32, device=self.device) + self.nearzero
        WATER_IN_SNOWPACK = torch.zeros([n_grid, self.nmul, 3], dtype=torch.float32, device=self.device) + self.nearzero
        CUMSNOWMELT = torch.zeros([n_grid, self.nmul, 3], dtype=torch.float32, device=self.device) + self.nearzero
        PHREATIC_LEVEL = torch.zeros([n_grid, self.nmul], dtype=torch.float32, device=self.device) + self.nearzero

        # 获取初始水箱水位（基于最大水位的比例）
        vadose_max_init = change_param_range(
            torch.sigmoid(phy_params[:, self.phy_param_names.index('vadose_max_level')]),
            self.parameter_bounds['vadose_max_level'])
        VADOSE_LEVEL = 0.5 * vadose_max_init

        if warm_up > 0:
            with torch.no_grad():
                initial_states = [SNOW_ON_GROUND, WATER_IN_SNOWPACK, CUMSNOWMELT, VADOSE_LEVEL, PHREATIC_LEVEL]
                weights_params_warmup = self.descale_weight_parameters(weights_params[:warm_up, :, :])
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
                    full_param_dict={"weights_params": weights_params_warmup,
                                     "phy_params": phy_params_dict},
                    is_warmup=True
                )
                SNOW_ON_GROUND, WATER_IN_SNOWPACK, CUMSNOWMELT, VADOSE_LEVEL, PHREATIC_LEVEL = final_states

                # Restore model settings.
                self.initialize = initialize
                self.routing = routing

        out_dict = self.PBM(
            forcing=x[warm_up:, :, :],
            initial_states=[SNOW_ON_GROUND, WATER_IN_SNOWPACK, CUMSNOWMELT, VADOSE_LEVEL, PHREATIC_LEVEL],
            full_param_dict={"weights_params": self.descale_weight_parameters(weights_params[warm_up:, :, :]),
                             "phy_params": phy_params_dict},
            is_warmup=False,
            muwts=x_dict.get('muwts', None)
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
            muwts=None
    ) -> Union[dict, list]:
        SNOW_ON_GROUND, WATER_IN_SNOWPACK, CUMSNOWMELT, VADOSE_LEVEL, PHREATIC_LEVEL = initial_states

        Prcp = forcing[:, :, self.variables.index('Prcp')]
        Tmean = forcing[:, :, self.variables.index('Tmean')]
        Pet = forcing[:, :, self.variables.index('Pet')]
        n_steps, n_grid = Prcp.size()

        Pm = Prcp.unsqueeze(2).repeat(1, 1, self.nmul)
        Tm = Tmean.unsqueeze(2).repeat(1, 1, self.nmul)
        Petm = Pet.unsqueeze(2).repeat(1, 1, self.nmul)

        horizontal_transfert_mu = torch.zeros(n_steps, n_grid, self.nmul, 4, device=self.device, dtype=Pm.dtype)
        RET_sim_mu = torch.zeros(n_steps, n_grid, self.nmul, device=self.device, dtype=Pm.dtype)
        vadose_level_sim_mu = torch.zeros(n_steps, n_grid, self.nmul, device=self.device, dtype=Pm.dtype)
        phreatic_level_sim_mu = torch.zeros(n_steps, n_grid, self.nmul, device=self.device, dtype=Pm.dtype)

        dynamic_weights_placeholder = torch.zeros(n_steps, n_grid, self.nmul, 9, device=self.device, dtype=Pm.dtype)
        overflow_placeholder = torch.zeros(n_steps, n_grid, self.nmul, 3, device=self.device, dtype=Pm.dtype)
        infil_placeholder = torch.zeros(n_steps, n_grid, self.nmul, 3, device=self.device, dtype=Pm.dtype)
        quickflow_placeholder = torch.zeros(n_steps, n_grid, self.nmul, 3, device=self.device, dtype=Pm.dtype)

        param_dict = full_param_dict['phy_params']

        for t in range(n_steps):
            for key in full_param_dict['weights_params'].keys():
                param_dict[key] = full_param_dict['weights_params'][key][t, :, :]

            # -- 雨雪划分 --
            prcp_, temp_, pet_ = Pm[t], Tm[t], Petm[t]
            rainfall_ = torch.mul(prcp_, (temp_ >= 0.0).float())
            snowfall_ = torch.mul(prcp_, (temp_ < 0.0).float())
            # -- 融雪模块 --
            # todo 控制不设置refreeze
            refreeze_ = refreeze_hbv(temp_, liquidwater_, param_dict['Tbf'], param_dict['Kf'])
            snowpack_ = snowpack_ + snowfall_ + refreeze_
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
            inf_gr4j_ = infiltration_gr4j(soilwater_, overflow_, param_dict['field_capacity'])
            inf_pc_ = infiltration_partitioning_coefficient(overflow_, param_dict['inf_pc'])
            inf_hbv_ = infiltration_hbv(overflow_, soilwater_, param_dict['field_capacity'], param_dict['inf_hbv_beta'])
            inf_vicarno_ = infiltration_vic_arno(overflow_, soilwater_,
                                                 param_dict['field_capacity'], param_dict['inf_vic_bexp'])
            inf_hmets_ = infiltration_hmets(overflow_, soilwater_,
                                            param_dict['field_capacity'], param_dict['inf_hmets_alpha'])

            inf_weight_avg_ = param_dict['inf_w1'] * inf_flush_ + \
                              param_dict['inf_w2'] * inf_gr4j_ + \
                              param_dict['inf_w3'] * inf_pc_ + \
                              param_dict['inf_w4'] * inf_hbv_ + \
                              param_dict['inf_w5'] * inf_hmets_

            surface_flow_ = overflow_ - inf_weight_avg_
            soilwater_ = soilwater_ + inf_weight_avg_
            soilevap_ = soil_evaporation_hbv(pet_, soilwater_, param_dict['field_capacity'])
            """
            'bf1_gr4j_x3': [20.0, 300.0],  # [BF1-GR4J] GR4J Method X3-1
            'bf1_pl_bfc': [0.001, 0.5],  # [BF1-PL] Power Law Baseflow BFc1
            'bf1_pl_bfn': [1.0, 5.0],  # [BF1-PL] Power Law Baseflow BFn1
            'bf1_vic_bfmax': [0.1, 50.0],  # [BF1-VIC] VIC Method BFmax1
            'bf1_vic_bfn': [1.0, 5.0],  # [BF1-VIC] VIC Method BFn1
            'bf1_top_bfmax': [0.1, 50.0],  # [BF1-TOPMOD] TOPMODEL Method BFmax1
            'bf1_top_bfn': [1.0, 5.0],  # [BF1-TOPMOD] TOPMODEL Method BFn1
            'bf1_top_lam': [5.0, 10.0],  # [BF1-TOPMOD] TOPMODEL Method lam1
            'bf1_thresh_bfmax': [0.1, 50.0],  # [BF1-THRESH] Threshold Baseflow BFmax1
            'bf1_thresh_bfn': [1.0, 5.0],  # [BF1-THRESH] Threshold Baseflow BFn1
            'bf1_thresh_bfthresh': [0.0, 1.0],  # [BF1-THRESH] Threshold Baseflow BFThresh1
            """
            baseflow1_noflow_ = torch.zeros_like(soilwater_)
            baseflow1_gr4j_ = baseflow_gr4j_exchange(soilwater_, param_dict['bf1_gr4j_x3'])
            baseflow1_power = baseflow_power_law(soilwater_, param_dict['bf1_pl_bfc'], param_dict['bf1_pl_bfn'])
            


            # -- 土壤蒸发计算 --
            RET = clamp(param_dict['ET_efficiency'] * Petm[t], max=water_available)
            water_available = water_available - RET
            # -- 下渗计算 --
            VADOSE_PROP = clamp(VADOSE_LEVEL / param_dict['vadose_max_level'],
                                max=1.0 - self.nearzero, min=self.nearzero)
            infil_1 = infil_hmets(VADOSE_PROP, param_dict)
            infil_2 = infil_vic_arno(VADOSE_PROP, param_dict)
            infil_3 = infil_hbv(VADOSE_PROP, param_dict)
            nearzero_arr = torch.ones_like(water_available) * self.nearzero
            infiltration = water_available * clamp((param_dict['w4'] * infil_1 + param_dict['w5'] * infil_2
                                                    + param_dict['w6'] * infil_3), min=self.nearzero, max=1.0)

            horizontal_transfert_mu[t, :, :, 0] = water_available - infiltration
            VADOSE_LEVEL = clamp(VADOSE_LEVEL + infiltration - RET, min=self.nearzero)
            # -- 快速流计算 --
            VADOSE_PROP_new = clamp(VADOSE_LEVEL / param_dict['vadose_max_level'],
                                    max=1.0 - self.nearzero, min=self.nearzero)
            quickflow_1 = quickflow_exphydro(VADOSE_LEVEL, param_dict, self.nearzero)
            quickflow_2 = quickflow_vic(VADOSE_PROP_new, param_dict, self.nearzero)
            quickflow_3 = quickflow_thresh(VADOSE_PROP_new, param_dict, self.nearzero)
            horizontal_transfert_mu[t, :, :, 1] = clamp(param_dict['mmax'] *
                                                        (param_dict['w7'] * quickflow_1
                                                         + param_dict['w8'] * quickflow_2
                                                         + param_dict['w9'] * quickflow_3),
                                                        min=nearzero_arr, max=VADOSE_LEVEL)
            # -- 基流计算 -- 仅适用一个
            horizontal_transfert_mu[t, :, :, 2] = baseflow_linear(VADOSE_LEVEL, param_dict)

            vadose2phreatic = param_dict['coef_vadose2phreatic'] * VADOSE_LEVEL
            VADOSE_LEVEL = clamp(VADOSE_LEVEL - horizontal_transfert_mu[t, :, :, 1] \
                                 - horizontal_transfert_mu[t, :, :, 2] - vadose2phreatic, min=self.nearzero)

            vadose_pos_mask = VADOSE_LEVEL > param_dict['vadose_max_level']
            vadose_excess = VADOSE_LEVEL - param_dict['vadose_max_level']
            horizontal_transfert_mu[t, :, :, 0][vadose_pos_mask] = horizontal_transfert_mu[t, :, :, 0][vadose_pos_mask] \
                                                                   + vadose_excess[vadose_pos_mask]
            VADOSE_LEVEL = clamp(VADOSE_LEVEL, max=param_dict['vadose_max_level'])

            horizontal_transfert_mu[t, :, :, 3] = param_dict['coef_phreatic'] * PHREATIC_LEVEL
            PHREATIC_LEVEL = clamp(PHREATIC_LEVEL + vadose2phreatic - horizontal_transfert_mu[t, :, :, 3],
                                   min=self.nearzero)

            RET_sim_mu[t], vadose_level_sim_mu[t], phreatic_level_sim_mu[t] = RET, VADOSE_LEVEL, PHREATIC_LEVEL

            # -- 存储动态参数值 --
            for i in range(9):
                dynamic_weights_placeholder[t, :, :, i] = param_dict[f'w{i + 1}']

            # -- 存储各个变量的值 --
            for i, overflow in enumerate([overflow_1, overflow_2, overflow_3]):
                overflow_placeholder[t, :, :, i] = overflow

            for i, infil in enumerate([infil_1, infil_2, infil_3]):
                infil_placeholder[t, :, :, i] = water_available * infil

            for i, quickflow in enumerate([quickflow_1, quickflow_2, quickflow_3]):
                quickflow_placeholder[t, :, :, i] = param_dict['mmax'] * quickflow

        weights_dict = {f'w{i + 1}': dynamic_weights_placeholder[:, :, :, i] for i in range(9)}
        overflow_dict = {f'overflow_{i + 1}': overflow_placeholder[:, :, :, i] for i in range(3)}
        infil_dict = {f'infil_{i + 1}': infil_placeholder[:, :, :, i] for i in range(3)}
        quickflow_dict = {f'quickflow_{i + 1}': quickflow_placeholder[:, :, :, i] for i in range(3)}

        if is_warmup:
            return [SNOW_ON_GROUND, WATER_IN_SNOWPACK, CUMSNOWMELT, VADOSE_LEVEL, PHREATIC_LEVEL]

        if muwts is None:
            horizontal_transfert = horizontal_transfert_mu.mean(dim=2)
        else:
            horizontal_transfert = (horizontal_transfert_mu * self.muwts).sum(-1)

        # --- 流量演算 ---
        if self.routing:
            UH1 = uh_gamma(
                self.routing_param_dict['alpha1'].repeat(n_steps, 1).unsqueeze(-1),  # Shape: [time, n_grid]
                self.routing_param_dict['beta1'].repeat(n_steps, 1).unsqueeze(-1),
                lenF=50
            ).to(horizontal_transfert.dtype)
            UH2 = uh_gamma(
                self.routing_param_dict['alpha2'].repeat(n_steps, 1).unsqueeze(-1),
                self.routing_param_dict['beta2'].repeat(n_steps, 1).unsqueeze(-1),
                lenF=50
            ).to(horizontal_transfert.dtype)

            rf_delay = horizontal_transfert[:, :, 1].permute(1, 0).unsqueeze(1)  # Shape: [n_grid, 1, time]
            rf_base = horizontal_transfert[:, :, 2].permute(1, 0).unsqueeze(1)  # Shape: [n_grid, 1, time]

            UH1_permuted = UH1.permute(1, 2, 0)  # Shape: [n_grid, 1, time]
            UH2_permuted = UH2.permute(1, 2, 0)  # Shape: [n_grid, 1, time]

            rf_delay_rout = uh_conv(rf_delay, UH1_permuted).permute(2, 0, 1)
            rf_base_rout = uh_conv(rf_base, UH2_permuted).permute(2, 0, 1)

            rf_ruf = horizontal_transfert[:, :, 0].unsqueeze(-1)
            rf_gwd = horizontal_transfert[:, :, 3].unsqueeze(-1)

            Qsim = rf_delay_rout + rf_base_rout + rf_ruf + rf_gwd
        else:
            Qsim = horizontal_transfert.sum(dim=2).unsqueeze(-1)
            rf_ruf = horizontal_transfert[:, :, 0].unsqueeze(-1)
            rf_delay_rout = horizontal_transfert[:, :, 1].unsqueeze(-1)
            rf_base_rout = horizontal_transfert[:, :, 2].unsqueeze(-1)
            rf_gwd = horizontal_transfert[:, :, 3].unsqueeze(-1)

        out_dict = {
            'streamflow': Qsim,
            'srflow': rf_ruf,
            'interflow': rf_delay_rout,
            'ssflow': rf_base_rout,
            'gwflow': rf_gwd,
            'AET_hydro': RET_sim_mu.mean(dim=2).unsqueeze(-1),
            'vadose_storage': vadose_level_sim_mu.mean(dim=2).unsqueeze(-1),
            'phreatic_storage': phreatic_level_sim_mu.mean(dim=2).unsqueeze(-1),
        }
        for d in [weights_dict, overflow_dict, infil_dict, quickflow_dict]:
            out_dict.update(d)
        return out_dict
