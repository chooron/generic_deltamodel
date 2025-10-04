from typing import Any, Optional, Union

import torch
from dmg.models.hydrodl2 import change_param_range, uh_conv, uh_gamma


class hmets(torch.nn.Module):
    """
    HMETS 模型

    这是一个功能完整的、可微分的 PyTorch HMETS 模型，支持状态预热和动态参数。

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
        self.name = 'HMETS 1.0'
        self.config = config
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # --- 从配置或默认值设置模型属性 ---
        self.warm_up = 0
        self.warm_up_states = True
        self.pred_cutoff = 0
        self.dynamic_params = []
        self.dy_drop = 0.0
        # HMETS 需要的输入变量
        self.variables = ['Prcp', 'Tmean', 'Pet']
        self.routing = True  # 在 HMETS 中，路由是不可分割的一部分
        self.initialize = False
        self.nearzero = 1e-5
        self.nmul = 1

        # HMETS 的参数边界
        self.parameter_bounds = {
            # Snow Model (10)
            'ddf_min': [0.0, 20.0], 'ddf_plus': [0.0, 20.0],
            'Tbm': [-2.0, 3.0], 'Kcum': [0.01, 0.2],
            'fcmin': [0.0, 0.1], 'fcmin_plus': [0.01, 0.25],
            'Ccum': [0.005, 0.05], 'Tbf': [-5.0, 2.0],
            'Kf': [0.0, 5.0], 'exp_fe': [0.0, 1.0],
            # ET (1)
            'ET_efficiency': [0.0, 3.0],
            # Subsurface (6)
            'coef_runoff': [0.0, 1.0], 'coef_vadose2phreatic': [0.00001, 0.02],
            'coef_vadose': [0.0, 0.1], 'coef_phreatic': [0.00001, 0.01],
            'vadose_max_level': [0.001, 500.0]
        }

        # HMETS 没有像 HBV 那样独立的路由参数，它们是主参数集的一部分
        self.routing_parameter_bounds = {
            # Unit Hydrograph (4)
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
        self.learnable_param_count = len(self.phy_param_names) * self.nmul + len(self.routing_param_names)

    def unpack_parameters(self, parameters: torch.Tensor) -> tuple[torch.Tensor, None]:
        """从神经网络输出中提取物理参数。"""
        phy_param_count = len(self.parameter_bounds)
        # Physical parameters
        phy_params = torch.sigmoid(
            parameters[:, :, :phy_param_count * self.nmul]).view(
            parameters.shape[0],
            parameters.shape[1],
            phy_param_count,
            self.nmul,
        )
        # Routing parameters
        routing_params = None
        if self.routing:
            routing_params = torch.sigmoid(
                parameters[-1, :, phy_param_count * self.nmul:],
            )
        return phy_params, routing_params

    def descale_phy_parameters(self, phy_params: torch.Tensor, dy_list: list) -> dict:
        """将归一化的物理参数去归一化到其物理范围。"""
        n_steps, n_grid, _, _ = phy_params.shape
        param_dict = {}
        pmat = torch.ones([1, n_grid, 1], device=self.device) * self.dy_drop

        for i, name in enumerate(self.phy_param_names):
            # 静态参数使用最后一个时间步的值
            staPar = phy_params[-1, :, i, :].unsqueeze(0).repeat([n_steps, 1, 1])
            if name in dy_list:
                dynPar = phy_params[:, :, i, :]
                # 使用 dropout 组合动态和静态参数
                drmask = torch.bernoulli(pmat).detach().cuda()
                comPar = dynPar * (1 - drmask) + staPar * drmask
                param_dict[name] = change_param_range(comPar, self.parameter_bounds[name])
            else:
                param_dict[name] = change_param_range(staPar, self.parameter_bounds[name])
        return param_dict

    def descale_rout_parameters(self, routing_params: torch.Tensor) -> torch.Tensor:
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
            parameters: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """HMETS 的前向传播。"""
        # 解包输入数据
        x = x_dict['x_phy']  # shape: [time, grids, vars]
        self.muwts = x_dict.get('muwts', None)
        # watershed_area = x_dict['area'] # shape: [grids]

        # 解包参数
        phy_params, routing_params = self.unpack_parameters(parameters)

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
        SNOW_ON_GROUND = torch.zeros([n_grid, self.nmul], dtype=torch.float32, device=self.device) + self.nearzero
        WATER_IN_SNOWPACK = torch.zeros([n_grid, self.nmul], dtype=torch.float32, device=self.device) + self.nearzero
        CUMSNOWMELT = torch.zeros([n_grid, self.nmul], dtype=torch.float32, device=self.device) + self.nearzero

        # 获取初始水箱水位（基于最大水位的比例）
        phy_params_last_step = phy_params[-1, :, :]
        vadose_max_init = change_param_range(
            torch.sigmoid(phy_params_last_step[:, self.phy_param_names.index('vadose_max_level')]),
            self.parameter_bounds['vadose_max_level'])

        VADOSE_LEVEL = 0.5 * vadose_max_init
        PHREATIC_LEVEL = torch.zeros([n_grid, self.nmul], dtype=torch.float32, device=self.device) + self.nearzero

        if warm_up > 0:
            with torch.no_grad():
                phy_param_warmup_dict = self.descale_phy_parameters(
                    phy_params[:warm_up, :, :],
                    dy_list=[],  # 预热期间不使用动态参数
                )
                initial_states = [SNOW_ON_GROUND, WATER_IN_SNOWPACK, CUMSNOWMELT, VADOSE_LEVEL, PHREATIC_LEVEL]

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
                    full_param_dict=phy_param_warmup_dict,
                    is_warmup=True  # 标记为预热运行
                )
                SNOW_ON_GROUND, WATER_IN_SNOWPACK, CUMSNOWMELT, VADOSE_LEVEL, PHREATIC_LEVEL = final_states

                # Restore model settings.
                self.initialize = initialize
                self.routing = routing

        # --- 正式模拟 ---
        phy_params_dict = self.descale_phy_parameters(
            phy_params[warm_up:, :, :],
            dy_list=self.dynamic_params,
        )

        out_dict = self.PBM(
            forcing=x[warm_up:, :, :],
            initial_states=[SNOW_ON_GROUND, WATER_IN_SNOWPACK, CUMSNOWMELT, VADOSE_LEVEL, PHREATIC_LEVEL],
            full_param_dict=phy_params_dict,
            is_warmup=False
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
            is_warmup: bool
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

        param_dict = {}

        for t in range(n_steps):
            for key in full_param_dict.keys():
                param_dict[key] = full_param_dict[key][t, :, :]
            # -- 雨雪划分 --
            PRECIP = Pm[t]
            RAIN = torch.mul(PRECIP, (Tm[t] >= 0.0).float())
            SNOW = torch.mul(PRECIP, (Tm[t] < 0.0).float())

            # -- 融雪模块 --
            Tdiurnal = Tm[t]
            potential_freezing = param_dict['Kf'] * torch.clamp(param_dict['Tbf'] - Tdiurnal, min=self.nearzero) ** \
                                 param_dict['exp_fe']
            overnight_freezing = torch.min(potential_freezing, WATER_IN_SNOWPACK)

            WATER_IN_SNOWPACK = WATER_IN_SNOWPACK - overnight_freezing
            SNOW_ON_GROUND = SNOW_ON_GROUND + overnight_freezing

            ddf = torch.min(param_dict['ddf_min'] + param_dict['ddf_plus'], param_dict['ddf_min'] \
                            * (1 + param_dict['Kcum'] * CUMSNOWMELT))
            snowmelt_potential = torch.clamp(ddf * (Tm[t] - param_dict['Tbm']), min=0)

            snowmelt = torch.min(snowmelt_potential, SNOW_ON_GROUND + SNOW)
            SNOW_ON_GROUND = SNOW_ON_GROUND + SNOW - snowmelt

            CUMSNOWMELT = (CUMSNOWMELT + snowmelt) * (SNOW_ON_GROUND > self.nearzero).float()

            water_retention_fraction = torch.clamp((param_dict['fcmin'] + param_dict['fcmin_plus']) \
                                                   * (1 - param_dict['Ccum'] * CUMSNOWMELT),
                                                   min=param_dict['fcmin'])
            water_retention = water_retention_fraction * SNOW_ON_GROUND

            water_in_snowpack_temp = WATER_IN_SNOWPACK + snowmelt + RAIN
            water_available = torch.clamp(water_in_snowpack_temp - water_retention, min=0)
            WATER_IN_SNOWPACK = torch.where(water_available > 0, water_retention, water_in_snowpack_temp)

            # -- 土壤含水模块 --
            RET = param_dict['ET_efficiency'] * Petm[t]

            horizontal_transfert_mu[t, :, :, 0] = param_dict['coef_runoff'] * (
                    VADOSE_LEVEL / param_dict['vadose_max_level']) * water_available
            infiltration = water_available - horizontal_transfert_mu[t, :, :, 0] - RET

            delayed_runoff_frac = param_dict['coef_runoff'] * (VADOSE_LEVEL / param_dict['vadose_max_level']) ** 2
            horizontal_transfert_mu[t, :, :, 2] = param_dict['coef_vadose'] * VADOSE_LEVEL
            vadose2phreatic = param_dict['coef_vadose2phreatic'] * VADOSE_LEVEL

            horizontal_transfert_mu[t, :, :, 1] = torch.where(infiltration > 0, delayed_runoff_frac * infiltration,
                                                              torch.tensor(0.0, device=self.device))
            infiltration = torch.clamp(infiltration, min=0)

            VADOSE_LEVEL = VADOSE_LEVEL + infiltration - RET - horizontal_transfert_mu[t, :, :, 1] \
                           - horizontal_transfert_mu[t, :, :, 2] - vadose2phreatic

            vadose_neg_mask = VADOSE_LEVEL < 0
            RET[vadose_neg_mask] = RET[vadose_neg_mask] + VADOSE_LEVEL[vadose_neg_mask]
            RET = torch.clamp(RET, min=0)
            VADOSE_LEVEL = torch.clamp(VADOSE_LEVEL, min=0)

            vadose_pos_mask = VADOSE_LEVEL > param_dict['vadose_max_level']
            vadose_excess = VADOSE_LEVEL - param_dict['vadose_max_level']
            horizontal_transfert_mu[t, :, :, 0][vadose_pos_mask] = horizontal_transfert_mu[t, :, :, 0][vadose_pos_mask] \
                                                                   + vadose_excess[vadose_pos_mask]
            VADOSE_LEVEL = torch.clamp(VADOSE_LEVEL, max=param_dict['vadose_max_level'])

            horizontal_transfert_mu[t, :, :, 3] = param_dict['coef_phreatic'] * PHREATIC_LEVEL
            PHREATIC_LEVEL = PHREATIC_LEVEL + vadose2phreatic - horizontal_transfert_mu[t, :, :, 3]

            # 记录每个mu的模拟变量
            RET_sim_mu[t], vadose_level_sim_mu[t], phreatic_level_sim_mu[t] = RET, VADOSE_LEVEL, PHREATIC_LEVEL

        if is_warmup:
            return [SNOW_ON_GROUND, WATER_IN_SNOWPACK, CUMSNOWMELT, VADOSE_LEVEL, PHREATIC_LEVEL]

        if self.muwts is None:
            horizontal_transfert = horizontal_transfert_mu.mean(dim=2)
        else:
            horizontal_transfert = (horizontal_transfert_mu * self.muwts).sum(-1)

        # --- 流量演算 (Routing) ---
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
        return out_dict
