import torch


# ----------------------------------------------------------------------------
# Canopy Process Functions (冠层过程函数)
# ----------------------------------------------------------------------------

def canopy_interception_evaporation_max(pet: torch.Tensor, Fc: torch.Tensor, fs: torch.Tensor) -> torch.Tensor:
    """
    计算冠层最大截留/蒸发量 (Maximum Method)。
    :param pet: 降雨或降雪张量。
    :param Fc: 冠层最大截留容量。
    :param fs: 冠层最大截留容量。
    :return: 截留/蒸发量。
    """
    # 截留量是降水量和最大存储容量中的较小者。
    return pet * Fc * (1 - fs)


# ----------------------------------------------------------------------------
# Soil Evaporation Functions (土壤蒸发函数)
# ----------------------------------------------------------------------------

def soil_evaporation_hbv(pet: torch.Tensor, soil_moisture: torch.Tensor, field_capacity: torch.Tensor) -> torch.Tensor:
    """
    根据土壤湿度计算实际蒸发量 (HBV method)。
    :param pet: 潜在蒸发量。
    :param soil_moisture: 当前土壤湿度。
    :param field_capacity: 土壤田间持水量（一个重要的土壤参数）。
    :return: 实际土壤蒸发量。
    """
    # 实际蒸发量与土壤湿润程度成正比。
    return pet * soil_moisture / field_capacity


# ----------------------------------------------------------------------------
# Infiltration Functions (入渗函数)
# ----------------------------------------------------------------------------

def infiltration_flush(available_water: torch.Tensor) -> torch.Tensor:
    """
    所有有效水分全部入渗 (100% Infiltration)。
    :param available_water: 可用于入渗的水量 (如：穿透雨)。
    :return: 入渗量。
    """
    return available_water


def infiltration_gr4j(soil_moisture: torch.Tensor,
                      ponded_water: torch.Tensor,
                      field_capacity: torch.Tensor) -> torch.Tensor:
    """
    模拟GR4J产流模型中的入渗过程 (GR4J Method)。
    :param ponded_water: 可用于入渗的水量。
    :param soil_moisture: 当前产流层土壤湿度。
    :param field_capacity: 产流层最大容量 (GR4J中的X1参数)。
    :return: 入渗量。
    """
    # 计算有效入渗量，当土壤越湿时，入渗比例越小。
    alpha = torch.tanh(ponded_water / field_capacity)
    infiltration = field_capacity * (alpha * (1 - torch.pow(soil_moisture / field_capacity, 2))
                                     / (1 + alpha * soil_moisture / field_capacity))
    return torch.clamp(infiltration, max=ponded_water)


def infiltration_partitioning_coefficient(available_water: torch.Tensor, pc: float) -> torch.Tensor:
    """
    使用分配系数来计算入渗量 (Partitioning Coefficient)。
    :param available_water: 可用于入渗的水量。
    :param pc: 入渗分配系数 [0, 1]。
    :return: 入渗量。
    """
    return available_water * (1 - pc)


def infiltration_hbv(available_water: torch.Tensor, soil_moisture: torch.Tensor,
                     field_capacity: torch.Tensor, beta: float) -> torch.Tensor:
    """
    使用HBV模型的非线性关系计算入渗/补给量 (HBV Method)。
    :param available_water: 可用于入渗的水量。
    :param soil_moisture: 当前土壤湿度。
    :param field_capacity: 田间持水量。
    :param beta: 控制非线性关系的指数参数 [0.5, 3]。
    :return: 入渗量。
    """
    # 入渗量与土壤湿润程度的beta次幂成正比。
    infiltration_ratio = torch.pow(soil_moisture / field_capacity, beta)
    return available_water * (1 - infiltration_ratio)


def infiltration_vic_arno(available_water: torch.Tensor, soil_moisture: torch.Tensor, max_soil_moisture: torch.Tensor,
                          bexp: float) -> torch.Tensor:
    """
    基于VIC/ARNO可变入渗曲线思想计算入渗 (VIC-ARNO Method)。
    :param available_water: 可用于入渗的水量。
    :param soil_moisture: 当前土壤湿度。
    :param max_soil_moisture: 最大土壤湿度。
    :param bexp: 曲线形状参数 [0.001, 3]。
    :return: 入渗量。
    """
    # 入渗发生在非饱和区域。
    infiltration = available_water * (
            1.0 - torch.pow((torch.clamp(1.0 - soil_moisture / max_soil_moisture, min=1e-6)), bexp))
    return infiltration


def infiltration_hmets(
        available_water: torch.Tensor,
        soil_moisture: torch.Tensor,
        field_capacity: torch.Tensor,
        alpha: torch.Tensor
) -> torch.Tensor:
    """
    HMETS模型中的简化入渗计算 (HMETS Method)。
    :param available_water: 可用于入渗的水量。
    :param soil_moisture: 控制入渗分布的形状参数 [0.3, 1]。
    :param field_capacity: 控制入渗分布的形状参数 [0.3, 1]。
    :param alpha: 控制入渗分布的形状参数 [0.3, 1]。
    :return: 入渗量。
    """
    # 这是一个简化的形式，alpha控制产流和入渗的非线性分配。
    return available_water * (1 - alpha * (soil_moisture / field_capacity))


# ----------------------------------------------------------------------------
# Surface Runoff Functions (地表径流函数)
# ----------------------------------------------------------------------------

def surface_runoff_infiltration_excess(available_water: torch.Tensor, infiltration: torch.Tensor) -> torch.Tensor:
    """
    计算超渗产流 (Infiltration Excess)。
    :param available_water: 可用于入渗的总水量。
    :param infiltration: 实际入渗量。
    :return: 地表径流量。
    """
    runoff = available_water - infiltration
    # 径流量不能为负。
    return torch.maximum(torch.tensor(0.0), runoff)


# ----------------------------------------------------------------------------
# Baseflow Soil Layer 1 Functions (土壤层1基流函数)
# ----------------------------------------------------------------------------

def baseflow_off(storage: torch.Tensor) -> torch.Tensor:
    """
    无基流产出 (No Flow)。
    :param storage: 储水量 (为保持接口一致性)。
    :return: 零张量。
    """
    return torch.zeros_like(storage)


def baseflow_constant_rate(storage: torch.Tensor, bfmax: torch.Tensor) -> torch.Tensor:
    """
    恒定速率的基流 (Constant Rate)。
    :param storage: 储水层水量。
    :param bfmax: 恒定基流速率 [mm/d]。
    :return: 基流量。
    """
    return torch.minimum(bfmax, storage)


def baseflow_linear(storage: torch.Tensor, bfk: torch.Tensor, bfmax: torch.Tensor) -> torch.Tensor:
    """
    线性水库基流 (Linear Baseflow)。
    :param storage: 储水层水量。
    :param bfk: 线性退水系数 [1/d]。
    :param bfmax: 线性退水系数 [1/d]。
    :return: 基流量。
    """
    return torch.minimum(bfmax, ((10 ** bfk) * storage))


def baseflow_exp(storage: torch.Tensor, max_storage: torch.Tensor, bfmax: torch.Tensor, bfexp: torch.Tensor):
    return torch.minimum(storage, bfmax * torch.exp(bfexp * torch.clamp(storage / max_storage, 0, 1) - 1))


def baseflow_gr4j_exchange(storage: torch.Tensor, gr4j_x3: torch.Tensor, bfmax: torch.Tensor) -> torch.Tensor:
    """
    模拟GR4J的汇流交换项 (GR4J Method)。这不是一个典型的基流函数。
    :param storage: 汇流层储水量。
    :param gr4j_x3: 交换系数 (X3-1) [mm]。
    :param bfmax: 交换系数 (X3-1) [mm]。
    :return: 交换水量 (正为增益，负为损失)。
    """
    # 该项描述的是与外部的交换，而非单纯的基流产出。
    # 计算基于储水量的四次方关系。
    return torch.minimum(bfmax, storage * torch.clamp(
        (1 - torch.pow(1 + torch.clamp(storage / gr4j_x3, min=1e-6) ** 4, -1 / 4)), min=0.0, max=1.0))


def baseflow_power_law(storage: torch.Tensor, bfc: torch.Tensor, bfn: torch.Tensor,
                       bfmax: torch.Tensor) -> torch.Tensor:
    """
    幂律形式的基流计算 (Power Law Baseflow)。
    :param storage: 储水层水量。
    :param bfc: 基流系数 [1/d]。
    :param bfn: 基流指数 [-]。
    :param bfmax: 基流指数 [-]。
    :return: 基流量。
    """
    return torch.minimum(bfmax, torch.minimum((10 ** (bfc)) * torch.pow(storage, bfn), storage))  # 确保基流不超过储水量


def baseflow_vic(storage: torch.Tensor, max_storage: torch.Tensor,
                 bfmax: torch.Tensor, bfn: torch.Tensor) -> torch.Tensor:
    """
    VIC模型的非线性基流计算 (VIC Method)。
    :param storage: 储水层水量。
    :param max_storage: 最大储水量。
    :param bfmax: 最大基流速率 [mm/d]。
    :param bfn: 非线性指数 [-]。
    :return: 基流量。
    """
    return torch.minimum(bfmax * torch.pow(storage / max_storage, bfn), storage)


def baseflow_topmodel(storage: torch.Tensor, max_storage: torch.Tensor,
                      bfmax: torch.Tensor, bflambda: torch.Tensor,
                      bfn: torch.Tensor) -> torch.Tensor:
    """
    TOPMODEL的指数形式基流计算 (TOPMODEL Method)。
    :param storage: 平均饱和水亏。
    :param bfmax: 饱和时的最大基流速率 [mm/d]。
    :param bflambda: 地形水文参数 [m]。
    :param bfn: 地形水文参数 [m]。
    :param max_storage: 地形水文参数 [m]。
    :return: 基流量。
    """
    # 基流随饱和水亏指数衰减。
    return torch.minimum(
        bfmax * torch.clamp(max_storage / (bfn * torch.pow(bflambda, bfn)), min=1e-6, max=1.0) *
        torch.pow(storage / max_storage, bfn), storage)


def baseflow_threshold(storage: torch.Tensor, max_storage: torch.Tensor,
                       bfmax: torch.Tensor, bfn: torch.Tensor, bfthresh: torch.Tensor) -> torch.Tensor:
    """
    带阈值的基流计算 (Threshold Baseflow)。
    :param storage: 储水层水量。
    :param max_storage: 最大储水量。
    :param bfmax: 最大基流速率 [mm/d]。
    :param bfn: 非线性指数 [-]。
    :param bfthresh: 产流阈值比例 [0, 1]。
    :return: 基流量。
    """
    return torch.minimum(
        torch.pow(torch.clamp(storage / max_storage - bfthresh, min=1e-6) / (1 - bfthresh), bfn) * bfmax, storage
    )


# ----------------------------------------------------------------------------
# Percolation & Capillary Rise Functions (渗漏与毛管上升函数)
# ----------------------------------------------------------------------------

def percolation_gawser(storage: torch.Tensor, max_perc: torch.Tensor,
                       max_storage: torch.Tensor, sfc: torch.Tensor) -> torch.Tensor:
    """
    GAWSER模型中的最大渗漏率法 (GAWSER Method)。
    :param available_water: 可用于渗漏的水量 (如：上层土壤排水)。
    :param max_perc: 最大渗漏速率 [mm/d]。
    :return: 渗漏量。
    """
    return torch.minimum(storage, max_perc *
                         (torch.clamp(storage - max_storage * sfc, min=1e-6) / (max_storage - max_storage * sfc)))


def capillary_rise_hbv(storage: torch.Tensor, max_storage: torch.Tensor, max_caprise: torch.Tensor) -> torch.Tensor:
    """
    HBV模型中的毛管水上升 (HBV Method)。
    :param storage: 上层土壤水亏 (max_moisture - current_moisture)。
    :param max_storage: 上层土壤水亏 (max_moisture - current_moisture)。
    :param max_caprise: 上层土壤水亏 (max_moisture - current_moisture)。
    :return: 毛管水上升量。
    """
    # 上升量不能超过土壤水亏，也不能超过最大速率。
    return torch.minimum(max_caprise * (1 - storage / max_storage), storage)


def snowmelt_hbv(tmean, snowpack, cumulmelt, ddf_min, ddf_plus, Kcum, Tbm):
    ddf = torch.minimum(ddf_min + ddf_plus, ddf_min * (1 + Kcum * cumulmelt))
    potenmelt = torch.clamp(ddf * (tmean - Tbm), min=0.0)
    snowmelt = torch.min(potenmelt, snowpack)
    return snowmelt


def refreeze_hbv(tmean, liquidwater, Tbf, Kf):
    return torch.min(Kf * torch.clamp(Tbf - tmean, min=0.0), liquidwater)


def overflow_hbv(snowpack, water_in_snowpack, swi):
    return torch.clamp(water_in_snowpack - swi * snowpack, 0.0)
