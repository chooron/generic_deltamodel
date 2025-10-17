1.1 Benchmark comparison

(1)  Predictive performance

比较在dPL-HBV 1.1架构下，替换了参数的动态估计器LSTM，选用GRU，Transformer以及SSM模型下，dPL-HBV 1.1的预测精度，并与benchmark的LSTM和SSM比较（共六个模型）。

【精度累积频率分布图】

【流域shp上点精度，以及与dPL-HBV-LSTM模型的差异图】

(2)  PUB performance

-       使用十折交叉检验下，dPL-HBV-LSTM与dPL-HBV-SSM模型的预测精度。

-       使用PUR分区，对大面积的无资料流域的预测精度进行率定

1.2 Inspect Model

parameters and inner fluxes

(1)  分析四种模型对于模型动态参数估计的结果特性

a)    不同HRU的参数动态估计结果的变化特性

b)    相似性

c)    统计特征

(2)  分析四种模型对于SWE和BFI两个中间变量的拟合精度



1.3 Explain Model

with post-hoc interpretability method

对比LSTM与其他深度学习在动态参数预测时的事后可解释性分析结果：

a)    对于历史数据的关注程度

b)    对于观测数据和流域属性数据的关注程度