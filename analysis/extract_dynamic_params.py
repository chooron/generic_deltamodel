import numpy as np
import matplotlib.pyplot as plt
from os.path import join

from shapely.lib import length

load_path = r"E:\PaperCode\generic_deltamodel\project\output\blendv3.1\camels_531\train1999-2008\no_multi\LstmMlpModel_E100_R365_B100_H128_128_n1_noLn_noWU_111111\blendv3\NseBatchLoss\stat\test1989-1999_Ep100"

w1_arr = np.load(join(load_path, 'all_infil.npy')).squeeze()
ww_arr = np.load(join(load_path, 'weight_params.npy')).squeeze()
s2_arr = np.load(join(load_path, 'soilwater2.npy')).squeeze()
tt = w1_arr[:,1,:]
ww = ww_arr[:,1,:]
w2_arr = np.load(join(load_path, 'w5.npy'))
w3_arr = np.load(join(load_path, 'w6.npy'))

var_name = 'infil'
quickflow1_arr = np.load(join(load_path, f'{var_name}_1.npy'))
quickflow2_arr = np.load(join(load_path, f'{var_name}_2.npy'))
quickflow3_arr = np.load(join(load_path, f'{var_name}_3.npy'))

basin_idx = 1
node_num = 12

# plt.plot(w1_arr[:, basin_idx, node_num])
# plt.plot(w2_arr[:, basin_idx, node_num])
# plt.plot(w3_arr[:, basin_idx, node_num])
# # plt.plot(w2_arr[:, 1, 2])
# # plt.plot(w3_arr[:, 1, 2])
# plt.show()
#
# plt.plot(w1_arr[:, basin_idx, node_num] * quickflow1_arr[:, basin_idx, node_num])
# plt.plot(w2_arr[:, basin_idx, node_num] * quickflow2_arr[:, basin_idx, node_num])
# plt.plot(w3_arr[:, basin_idx, node_num] * quickflow3_arr[:, basin_idx, node_num])
# # plt.plot(w2_arr[:, 1, 2])
# # plt.plot(w3_arr[:, 1, 2])
# plt.show()

sum_all = np.mean(w1_arr[:, basin_idx] * quickflow1_arr[:, basin_idx], axis=1) + \
          np.mean(w2_arr[:, basin_idx] * quickflow2_arr[:, basin_idx], axis=1) + \
          np.mean(w3_arr[:, basin_idx] * quickflow3_arr[:, basin_idx], axis=1)
seq1 = (np.mean(w1_arr[:, basin_idx] * quickflow1_arr[:, basin_idx], axis=1) / (sum_all + 1e-3))
seq2 = (np.mean(w2_arr[:, basin_idx] * quickflow2_arr[:, basin_idx], axis=1) / (sum_all + 1e-3))
seq3 = (np.mean(w3_arr[:, basin_idx] * quickflow3_arr[:, basin_idx], axis=1) / (sum_all + 1e-3))

node_num = 10
sum_all = w1_arr[:, basin_idx, node_num] * quickflow1_arr[:, basin_idx, node_num] + \
          w2_arr[:, basin_idx, node_num] * quickflow2_arr[:, basin_idx, node_num] + \
          w3_arr[:, basin_idx, node_num] * quickflow3_arr[:, basin_idx, node_num]
mean_sum_all = np.median(sum_all)
seq1 = w1_arr[:, basin_idx, node_num] * quickflow1_arr[:, basin_idx, node_num] / (sum_all + 1e-3)
seq2 = w2_arr[:, basin_idx, node_num] * quickflow2_arr[:, basin_idx, node_num] / (sum_all + 1e-3)
seq3 = w3_arr[:, basin_idx, node_num] * quickflow3_arr[:, basin_idx, node_num] / (sum_all + 1e-3)
mean_seq1, mean_seq2, mean_seq3 = np.mean(seq1), np.mean(seq2), np.mean(seq3)

plt.figure(figsize=(10, 6))
plt.stackplot(range(len(seq1)), seq1, seq2, seq3, labels=['Sequence 1', 'Sequence 2', 'Sequence 3'], alpha=1)
plt.xlabel('Time')
plt.ylabel('Proportion')
plt.title('Stacked Area Plot of Three Sequences')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
