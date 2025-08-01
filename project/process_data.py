import pickle
import numpy as np

with open(r"E:\PaperCode\generic_deltamodel\data\camels_data\camels_dataset", 'rb') as f:
    forcings, target, attributes = pickle.load(f)
gage_id = np.load(r"E:\PaperCode\generic_deltamodel\data\camels_data\gage_id.npy")
np.savez_compressed(
    r"E:\PaperCode\generic_deltamodel\data\camels_data\camels_dataset.npz",
    gage_ids=gage_id,
    forcings=forcings,
    target=target,
    attributes=attributes,
)
