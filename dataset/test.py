import os
from zarr import convenience
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.data import MapAPI
from Physical_data import PhysicalDataset
from Vectorizer import PhysicalVectorizer


os.environ["L5KIT_DATA_FOLDER"] = "/tmp/l5kit_data"
cfg = load_config_data("./vectorized_test.yaml")
dm = LocalDataManager(None)
mapAPI = MapAPI.from_cfg(dm, cfg)
train_cfg = cfg["train_data_loader"]
train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
vectorizer = PhysicalVectorizer(mapAPI, cfg)

train_dataset = PhysicalDataset(cfg, train_zarr, vectorizer)

fp = open("./vectorized_test.txt","a+")
print(train_dataset[50]["all_agents_polyline"], file=fp)
fp.close()