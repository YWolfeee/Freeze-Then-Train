"""Evaluate DFR on spurious correlations datasets."""

import numpy as np
import tqdm
from utils_func.wb_data import WaterBirdsDataset, get_loader, get_transform_cub


# WaterBirds
# C_OPTIONS = [1., 0.7, 0.3, 0.1, 0.07, 0.03, 0.01]
C_OPTIONS = [1.]
CLASS_WEIGHT_OPTIONS = [1., 2., 3., 10., 100., 300., 1000.]
# CelebA
REG = "l1"
# # REG = "l2"
# C_OPTIONS = [3., 1., 0.3, 0.1, 0.03, 0.01, 0.003]
# CLASS_WEIGHT_OPTIONS = [1., 2., 3., 10., 100, 300, 500]

CLASS_WEIGHT_OPTIONS = [{0: 1, 1: w} for w in CLASS_WEIGHT_OPTIONS] + [
        {0: w, 1: 1} for w in CLASS_WEIGHT_OPTIONS]

def get_balance_subset(sets, ratio = 1.0):
    X, labels, groups = sets["X"], sets["label"], sets["group"]
    n_groups = np.max(groups) + 1
    assert n_groups == 4
    g_idx = [np.where(np.logical_or(groups == i, groups == i + 2))[0] for i in range(2)]
    assert len(g_idx) == 2
    n_total = np.min([len(g) for g in g_idx])
    [np.random.shuffle(g) for g in g_idx]
    
    if type(ratio) == float:
        ratio = [ratio]
    assert type(ratio) == list and sum(ratio) <= 1.0
    s = 0.0
    ret_lis = []
    for rat in ratio:
        lower = int(n_total * s)
        s += rat
        upper = int(n_total * s)
        ret_lis.append((
            np.concatenate([X[g[lower:upper]] for g in g_idx]), 
            np.concatenate([labels[g[lower:upper]] for g in g_idx]), 
            np.concatenate([groups[g[lower:upper]] for g in g_idx])
        ))
    
    return ret_lis if len(ret_lis) > 1 else ret_lis[0]

## Load data
def get_sets(data_dir, core_noise: int = 0, drop_trian = False):
    target_resolution = (224, 224)
    train_transform = get_transform_cub(target_resolution=target_resolution,
                                        train=True, augment_data=False)
    test_transform = get_transform_cub(target_resolution=target_resolution,
                                    train=False, augment_data=False)

    if not drop_trian:
        trainset = WaterBirdsDataset(
            basedir= data_dir, split="train", core_noise=core_noise, transform=train_transform)
    valset = WaterBirdsDataset(
        basedir= data_dir, split="val", core_noise=core_noise, transform=test_transform)
    # for the testing set, we never add noise, because we use it to evalute.
    testset = WaterBirdsDataset(
        basedir= data_dir, split="test", core_noise=0, transform=test_transform)
    if not drop_trian:
        return {"train": trainset, "val": valset, "test": testset}
    return {"val": valset, "test": testset}

def get_loaders(batch_size, sets_dic):
    loader_kwargs = {'batch_size': batch_size,
                    'num_workers': 4, 'pin_memory': True,
                    "reweight_places": None}

    return {key: get_loader(
        sets, 
        train=(key=="train"), 
        reweight_groups=False if key == "train" else None, 
        reweight_classes=False if key == "train" else None,
        **loader_kwargs
    ) for key, sets in sets_dic.items()}

def torch_to_num(x):
    return x.detach().cpu().numpy()

def get_all_embiddings(loader_dic, feature_model):
    embeddings = {}
    for key, loader in loader_dic.items():
        embeddings[key] = {}
        feature_lis, label_lis, group_lis, _ = [], [], [], []
        # print("Start extracting {} features ...".format(key))
        for data, label, group, _ in tqdm.tqdm(loader):
            feature_lis.append(feature_model(data.cuda()))
            label_lis.append(torch_to_num(label))
            group_lis.append(torch_to_num(group))
        embeddings[key]["X"] = np.vstack(feature_lis)
        embeddings[key]["label"] = np.concatenate(label_lis)
        embeddings[key]["group"] = np.concatenate(group_lis)
    return embeddings