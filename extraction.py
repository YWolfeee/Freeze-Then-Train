from copy import deepcopy
import os
import numpy as np
import time
import argparse
import pandas as pd
import torch
from utils_func.models import FTT_resnet
from utils_func.dfr_utils import get_sets, get_loaders, get_all_embiddings, get_balance_subset
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

MY_NET_METHODS = ["FTT", "ERM", "PCA", "IRM"]
C_OPTIONS = [10., 5., 1., 0.5, 0.1, 0.05, 0.01]

def create_data_loaders(path, core_noise, batch_size = 256):
    all_split_datasets = get_sets(path, core_noise=core_noise, drop_trian=True)
    return get_loaders(batch_size, all_split_datasets)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type = str, default = "./data/waterbirds_v1.0")
    parser.add_argument("--base_dir", type = str)
    parser.add_argument("--metric", type = str, default = "worst")
    parser.add_argument("--method", type = str, required=True)
    parser.add_argument("--sup_fraction", type = float, default = 0.5)
    parser.add_argument("--unsup_fraction", type = float, default = 0.5)
    parser.add_argument("--core_noise", type = int, default = 0)
    parser.add_argument("--gen_embeddings", action="store_true",
                        help="Whether to generate embeddings from scratch.")
    parser.add_argument("--LR_avg_times", type = int, default = 10,
                        help="How many times we calculate LR and average.")
    parser.add_argument("--repeat_times", type = int, default = 10,
                        help="How many times do we repeat the whole setting.")
    parser.add_argument("--model_type", type = str, default = "resnet50")
    parser.add_argument("--projection_dim_list", nargs="+", type = int, default = [-1])

    args = parser.parse_args()
    return args

def construct_model(state, method = "FTT", model_name = "resnet50", n_classes = 2,
                    method_dic = None): 
    # method_dic should pass method specific argument
    # when method == FTT, sup_fraction and unsup_fraction should be passed.
    # Otherwise, hidden_dim should be pass
    if method in MY_NET_METHODS:     # load our own type of model
        model = FTT_resnet(model_name, method_dic["sup_fraction"], method_dic["unsup_fraction"], pretrained=False, n_classes=n_classes)
        model.load_state_dict(state)
    else:
        model = FTT_resnet(model_name, 1.0, 0.0, pretrained=False, n_classes=n_classes)
        state.fc = torch.nn.Identity()
        model.model = state
    model.cuda()
    model.eval()
    return model

def gen_embeddings(all_data_loaders, base_dir, method, method_dic, model_type = "resnet50", model_name_filter = None):
    em_dir = "embeddings" 
    os.makedirs(os.path.join(base_dir, em_dir), exist_ok=True)
    # name_dic = [w for w in os.listdir(os.path.join(base_dir, "models")) if w[-3:] == ".pt" and "checkpoint" in w]
    name_dic = [w for w in os.listdir(os.path.join(base_dir, "models")) if w[-3:] == ".pt"]
    if model_name_filter is not None:
        name_dic = [w for w in name_dic if w[:-3] in model_name_filter] # :-3 drops `.pt`
    name_dic = {w[:-3]: w for w in name_dic}        # convert to dic
    print("Generating embeddings for {}. Will save to {}".format(
        name_dic, em_dir))

    for tag, fname in name_dic.items():
        print("-------- Start extracting features for `{}` --------".format(tag))
        model_state = torch.load(os.path.join(base_dir, "models", fname))
        model = construct_model(model_state, method, model_type, method_dic=method_dic)
        feature_model = model.get_embed

        with torch.no_grad():
            embeddings = get_all_embiddings(all_data_loaders, feature_model)
        np.save(os.path.join(base_dir, em_dir, tag), embeddings)

        del model
        torch.cuda.empty_cache()
    print("Finished the whole embeddings generation.\n")


def get_acc(pred, y, group, print_acc = False):
    n_groups = np.max(group) + 1
    bc = np.bincount(group)
    test_accs = [(pred == y)[group == g].mean() * 100
                 for g in range(n_groups)]
    if n_groups == 4:
        worst_acc = round(np.min([
                (test_accs[0] * bc[0] + test_accs[2] * bc[2]) / (bc[0] + bc[2]),
                (test_accs[1] * bc[1] + test_accs[3] * bc[3]) / (bc[1] + bc[3])]
           ), 4)
    elif n_groups == 2:
        worst_acc = round(np.min(test_accs), 4)
    dic = {"mean": round((pred == y).mean() * 100, 4),
           "worst": worst_acc,
           "accs": [round(w, 4) for w in test_accs],
           "bincount": list(bc)
        }
    if print_acc:
        print("Mean: {:.2f}%, Worst: {:.2f}%, {}".format(
            dic["mean"], dic["worst"], dic["accs"]))
    return dic

@ignore_warnings(category=ConvergenceWarning)
def fit_model_train(embedding, kargs, down_sample_rate):
    (X, y, g), (Xtest, ytest, gtest) = get_balance_subset(embedding, [0.5 * down_sample_rate, 0.5 * down_sample_rate])
    model = LogisticRegression(n_jobs = 1, **kargs)
    model.fit(X, y)
    val_accs = get_acc(model.predict(Xtest), ytest, gtest)

    return model, val_accs

@ignore_warnings(category=ConvergenceWarning)
def fit_model_val(embedding, repeat, kargs, down_sample_rate):
    coef, intercept = [], []
    for _ in range(repeat):
        X, y, g = get_balance_subset(embedding, down_sample_rate)
        model = LogisticRegression(n_jobs = 1, **kargs)
        model.fit(X, y)
        coef.append(model.coef_)
        intercept.append(model.intercept_)
    model.coef_, model.intercept_ = np.mean(coef, axis = 0), np.mean(intercept, axis = 0)

    return model


def map_embeddings(embeddings: dict, pmodel: PCA):
    return {
        key:{
            k: pmodel.transform(v) if k == "X" else v
            for k,v in dic.items()
        } for key, dic in embeddings.items()}

def down_sample_set(sets, rate = 1.0, verbose = False):
    X, labels, groups = sets["X"], sets["label"], sets["group"]
    n_groups = np.max(groups) + 1
    assert n_groups == 4
    g_idx = [np.where(np.logical_or(groups == i, groups == i + 2))[0] for i in range(2)]
    assert len(g_idx) == 2
    n_total = np.min([len(g) for g in g_idx])
    [np.random.shuffle(g) for g in g_idx]

    if verbose:
        print("len(g) = {}, n_total = {}, rate = {}.".format(len(g_idx), n_total,rate))
    return {
        "X": np.concatenate([X[g[:int(rate * len(g))]] for g in g_idx]), 
        "label": np.concatenate([labels[g[:int(rate * len(g))]] for g in g_idx]), 
        "group": np.concatenate([groups[g[:int(rate * len(g))]] for g in g_idx])
    }

def get_low_embedding(embeddings, hidden_dim, down_sample_rate = 1.0, drop_diagonal = False):

    if hidden_dim != -1:
        pca_model = PCA(n_components=hidden_dim)
    else:
        pca_model = PCA()       # This will keep all features but still normalize

    pca_model.fit(embeddings["val"]["X"])
    if drop_diagonal:
        cov = np.cov(embeddings["val"]["X"].T)
        mask = (np.ones_like(cov) - np.eye(cov.shape[0]))
        _, diag, Vt = np.linalg.svd(cov * mask)
        
        pca_model.components_ = Vt[np.argsort(-np.abs(diag))[:hidden_dim]]

    embeddings = map_embeddings(embeddings, pca_model)
    embeddings["val"] = down_sample_set(embeddings["val"], down_sample_rate, verbose = True)
    print("down sample after PCA. bincount is {}".format(np.bincount(embeddings["val"]["group"])))
    return embeddings, pca_model


def one_calculation(embeddings, repeat, kargs, metric, down_sample_rate = 1.0, verbose = False, return_model = False ):
    # sweep the hyper parameters and select the best
    val_dic, best_acc, best_C = {}, -1, 1

    if "C_OPTIONS" in kargs.keys():
        OPTIONS = kargs["C_OPTIONS"]
        del kargs["C_OPTIONS"]
    else:
        OPTIONS = C_OPTIONS

    for C in OPTIONS:
        kargs_C = deepcopy(kargs)
        kargs_C["C"], val_dic[C] = C, []
        try:
            for _ in range(3):
                res = fit_model_train(embeddings["val"], kargs = kargs_C, down_sample_rate = down_sample_rate)[1]
                val_dic[C].append(res[metric])  # select hyper-parameters according to metric.
            val_dic[C] = np.mean(val_dic[C])
            if verbose:
                print("C = {:5}, worst_acc = {}".format(C, val_dic[C]))

            if val_dic[C] > best_acc:
                best_acc, best_C = val_dic[C], C
        except Exception as e:
            if verbose:
                print("Ignore C = {}. Encounter error = {}".format(
                    C, str(e)
                ))
    if verbose:
        print("Finish selection using worst acc, C = {}, avg_worst_acc = {}".format(best_C, best_acc))
    kargs["C"] = best_C

    model = fit_model_val(embeddings["val"], repeat, kargs, down_sample_rate = down_sample_rate)

    if verbose:
        print("Training:")
    val_accs = get_acc(model.predict(embeddings["val"]["X"]),
                            embeddings["val"]["label"],
                            embeddings["val"]["group"],
                            print_acc=verbose)

    if verbose:
        print("Testing:")
    test_results = get_acc(model.predict(embeddings["test"]["X"]), 
                            embeddings["test"]["label"], 
                            embeddings["test"]["group"], 
                            print_acc=verbose)
    res = {}
    for key, acc in val_accs.items():
        res["train_"+key] = acc
    for key, acc in test_results.items():
        res["test_"+key] = acc
    if not return_model:
        return res
    else:
        return res, model

def gen_df(base_dir, LR_avg_times = 10, repeat_times = 5, metric = "worst", projection_dim_list = [-1], step_filter = None, kargs = {}, drop_diagonal = False, down_sample_rate = 1.0, verbose = True):
    em_dir = "embeddings" 
    embed_dir = os.path.join(base_dir, em_dir)
    step_list = [w.split(".")[0] for w in os.listdir(embed_dir)]
    
    if step_filter is not None:
        step_list = [w for w in step_list if w in step_filter]
    print("load frm {}. step_list len = {}, Full list = {}".format(embed_dir, len(step_list), step_list))
    
    ret_step_list, typ_list, acc_list, proj_dim_list = [], [], [], []
    

    for projection_dim in projection_dim_list:
        start = time.time()
        for step in step_list:
            origin_embeddings = np.load(
                    os.path.join(embed_dir, str(step) + ".npy"), 
                    allow_pickle=True
                ).item()
            # Just use for records
            proj_dim = projection_dim if projection_dim != -1 else origin_embeddings["val"]["X"].shape[1]
            embeddings, pca_model = get_low_embedding(
                                        origin_embeddings, 
                                        hidden_dim = projection_dim, 
                                        down_sample_rate = down_sample_rate,
                                        drop_diagonal=drop_diagonal)
            print("## Project dim = {:4}, epoch = {}, captured variance = {:.4f} ##".format(
                projection_dim, step, sum(pca_model.explained_variance_ratio_)))
            
            for _ in range(repeat_times):
                A = one_calculation(embeddings, metric = metric, down_sample_rate = down_sample_rate, 
                                    repeat = LR_avg_times, kargs = kargs, verbose = verbose)
                if verbose:
                    print()
                for typ in ["train_mean","train_worst","test_mean", "test_worst"]:
                    ret_step_list.append(step)
                    typ_list.append(typ)
                    acc_list.append(A[typ])
                    proj_dim_list.append(proj_dim)
        duration = time.time() - start
        print(f"Dimension {projection_dim} takes {duration} seconds ({len(step_list)} step and {repeat_times} runs).")

    return pd.DataFrame({
        "step": ret_step_list, 
        "typ": typ_list,
        "acc": acc_list,
        "proj_dim": proj_dim_list
    })


def mymain(args):
    if args.gen_embeddings:
        all_loaders = create_data_loaders(args.data_dir, args.core_noise)
        if args.method in MY_NET_METHODS:
            method_dic = {"sup_fraction": args.sup_fraction, "unsup_fraction": args.unsup_fraction}
        else:
            method_dic = {"hidden_dim": -1}
        gen_embeddings(all_loaders, args.base_dir, args.method, method_dic,
                           model_type= args.model_type)

    df = gen_df(args.base_dir, args.LR_avg_times, args.repeat_times, metric = args.metric, projection_dim_list=args.projection_dim_list)
    fname = "probing_results+" + args.metric
    if args.projection_dim_list != [-1]:
        fname += "+projection"
    df.to_csv(os.path.join(args.base_dir, 
                f"{fname}.csv"), index=False)


if __name__== "__main__":
    args = get_args()
    mymain(args)