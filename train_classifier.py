import torch
import numpy as np
import pandas as pd
import os
import tqdm
import argparse
import sys
import json
from functools import partial
from copy import deepcopy
from sklearn.decomposition import PCA

from utils_func.wb_data import WaterBirdsDataset, get_loader, get_transform_cub, log_data
from utils_func.utils import (
    get_irm_penalty, update_dict, get_results,
    Logger, AverageMeter, set_seed, evaluate, get_y_p
)
from utils_func.models import FTT_resnet
from utils_func.dfr_utils import get_sets, get_loaders

parser = argparse.ArgumentParser()
# os path
parser.add_argument("--data_dir", type=str, default=None, help="Train dataset directory")
parser.add_argument("--output_dir", type=str, default="logs/", help="Output directory")

# train and model
parser.add_argument("--pretrained_model", action='store_true', help="Use pretrained model")
parser.add_argument("--model_type", type=str, default="resnet50")
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--reweight_classes", action='store_true', help="Reweight classes")
parser.add_argument("--reweight_places", action='store_true', help="Reweight based on place")
parser.add_argument("--reweight_groups", action='store_true', help="Reweight groups")
parser.add_argument("--augment_data", action='store_true', help="Train data augmentation")
parser.add_argument("--scheduler", action='store_true', help="Learning rate scheduler")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=300)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--momentum_decay", type=float, default=0.9)
parser.add_argument("--init_lr", type=float, default=0.001)
parser.add_argument("--save_middle_pt", action="store_true", help = "Whether to save the pt in the middle.")
parser.add_argument("--eval_freq", type=int, default=10)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--irm_weight", type = float, default = 10.0,
                    help="The weight on irm penalty term.")
parser.add_argument('--temperature', default=0.07, type=float,
                    help="softmax temperature (default: 0.07)")

# FTT relevant
parser.add_argument("--core_noise", type = int, choices = [0,2,4,6,8,10], default = 0,
                    help = "The noise of core features in percentage.")
parser.add_argument("--sup_fraction", type=float, default=0.5, 
                    help='The fraction of features that are allocated to supervised training. \
Only useful when both `supervised` and `unsupervised` is True.')
parser.add_argument("--unsup_fraction", type=float, default=0.5, 
                    help='The fraction of features that are allocated to unsupervised training. \
Only useful when both `supervised` and `unsupervised` is True.')
parser.add_argument("--unsup_method", type = str, default = "PCA", choices=["PCA", "SSL"],
                    help="Which algorithm is used for unsupervised training.")
parser.add_argument("--sup_method", type = str, default = "ERM", choices=["ERM", "IRM"],
                    help="Which algorithm is used for unsupervised training.")

args = parser.parse_args()
assert args.reweight_groups + args.reweight_classes <= 1

print('Preparing directory %s' % args.output_dir)
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, "embeddings"), exist_ok=True)
with open(os.path.join(args.output_dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
    args_json = json.dumps(vars(args))
    f.write(args_json)

set_seed(args.seed)
logger = Logger(os.path.join(args.output_dir, 'log.txt'))

# Data
splits = ["train", "test", "val"]
basedir = args.data_dir
target_resolution = (224, 224)
train_transform = get_transform_cub(target_resolution=target_resolution, train=True, augment_data=args.augment_data)
test_transform = get_transform_cub(target_resolution=target_resolution, train=False, augment_data=args.augment_data)

trainset = WaterBirdsDataset(basedir=basedir, split="train", core_noise=args.core_noise,
        transform=train_transform)
testset_dict = {
    'wb': WaterBirdsDataset(basedir=args.data_dir, split="test", core_noise=args.core_noise,
                            transform=test_transform),
}

loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 4, 'pin_memory': True}
train_loader = get_loader(
    trainset, train=True, reweight_groups=args.reweight_groups,
    reweight_classes=args.reweight_classes, reweight_places=args.reweight_places, **loader_kwargs)
test_loader_dict = {}
for test_name, testset_v in testset_dict.items():
    test_loader_dict[test_name] = get_loader(
        testset_v, train=False, reweight_groups=None,
        reweight_classes=None, reweight_places=None, **loader_kwargs)

get_yp_func = partial(get_y_p, n_places=trainset.n_places)
log_data(logger, trainset, testset_dict['wb'], get_yp_func=get_yp_func)

# Model
n_classes = trainset.n_classes
model = FTT_resnet(args.model_type, 
                     sup_fraction = args.sup_fraction, unsup_fraction = args.unsup_fraction,
                     pretrained = args.pretrained_model, n_classes = n_classes)
print("using {}. model dim = {}, bimodel dim = {}".format(
    args.model_type, model.model_dim, model.bimodel_dim
))


# TODO: fix resuming from a checkpoint
if args.resume is not None:
    print('Resuming from checkpoint at {}...'.format(args.resume))
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint)

model.cuda()

optimizer = torch.optim.SGD(
    model.parameters(), lr=args.init_lr, momentum=args.momentum_decay, weight_decay=args.weight_decay)
if args.scheduler:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs)
else:
    scheduler = None

criterion = torch.nn.CrossEntropyLoss()

logger.flush()

train_dict = {
    'epoch': [],
    'mean_accuracy': [],
    'worst_accuracy': [],
    'accuracy_0_0': [],
    'accuracy_0_1': [],
    'accuracy_1_0': [],
    'accuracy_1_1': [],
}
test_dict = deepcopy(train_dict)
train_dict["loss"] = []

def create_data_loaders(path, core_noise, batch_size = 256):
    all_split_datasets = get_sets(path, core_noise=core_noise, drop_trian=True)
    return get_loaders(batch_size, all_split_datasets)
embedding_used_loaders = create_data_loaders(args.data_dir, core_noise = args.core_noise)

def test_and_save(epoch):
    for test_name, test_loader in test_loader_dict.items():
        results = evaluate(model, test_loader, get_yp_func)
        logger.write("Test results \n")
        logger.write(str(results))
        for key in results.keys():
            test_dict[key].append(results[key])
        test_dict["epoch"].append(epoch)


def unsupervised_training():
    # will use the eval mode to conduct PCA
    # use the train split since we only have this during the training stage.
    print("Unsupervised training: preserve the highest varying features.")
    model.eval()
    with torch.no_grad():
        embeddings = []
        for x, _, _, _ in tqdm.tqdm(train_loader):
            embeddings.append(model.get_embed(x.cuda()))
        embeddings: np.ndarray = np.concatenate(embeddings, axis = 0)
        # PCA to find the most salient features
        pcamodel: PCA = PCA(n_components=model.bimodel_dim)
        pcamodel.fit(embeddings)
        if args.unsup_method == "SSL":
            cov = np.cov(embeddings.T)
            mask = (np.ones_like(cov) - np.eye(cov.shape[0]))
            _, diag, Vt = np.linalg.svd(cov * mask)
            pcamodel.components_ = Vt[np.argsort(-np.abs(diag))[:model.bimodel_dim]]
            print("## Project dim = {:4}, using SSL.".format(model.bimodel_dim))
        elif args.unsup_method == "PCA":
            print("## Project dim = {:4}, captured variance = {:.4f} ##".format(
            model.bimodel_dim, sum(pcamodel.explained_variance_ratio_)))
        weight, bias = pcamodel.components_, -np.dot(
            pcamodel.mean_.reshape((1, -1)), pcamodel.components_.T).reshape(-1)
        model.set_bimodel_fc(weight, bias)


def supervised_training():
    # Train loop
    for epoch in range(args.num_epochs):
        if epoch % args.eval_freq == 0:
            # Iterating over datasets we test on
            test_and_save(epoch)
            if args.save_middle_pt: # Save the checkpoint in the middle round
                torch.save(model.state_dict(), os.path.join(args.output_dir, "models",f'{epoch}.pt'))

        model.train()
        loss_meter = AverageMeter()
        acc_groups = {g_idx : AverageMeter() for g_idx in range(trainset.n_groups)}

        for batch in tqdm.tqdm(train_loader):
            x, y, g, p = batch
            x, y, p = x.cuda(), y.cuda(), p.cuda()

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            if args.sup_method == "IRM":
                penalty = get_irm_penalty(logits, y)
                loss += min(epoch, args.irm_weight) * penalty
            else:
                penalty = 0.0

            loss.backward()
            optimizer.step()
            
            loss_meter.update(loss, x.size(0))
            update_dict(acc_groups, y, g, logits)

        if args.scheduler:
            scheduler.step()
        logger.write(f"Epoch {epoch}\t Loss: {loss_meter.avg}\n")
        logger.write("Last batch\t loss: {:.4f}, penalty: {:.4f}".format(
            loss, penalty
        ))
        results = get_results(acc_groups, get_yp_func)
        logger.write(f"Train results \n")
        logger.write(str(results) + "\n")
        logger.write('\n')
        
        for key in results.keys():
            train_dict[key].append(results[key])
        train_dict["epoch"].append(epoch)
        train_dict["loss"].append(loss_meter.avg.cpu().item())

    test_and_save(args.num_epochs)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "models",'final_checkpoint.pt'))
    pd.DataFrame(train_dict).to_csv(os.path.join(args.output_dir, "train_stats.csv"), index = False)
    pd.DataFrame(test_dict).to_csv(os.path.join(args.output_dir, "test_stats.csv"), index = False)

if args.unsup_fraction > 0:
    unsupervised_training()
supervised_training()