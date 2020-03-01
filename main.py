import argparse
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.neighbors import DistanceMetric

import util
import model as Model


def eval_retrieval(vector, label, batch_size=1000, top_n=100) :
    distance = DistanceMetric.get_metric("hamming").pairwise(vector, vector)
    np.fill_diagonal(distance, np.Infinity)
    sort_idx = np.argsort(distance, axis=-1)
    predict = []
    for i in range(top_n) :
        predict.append(
            (np.equal(label[sort_idx[:, i].reshape(-1)], label)).reshape(-1, 1)
            )
    predict = np.concatenate(predict, axis=1)
    precision = np.mean(np.sum(predict, axis=1)/top_n)
    precision = round(precision, 2)
    print("Top {} precision : {}".format(top_n, precision))
    return precision


def encode(model, feature, batch_size=512, use_cuda=True) :
    vector = []
    model.eval()
    for i in range(0, feature.shape[0], batch_size) :
        x = feature[i:i+batch_size].toarray()
        x = torch.from_numpy(x).float()
        if use_cuda :
            x = x.cuda()
        with torch.no_grad() :
            vector.append(
                model.encode(x).detach().cpu().numpy().astype("uint8")
            )
    vector = np.vstack(vector)
    model.train()
    return vector


def main(config) :

    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    data = util.load_data(feature_type=config.feature_type)
    label = data["category"]
    feature = data["feature"]
    input_size = feature.shape[-1]

    dataloader = DataLoader(TensorDataset(
         torch.from_numpy(feature.toarray()).float()
    ), batch_size=config.batch_size, shuffle=True, drop_last=True)

    model = Model.NASH(config, input_size)
    if config.use_cuda :
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_step, gamma=config.lr_decay_rate)

    max_precision = 0
    best_weight = None
    for e in range(config.epoch) :
        avg_loss = 0
        for batch in tqdm(dataloader) :
            x = batch[0]
            if config.use_cuda :
                x = x.cuda()
            tot_loss = model(x)
            avg_loss += tot_loss.item()
            tot_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        print("train epoch {} : {}".format(e+1, avg_loss/len(dataloader)))
        vector = encode(model, feature, use_cuda=config.use_cuda)
        precision = eval_retrieval(vector, label, top_n=config.top_n)
        if precision > max_precision :
            max_precision = precision
            best_weight = model.state_dict()
        print("")
        print("")
    torch.save(best_weight, "best.w")


if __name__=="__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cuda", default=True, type=bool, help="use cuda or not")
    parser.add_argument("--feature_type" , default="tfidf", type=str, help="tfidf | onehot")
    parser.add_argument("--epoch" , default=10, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--random_seed", default=0, type=int)
    parser.add_argument("--hidden_size" , default=500, type=int)
    parser.add_argument("--output_size" , default=64, type=int)
    parser.add_argument("--dropout" , default=0.1, type=float)
    parser.add_argument("--deterministic" , default=True, type=bool, help="use deterministic binarization or not")
    parser.add_argument("--lr_decay_step" , default=1e4, type=int)
    parser.add_argument("--lr_decay_rate" , default=0.96, type=float)
    parser.add_argument("--top_n" , default=100, type=int, help="number of top n retrieved number")

    config = parser.parse_args()
    main(config)