import torch
from torch.utils.data import DataLoader, TensorDataset

import util
import model as Model

if __name__=="__main__" :
    print("hello")
    print(util.get_word_freq("unique"))
    data = util.load_data(feature_type="tfidf")
    label = data["category"]
    feature = data["feature"]
    input_size = feature.shape[-1]

    x = torch.from_numpy(feature[:128].toarray()).float()
    dataloader = DataLoader(TensorDataset(x), batch_size=4, shuffle=True, drop_last=True)

    model = Model.NASH(input_size)
    optimizer = torch.optim.Adam(model.parameters())

    for e, batch in enumerate(dataloader) :
        loss, kl_loss = model(x)
        tot_loss = loss - kl_loss
        tot_loss.backward()
        optimizer.zero_grad()
        optimizer.step()
        print(tot_loss.item())