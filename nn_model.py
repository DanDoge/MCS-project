import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, out_size):
        super(MLP, self).__init__()
        self.sizes = [input_size] + hidden_layers + [out_size]
        self.linears = [nn.Linear(in_dim, out_dim, True) for in_dim, out_dim in zip(self.sizes[: -1], self.sizes[1:])]
        self.linears = nn.ModuleList(self.linears)
        self.weight_init()

    def forward(self, x):
        for layer in self.linears[:-1]:
            x = F.relu(layer(x))
        x = self.linears[-1](x)
        return x

    def weight_init(self):
        for layer in self.linears:
            torch.nn.init.xavier_uniform(layer.weight)
            torch.nn.init.zeros_(layer.bias)

class MLP_bn(nn.Module):
    def __init__(self, input_size, hidden_layers, out_size):
        super(MLP_bn, self).__init__()
        self.sizes = [input_size] + hidden_layers + [out_size]
        self.linears = [nn.Sequential(nn.Linear(in_dim, out_dim, True), nn.BatchNorm1d(out_dim)) for in_dim, out_dim in zip(self.sizes[: -1], self.sizes[1:])]
        self.linears = nn.ModuleList(self.linears)
        self.weight_init()

    def forward(self, x):
        for layer in self.linears[:-1]:
            x = F.relu(layer(x))
        x = self.linears[-1][0](x)
        return x

    def weight_init(self):
        for layer in self.linears:
            torch.nn.init.xavier_uniform(layer[0].weight)
            torch.nn.init.zeros_(layer[0].bias)

class MLP_drop(nn.Module):
    def __init__(self, input_size, hidden_layers, out_size):
        super(MLP_drop, self).__init__()
        self.sizes = [input_size] + hidden_layers + [out_size]
        self.linears = [nn.Sequential(nn.Linear(in_dim, out_dim, True), nn.Dropout(0.5)) for in_dim, out_dim in zip(self.sizes[: -1], self.sizes[1:])]
        self.linears = nn.ModuleList(self.linears)
        self.weight_init()

    def forward(self, x):
        for layer in self.linears[:-1]:
            x = F.relu(layer(x))
        x = self.linears[-1][0](x)
        return x

    def weight_init(self):
        for layer in self.linears:
            torch.nn.init.xavier_uniform(layer[0].weight)
            torch.nn.init.zeros_(layer[0].bias)

def train_nn(model, data, num_epoch=5000):
    train_dataset = TensorDataset(torch.Tensor(data.Xtrain), torch.Tensor(data.Ytrain))
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=1024, shuffle=True)

    test_dataset = TensorDataset(torch.Tensor(data.Xtest), torch.Tensor(data.Ytest))
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1024)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

    test_losses = []
    train_losses = []
    for epoch in range(num_epoch):
        for inputs, targets in train_dataloader:
            optimizer.zero_grad()
            inputs = inputs.reshape([-1, 1, 100, 100])
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
        model.eval()
        te_loss = 0.
        for inputs, targets in test_dataloader:
            inputs = inputs.reshape([-1, 1, 100, 100])
            outputs = model(inputs)
            te_loss += torch.nn.L1Loss(reduction="mean")(outputs * 180 + 180, targets * 180 + 180).data
        te_loss = te_loss.item() / len(test_dataloader)
        test_losses.append(te_loss)
        tr_loss = 0.
        for inputs, targets in train_dataloader:
            inputs = inputs.reshape([-1, 1, 100, 100])
            outputs = model(inputs)
            tr_loss += torch.nn.L1Loss(reduction="mean")(outputs * 180 + 180, targets * 180 + 180).data
        tr_loss = tr_loss.item() / len(train_dataloader)
        train_losses.append(tr_loss)
        print(tr_loss, te_loss)
        model.train()

    return train_losses, test_losses

def train_nn_memory(model, data1, data2, num_epoch=5000):
    train_dataset = TensorDataset(torch.Tensor(data1.Xtrain), torch.Tensor(data1.Ytrain))
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

    test_dataset = TensorDataset(torch.Tensor(data1.Xtest), torch.Tensor(data1.Ytest))
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=128)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    test_losses = []
    train_losses = []
    for epoch in range(num_epoch):
        if epoch == num_epoch / 2:
            print("switch dataset...")
            train_dataset = TensorDataset(torch.Tensor(data2.Xtrain), torch.Tensor(data2.Ytrain))
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
        for inputs, targets in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        model.eval()
        te_loss = 0.
        for inputs, targets in test_dataloader:
            outputs = model(inputs)
            te_loss += torch.nn.MSELoss(reduction="sum")(outputs, targets).data
        te_loss = te_loss.data / len(test_dataloader)
        test_losses.append(te_loss)
        tr_loss = 0.
        for inputs, targets in train_dataloader:
            outputs = model(inputs)
            tr_loss += torch.nn.MSELoss(reduction="sum")(outputs, targets).data
        tr_loss = tr_loss.data / len(train_dataloader)
        train_losses.append(tr_loss)
        print(tr_loss, te_loss)
        model.train()

    return train_losses, test_losses
