import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

df = pd.read_csv('data.csv')
df = df.drop('Sample', axis=1)
data = np.array(df)
data = np.float32(data)

data = torch.from_numpy(data)
# print(data.dtype)
x = data[:, 5:]
y = data[:, 0:5]
y = torch.unsqueeze(y, dim=1)
print(f'Input data shape: {x.shape}')
print(f'Output data shape: {y.shape}')


class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

    def __getitem__(self, index):

        ID = self.x[index]

        # Load data and get label
        X = self.x[index]
        Y = self.y[index]

        return X, Y

# train_dataset = Dataset(x[:40], y[:40])
# vali_dataset = Dataset(x[40:], y[40:])
# # print(dataset[5])

# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
# vali_loader = torch.utils.data.DataLoader(dataset=vali_dataset, batch_size=100, shuffle=True)

# a, b = next(iter(train_loader))
# print(f'Training samples: {len(train_dataset)}')
# print(f'Validation samples: {len(vali_dataset)}')





# class Net(torch.nn.Module):
#     def __init__(self, n_feature, n_hidden, n_output):
#         super(Net, self).__init__()#继承
#         self.hidden_1 = torch.nn.Linear(n_feature,n_hidden)   # hidden layer
#         self.hidden_2 = torch.nn.Linear(n_hidden,100)
#         self.hidden_3 = torch.nn.Linear(100,n_hidden)
#         self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

#     def forward(self, x):
#         # print(x.shape)
#         x1 = self.hidden_1(x)
#         x2 = F.relu(x1)
#         x3 = self.hidden_2(x2)
#         x4 = F.relu(x3)
#         x5 = self.hidden_3(x4)
#         x6 = F.relu(x5)
#         y = self.predict(x6)
#         return y

# net = Net(n_feature=12, n_hidden=10, n_output=1)     # define the network
# # print(net)  # net architecture

# optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
# loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

# # plt.ion()   # something about plotting

# for t in range(1000):
#     net.train()
#     x_, y_ = next(iter(train_loader))
#     prediction = net(x_)     # input x and predict based on x

#     # print(prediction.shape, y.shape)
#     loss = loss_func(prediction, y_)     # must be (1. nn output, 2. target)
#     # print(loss)

#     optimizer.zero_grad()   # clear gradients for next train
#     loss.backward()         # backpropagation, compute gradients
#     optimizer.step()        # apply gradients

#     if t % 500 == 0:
#         net.eval()
#         x_val, y_val = next(iter(vali_loader))
#         prediction = net(x_val)     # input x and predict based on x
#         loss_val = loss_func(prediction, y_val)     # must be (1. nn output, 2. target)
#         print(f'Epoch {t}: Validation loss: {loss}')


from kan import *
# torch.set_default_dtype(torch.float64)
# create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
model = KAN(width=[8, 8, 4, 5], grid=8, k=5, seed=42)

from kan.utils import create_dataset
# create dataset f(x,y) = exp(sin(pi*x)+y^2)
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
dataset = create_dataset(f, n_var=8)
print(dataset['train_input'].shape, dataset['train_label'].shape)

dataset['train_input'] = x[:40]
dataset['train_label'] = y[:40]

print(dataset['train_input'].shape, dataset['train_label'].shape)


print(model)
# plot KAN at initialization
model(dataset['train_input'])
model.plot(beta=100, in_vars=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'], out_vars=['y1', 'y2', 'y3', 'y4', 'y5'], title = 'My KAN', metric='act')
model.plot(beta=100)
plt.show()


model.fit(dataset, opt="LBFGS", steps=50, lamb=0.001)
# model = model.prune()
# model.fit(dataset, opt="LBFGS", steps=50)
# model = model.refine(10)
# model.fit(dataset, opt="LBFGS", steps=50)

# mode = "auto" # "manual"

# if mode == "manual":
#     # manual mode
#     model.fix_symbolic(0,0,0,'sin')
#     model.fix_symbolic(0,1,0,'x^2')
#     model.fix_symbolic(1,0,0,'exp')
# elif mode == "auto":
#     # automatic mode
#     lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']
#     model.auto_symbolic(lib=lib)

# model.fit(dataset, opt="LBFGS", steps=50)

# from kan.utils import ex_round

# print(ex_round(model.symbolic_formula()[0][0],4))
# model.plot()
# plt.show()
