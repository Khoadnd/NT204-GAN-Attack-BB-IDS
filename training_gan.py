#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn


# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms


# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split

X = pd.read_feather('dataset/train.feather')
y = X.iloc[:, -1]
X = X.iloc[:, :-1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.25, stratify=y)

train_target = torch.tensor(y_train.values.astype('float32'))
train = torch.tensor(X_train.values.astype('float32'))
train_tensor = TensorDataset(train, train_target)
train_loader = DataLoader(train_tensor, batch_size=64, shuffle=True)

test_target = torch.tensor(y_test.values.astype('float32'))
test = torch.tensor(X_test.values.astype('float32'))
test_tensor = TensorDataset(test, test_target)
test_loader = DataLoader(test_tensor, batch_size=64, shuffle=True)


# In[4]:


class Critic(nn.Module):
  def __init__(self, in_feature) -> None:
    super().__init__()
    self.disc = nn.Sequential(
      nn.Linear(in_feature, 128),
      nn.LeakyReLU(),
      nn.Linear(128, 1),
    )
    
  def forward(self, x):
    return self.disc(x)
  
class Generator(nn.Module):
  def __init__(self, z_dim, out_dim) -> None:
    super().__init__()
    self.gen = nn.Sequential(
      nn.Linear(z_dim, 128),
      nn.LeakyReLU(),
      nn.Linear(128, out_dim),
      nn.Tanh()
    )
    
  def forward(self, x):
    return self.gen(x)
  
def initialize_weight(model):
  for m in model.modules():
    if isinstance(m, nn.Linear):
      nn.init.normal_(m.weight.data, 0.0, 0.02)
  
# load blackbox model
import pickle

with open('models/ExtraTrees.pickle', 'rb') as f:
  blackbox = pickle.load(f)


# In[5]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'

z_dim = in_feature = train.shape[1]

critic = Critic(in_feature).to(device)
gen = Generator(z_dim, in_feature).to(device)
initialize_weight(critic)
initialize_weight(gen)

num_epochs = 15
batch_size = 64
n_critics = 5
clipping_value = 0.01

ids_loss = nn.CrossEntropyLoss()
lambda_ = 0.3

lr = 1e-4
opt_critic = optim.RMSprop(critic.parameters(), lr=lr)
opt_gen = optim.RMSprop(gen.parameters(), lr=lr)

gen.train()
critic.train()


# In[6]:


for epoch in range(num_epochs):
  for batch_idx, (real, _) in enumerate(train_loader):
    real = real.to(device)
    
    for _ in range(n_critics):
      noise = torch.randn(batch_size, z_dim).to(device)
      fake = gen(noise)
      critic_real = critic(real)
      critic_fake = critic(fake)
      loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
      critic.zero_grad()
      loss_critic.backward(retain_graph=True)
      opt_critic.step()
      
      for p in critic.parameters():
        p.data.clamp_(-clipping_value, clipping_value)
    
    output = critic(fake)
    with torch.no_grad():
      bb = torch.tensor(blackbox.predict(
          fake).astype('float32')).to(device)
    loss_gen = -torch.mean(output) + lambda_ * ids_loss(bb, torch.zeros_like(bb))
    gen.zero_grad()
    loss_gen.backward()
    opt_gen.step()
    
    
    if batch_idx == 0:
      print(
        f"Epoch [{epoch}/{num_epochs}] \ "
        f"Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
      )

