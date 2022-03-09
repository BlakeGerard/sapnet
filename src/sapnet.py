import math
import random
import numpy as np
from action import SAP_ACTION_SPACE
from collections import namedtuple, deque
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tv

MASK_VAL = -float("inf")
EPS = np.finfo(np.float32).eps.item()
N_ACTIONS = len(SAP_ACTION_SPACE)

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class MaskedSoftmax(nn.Module):
    def __init__(self):
        super(MaskedSoftmax, self).__init__()
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, state, mask):
        state = state - torch.max(state)
        state[mask == 0] = MASK_VAL
        out = self.softmax(state)
        return out

class SAPNetActorCritic(nn.Module):

    def __init__(self, name):
        super(SAPNetActorCritic, self).__init__()
        self.name = name
        self.gru_layers = 256
        self.hidden_size = 128

        self.transform = tv.Compose([
            tv.ToTensor(),
            tv.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ])

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size = 3, stride = 1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.LeakyReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size = 3, stride = 1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.LeakyReLU()
        )

        #self.layer3 = nn.Sequential(
        #    nn.Conv2d(16, 32, kernel_size = 3, stride = 1),
        #    nn.MaxPool2d(kernel_size = 2, stride = 2),
        #    nn.LeakyReLU()
        #)

        self.flatten = nn.Flatten()
        self.gru = nn.GRU(16 * 148 * 238, self.hidden_size, self.gru_layers, batch_first=True)
        
        self.layer4 = nn.Sequential(
            nn.Linear(self.hidden_size, N_ACTIONS),
            nn.LeakyReLU()
        )
        
        self.action_head = MaskedSoftmax()
        self.critic_head = nn.Linear(N_ACTIONS, 1)
    
    def forward(self, image, hidden, mask):
        state = self.transform(image)
        state = state.unsqueeze(0)
        state = self.layer1(state)
        state = self.layer2(state)
        #state = self.layer3(state)
        state = self.flatten(state)
        state = state.unsqueeze(0)
        state, hidden = self.gru(state, hidden)
        hidden = hidden.detach()
        state = state.squeeze(0)
        state = self.layer4(state)
        action_prob = self.action_head(state, mask)
        state_value = self.critic_head(state)
       	return action_prob, state_value, hidden

    def init_hidden(self, batch_size=1):
        weight = next(self.parameters()).data
        hidden = weight.new(self.gru_layers, batch_size, self.hidden_size).zero_()
        return hidden

    def save(self):
	    path = "models/" + self.name + ".pt"
	    torch.save(self.state_dict(), path)

    def save_old(self):
        path = "models/" + self.name + ".old.pt"
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
