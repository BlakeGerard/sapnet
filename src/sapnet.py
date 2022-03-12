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

class RnnPreproc(nn.Module):
    def __init__(self):
        super(RnnPreproc, self).__init__()
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, state):
        state = self.flatten(state)
        state = state.unsqueeze(1)
        return state

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
        self.hidden_size = 256
        self.hidden = None

        self.transform = tv.Compose([
            tv.ToTensor(),
            tv.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ])

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size = 3, stride = 1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.LeakyReLU()
        )
        self.layer1.apply(init_weights)

        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size = 3, stride = 1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.LeakyReLU()
        )
        self.layer2.apply(init_weights)

        self.rnn_preproc = RnnPreproc()

        self.gru = nn.GRU(16 * 95 * 118, self.hidden_size, self.gru_layers)
        self.gru.apply(init_weights)

        self.fc = nn.Linear(self.hidden_size, N_ACTIONS)  
        self.fc.apply(init_weights)

        self.action_head = MaskedSoftmax()
        self.action_head.apply(init_weights)

        self.value_head = nn.Linear(N_ACTIONS, 1)
        self.value_head.apply(init_weights)

    def forward(self, image, mask):
        state = self.transform(image)
        state = state.unsqueeze(0)    # Add batch dimension
        state = self.layer1(state)
        state = self.layer2(state)
        state = self.rnn_preproc(state)
        state, self.hidden = self.gru(state, self.hidden)
        state = state.squeeze(0)      # Remove batch dimension
        state = self.fc(state)
        action_prob = self.action_head(state, mask)
        state_value = self.value_head(state)
       	return action_prob, state_value

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
