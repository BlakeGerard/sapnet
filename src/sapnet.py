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

class MaskedSoftmax(nn.Module):
    def __init__(self):
        super(MaskedSoftmax, self).__init__()
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, state, mask):
        state_normed = state - torch.max(state)
        state_normed[mask == 0] = MASK_VAL
        return self.softmax(state_normed)

class SAPNetActorCritic(nn.Module):

    def __init__(self, name):
        super(SAPNetActorCritic, self).__init__()
        self.name = name
        self.gru_layers = 1
        self.hidden_size = 64

        self.transform = tv.ToTensor()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size = 5, stride = 1, padding='same'),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = 5, stride = 1, padding='same'),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()
        self.gru = nn.GRU(32 * 150 * 240, self.hidden_size, self.gru_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, N_ACTIONS)
        self.action_head = MaskedSoftmax()
        self.critic_head = nn.Linear(N_ACTIONS, 1)
    
    def forward(self, image, hidden, mask):
        state = self.transform(image).unsqueeze(0)
        #print(state)
        state = self.layer1(state)
        #print(state)
        state = self.layer2(state)
        #print(state.shape)
        state = self.flatten(state).unsqueeze(0)
        #print(state.shape)
        state, hidden = self.gru(state, hidden)
        #print(state.shape)
        state = state.squeeze(0)
        #print(state.shape)
        state = self.fc(state)
        #print(state.shape)
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
