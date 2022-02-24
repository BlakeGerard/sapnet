import math
import random
import numpy as np
from action import SAP_ACTION_SPACE
from collections import namedtuple, deque
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import get_image_num_channels, pil_to_tensor, normalize

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MASK_VAL = -1000.0

class MaskedSoftmax(nn.Module):
    def __init__(self):
        super(MaskedSoftmax, self).__init__()
        self.softmax = nn.Softmax(1)

    def forward(self, state, mask):
        state_masked = state.clone()
        state_masked[mask == 0] = MASK_VAL
        state_masked_normed = state_masked - torch.max(state_masked)
        return self.softmax(state_masked_normed)

class SAPNetActorCritic(nn.Module):

    def __init__(self, name):
        super(SAPNetActorCritic, self).__init__()
        self.name = name
        n_actions = len(SAP_ACTION_SPACE)

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 5, stride = 1, padding = "same"),
            nn.GELU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 5, stride = 1, padding = "same"),
            nn.GELU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 150 * 240, n_actions)
        self.dropout = nn.Dropout()
        self.action_head = MaskedSoftmax()
        self.critic_head = nn.Linear(n_actions, 1)
    
    def forward(self, image, mask):
        state = pil_to_tensor(image).float()
        means = [state[i:,:,].mean() for i in range(get_image_num_channels(state))]
        stds = [state[i:,:,].std()  for i in range(get_image_num_channels(state))]
        normalize(state, means, stds, True)
        state = state.unsqueeze(0)
        state = self.layer1(state)
        state = self.layer2(state)
        state = self.flatten(state)
        state = self.fc1(state)
        state = self.dropout(state)
        action_prob = self.action_head(state, mask)
        state_value = self.critic_head(state)
       	return action_prob, state_value

    def save(self):
	    path = "models/" + self.name + ".pt"
	    torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        #self.eval()
