from sapnet import *
from server import *
from action import SAP_ACTION_SPACE
from collections import namedtuple
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

RUNS = 1000
GAMMA = 0.999
ACTION_LIMIT = 15
LEARNING_RATE = 1e-5
GRAD_CLIP_NORM = 5

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class ActorCriticTrainer:
    def __init__(self, model, role):
        self.model = model
        self.server = SAPServer(role)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.action_history = deque([], maxlen=ACTION_LIMIT)
        self.reward_history = deque([], maxlen=ACTION_LIMIT)

    def select_action(self, image, hidden, mask):
        action_prob, state_value, hidden = self.model(image, hidden, mask)
        dist = Categorical(action_prob)
        index = dist.sample()
        print(action_prob)
        self.action_history.append(SavedAction(dist.log_prob(index), state_value))
        return SAP_ACTION_SPACE[index], hidden

    def update_model(self):
        R = 0
        policy_losses = []
        value_losses = []
        returns = []

        print("Reward history: ", self.reward_history)

        for r in self.reward_history[::-1]:
            R = r + GAMMA * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + EPS)

        print("Returns: ", returns)

        for (log_prob, value), R in zip(self.action_history, returns):
            advantage = R - value.item()
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(value.squeeze(0), torch.tensor([R])))


        policy_loss = torch.stack(policy_losses).sum()
        value_loss = torch.stack(value_losses).sum()
        loss = policy_loss + value_loss

        print("Policy loss: ", policy_loss.item()),
        print("Value loss:  ", value_loss.item())

        print("detect_anomaly enabled: ", torch.is_anomaly_enabled()) 
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_NORM)
        self.optimizer.step()

        self.action_history.clear()
        self.reward_history.clear()

    def train(self):
        torch.autograd.set_detect_anomaly(True)

        cumulative_reward = 0

        # For RUNS many Arena runs.
        for _ in range(RUNS):

            # Start an Arena run
            self.server.start_run()
            run_complete = False
            run_reward = 0
            turn = 1
            hidden = self.model.init_hidden(1)

            # We'll refer to one Arena run as an Episode in classic RL terms.
            while(run_complete is False):

                print(self.model.layer1[0].weight)

                action = None
                mask = None
                state = self.server.get_state()
                action_counter = 0
                shop_time_ended = False

                while(self.server.shop_ready(state) is False):
                     print("Waiting for shop to be ready")
                     self.server.click_center()
                     time.sleep(1)
                     state = self.server.get_state()

                print("-------------------")
                print("Beginning turn", turn)
                print("-------------------")

                # SHOP PHASE
                while(1):

                    # Fetch the shop state
                    state = self.server.get_state()

                    # Select an action mask based on turn and gold amount
                    mask = self.server.get_appropriate_mask(state, turn)

                    # Feed the shop state to the network
                    action, hidden = self.select_action(state, hidden, mask)
                    hidden = hidden.detach()

                    if (action == Action.A68):
                        print("Agent chose to end turn")        
                        self.server.start_battle(state)
                        break
                    if (action_counter == ACTION_LIMIT):
                        print("Reached action limit")
                        self.server.start_battle(state)
                        break
                    if (self.server.shop_ready(state) is False):
                        print("Ran out of time")
                        break

                    # Apply the action (also waits 1 second)
                    print("Applying: ", action)
                    self.server.apply(action)

                    # Increment action_counter
                    action_counter += 1

                print("----------------------")
                print("Beginning battle phase")
                print("----------------------")

                battle_status = Battle.ONGOING

                # BATTLE PHASE
                while(battle_status is Battle.ONGOING):
                    state = self.server.get_state()
                    battle_status = self.server.battle_status(state)
                    
                # UPDATE PHASE
                reward = self.server.reward_default(battle_status)
                print("Reward: ", reward)
                run_reward += reward
                self.reward_history = [0] * len(self.action_history)
                self.reward_history[-1] = reward

                self.model.save_old()

                # Update the model
                if (battle_status is not Battle.GAMEOVER):
                    self.update_model()
                    self.model.save()

                turn += 1
                if (battle_status is Battle.GAMEOVER):
                    while(self.server.run_complete(self.server.get_state()) is False):
                        self.server.click_top()
                        time.sleep(1)
                    run_complete = True

            cumulative_reward = 0.05 * run_reward + (1 - 0.05) * cumulative_reward
            print("Cumulative reward: ", cumulative_reward) 
