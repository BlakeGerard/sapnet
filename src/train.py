from sapnet import *
from server import *
from action import SAP_ACTION_SPACE
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

RUNS = 1000
GAMMA = 0.999
EPS = np.finfo(np.float32).eps.item()
ACTION_LIMIT = 20

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class ActorCriticTrainer:
    def __init__(self, model, role):
        self.model = model
        self.server = SAPServer(role)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-6)
        self.action_history = deque([], maxlen=ACTION_LIMIT)
        self.reward_history = deque([], maxlen=ACTION_LIMIT)

    def select_action(self, state, mask):
        action_prob, state_value = self.model(state, mask)
        dist = Categorical(action_prob)
        index = dist.sample()
        self.action_history.append(SavedAction(dist.log_prob(index), state_value))
        return SAP_ACTION_SPACE[index]

    def update_model(self):
        R = 0
        policy_losses = []
        value_losses = []
        returns = []

        print("Action history: ", self.action_history)
        print("Reward history: ", self.reward_history)
        print("Updating model")

        for r in self.reward_history[::-1]:
            R = R * GAMMA + r
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + EPS)

        for (log_prob, value), R in zip(self.action_history, returns):
            advantage = R - value.item()
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(value.squeeze(0), torch.tensor([R])))

        self.optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        print("Loss: ", loss.item())
        loss.backward()
        self.optimizer.step()
        self.action_history.clear()
        self.reward_history.clear()

    def train(self):
        cumulative_reward = 0

        # For RUNS many Arena runs.
        for _ in range(RUNS):

            # Start an Arena run
            self.server.start_run()
            run_complete = False
            run_reward = 0
            turn = 1

            # We'll refer to one Arena run as an Episode in classic RL terms.
            while(run_complete is False):
                
                action = None
                mask = None
                state = self.server.get_state()
                action_counter = 0

                print("-------------------")
                print("Beginning buy phase")
                print("-------------------")

                # Query network for actions while buy phase is ongoing
                while(1):

                    # Select an action mask based on turn and gold amount
                    mask = self.server.get_appropriate_mask(state, turn)

                    # Select and record an action based on the current shop state
                    action = self.select_action(state, mask)

                    if (action == Action.A58 or action_counter == ACTION_LIMIT or self.server.zero_gold(state)):
                        self.server.start_battle(state)
                        break

                    # Apply the action (also waits 1 second)
                    print("Applying: ", action)
                    self.server.apply(action)

                    # Observe the new state
                    state = self.server.get_state()

                    # Check if agent as zero gold or has selected End Turn
                    action_counter += 1

                print("----------------------")
                print("Beginning battle phase")
                print("----------------------")

                battle_status = Battle.ONGOING

                # Wait for the battle to finish
                while(battle_status is Battle.ONGOING):
                    time.sleep(5)

                    # Observe the new state
                    state = self.server.get_state()
                    battle_status = self.server.battle_status(state)
                    print(battle_status)
                    
                # Grant rewards
                reward = self.server.reward(battle_status)
                run_reward += reward
                self.reward_history = [reward] * len(self.action_history)

                # Update the model
                self.update_model()

                turn += 1
                if (battle_status is Battle.GAMEOVER):
                    run_complete = True

                # Click off the end-of-battle screen and level pop-ups
               	self.server.click_center()
                time.sleep(1)
                self.server.click_center()
                time.sleep(1)
                self.server.click_center()

            cumulative_reward = 0.05 * run_reward + (1 - 0.05) * cumulative_reward
            print("Cumulative reward: ", cumulative_reward) 
            self.model.save()
