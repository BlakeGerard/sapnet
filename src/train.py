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
ACTION_LIMIT = 20
LEARNING_RATE = 1e-4
GRAD_CLIP_VAL = 5
E = 0.2

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

def animate_loss(iteration, policy_loss, value_loss):
    total_loss = policy_loss + value_loss
    x.append(iteration)

class ActorCriticTrainer:
    def __init__(self, model, role):
        self.model = model
        self.server = SAPServer(role)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.old_action_history = deque([], maxlen=ACTION_LIMIT)
        self.action_history = deque([], maxlen=ACTION_LIMIT)
        self.reward_history = deque([], maxlen=ACTION_LIMIT)

    def select_action(self, image, mask):
        action_prob, state_value = self.model(image, mask)
        dist = Categorical(action_prob)
        index = dist.sample()
        print(action_prob)
        self.action_history.append(SavedAction(dist.log_prob(index), state_value))
        return SAP_ACTION_SPACE[index]

    def animate_loss(self, turn):
        x = np.arange(0, turn)
        self.loss_ax.clear()
        self.loss_ax.plot(x, self.policy_loss_history)
        self.loss_ax.plot(x, self.value_loss_history)
        self.loss_ax.plot(x, np.asarray(self.policy_loss_history) + np.asarray(self.value_loss_history))

    def update_model_pg(self):
        R = 0
        policy_losses = []
        value_losses = []
        returns = []

        for r in self.reward_history[::-1]:
            R = r + GAMMA * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        print("Returns: ", returns)

        for (log_prob, value), R in zip(self.action_history, returns):
            advantage = R - value.item()
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.huber_loss(value, torch.tensor([R]).unsqueeze(0)))

        print(policy_losses)
        print(value_losses)

        policy_loss = torch.stack(policy_losses).sum()
        value_loss = torch.stack(value_losses).sum()
        loss = policy_loss + value_loss

        loss_file = open("loss.txt", "a") 
        loss_file.write("Policy loss: {}\n".format(policy_loss.item()))
        loss_file.write("Value loss: {}\n".format(value_loss.item()))
        loss_file.write("Total loss: {}\n".format(loss.item()))
        loss_file.write("---------------\n")
        loss_file.close()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_VAL)
        self.optimizer.step()

        self.action_history.clear()
        self.reward_history.clear()

    def update_model_lclip(self):
        R = 0
        returns = []

        it = min(len(self.old_action_history), len(self.action_history))
        if (it == 0):
            self.old_action_history = self.action_history
            print("Updating with pg method")
            self.update_model_pg()        
            return

        print("Updating with lclip method")
        for r in self.reward_history[::-1]:
            R = r + GAMMA * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        advantage = []
        for i in range(it):
            advantage.append(R - self.action_history[i][1].item())

        policy_ratio = []
        for i in range(it):
            policy_ratio.append(self.action_history[i][0] / self.old_action_history[i][0])
        
        policy_losses = []
        value_losses = []
        for i in range(it):
            policy_losses.append(min( (policy_ratio[i] * advantage[i]), 
                                      (torch.clip(policy_ratio[i], 1-E, 1+E) * advantage[i])
                                    ))
            value_losses.append(F.huber_loss(self.action_history[i][1], torch.tensor([R]).unsqueeze(0)))

        print(policy_losses)
        print(value_losses)

        policy_loss = torch.stack(policy_losses).sum()
        value_loss = torch.stack(value_losses).sum()
        loss = policy_loss - value_loss

        loss_file = open("loss.txt", "a") 
        loss_file.write("Policy loss: {}\n".format(policy_loss.item()))
        loss_file.write("Value loss: {}\n".format(value_loss.item()))
        loss_file.write("Total loss: {}\n".format(loss.item()))
        loss_file.write("---------------\n")
        loss_file.close()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.old_action_history = self.action_history
        self.action_history.clear()
        self.reward_history.clear()

    def train(self):
        cumulative_reward = 0

        # For RUNS many Arena runs.
        for _ in range(RUNS):

            # Start an Arena run
            self.server.start_run()
            time.sleep(1)
            run_complete = False
            run_reward = 0
            turn = 1

            # We'll refer to one Arena run as an Episode in classic RL terms.
            while(run_complete is False):

                action = None
                mask = None
                action_counter = 0
                shop_time_ended = False

                while(self.server.shop_ready(self.server.get_full_state()) is False):
                     print("Waiting for shop to be ready")
                     self.server.click_center()
                     time.sleep(1)
                     state = self.server.get_full_state()

                print("-------------------")
                print("Beginning turn", turn)
                print("-------------------")

                self.model.hidden = self.model.init_hidden(1)

                # SHOP PHASE
                while(1):

                    # Select an action mask based on turn and gold amount
                    mask = self.server.get_appropriate_mask(self.server.get_full_state(), turn, action_counter)

                    # Feed the shop state to the network
                    state = self.server.get_shop_state()
                    action = self.select_action(state, mask)

                    if (action == Action.A68):
                        print("Agent chose to end turn")        
                        self.server.start_battle(state)
                        break
                    if (action_counter == ACTION_LIMIT):
                        print("Reached action limit")
                        self.server.start_battle(state)
                        break
                    if (self.server.shop_ready(self.server.get_full_state()) is False):
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

                while(self.server.battle_ready(self.server.get_full_state()) is False):
                    print("Waiting for battle to start")

                battle_start = time.time()

                # BATTLE PHASE
                battle_status = Battle.ONGOING
                while(battle_status is Battle.ONGOING):
                    state = self.server.get_full_state()
                    battle_status = self.server.battle_status(state)
                    
                battle_duration = time.time() - battle_start

                # UPDATE PHASE
                reward = self.server.reward_duration(battle_status, battle_duration)
                print("Reward: ", reward)
                run_reward += reward
                self.reward_history = [0] * len(self.action_history)
                self.reward_history[-1] = reward


                # Update the model
                if (battle_status is not Battle.GAMEOVER):
                    self.model.save_old()
                    self.update_model_pg()
                    self.model.save()

                turn += 1
                if (battle_status is Battle.GAMEOVER):
                    while(self.server.run_complete(self.server.get_full_state()) is False):
                        self.server.click_top()
                        time.sleep(1)
                    run_complete = True

            cumulative_reward = 0.05 * run_reward + (1 - 0.05) * cumulative_reward
            print("Cumulative reward: ", cumulative_reward) 
