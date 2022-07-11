from sapnet import *
from server import *
from action import SAP_ACTION_SPACE
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Training parameters
RUNS = 1000
GAMMA = 0.90
ACTION_LIMIT = 20
LEARNING_RATE = 5e-4
GRAD_CLIP_VAL = 10
E = 0.2

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class ActorCriticTrainer:
    def __init__(self, model, role):
        self.model = model
        self.server = SAPServer(role)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)
        self.action_history = deque([], maxlen=ACTION_LIMIT)
        self.reward_history = deque([], maxlen=ACTION_LIMIT)

    # Invoke the model to select an action
    def select_action(self, image, mask):
        action_prob, state_value = self.model(image, mask)
        dist = Categorical(action_prob)
        index = dist.sample()
        print(action_prob)
        self.action_history.append(SavedAction(dist.log_prob(index), state_value))
        return SAP_ACTION_SPACE[index]

    # Default integer reward function
    def reward_default(self, battle_status):
        if (battle_status is Battle.WIN):
            return 1
        if (battle_status is Battle.DRAW):
            return 1
        if (battle_status is Battle.LOSS):
            return -1
        if (battle_status is Battle.GAMEOVER):
            return 0

    # Reward based on the duration of the battle
    def reward_duration(self, battle_status, duration):
        base = 0
        if (battle_status is Battle.WIN or battle_status is Battle.RUN_WIN or battle_status >
            base = 1
        elif (battle_status is Battle.LOSS or battle_status is Battle.RUN_LOSS):
            base = -1
        return base * (20.0 - duration)

    # Update the model
    def update_model(self):
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

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_VAL)
        self.optimizer.step()

        self.action_history.clear()
        self.reward_history.clear()

    def train(self):
        cumulative_reward = 0

        # For RUNS many Arena runs.
        for _ in range(RUNS):

            # Start an Arena run
            self.server.start_arena_run()
            run_complete = False
            run_reward = 0
            turn = 1

            # We'll refer to one Arena run as an Episode in classic RL terms.
            while(run_complete is False):

                action = None
                mask = None
                action_counter = 0
                shop_time_ended = False
                self.model.hidden = self.model.init_hidden(1)

                print("-------------------")
                print("Beginning turn", turn)
                print("-------------------")

                # SHOP PHASE
                while(1):

                    shop_state = self.server.get_full_state()

                    # Check if we ran out of time
                    if (self.server.shop_ready(shop_state) is False):
                        print("Ran out of time")
                        break

                    # Select an action mask based on turn and gold amount
                    mask = self.server.get_appropriate_mask(shop_state, turn, action_counter)

                    # Feed the shop state to the network
                    action = self.select_action(shop_state, mask)

                    # Check if we hit any shop terminal states
                    if (action == Action.A68):
                        print("Agent chose to end turn")
                        self.server.start_battle(state)
                        break
                    if (action_counter == ACTION_LIMIT):
                        print("Reached action limit")
                        self.server.start_battle(state)
                        break

                    # Apply the action
                    print("Applying: ", action)
                    self.server.apply(action)

                    # Increment action_counter
                    action_counter += 1


                # Wait for the battle to start
                while(self.server.battle_ready(self.server.get_full_state()) is False):
                    print("Waiting for battle to start")

                print("----------------------")
                print("Beginning battle phase")
                print("----------------------")

                 # Start battle timer
                battle_start = time.time()
                print("Battle timer started")

                # Wait for the battle to complete
                battle_status = Battle.ONGOING
                while(battle_status is Battle.ONGOING):
                    state = self.server.get_full_state()
                    battle_status = self.server.battle_status(state)

                # End battle timer
                battle_duration = time.time() - battle_start
                print("Battle timer stopped")

                # UPDATE PHASE
                reward = self.reward_default(battle_status)
                print("Reward: ", reward)
                run_reward += reward

                # 0 reward for all actions except the last
                self.reward_history = [0] * len(self.action_history)
                self.reward_history[-1] = reward

                # Update the model
                self.update_model_pg()
                self.model.save()

                turn += 1

                # If the run is over, signal run complete
                if (battle_status is Battle.RUN_WIN or battle_status is Battle.RUN_LOSS):
                    while(self.server.run_complete(self.server.get_full_state()) is False):
                        self.server.click_top()
                        time.sleep(1)
                    run_complete = True

            cumulative_reward = 0.05 * run_reward + (1 - 0.05) * cumulative_reward
            print("Cumulative reward: ", cumulative_reward) 
