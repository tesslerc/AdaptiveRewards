import numpy as np
import torch
import torch.nn.functional as F
import argparse
import os

from utils import ReplayBuffer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PuddleWorld(object):
    def __init__(self, start=None, goal=None, height=10, width=10, reward_goal=0.0, step_reward=-1.0,
                 puddle_reward=-0.1, puddle_center=None, puddle_width=None, horizon=20):
        assert len(start) == 2
        assert len(goal) == 2
        assert len(puddle_center) == 2
        assert puddle_width > 0

        self.start = start
        self.goal = goal
        self.height = height
        self.width = width
        self.puddle_center = puddle_center
        self.puddle_width = puddle_width
        self.reward_goal = reward_goal
        self.step_reward = step_reward
        self.puddle_reward = puddle_reward
        self.horizon = horizon
        self.x, self.y = self.start
        self.t = 0

    def reset(self):
        self.x, self.y = self.start
        self.t = 0
        return self.x + self.width * self.y + self.width * self.height * self.t, self.x, self.y

    def step(self, action):
        if action == 0:     # left
            self.x = max(0, self.x - 1)
        elif action == 1:   # right
            self.x = min(self.width - 1, self.x + 1)
        elif action == 2:   # up
            self.y = max(0, self.y - 1)
        elif action == 3:   # down
            self.y = min(self.height - 1, self.y + 1)

        reward = self.step_reward
        terminal = False
        if np.abs(self.x - self.puddle_center[0]) <= (self.puddle_width - 1) * 1.0 / 2 and \
                np.abs(self.y - self.puddle_center[1]) <= (self.puddle_width - 1) * 1.0 / 2:
            reward = self.puddle_reward
        elif self.y == self.goal[1]:  # and self.x == self.goal[0]:
            reward = self.reward_goal
            terminal = True

        self.t += 1

        if self.t >= (self.horizon - 1):
            terminal = True

        return self.x + self.width * self.y + self.width * self.height * self.t, self.x, self.y, reward, terminal
        # return self.x + self.width * self.y, self.x, self.y, reward, terminal


class RewardModel(torch.nn.Module):
    def __init__(self, input_dim):
        super(RewardModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs


def evaluate(q, puddle):
    terminal = False
    episode_reward = 0
    state, x, y = puddle.reset()

    path_matrix = np.zeros((6, 6))
    t = 0

    while not terminal:
        print(q[state])
        path_matrix[x, y] = t + 1
        t += 1

        action = np.argmax(q[state, :])
        state, x, y, reward, terminal = puddle.step(action)
        print(reward)

        episode_reward += reward

    path_matrix[x, y] = t + 1

    print(path_matrix)

    return episode_reward


def learn_reward(memory, reward_tilde, criterion, optimizer, discount):
    reward_loss = 0
    batch_size = 8
    for _ in range(batch_size):
        states, rewards = memory.sample_trajectory(2, sample_whole_trajectory=True, min_length=0)

        discounts_0 = torch.cumprod(torch.zeros(len(states[0])).to(device) + discount, dim=0) / discount
        discounts_1 = torch.cumprod(torch.zeros(len(states[1])).to(device) + discount, dim=0) / discount

        len_traj_0 = len(rewards[0])
        len_traj_1 = len(rewards[1])

        for i in [0]:  # range(0, len_traj_0, 1):
            for j in [0]:  # range(0, len_traj_1, 1):
                features_0 = (states[0][i:] * discounts_0[:len_traj_0 - i].unsqueeze(-1)).sum(dim=0).unsqueeze(0)
                features_1 = (states[1][j:] * discounts_1[:len_traj_1 - j].unsqueeze(-1)).sum(dim=0).unsqueeze(0)

                rew_traj_0 = rewards[0][i:].sum().item()
                rew_traj_1 = rewards[1][j:].sum().item()

                predicted_rew_traj_0 = reward_tilde(features_0)
                predicted_rew_traj_1 = reward_tilde(features_1)

                concat_rews = torch.cat((predicted_rew_traj_0, predicted_rew_traj_1), 1)

                logit = 0 if rew_traj_0 > rew_traj_1 else 1

                reward_loss += criterion(concat_rews, torch.Tensor([logit]).long().to(device)) / batch_size
                # reward_loss += F.mse_loss(predicted_rew_traj_0, torch.Tensor([[rew_traj_0]]))
                # reward_loss += F.mse_loss(predicted_rew_traj_1, torch.Tensor([[rew_traj_1]]))

    optimizer.zero_grad()
    reward_loss.backward()
    optimizer.step()


def solve_mdp(adaptive_reward, reward_tilde, q, height, width, horizon, puddle, discount):
    for _ in range(20):
        for x in range(width):
            for y in range(height):
                for t in range(horizon - 1):
                    for a in range(4):
                        puddle.x = x
                        puddle.y = y
                        puddle.t = t

                        state = x + width * y + width * height * t
                        next_state, _, _, reward, terminal = puddle.step(a)

                        if adaptive_reward:
                            with torch.no_grad():
                                np_next_state = np.zeros(width * height * horizon)
                                np_next_state[next_state] = 1
                                torch_state = torch.FloatTensor(np_next_state.reshape(1, -1)).to(device)
                                reward = reward_tilde(torch_state).cpu().item()

                        if terminal:
                            q[state, a] = reward
                        else:
                            q[state, a] = reward + discount * np.max(q[next_state, :])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--adaptive_reward", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    args = parser.parse_args()

    file_name = f"{args.discount}_{args.adaptive_reward}"

    width, height, horizon = 6, 6, 20
    eval_freq = 1000

    puddle = PuddleWorld(start=(0, 0), goal=(5, 5), height=height, width=width, reward_goal=20.0, step_reward=-1.0,
                         puddle_reward=0.0, puddle_center=(2, 1), puddle_width=3, horizon=horizon)

    memory = ReplayBuffer(width * height * horizon, 1, max_size=10000)
    reward_tilde = RewardModel(width * height * horizon).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(reward_tilde.parameters(), lr=1e-1)  # , weight_decay=0.001)

    grid_matrix = np.zeros((6, 6))
    with torch.no_grad():
        for i in range(6):
            for j in range(6):
                val = 0
                if i == 0 and j == 0:
                    val = 1
                elif np.abs(i - puddle.puddle_center[0]) <= (puddle.puddle_width - 1) * 1.0 / 2 and \
                        np.abs(j - puddle.puddle_center[1]) <= (puddle.puddle_width - 1) * 1.0 / 2:
                    val = 2
                elif i == 5 and j == 5:
                    val = 3
                grid_matrix[i, j] = val

    q = np.zeros((width * height * horizon, 4)) + 5

    terminal = True
    state = None
    eval = True
    path = [1, 1, 1, 3, 3, 3, 1, 1, 3, 3]

    evaluations = []
    for t in range(1000000):
        if terminal:
            if eval:
                solve_mdp(args.adaptive_reward, reward_tilde, q, height, width, horizon, puddle, args.discount)
                evaluation = evaluate(q, puddle)
                evaluations.append(evaluation)

                print(evaluation)

            state, _, _ = puddle.reset()
        if np.random.rand() < max(0.05, 1 - t / 100000):
            action = np.random.randint(4)
        else:
            action = np.argmax(q[state, :])

        # if t < 20:
        #     action = path[t % 10]

        next_state, _, _, reward, terminal = puddle.step(action)

        np_state = np.zeros(width * height * horizon)
        np_state[state] = 1

        np_next_state = np.zeros(width * height * horizon)
        np_next_state[next_state] = 1

        memory.add(np_state, action, np_next_state, reward, terminal)

        state = next_state

        if (t + 1) % eval_freq == 0:
            eval = True

        if t > 100 and args.adaptive_reward:
            learn_reward(memory, reward_tilde, criterion, optimizer, args.discount)

    torch.save(reward_tilde.state_dict(), f"./results/{file_name}" + '_reward_tilde')
    np.save(f"./results/{file_name}" + '_evaluations', evaluations)


if __name__ == "__main__":
    main()
