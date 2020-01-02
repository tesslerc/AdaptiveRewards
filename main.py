import numpy as np
import torch
import gym
import argparse
import os

import utils
import TD3
import DDPG


# Runs policy for X episodes and returns average *total* reward
def eval_policy(policy, env_name, eval_episodes=10, max_steps=0):
    eval_env = gym.make(env_name)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        remaining_steps = max_steps * 1.0
        if max_steps > 0:
            state = np.append(state, [remaining_steps / max_steps])

        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)

            remaining_steps -= 1

            if max_steps > 0:
                state = np.append(state, [remaining_steps / max_steps])

            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG)
    parser.add_argument("--env", default="HalfCheetah-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--infinite_horizon", action="store_true")  # Consider infinite or finite horizon task
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    max_steps = 0.0

    if not args.infinite_horizon:
        max_steps = env._max_episode_steps
        state_dim += 1

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "max_steps": max_steps
    }

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, max_steps=max_steps)]
    best_performance = evaluations[-1]

    state, done = env.reset(), False
    remaining_steps = max_steps * 1.0
    if not args.infinite_horizon:
        state = np.append(state, [remaining_steps / max_steps])
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (policy.select_action(np.array(state)) + np.random.normal(0, max_action * args.expl_noise, size=action_dim)).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps or not args.infinite_horizon else 0

        remaining_steps -= 1

        if not args.infinite_horizon:
            next_state = np.append(next_state, [remaining_steps / max_steps])

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False

            remaining_steps = max_steps * 1.0
            if not args.infinite_horizon:
                state = np.append(state, [remaining_steps / max_steps])

            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, max_steps=max_steps))
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model and evaluations[-1] > best_performance:
                best_performance = evaluations[-1]
                policy.save(f"./models/{file_name}")

    # After training, evaluate the "seemingly best" policy for 100 episodes.
    # The 100 evals are for better estimation of empirical mean.
    # This is an unbiased estimator of the performance, as opposed to taking the max over the process itself.
    if args.save_model:
        policy.load(f"./models/{file_name}")
        evaluations.append(eval_policy(policy, args.env, eval_episodes=100, max_steps=max_steps))
        np.save(f"./results/{file_name}", evaluations)


if __name__ == "__main__":
    main()
