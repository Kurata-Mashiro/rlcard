'''Minimal runnable experiment setup for Dou Dizhu in RLCard.
'''
import argparse
import os
import random

import numpy as np

import rlcard
from rlcard import models
from rlcard.agents import RandomAgent
from rlcard.utils import reorganize, set_seed


def masked_action_probs(scores, legal_actions, temperature=1.0):
    '''Return normalized probabilities over legal actions only.'''
    probs = np.zeros_like(scores, dtype=np.float64)
    legal_actions = list(legal_actions)
    if len(legal_actions) == 1:
        probs[legal_actions[0]] = 1.0
        return probs

    legal_scores = scores[legal_actions].astype(np.float64)
    if temperature <= 0:
        best_idx = legal_actions[int(np.argmax(legal_scores))]
        probs[best_idx] = 1.0
        return probs

    legal_scores = legal_scores / temperature
    legal_scores = legal_scores - np.max(legal_scores)
    legal_probs = np.exp(legal_scores)
    legal_probs = legal_probs / np.sum(legal_probs)
    probs[legal_actions] = legal_probs
    return probs


def build_learning_agent(args, env, device):
    from rlcard.agents import DQNAgent, NFSPAgent

    if args.algorithm == 'dqn':
        return DQNAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            mlp_layers=[64, 64],
            device=device,
            save_path=args.log_dir,
            save_every=args.save_every,
        )
    return NFSPAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        hidden_layers_sizes=[64, 64],
        q_mlp_layers=[64, 64],
        device=device,
        save_path=args.log_dir,
        save_every=args.save_every,
    )


def build_opponents(env, baseline, rng):
    random_agents = [RandomAgent(env.num_actions) for _ in range(env.num_players)]
    rule_agents = models.load('doudizhu-rule-v1').agents
    if baseline == 'random':
        return random_agents
    if baseline == 'rule':
        return rule_agents
    return random_agents if rng.random() < 0.5 else rule_agents


def evaluate(env, learner, args, baseline, rng):
    role_stats = {
        'landlord': {'sum': 0.0, 'games': 0},
        'peasant': {'sum': 0.0, 'games': 0},
    }
    total_reward = 0.0

    for _ in range(args.num_eval_games):
        learner_pos = rng.randrange(env.num_players)
        agents = build_opponents(env, baseline, rng)
        agents[learner_pos] = learner
        env.set_agents(agents)
        _, payoffs = env.run(is_training=False)

        learner_role = env.game.players[learner_pos].role
        learner_reward = float(payoffs[learner_pos])
        total_reward += learner_reward
        role_stats[learner_role]['sum'] += learner_reward
        role_stats[learner_role]['games'] += 1

    avg_reward = total_reward / args.num_eval_games
    landlord_avg = role_stats['landlord']['sum'] / max(role_stats['landlord']['games'], 1)
    peasant_avg = role_stats['peasant']['sum'] / max(role_stats['peasant']['games'], 1)
    return {
        'avg_reward': avg_reward,
        'landlord_avg_reward': landlord_avg,
        'landlord_games': role_stats['landlord']['games'],
        'peasant_avg_reward': peasant_avg,
        'peasant_games': role_stats['peasant']['games'],
    }


def train(args):
    os.makedirs(args.log_dir, exist_ok=True)
    try:
        import torch
        from rlcard.utils import get_device
    except ImportError as exc:
        raise ImportError(
            'This script requires PyTorch. Install with: pip install -e ".[torch]"'
        ) from exc

    device = get_device()
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    env = rlcard.make('doudizhu', config={'seed': args.seed})
    learner = build_learning_agent(args, env, device)
    rng = random.Random(args.seed)

    for episode in range(args.num_episodes):
        if args.algorithm == 'nfsp':
            learner.sample_episode_policy()

        learner_pos = rng.randrange(env.num_players)
        agents = build_opponents(env, args.train_opponents, rng)
        agents[learner_pos] = learner
        env.set_agents(agents)

        trajectories, payoffs = env.run(is_training=True)
        trajectories = reorganize(trajectories, payoffs)
        for ts in trajectories[learner_pos]:
            learner.feed(ts)

        if (episode + 1) % args.evaluate_every == 0 or episode == 0:
            random_eval = evaluate(env, learner, args, baseline='random', rng=rng)
            rule_eval = evaluate(env, learner, args, baseline='rule', rng=rng)
            print(
                f'[Episode {episode + 1}] '
                f'vs random avg={random_eval["avg_reward"]:.4f} '
                f'(L={random_eval["landlord_avg_reward"]:.4f}/{random_eval["landlord_games"]}, '
                f'P={random_eval["peasant_avg_reward"]:.4f}/{random_eval["peasant_games"]}) | '
                f'vs rule avg={rule_eval["avg_reward"]:.4f} '
                f'(L={rule_eval["landlord_avg_reward"]:.4f}/{rule_eval["landlord_games"]}, '
                f'P={rule_eval["peasant_avg_reward"]:.4f}/{rule_eval["peasant_games"]})'
            )

    save_path = os.path.join(args.log_dir, f'doudizhu_{args.algorithm}_model.pth')
    torch.save(learner, save_path)
    print('Model saved in', save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Minimal Dou Dizhu experiment for RLCard')
    parser.add_argument('--algorithm', type=str, default='nfsp', choices=['dqn', 'nfsp'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--num_eval_games', type=int, default=100)
    parser.add_argument('--evaluate_every', type=int, default=100)
    parser.add_argument('--train_opponents', type=str, default='random', choices=['random', 'rule', 'mix'])
    parser.add_argument('--log_dir', type=str, default='experiments/doudizhu_minimal_experiment')
    parser.add_argument('--save_every', type=int, default=-1)
    parser.add_argument('--cuda', type=str, default='')
    _args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = _args.cuda
    train(_args)
