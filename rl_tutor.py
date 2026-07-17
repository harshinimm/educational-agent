"""A real RL tutor: a small DQN trained inside a BKT-based simulator.

Real interaction data (a few hundred rows across ~25 concepts) isn't
enough to train an RL agent directly — which is exactly why the guide
wants a simulator. Each concept's own fitted BKT parameters (P(L0), P(T),
P(S), P(G)) already form a generative model of a student answering and
learning that concept, so we use them as the environment instead of
standing up a separate DKT-based one: BKT is naturally generative, where
DKT is a discriminative predictor and would need extra machinery to
sample from as a stochastic environment.

State: current P(known) per concept. Action: which concept to study next.
Reward: the mastery gain from that one study step. A myopic "always study
your weakest concept" policy isn't actually optimal here, because concepts
differ in learn rate P(T) — a low-P(T) "hard" concept can look most
urgent while returning the least mastery per step studied. That gap is
what the learned Q-function has room to beat a naive heuristic on.
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

import bkt


class StudySimulator:
    """One episode = a simulated study session across all fitted concepts."""

    def __init__(self, concept_params, episode_length=20):
        self.concepts = list(concept_params.keys())
        self.params = concept_params  # concept -> {p_l0, p_t, p_s, p_g}
        self.episode_length = episode_length

    def reset(self):
        base = np.array([self.params[c]["p_l0"] for c in self.concepts], dtype=np.float32)
        # jitter the start so training sees varied initial knowledge states
        self.state = np.clip(base + np.random.normal(0, 0.05, size=base.shape), 0, 1).astype(np.float32)
        self.t = 0
        return self.state.copy()

    def step(self, action_idx):
        concept = self.concepts[action_idx]
        p = self.params[concept]
        p_known = float(self.state[action_idx])
        correct = random.random() < bkt.predict_correct(p_known, p["p_s"], p["p_g"])
        new_p_known = bkt.forward_update(p_known, correct, p["p_t"], p["p_s"], p["p_g"])
        reward = new_p_known - p_known  # mastery gain from this one study step
        self.state[action_idx] = new_p_known
        self.t += 1
        done = self.t >= self.episode_length
        return self.state.copy(), reward, done


class QNet(nn.Module):
    def __init__(self, n_concepts, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_concepts, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_concepts),
        )

    def forward(self, x):
        return self.net(x)


def _fitted_concept_params():
    results = bkt.evaluate_all_concepts()
    return {c: r["params"] for c, r in results.items()}


def train_dqn(episodes=300, episode_length=20, gamma=0.9, lr=1e-3, batch_size=64, seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    concept_params = _fitted_concept_params()
    if len(concept_params) < 2:
        raise ValueError("Need at least 2 concepts with fitted BKT params to train the RL tutor.")

    env = StudySimulator(concept_params, episode_length=episode_length)
    n = len(env.concepts)
    q_net = QNet(n)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
    buffer = deque(maxlen=5000)

    epsilon = 1.0
    for _ in range(episodes):
        state = env.reset()
        for _ in range(episode_length):
            if random.random() < epsilon:
                action = random.randrange(n)
            else:
                with torch.no_grad():
                    action = int(torch.argmax(q_net(torch.tensor(state))).item())
            next_state, reward, done = env.step(action)
            buffer.append((state, action, reward, next_state, done))
            state = next_state

            if len(buffer) >= batch_size:
                batch = random.sample(buffer, batch_size)
                states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
                actions = torch.tensor([b[1] for b in batch], dtype=torch.long)
                rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
                next_states = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32)
                dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)

                q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = q_net(next_states).max(1).values
                    target = rewards + gamma * next_q * (1 - dones)
                loss = nn.functional.mse_loss(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(0.05, epsilon * 0.97)

    return q_net, env


def _rollout(env, policy_fn, n_episodes=30):
    """Average final mean-mastery across concepts, over n_episodes simulated sessions."""
    finals = []
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            action = policy_fn(state)
            state, _, done = env.step(action)
        finals.append(float(state.mean()))
    return float(np.mean(finals))


def evaluate(q_net, env, n_episodes=30):
    n = len(env.concepts)

    def random_policy(state):
        return random.randrange(n)

    def greedy_weakest_policy(state):
        return int(np.argmin(state))

    def rl_policy(state):
        with torch.no_grad():
            return int(torch.argmax(q_net(torch.tensor(state, dtype=torch.float32))).item())

    return {
        "random": _rollout(env, random_policy, n_episodes),
        "greedy_weakest": _rollout(env, greedy_weakest_policy, n_episodes),
        "rl_agent": _rollout(env, rl_policy, n_episodes),
    }


def recommend_next_real(q_net, concepts):
    """Feed the CURRENT real P(known) vector (from BKT) into the trained
    Q-network and return its top pick for what to study next, for real."""
    results = bkt.evaluate_all_concepts()
    state = np.array(
        [results[c]["final_p_known"] if c in results else 0.0 for c in concepts], dtype=np.float32
    )
    with torch.no_grad():
        q_values = q_net(torch.tensor(state, dtype=torch.float32))
    action = int(torch.argmax(q_values).item())
    return concepts[action], q_values.numpy()


if __name__ == "__main__":
    q_net, env = train_dqn()
    print("Trained on", len(env.concepts), "concepts.")
    print(evaluate(q_net, env))
    print(recommend_next_real(q_net, env.concepts)[0])
