# -*- coding: utf-8 -*-
import random
import numpy as np
import pickle

from environment.environment import Environment
from environment.environment_node_data import Mode
import action_mapper


# ============================================================
#                     PREPROCESSING
# ============================================================

def compress_laserscan(scan, num_sectors=5):
    scan = np.array(scan).flatten()
    sector_size = len(scan) // num_sectors
    compressed = [
        np.min(scan[i * sector_size:(i + 1) * sector_size])
        for i in range(num_sectors)
    ]
    return np.array(compressed, dtype=float)


def compress_orientation(orientation, target_size=8):
    orientation = np.array(orientation).flatten()
    step = len(orientation) // target_size
    return orientation[::step][:target_size]


def preprocess_state(raw):
    lasers = raw[:1081]
    orient = raw[1081:]

    lasers = compress_laserscan(lasers, 5)
    orient = compress_orientation(orient, 8)

    final = np.concatenate([lasers, orient])
    return final.astype(float)


# ============================================================
#             FUZZY SAFETY LAYER (OBSTACLE AVOIDANCE)
# ============================================================

def fuzzy_safety_action(laser_sectors):
    """
    laser_sectors: 5 values (FL, L, C, R, FR)
    returns: override_action or None
    """
    FL, L, C, R, FR = laser_sectors

    # allow robot to go closer to walls before safety kicks in
    danger = 0.25     # immediate collision region (reduced from 0.35)
    warning = 0.5     # soft steering (reduced from 0.6)

    # HARD STOPS / EMERGENCY EVASION
    if C < danger:
        return "turn_right" if L > R else "turn_left"

    # Strong left wall
    if L < danger or FL < danger:
        return "turn_right"

    # Strong right wall
    if R < danger or FR < danger:
        return "turn_left"

    # SOFT steering in narrow corridor
    if C < warning:
        return "slight_left" if L > R else "slight_right"

    return None


def action_to_vel(action_name):
    if action_name == "turn_right":
        return (0.05, -0.8)
    if action_name == "turn_left":
        return (0.05, 0.8)
    if action_name == "slight_left":
        return (0.1, 0.3)
    if action_name == "slight_right":
        return (0.1, -0.3)
    return None


# ============================================================
#                     REWARD SHAPING
# ============================================================

def shape_reward(env_reward,
                 state, next_state,
                 linear, angular,
                 safety, done):
    """
    env_reward : reward returned by env.step(...)
    state, next_state : current and next preprocessed states (1 x state_size)
    linear, angular   : executed velocities
    safety            : None or a fuzzy override string
    done              : episode finished flag
    """
    # Use next state's lasers to judge how "safe" the motion was
    lasers_next = next_state[0][:5]
    min_dist = float(np.min(lasers_next))

    # Base small step penalty to encourage faster solutions
    r = -0.01

    # Encourage forward motion
    r += 0.4 * float(linear)

    # Discourage excessive spinning
    r -= 0.1 * abs(float(angular))

    # Distance-based shaping
    if min_dist < 0.25:
        # Very close to obstacles
        r -= 0.7
    elif min_dist < 0.5:
        # Moderately close
        r -= 0.3
    else:
        # Nicely far from walls / in wide corridor
        r += 0.05

    # Punish when fuzzy safety had to intervene
    if safety is not None:
        r -= 1.0

    # Keep environment's terminal signal strong
    if done:
        r += float(env_reward)

    return r


# ============================================================
#             NEURO-FUZZY ACTOR–CRITIC (HIGH LEVEL RL)
# ============================================================

class NeuroFuzzyActorCritic:

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        # Learning rates / discount
        self.gamma = 0.95      # you can push this higher for very long episodes
        self.alpha_v = 0.1
        self.alpha_pi = 0.01

        # Tables
        self.policy_prefs = {}   # key → action preferences
        self.value_table = {}    # key → V(s)

    # -------------- Softmax with temperature -----------------
    def softmax(self, x, temperature=1.0):
        """
        temperature > 1.0  -> more random (flatter distribution)
        temperature = 1.0  -> normal softmax
        temperature < 1.0  -> more greedy / peaky
        """
        x = np.array(x, dtype=float) / float(temperature)
        x -= np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    # -------------- RL Action ----------------
    def act(self, state, greedy=False, temperature=1.0, epsilon=0.0):
        """
        greedy=True:
            deterministic argmax over preferences (no exploration)
        greedy=False:
            softmax with temperature, plus epsilon-greedy randomization
        """
        key = tuple((state[0] * 10).astype(int))

        if key not in self.policy_prefs:
            self.policy_prefs[key] = np.zeros(self.action_size)

        prefs = self.policy_prefs[key]

        if greedy:
            # Pure exploitation
            action = int(np.argmax(prefs))
            return action, key

        # Softmax sampling with temperature
        probs = self.softmax(prefs, temperature=temperature)
        action = int(np.random.choice(self.action_size, p=probs))

        # ε-greedy on top: with prob epsilon, override with a random action
        if random.random() < float(epsilon):
            action = random.randint(0, self.action_size - 1)

        return action, key

    # -------------- RL Learning --------------
    def learn(self, key_s, action, reward, key_sp, done):

        # Ensure all keys exist
        if key_s not in self.policy_prefs:
            self.policy_prefs[key_s] = np.zeros(self.action_size)
        if key_s not in self.value_table:
            self.value_table[key_s] = 0.0

        if key_sp not in self.policy_prefs:
            self.policy_prefs[key_sp] = np.zeros(self.action_size)
        if key_sp not in self.value_table:
            self.value_table[key_sp] = 0.0

        # Critic values
        v_s = self.value_table[key_s]
        v_sp = 0.0 if done else self.value_table[key_sp]

        # TD Error
        delta = reward + self.gamma * v_sp - v_s

        # Critic update
        self.value_table[key_s] = v_s + self.alpha_v * delta

        # Actor update
        prefs = self.policy_prefs[key_s]
        probs = self.softmax(prefs)  # temperature=1 here is fine for gradient

        for a in range(self.action_size):
            grad = (1.0 if a == action else 0.0) - probs[a]
            prefs[a] += self.alpha_pi * delta * grad

        self.policy_prefs[key_s] = prefs

    # -------------- Save / Load --------------
    def save(self, path):
        data = {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "policy_prefs": self.policy_prefs,
            "value_table": self.value_table,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = NeuroFuzzyActorCritic(
            data["state_size"],
            data["action_size"]
        )
        agent.policy_prefs = data["policy_prefs"]
        agent.value_table = data["value_table"]
        return agent


# ============================================================
#                     MAIN TRAINING + EVAL
# ============================================================

# Set this to True if you want to train,
# False if you only want to run a saved policy.
TRAINING = True

TRAIN_EPISODES = 1
MAX_STEPS = 20_000
EVAL_EPISODES = 1          # how many greedy runs
MODEL_PATH = "nfac_agent.pkl"

# Exploration knobs
SOFTMAX_TEMPERATURE = 2.0   # >1.0 => more random action distribution
EPSILON = 1.0               # probability of fully random action on top of softmax


if __name__ == "__main__":

    env = Environment("../Simulation2d/world/maze")
    env.use_observation_rotation_size(True)
    env.set_observation_rotation_size(128)

    state_size = 5 + 8
    action_size = action_mapper.ACTION_SIZE

    if TRAINING:
        # -------------------- TRAINING --------------------
        agent = NeuroFuzzyActorCritic(state_size, action_size)
        print("START TRAINING: HYBRID FUZZY-SAFETY + ACTOR-CRITIC WITH SHAPED REWARD")

        for e in range(TRAIN_EPISODES):

            visualize = True
            total_reward = 0.0

            raw_state, _, _, _ = env.reset()
            state = preprocess_state(raw_state)
            state = np.reshape(state, (1, state_size))

            # You could also decay epsilon over episodes, e.g.:
            # eps = max(EPSILON * (1.0 - e / TRAIN_EPISODES), 0.01)
            eps = EPSILON

            for it in range(MAX_STEPS):

                # Extract LiDAR sectors for safety
                laser_sectors = state[0][:5]

                # ------ FUZZY SAFETY LAYER ------
                safety = fuzzy_safety_action(laser_sectors)

                if safety is not None:
                    # Use fuzzy override, RL does NOT choose action
                    linear, angular = action_to_vel(safety)
                    rl_action = None
                    key_s = tuple((state[0] * 10).astype(int))

                else:
                    # ------ RL ACTOR (exploration) -------
                    rl_action, key_s = agent.act(
                        state,
                        greedy=False,
                        temperature=SOFTMAX_TEMPERATURE,
                        epsilon=eps,
                    )
                    linear, angular = action_mapper.map_action(rl_action)

                # ------- Step environment -------
                next_raw, env_reward, done, info = env.step(linear, angular, 20)
                next_state = preprocess_state(next_raw)
                next_state = np.reshape(next_state, (1, state_size))

                # ------- Reward shaping -------
                shaped_reward = shape_reward(env_reward,
                                             state, next_state,
                                             linear, angular,
                                             safety, done)

                total_reward += shaped_reward
                key_sp = tuple((next_state[0] * 10).astype(int))

                # ------- RL learns ONLY if RL took action -------
                if safety is None and rl_action is not None:
                    agent.learn(key_s, rl_action, shaped_reward, key_sp, done)

                state = next_state

                if visualize:
                    env.visualize()

                if done:
                    print(f"[TRAIN] EP {e}/{TRAIN_EPISODES} | "
                          f"shaped_return={total_reward:.2f} | steps={it}")
                    break

        # Save trained agent
        agent.save(MODEL_PATH)
        print(f"Training finished. Policy saved to '{MODEL_PATH}'.")

    else:
        # -------------------- LOAD PRETRAINED --------------------
        print(f"LOADING trained agent from '{MODEL_PATH}'...")
        agent = NeuroFuzzyActorCritic.load(MODEL_PATH)

    # -------------------- EVALUATION (GREEDY RUN) --------------------
    print("Starting greedy evaluation run(s) with learned policy...")

    for eval_id in range(EVAL_EPISODES):

        visualize = True
        total_env_reward = 0.0  # raw env reward

        raw_state, _, _, _ = env.reset()
        state = preprocess_state(raw_state)
        state = np.reshape(state, (1, state_size))

        for it in range(MAX_STEPS):

            laser_sectors = state[0][:5]

            # Fuzzy safety still active during eval
            safety = fuzzy_safety_action(laser_sectors)

            if safety is not None:
                linear, angular = action_to_vel(safety)
            else:
                # PURE EXPLOITATION: greedy=True (no exploration here)
                rl_action, key_s = agent.act(state, greedy=True)
                linear, angular = action_mapper.map_action(rl_action)

            next_raw, env_reward, done, info = env.step(linear, angular, 20)
            next_state = preprocess_state(next_raw)
            next_state = np.reshape(next_state, (1, state_size))

            total_env_reward += env_reward
            state = next_state

            if visualize:
                env.visualize()

            if done:
                print(f"[EVAL] EP {eval_id} | env_return={total_env_reward:.2f} "
                      f"| steps={it}")
                break

    print("Done.")