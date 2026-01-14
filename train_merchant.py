from gymnasium.wrappers import FlattenObservation, TimeLimit
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from norm_rl_gym.envs.merchant.merchant import MerchantEnv
from utils.stats import MerchantEvaluationStats
from algorithms.qlearning import QLearning
from pprint import pprint
from os import getenv
import numpy as np


LAYOUT = getenv("LAYOUT", "basic")


def make_merchant_env():
    env = MerchantEnv(layout=LAYOUT, risk_fight=0.9, risk_death=0.1, capacity=5, sunset=28)
    env = FlattenObservation(env)
    env = TimeLimit(env, max_episode_steps=50)
    return env


# we use separate environments for training and evaluation
train_env = make_vec_env(make_merchant_env)
eval_env = make_vec_env(make_merchant_env)

# some evaluation settings
evaluation_stats = MerchantEvaluationStats(
    trial=None,  # or optuna trial
    eval_envs=eval_env,
    train_envs=train_env,
    n_eval_episodes=100,
    monitor_names=[
        "ContradictionMonitor",
        "DangerMonitor",
        "DeliveryMonitor",
        "EnvFriendlyMonitor",
        "EvolvingMonitor",
        "PacifistMonitor",
    ],
    int_eval_episodes=10,
    csv_prefix="logfile",
    int_eval_frequency=100,
)

N_SEEDS = 5
N_EVAL_EPISODES = 1000
all_returns = []
for i in range(N_SEEDS):
    # reset envs
    train_env.reset()
    eval_env.reset()
    # learn model
    model = QLearning(train_env, exploration_fraction=0.9, verbose=0, seed=i)
    model.learn(total_timesteps=5_000_000)
    # saving and loading
    model.save(f"merchant_{LAYOUT}_{i}.model")
    model.set_parameters(f"merchant_{LAYOUT}_{i}.model")
    # evaluate model
    evaluation_stats.init_eval_step(model)
    mean_return, std_return = evaluate_policy(
        model, eval_env, n_eval_episodes=N_EVAL_EPISODES, callback=evaluation_stats.eval_callback
    )
    all_returns.append(mean_return)
    print(f"Seed {i}: {mean_return}±{std_return}", flush=True)

mean_all = np.mean(all_returns)
std_all = np.std(all_returns)
print(f"Mean return over all seeds: {mean_all}±{std_all}")
print("Avg. number of norm violations per episode:")
pprint(evaluation_stats.get_stats(N_SEEDS * N_EVAL_EPISODES))
