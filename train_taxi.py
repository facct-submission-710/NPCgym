from gymnasium.wrappers import FlattenObservation, TimeLimit
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from norm_rl_gym.envs.taxi import TaxiEnv
from utils.stats import TaxiEvaluationStats
from algorithms.qlearning import QLearning
from pprint import pprint
import numpy as np


def make_taxi_env():
    env = TaxiEnv(is_rainy=False, fickle_passenger=False, storm_risk=False)
    env = FlattenObservation(env)
    env = TimeLimit(env, max_episode_steps=50)
    return env


# we use separate environments for training and evaluation
train_env = make_vec_env(make_taxi_env)
eval_env = make_vec_env(make_taxi_env)

# some evaluation settings
evaluation_stats = TaxiEvaluationStats(
    trial=None,  # or optuna trial
    eval_envs=eval_env,
    train_envs=train_env,
    n_eval_episodes=100,
    monitor_names=[
        "EmergencyMonitor",
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
    model = QLearning(train_env, verbose=0, seed=i)
    model.learn(total_timesteps=1_000_000)
    # saving and loading
    model.save(f"taxi_model_{i}.model")
    model.set_parameters(f"taxi_model_{i}.model")
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
