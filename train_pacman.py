import torch

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env

from utils.file import save_model
from utils.pacman import create_pacman_env
from utils.images import ImageFeaturesExtractor

import norm_rl_gym.envs.pacman.pacman_env


def train(
    env_name,
    algo,
    feature_extractor,
    n_steps,
    level,
    greyscale=True,
    image_stack_size=1,
    tb_name=None,
    features_dims=[256, 128],
    verbose=True,
    n_vec=8,
):
    env = make_vec_env(
        lambda: create_pacman_env(
            env_name,
            feature_extractor=feature_extractor,
            level=level,
            render_mode="none",
            greyscale=greyscale,
            image_stack_size=image_stack_size,
            scale="ppo" in algo or "image" in feature_extractor,
        ),
        8,
    )

    dfa_states = 0
    if "image" in feature_extractor:
        policy_kwargs_dqn = dict(
            features_extractor_class=ImageFeaturesExtractor,
            features_extractor_kwargs=dict(
                features_dims=features_dims,
                dfa_states=dfa_states,
                greyscale=greyscale,
                image_stack_size=image_stack_size,
            ),
            net_arch=[128, 128],
        )
        policy_kwargs_ppo = dict(
            features_extractor_class=ImageFeaturesExtractor,
            features_extractor_kwargs=dict(
                features_dims=features_dims,
                dfa_states=dfa_states,
                greyscale=greyscale,
                image_stack_size=image_stack_size,
            ),
            net_arch=[128],
        )

    else:
        policy_kwargs_dqn = dict(net_arch=[256, 256], activation_fn=torch.nn.ReLU)
        policy_kwargs_ppo = dict(net_arch=[256, 256], activation_fn=torch.nn.ReLU)

    tb_path = f"./tb_log/{tb_name}/" if tb_name is not None else None
    if "image" in feature_extractor:
        if algo == "dqn":
            model = DQN(
                "MlpPolicy",
                env,
                verbose=verbose,
                tensorboard_log=tb_path,
                policy_kwargs=policy_kwargs_dqn,
                batch_size=2**5,
                buffer_size=2**14,
                exploration_fraction=0.1,
                gamma=0.95,
                gradient_steps=-1,
                train_freq=2,
                target_update_interval=75000,
                exploration_final_eps=0.01,
                tau=1,
                learning_rate=0.57e-4,
                learning_starts=10000,
                # learning_rate=0.57e-4, learning_starts=10000,
            )
        else:
            model = PPO(
                "CnnPolicy",
                env,
                verbose=verbose,
                tensorboard_log=tb_path,
                policy_kwargs=policy_kwargs_ppo,
                # batch_size=2 ** 5, gamma=0.9,
                learning_rate=2.5e-4,
                n_steps=128,
                batch_size=64,
                n_epochs=5,
                gamma=0.9,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.05,  # increase to 0.05 if sparse rewards make policy collapse
                vf_coef=0.5,
                max_grad_norm=0.5,
            )

    else:
        if algo == "dqn":
            model = DQN(
                "MlpPolicy",
                env,
                verbose=verbose,
                tensorboard_log=tb_path,
                policy_kwargs=policy_kwargs_dqn,
                batch_size=64,
                buffer_size=50_000,
            )
        elif algo == "ppo":
            model = PPO("MlpPolicy", env, verbose=verbose, tensorboard_log=tb_path, policy_kwargs=policy_kwargs_ppo)
        else:
            raise NotImplementedError()
    model.learn(n_steps)

    return model


def train_pacman(algo, env_name, steps, level, feature_extractor, greyscale, image_stack_size=1):
    model_name = f"pickles/models/{algo}_{env_name.replace('/', '_')}_{steps}_level_{level}_{feature_extractor}"
    tb_name = f"{algo}_{env_name}_mode_{level}_{feature_extractor}_unshielded"

    model = train(
        env_name,
        algo,
        feature_extractor=feature_extractor,
        n_steps=steps,
        tb_name=tb_name,
        level=level,
        greyscale=greyscale,
        image_stack_size=image_stack_size,
    )
    save_model(model_name, model)
    return model


if __name__ == "__main__":
    import sys

    env_name = "BerkeleyPacmanPO-v0"
    steps = 2500_000
    level = "smallClassic"
    feature_extractor = "complete"  # "image-full"

    algo = sys.argv[1]
    image_stack_size = 2
    greyscale = True
    train_pacman(algo, env_name, steps, level, feature_extractor, greyscale, image_stack_size)
