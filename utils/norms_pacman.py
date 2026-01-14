import json
import sys

from utils.file import load_model
from utils.pacman import create_pacman_env

import norm_rl_gym.envs.pacman.pacman_env


def run_tests_with_monitor(env, model, eval_names, n_eval):
    env.reset()
    module = __import__("norm_rl_gym")
    mons = getattr(module, "monitors")
    pac = getattr(mons, "pacman_monitors")
    monitors = dict()
    for eval_name in eval_names:
        monitor_name = monitor_dict[eval_name]
        cls = getattr(pac, monitor_name)
        monitor = cls(env)
        monitors[eval_name] = monitor
    obs, info = env.reset()
    for name in eval_names:
        monitors[name].detectViolation(env.unwrapped.get_state(), 0)
    fields = [f"Violations_{name}" for name in list(eval_names)]
    fields.append("Win/Lose")
    fields.append("Score")
    fields.append("Blue Eaten")
    fields.append("Orange Eaten")
    fields.append("Ep. Length")
    entries = []
    for i in range(n_eval):
        episode_over = False
        ep_len = 0
        while not episode_over:
            action, _states = model.predict(obs)
            action = action.item()
            for name in eval_names:
                monitors[name].detectViolation(env.unwrapped.get_state(), action)
            obs, rewards, term, trunc, info = env.step(action)
            episode_over = term or trunc
            ep_len += 1
            if episode_over:
                dic = dict()
                for name in eval_names:
                    monitors[name].detectViolation(env.unwrapped.get_state(), action)
                    dic_single = monitors[name].export()
                    dic.update(dic_single)
                dic["Win/Lose"] = info["episode"][0]["w"]
                dic["Score"] = env.unwrapped.get_state().getScore()
                eaten = env.unwrapped.get_state().getGhostsEaten()
                dic["Blue Eaten"] = eaten[0]
                dic["Orange Eaten"] = eaten[1]
                dic["Ep. Length"] = ep_len

                for name in eval_names:
                    dic[f"Violations_{name}"] = monitors[name].violations
                entries.append(dic)
                obs, info = env.reset()

                for name in eval_names:
                    monitors[name].reset()
    return entries


monitor_desc = [
    ("Vegan", "VeganMonitor"),
    ("Vegetarian", "VegetarianOrangeMonitor"),
    ("Conditional_Vegan", "ConditionalVeganMonitor"),
    ("Penalty", "PenaltyMonitor"),
    ("Visit", "VisitMonitor"),
    ("EarlyBird", "EarlyBirdMonitor1"),
    ("Trapped", "TrappedMonitor"),
    ("OneTaste", "OneTasteMonitor"),
    ("AllOrNothing", "AllOrNothingMonitor"),
    ("VeganConflict", "VeganConflictMonitor"),
    ("Contradiction", "ContradictionMonitor"),
    ("VeganPref", "VeganPreferenceMonitor"),
    ("Benevolent", "VeganMonitor"),
    ("Switch", "SwitchMonitor"),
    ("Solution", "SolutionMonitor"),
    ("Guilt", "GuiltMonitor"),
    ("Maximum", "MaximumMonitor"),
]
monitor_dict = dict(monitor_desc)


def main(algo, test, exp_id, n_eval):
    env_name = "BerkeleyPacmanPO-v0"
    steps = 2500_000
    level = "smallClassic"
    feature_extractor_model = "complete"
    if test == "all":
        eval_names = [m[0] for m in monitor_desc]
    else:
        eval_names = [test]
    image_stack_size = 2
    greyscale = True
    model_name = (
        f"pickles/models/{algo}_{env_name.replace('/', '_')}_{steps}_level_"
        f"{level}_{feature_extractor_model}_{exp_id}.zip"
    )
    model = load_model(model_name, algo, exact_match=True)
    env = create_pacman_env(
        env_name,
        feature_extractor=feature_extractor_model,
        level=level,
        render_mode="none",
        greyscale=greyscale,
        image_stack_size=image_stack_size,
        scale="ppo" in algo or "image" in feature_extractor_model,
    )

    results = run_tests_with_monitor(env, model, eval_names, n_eval)
    print(f"Finished {exp_id}")
    with open(f"eval_results/pacman_{algo}_{exp_id}.json", "w") as fp:
        json.dump(results, fp)


if __name__ == "__main__":
    algo = sys.argv[1]
    test = sys.argv[2]
    exp_id = sys.argv[3]
    n_eval = 10000
    if exp_id == "all":
        for i in range(1, 6):
            main(algo, test, i, n_eval)
    else:
        exp_id = int(exp_id)
        main(algo, test, exp_id, n_eval)
