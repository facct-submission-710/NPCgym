import glob
import json
from collections import defaultdict
from utils.norms_pacman import monitor_desc

algos = ["dqn", "ppo"]
results_path = "eval_results/pacman_"

results = defaultdict(list)
for algo in algos:
    for file in glob.glob(results_path + algo + "*"):
        with open(file, "rb") as f:
            single_result = json.load(f)
            results[algo].append(single_result)

norm_bases = [n[0] for n in monitor_desc]

table_preamble = r"""
\begin{table}[]
\begin{tabular}{|c|l|l|}
\hline
Norm Bases & DQN & PPO \\ \hline 
"""
table_end = r"""
\end{tabular}
\end{table}
"""
table_content = []
for norm_base in norm_bases:
    row = f"{norm_base}"
    for algo in algos:
        results_for_nb = [r[f"Violations_{norm_base}"] for ep_r in results[algo] for r in ep_r]
        avg_violations = sum(results_for_nb) / len(results_for_nb)
        col_start = r"\multicolumn{1}{l|}{"
        col_end = r"}"
        row += f"& {col_start}{avg_violations:.2f}{col_end}"
    row += r"\\ \hline"
    table_content.append(row)
print(table_preamble)
print("\n".join(table_content))
print(table_end)

for algo in algos:
    scores = [r["Score"] for ep_r in results[algo] for r in ep_r]
    wins = [r["Win/Lose"] for ep_r in results[algo] for r in ep_r]
    avg_score = sum(scores) / len(scores)
    avg_win = sum(wins) / len(wins)
    print(f"{algo}: {avg_win:.2f} / {avg_score:.2f}")
