GSM8K数据集llama3.18b：

2026-09-30 02:01:10,927 | INFO     |   student_1 complete, PPL: 1.2274
2026-09-30 02:01:10,957 | INFO     |   student_2 complete, PPL: 1.1986
2026-09-30 02:01:11,288 | INFO     |   student_3 complete, PPL: 1.1787
2026-09-30 02:01:15,326 | INFO     |   student_4 complete, PPL: 1.1967
2026-09-30 02:01:15,326 | INFO     | [Stage 3] Selection & Summary...
2026-09-30 02:01:15,328 | INFO     |   Selection result: Case 2: Complete Consensus - All answers are equivalent
2026-09-30 02:01:15,328 | INFO     |   Best: student_3 (PPL: 1.1787)
2026-09-30 02:01:15,728 | INFO     |     [+] Positive rule: Calculate the total quantity and divide by the number of recipients to find the equal share for each individual.

{14}
2026-09-30 02:01:15,731 | INFO     | Shutting down thread pool...
2026-09-30 02:01:15,732 | INFO     | Thread pool shut down
2026-09-30 02:01:15,734 | INFO     |
============================================================
2026-09-30 02:01:15,735 | INFO     | Experiment Summary
2026-09-30 02:01:15,735 | INFO     | ============================================================
2026-09-30 02:01:15,735 | INFO     | Total questions: 1319
2026-09-30 02:01:15,735 | INFO     | Successfully processed: 1319
2026-09-30 02:01:15,735 | INFO     | Role distribution: {'student_3': 17, 'teacher': 49, 'student_1': 9, 'student_4': 16, 'student_2': 9}
2026-09-30 02:01:15,735 | INFO     | Rule base: 100 total rules
2026-09-30 02:01:15,735 | INFO     | PPL statistics: min=1.0104, max=1.3474, avg=1.1166
2026-09-30 02:01:19,957 | INFO     | Standard result file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20260929_185046/result.jsonl
2026-09-30 02:01:19,957 | INFO     | Detailed output file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20260929_185046/output.jsonl
2026-09-30 02:01:19,957 | INFO     | Rule base file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20260929_185046/rules.json
2026-09-30 02:01:19,957 | INFO     | Summary file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20260929_185046/summary.json
2026-09-30 02:01:19,957 | INFO     | ============================================================
2026-09-30 02:01:19,957 | INFO     | Experiment Completed!
2026-09-30 02:01:19,957 | INFO     | Duration: 54628.99 seconds
2026-09-30 02:01:19,957 | INFO     | Results saved to: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20260929_185046
2026-09-30 02:01:19,957 | INFO     | ============================================================

========================================
Role Selection Distribution:
  student_3: 17 (1.3%)
  teacher: 49 (3.7%)
  student_1: 9 (0.7%)
  student_4: 16 (1.2%)
  student_2: 9 (0.7%)
========================================


结果：
 python -c "import eval_accuracy as e; acc,n=e.get_acc('/home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20260929_185046//result.jsonl'
, dataset_type='gsm8k1', use_gold_patch=True); print(f'accuracy={acc:.4f}
, n={n}')"
use dataset_type: gsm8k1
CSV written: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20260929_185046/logs_legacy/all_results_result.csv (1319 rows)
CSV written: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20260929_185046/logs_legacy/wrong_cases_result.csv (160 rows)
CSV written: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20260929_185046/logs_legacy/right_cases_result.csv (1159 rows)
Log written: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20260929_185046/logs_legacy/detailed_eval.log
unified table log: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20260929_185046/logs_legacy/new.log
All Results CSV: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20260929_185046/logs_legacy/all_results_result.csv
Wrong Cases CSV: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20260929_185046/logs_legacy/wrong_cases_result.csv
Right Cases CSV: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20260929_185046/logs_legacy/right_cases_result.csv
Detailed Log: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20260929_185046/logs_legacy/detailed_eval.log
accuracy=0.8787, n=1319	







 cp core/config/local.yaml \
>   core/exp_res/gsm8k_close_20260929_185046/config_used.yaml
(ttt) lthpc@lthpc:~/private/code/zcp/TTT/TF-TTCL$
(ttt) lthpc@lthpc:~/private/code/zcp/TTT/TF-TTCL$
(ttt) lthpc@lthpc:~/private/code/zcp/TTT/TF-TTCL$ git rev-parse HEAD
69451340bb50f2f2f94e1b61f06eaf04597e553e
(ttt) lthpc@lthpc:~/private/code/zcp/TTT/TF-TTCL$ vllm --version
0.30.0


-----------------------------------------------------------------------------------------------
GSM8K数据集llama3.2-3B：
--------------------------------------------------------------------------------
2026-09-30 08:36:41,115 | INFO     |   student_1 complete, PPL: 1.1206
2026-09-30 08:36:41,514 | INFO     |   student_2 complete, PPL: 1.1383
2026-09-30 08:36:41,514 | INFO     |   student_3 complete, PPL: 1.2066
2026-09-30 08:36:41,514 | INFO     |   student_4 complete, PPL: 1.0712
2026-09-30 08:36:41,514 | INFO     | [Stage 3] Selection & Summary...
2026-09-30 08:36:41,515 | INFO     |   Selection result: Case 2: Complete Consensus - All answers are equivalent
2026-09-30 08:36:41,515 | INFO     |   Best: student_4 (PPL: 1.0712)
2026-09-30 08:36:41,982 | INFO     |     [+] Positive rule: To identify the key reasoning step, we can extract the following Positive Rule:

"Apply the order of operations (PEMDAS) to simplify expressions and solve problems logically, ensuring accuracy and fairness in sharing resources."

This rule is helpful for solving similar future problems involving mathematical expressions, sharing resources, or dividing quantities equally.
2026-09-30 08:36:41,984 | INFO     | Shutting down thread pool...
2026-09-30 08:36:41,985 | INFO     | Thread pool shut down
2026-09-30 08:36:41,987 | INFO     |
============================================================
2026-09-30 08:36:41,987 | INFO     | Experiment Summary
2026-09-30 08:36:41,987 | INFO     | ============================================================
2026-09-30 08:36:41,987 | INFO     | Total questions: 1319
2026-09-30 08:36:41,987 | INFO     | Successfully processed: 1319
2026-09-30 08:36:41,987 | INFO     | Role distribution: {'student_1': 13, 'teacher': 50, 'student_2': 15, 'student_4': 13, 'student_3': 9}
2026-09-30 08:36:41,987 | INFO     | Rule base: 100 total rules
2026-09-30 08:36:41,988 | INFO     | PPL statistics: min=1.0141, max=1.4696, avg=1.1396
2026-09-30 08:36:46,111 | INFO     | Standard result file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20260930_135538/result.jsonl
2026-09-30 08:36:46,111 | INFO     | Detailed output file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20260930_135538/output.jsonl
2026-09-30 08:36:46,111 | INFO     | Rule base file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20260930_135538/rules.json
2026-09-30 08:36:46,111 | INFO     | Summary file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20260930_135538/summary.json
2026-09-30 08:36:46,111 | INFO     | ============================================================
2026-09-30 08:36:46,111 | INFO     | Experiment Completed!
2026-09-30 08:36:46,111 | INFO     | Duration: 9663.06 seconds
2026-09-30 08:36:46,111 | INFO     | Results saved to: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20260930_135538
2026-09-30 08:36:46,111 | INFO     | ============================================================

========================================
Role Selection Distribution:
  student_1: 13 (1.0%)
  teacher: 50 (3.8%)
  student_2: 15 (1.1%)
  student_4: 13 (1.0%)
  student_3: 9 (0.7%)
========================================




结果：

cd ~/private/code/zcp/TTT/TF-TTCL/evaluate/accuracy
(ttt) lthpc@lthpc:~/private/code/zcp/TTT/TF-TTCL/evaluate/accuracy$ python -c "import eval_accuracy as e; acc,n=e.get_acc('/home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20260930_135538/result.jsonl', dataset_type='gsm8k1', use_gold_patch=True); print(f'accuracy={acc:.4f}, n={n}')"
use dataset_type: gsm8k1
CSV written: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20260930_135538/logs_legacy/all_results_result.csv (1319 rows)
CSV written: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20260930_135538/logs_legacy/wrong_cases_result.csv (222 rows)
CSV written: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20260930_135538/logs_legacy/right_cases_result.csv (1097 rows)
Log written: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20260930_135538/logs_legacy/detailed_eval.log
unified table log: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20260930_135538/logs_legacy/new.log
All Results CSV: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20260930_135538/logs_legacy/all_results_result.csv
Wrong Cases CSV: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20260930_135538/logs_legacy/wrong_cases_result.csv
Right Cases CSV: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20260930_135538/logs_legacy/right_cases_result.csv
Detailed Log: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20260930_135538/logs_legacy/detailed_eval.log
accuracy=0.8317, n=1319

------------------------------------------------------------------------------------------------


Math500数据集llama3.1 8b：
2026-09-30 16:59:53,295 | INFO     | Shutting down thread pool...
2026-09-30 16:59:53,296 | INFO     | Thread pool shut down
2026-09-30 16:59:53,296 | INFO     |
============================================================
2026-09-30 16:59:53,297 | INFO     | Experiment Summary
2026-09-30 16:59:53,297 | INFO     | ============================================================
2026-09-30 16:59:53,297 | INFO     | Total questions: 500
2026-09-30 16:59:53,297 | INFO     | Successfully processed: 500
2026-09-30 16:59:53,297 | INFO     | Role distribution: {'teacher': 48, 'student_4': 16, 'student_2': 15, 'student_3': 12, 'student_1': 9}
2026-09-30 16:59:53,297 | INFO     | Rule base: 100 total rules
2026-09-30 16:59:53,297 | INFO     | PPL statistics: min=1.0063, max=1.3638, avg=1.1303
2026-09-30 16:59:54,872 | INFO     | Standard result file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/math500_close_20260930_131435/result.jsonl
2026-09-30 16:59:54,873 | INFO     | Detailed output file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/math500_close_20260930_131435/output.jsonl
2026-09-30 16:59:54,873 | INFO     | Rule base file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/math500_close_20260930_131435/rules.json
2026-09-30 16:59:54,873 | INFO     | Summary file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/math500_close_20260930_131435/summary.json
2026-09-30 16:59:54,873 | INFO     | ============================================================
2026-09-30 16:59:54,873 | INFO     | Experiment Completed!
2026-09-30 16:59:54,873 | INFO     | Duration: 42317.68 seconds
2026-09-30 16:59:54,873 | INFO     | Results saved to: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/math500_close_20260930_131435
2026-09-30 16:59:54,873 | INFO     | ============================================================

========================================
Role Selection Distribution:
  teacher: 48 (9.6%)
  student_4: 16 (3.2%)
  student_2: 15 (3.0%)
  student_3: 12 (2.4%)
  student_1: 9 (1.8%)
========================================





结果；
 python -c "import eval_accuracy as e; acc,n=e.get_acc('/home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/math500_close_20260930_131435/result.jsonl', dataset_type='math_500', use_gold_patch=True); print(f'accuracy={acc:.4f}, n={n}')"
use dataset_type: math_500
  [Gold Patch] #5: '\text{Evelyn}' -> 'Evelyn'
  [Gold Patch] #100: '\begin{pmatrix} -1/3 \\ 2/3 \\ 5/3 \end{pmatrix}' -> '\begin{pmatrix} \frac{5}{3} \ \frac{2}{3} \ -\frac{1}{3} \end{pmatrix}'
  [Gold Patch] #147: '\frac{\sqrt{3}}{3}' -> '3'
  [Gold Patch] #199: '10,\!080' -> '10080'
  [Gold Patch] #228: '\text{(C)}' -> '(C)s'
  [Gold Patch] #256: '\text{(E)}' -> '(E)'
  [Gold Patch] #258: '864 \mbox{ inches}^2' -> '864'
  [Gold Patch] #294: '\left(\frac{3}{5},\frac{8}{3}\right]' -> '\left(\frac{3}{5}, \frac{8}{3}\right)'
  [Gold Patch] #327: '\frac{9}{100}' -> '0.0897'
  [Gold Patch] #348: '6r^2-4r-24' -> '$6r^2 - 4r - 24$'
  [Gold Patch] #365: '(1,-16,-4,43)' -> '(1, -16, -4, 39)'
  [Gold Patch] #384: 'x \in [-2,7]' -> '[-2, 7]'
  [Gold Patch] #419: '2\sqrt{5}' -> '$2\sqrt{5}$'
  [Gold Patch] #421: '\begin{pmatrix} 16/49 \\ 48/49 \\ 24/49 \end{pmatrix}' -> '\begin{pmatrix} \frac{16}{49} \ \frac{48}{49} \ \frac{24}{49} \end{pmatrix}'
  [Gold Patch] #473: '137 \frac{1}{2}' -> '$137\frac{1}{2}$'
  [Gold Patch] #497: '\frac{5}{13}' -> '\dfrac{65}{169}'
CSV written: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/math500_close_20260930_131435/logs_legacy/all_results_result.csv (500 rows)
CSV written: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/math500_close_20260930_131435/logs_legacy/wrong_cases_result.csv (240 rows)
CSV written: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/math500_close_20260930_131435/logs_legacy/right_cases_result.csv (260 rows)
Log written: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/math500_close_20260930_131435/logs_legacy/detailed_eval.log
unified table log: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/math500_close_20260930_131435/logs_legacy/new.log
All Results CSV: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/math500_close_20260930_131435/logs_legacy/all_results_result.csv
Wrong Cases CSV: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/math500_close_20260930_131435/logs_legacy/wrong_cases_result.csv
Right Cases CSV: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/math500_close_20260930_131435/logs_legacy/right_cases_result.csv
Detailed Log: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/math500_close_20260930_131435/logs_legacy/detailed_eval.log
accuracy=0.5200, n=500




Finance/wealth数据集llama3.2-3B：
2026-09-30 17:19:02,080 | INFO     | Experiment Summary
2026-09-30 17:19:02,080 | INFO     | ============================================================
2026-09-30 17:19:02,080 | INFO     | Total questions: 5000
2026-09-30 17:19:02,080 | INFO     | Successfully processed: 5000
2026-09-30 17:19:02,080 | INFO     | Role distribution: {'student_4': 13, 'student_3': 18, 'student_1': 18, 'student_2': 22, 'teacher': 29}
2026-09-30 17:19:02,080 | INFO     | Rule base: 100 total rules
2026-09-30 17:19:02,081 | INFO     | PPL statistics: min=1.0026, max=2.5619, avg=1.2921
2026-09-30 17:19:16,576 | INFO     | Standard result file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/wealth_open_20260930_182826/result.jsonl
2026-09-30 17:19:16,576 | INFO     | Detailed output file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/wealth_open_20260930_182826/output.jsonl
2026-09-30 17:19:16,576 | INFO     | Rule base file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/wealth_open_20260930_182826/rules.json
2026-09-30 17:19:16,576 | INFO     | Summary file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/wealth_open_20260930_182826/summary.json
2026-09-30 17:19:16,576 | INFO     | ============================================================
2026-09-30 17:19:16,576 | INFO     | Experiment Completed!
2026-09-30 17:19:16,576 | INFO     | Duration: 24635.12 seconds
2026-09-30 17:19:16,576 | INFO     | Results saved to: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/wealth_open_20260930_182826
2026-09-30 17:19:16,576 | INFO     | ============================================================

========================================
Role Selection Distribution:
  student_4: 13 (0.3%)
  student_3: 18 (0.4%)
  student_1: 18 (0.4%)
  student_2: 22 (0.4%)
  teacher: 29 (0.6%)
========================================



结果：
> print(f"ROUGE-Lsum={sum(scores)/len(scores):.4f}, n={len(scores)}")
> PY
ROUGE-Lsum=0.2378, n=5000


------------------------------------------------------------------------------------------------

aime24数据集llama3.18B：

--------------------------------------------------------------------------------
2026-10-01 08:46:08,498 | INFO     |   student_1 complete, PPL: 1.1615
2026-10-01 08:46:10,103 | INFO     |   student_2 complete, PPL: 1.1942
2026-10-01 08:46:17,262 | INFO     |   student_3 complete, PPL: 1.2226
2026-10-01 08:46:17,262 | INFO     |   student_4 complete, PPL: 1.2513
2026-10-01 08:46:17,263 | INFO     | [Stage 3] Selection & Summary...
2026-10-01 08:46:17,288 | INFO     |   Selection result: Case 3: Partial Consensus (Max Count: 2, Groups: 4)
2026-10-01 08:46:17,288 | INFO     |   Best: student_2 (PPL: 1.1942)
2026-10-01 08:46:17,289 | INFO     |   Worst: teacher (PPL: 1.0103)
2026-10-01 08:46:17,929 | INFO     |     [+] Positive rule: Apply Burnside's Lemma when counting distinct colorings under rotational symmetry, considering the average number of colorings fixed by each rotation to solve problems involving combinatorial arrangements with symmetry constraints.
2026-10-01 08:46:20,575 | INFO     |     [-] Negative rule: <rule>
\textbf{Negative Rule:} Do not overcomplicate combinatorial counting by ignoring constraints and considering all possible cases, especially when dealing with edge cases such as grid sizes or chip placements.
2026-10-01 08:46:20,581 | INFO     | Shutting down thread pool...
2026-10-01 08:46:20,583 | INFO     | Thread pool shut down
2026-10-01 08:46:20,583 | INFO     |
============================================================
2026-10-01 08:46:20,583 | INFO     | Experiment Summary
2026-10-01 08:46:20,583 | INFO     | ============================================================
2026-10-01 08:46:20,583 | INFO     | Total questions: 30
2026-10-01 08:46:20,583 | INFO     | Successfully processed: 30
2026-10-01 08:46:20,584 | INFO     | Role distribution: {'student_1': 3, 'teacher': 4, 'student_3': 2, 'student_4': 1, 'student_2': 1}
2026-10-01 08:46:20,584 | INFO     | Rule base: 11 total rules
2026-10-01 08:46:20,584 | INFO     | PPL statistics: min=1.0048, max=1.3195, avg=1.0943
2026-10-01 08:46:20,649 | INFO     | Standard result file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/aime24_close_20261001_143332/result.jsonl
2026-10-01 08:46:20,649 | INFO     | Detailed output file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/aime24_close_20261001_143332/output.jsonl
2026-10-01 08:46:20,649 | INFO     | Rule base file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/aime24_close_20261001_143332/rules.json
2026-10-01 08:46:20,650 | INFO     | Summary file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/aime24_close_20261001_143332/summary.json
2026-10-01 08:46:20,650 | INFO     | ============================================================
2026-10-01 08:46:20,650 | INFO     | Experiment Completed!
2026-10-01 08:46:20,650 | INFO     | Duration: 7967.56 seconds
2026-10-01 08:46:20,650 | INFO     | Results saved to: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/aime24_close_20261001_143332
2026-10-01 08:46:20,650 | INFO     | ============================================================

========================================
Role Selection Distribution:
  student_1: 3 (10.0%)
  teacher: 4 (13.3%)
  student_3: 2 (6.7%)
  student_4: 1 (3.3%)
  student_2: 1 (3.3%)
========================================



结果：
 python -c "import eval_accuracy as e; acc,n=e.get_acc('/home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/aime24_close_20261001_143332/result.jsonl', dataset_type='aime', use_gold_patch=True); print(f'accuracy={acc:.4f}, n={n}')"de/zcp/TTT/TF-TTCL/evaluate/accuracy$ python -c "import eval_accuracy as e;
use dataset_type: aime
CSV written: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/aime24_close_20261001_143332/logs_legacy/all_results_result.csv (30 rows)
CSV written: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/aime24_close_20261001_143332/logs_legacy/wrong_cases_result.csv (26 rows)
CSV written: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/aime24_close_20261001_143332/logs_legacy/right_cases_result.csv (4 rows)
Log written: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/aime24_close_20261001_143332/logs_legacy/detailed_eval.log
unified table log: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/aime24_close_20261001_143332/logs_legacy/new.log
All Results CSV: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/aime24_close_20261001_143332/logs_legacy/all_results_result.csv
Wrong Cases CSV: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/aime24_close_20261001_143332/logs_legacy/wrong_cases_result.csv
Right Cases CSV: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/aime24_close_20261001_143332/logs_legacy/right_cases_result.csv
Detailed Log: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/aime24_close_20261001_143332/logs_legacy/detailed_eval.log
accuracy=0.1333, n=30


------------------------------------------------------------------------------------------------
Minerva数据集llama3.18B：

--------------------------------------------------------------------------------
2026-10-01 22:33:55,641 | INFO     |   student_1 complete, PPL: 1.0769
2026-10-01 22:33:55,642 | INFO     |   student_2 complete, PPL: 1.1215
2026-10-01 22:33:55,642 | INFO     |   student_3 complete, PPL: 1.1360
2026-10-01 22:33:55,642 | INFO     |   student_4 complete, PPL: 1.1764
2026-10-01 22:33:55,642 | INFO     | [Stage 3] Selection & Summary...
2026-10-01 22:33:55,645 | INFO     |   Selection result: Case 3: Partial Consensus (Max Count: 3, Groups: 2)
2026-10-01 22:33:55,645 | INFO     |   Best: student_1 (PPL: 1.0769)
2026-10-01 22:33:55,645 | INFO     |   Worst: student_2 (PPL: 1.1215)
2026-10-01 22:33:57,214 | INFO     |     [+] Positive rule: Apply non-degenerate perturbation theory to find the energy correction of a non-degenerate state by computing the sum of first-order and second-order corrections, where the first-order correction is given by the expectation value of the perturbing Hamiltonian with respect to the unperturbed state and the second-order correction is given by the sum of the squares of the matrix elements of the perturbing Hamiltonian with respect to other states, divided by the energy differences between the unperturbed states.
2026-10-01 22:33:58,100 | INFO     |     [-] Negative rule: Check the validity of assumptions in non-degenerate perturbation theory, particularly when diagonalizing the matrix or calculating eigenstates and eigenvectors, to avoid incorrect conclusions and hallucinated theorems.
2026-10-01 22:33:58,102 | INFO     | Shutting down thread pool...
2026-10-01 22:33:58,102 | INFO     | Thread pool shut down
2026-10-01 22:33:58,103 | INFO     |
============================================================
2026-10-01 22:33:58,103 | INFO     | Experiment Summary
2026-10-01 22:33:58,103 | INFO     | ============================================================
2026-10-01 22:33:58,103 | INFO     | Total questions: 272
2026-10-01 22:33:58,104 | INFO     | Successfully processed: 272
2026-10-01 22:33:58,104 | INFO     | Role distribution: {'student_1': 18, 'student_4': 16, 'teacher': 40, 'student_3': 14, 'student_2': 12}
2026-10-01 22:33:58,104 | INFO     | Rule base: 100 total rules
2026-10-01 22:33:58,104 | INFO     | PPL statistics: min=1.0102, max=1.3721, avg=1.1174
2026-10-01 22:33:58,968 | INFO     | Standard result file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/minerva_close_20261002_014528/result.jsonl
2026-10-01 22:33:58,968 | INFO     | Detailed output file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/minerva_close_20261002_014528/output.jsonl
2026-10-01 22:33:58,968 | INFO     | Rule base file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/minerva_close_20261002_014528/rules.json
2026-10-01 22:33:58,968 | INFO     | Summary file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/minerva_close_20261002_014528/summary.json
2026-10-01 22:33:58,968 | INFO     | ============================================================
2026-10-01 22:33:58,968 | INFO     | Experiment Completed!
2026-10-01 22:33:58,968 | INFO     | Duration: 17309.61 seconds
2026-10-01 22:33:58,969 | INFO     | Results saved to: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/minerva_close_20261002_014528
2026-10-01 22:33:58,969 | INFO     | ============================================================

========================================
Role Selection Distribution:
  student_1: 18 (6.6%)
  student_4: 16 (5.9%)
  teacher: 40 (14.7%)
  student_3: 14 (5.1%)
  student_2: 12 (4.4%)
========================================

结果：
 python -c "import eval_accuracy as e; acc,n=e.get_acc('/home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/minerva_close_20261002_014528/result.jsonl', dataset_type='minerva', use_gold_patch=True); print(f'accuracy={acc:.4f}, n={n}')"zcp/TTT/TF-TTCL/evaluate/accuracy$ python -c "import eval_accuracy as e;
use dataset_type: minerva
CSV written: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/minerva_close_20261002_014528/logs_legacy/all_results_result.csv (272 rows)
CSV written: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/minerva_close_20261002_014528/logs_legacy/wrong_cases_result.csv (205 rows)
CSV written: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/minerva_close_20261002_014528/logs_legacy/right_cases_result.csv (67 rows)
Log written: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/minerva_close_20261002_014528/logs_legacy/detailed_eval.log
unified table log: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/minerva_close_20261002_014528/logs_legacy/new.log
All Results CSV: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/minerva_close_20261002_014528/logs_legacy/all_results_result.csv
Wrong Cases CSV: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/minerva_close_20261002_014528/logs_legacy/wrong_cases_result.csv
Right Cases CSV: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/minerva_close_20261002_014528/logs_legacy/right_cases_result.csv
Detailed Log: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/minerva_close_20261002_014528/logs_legacy/detailed_eval.log
accuracy=0.2463, n=272

------------------------------------------------------------------------------------------------

Gsm8k数据集qwen3-32B:
 python core/main.py --config core/config/qwen32_gsm.yaml --limit 1319

--------------------------------------------------------------------------------
2026-10-03 12:17:06,480 | INFO     |   student_1 complete, PPL: 1.1691
2026-10-03 12:17:08,379 | INFO     |   student_2 complete, PPL: 1.1559
2026-10-03 12:17:08,888 | INFO     |   student_3 complete, PPL: 1.1642
2026-10-03 12:17:09,981 | INFO     |   student_4 complete, PPL: 1.1671
2026-10-03 12:17:09,981 | INFO     | [Stage 3] Selection & Summary...
2026-10-03 12:17:09,982 | INFO     |   Selection result: Case 2: Complete Consensus - All answers are equivalent
2026-10-03 12:17:09,982 | INFO     |   Best: teacher (PPL: 1.1328)
2026-10-03 12:17:13,593 | INFO     |     [+] Positive rule: <think>
Okay, let's see. The user wants me to extract a general rule from the example given. The example is about Henry and his friends sharing pizzas. The answer involved calculating the total number of slices and then dividing by the number of people.

So, the key steps here are: first, find the total amount by multiplying the number of items (pizzas) by the number of parts (slices) each has. Then, divide that total by the number of people sharing.
2026-10-03 12:17:13,595 | INFO     | Shutting down thread pool...
2026-10-03 12:17:13,596 | INFO     | Thread pool shut down
2026-10-03 12:17:13,597 | INFO     |
============================================================
2026-10-03 12:17:13,597 | INFO     | Experiment Summary
2026-10-03 12:17:13,597 | INFO     | ============================================================
2026-10-03 12:17:13,597 | INFO     | Total questions: 1319
2026-10-03 12:17:13,598 | INFO     | Successfully processed: 1319
2026-10-03 12:17:13,598 | INFO     | Role distribution: {'teacher': 82, 'student_2': 5, 'student_4': 4, 'student_1': 2, 'student_3': 7}
2026-10-03 12:17:13,598 | INFO     | Rule base: 100 total rules
2026-10-03 12:17:13,598 | INFO     | PPL statistics: min=1.0455, max=1.5182, avg=1.1237
2026-10-03 12:17:16,801 | INFO     | Standard result file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20261002_121133/result.jsonl
2026-10-03 12:17:16,802 | INFO     | Detailed output file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20261002_121133/output.jsonl
2026-10-03 12:17:16,802 | INFO     | Rule base file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20261002_121133/rules.json
2026-10-03 12:17:16,802 | INFO     | Summary file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20261002_121133/summary.json
2026-10-03 12:17:16,802 | INFO     | ============================================================
2026-10-03 12:17:16,802 | INFO     | Experiment Completed!
2026-10-03 12:17:16,802 | INFO     | Duration: 115540.10 seconds
2026-10-03 12:17:16,802 | INFO     | Results saved to: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20261002_121133
2026-10-03 12:17:16,802 | INFO     | ============================================================

========================================
Role Selection Distribution:
  teacher: 82 (6.2%)
  student_2: 5 (0.4%)
  student_4: 4 (0.3%)
  student_1: 2 (0.2%)
  student_3: 7 (0.5%)

结果：

python -c "import eval_accuracy as e; acc,n=e.get_acc('/home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20261002_121133/result.jsonl', dataset_type='gsm8k1', use_gold_patch=True); print(f'accuracy={acc:.4f},
 n={n}')"
use dataset_type: gsm8k1
CSV written: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20261002_121133/logs_legacy/all_results_result.csv (1319 rows)
CSV written: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20261002_121133/logs_legacy/wrong_cases_result.csv (88 rows)
CSV written: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20261002_121133/logs_legacy/right_cases_result.csv (1231 rows)
Log written: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20261002_121133/logs_legacy/detailed_eval.log
unified table log: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20261002_121133/logs_legacy/new.log
All Results CSV: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20261002_121133/logs_legacy/all_results_result.csv
Wrong Cases CSV: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20261002_121133/logs_legacy/wrong_cases_result.csv
Right Cases CSV: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20261002_121133/logs_legacy/right_cases_result.csv
Detailed Log: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20261002_121133/logs_legacy/detailed_eval.log
accuracy=0.9333, n=1319

------------------------------------------------------------------------------------------------

Llama3.3——70B：
 python core/main.py --config core/config/llama33_gsm.yaml --limit 1319

--------------------------------------------------------------------------------
2026-10-04 23:40:43,538 | INFO     |   student_1 complete, PPL: 1.0800
2026-10-04 23:40:43,538 | INFO     |   student_2 complete, PPL: 1.0895
2026-10-04 23:40:44,443 | INFO     |   student_3 complete, PPL: 1.0380
2026-10-04 23:40:44,443 | INFO     |   student_4 complete, PPL: 1.1022
2026-10-04 23:40:44,443 | INFO     | [Stage 3] Selection & Summary...
2026-10-04 23:40:44,444 | INFO     |   Selection result: Case 2: Complete Consensus - All answers are equivalent
2026-10-04 23:40:44,444 | INFO     |   Best: student_3 (PPL: 1.0380)
2026-10-04 23:40:45,250 | INFO     |     [+] Positive rule: Divide total resources by the number of recipients to ensure equal distribution.
2026-10-04 23:40:45,252 | INFO     | Shutting down thread pool...
2026-10-04 23:40:45,253 | INFO     | Thread pool shut down
2026-10-04 23:40:45,255 | INFO     |
============================================================
2026-10-04 23:40:45,255 | INFO     | Experiment Summary
2026-10-04 23:40:45,255 | INFO     | ============================================================
2026-10-04 23:40:45,255 | INFO     | Total questions: 1319
2026-10-04 23:40:45,255 | INFO     | Successfully processed: 1319
2026-10-04 23:40:45,255 | INFO     | Role distribution: {'student_1': 12, 'teacher': 51, 'student_2': 14, 'student_3': 14, 'student_4': 9}
2026-10-04 23:40:45,255 | INFO     | Rule base: 100 total rules
2026-10-04 23:40:45,256 | INFO     | PPL statistics: min=1.0076, max=1.2277, avg=1.0319
2026-10-04 23:40:48,546 | INFO     | Standard result file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20261004_183059/result.jsonl
2026-10-04 23:40:48,546 | INFO     | Detailed output file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20261004_183059/output.jsonl
2026-10-04 23:40:48,546 | INFO     | Rule base file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20261004_183059/rules.json
2026-10-04 23:40:48,546 | INFO     | Summary file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20261004_183059/summary.json
2026-10-04 23:40:48,546 | INFO     | ============================================================
2026-10-04 23:40:48,546 | INFO     | Experiment Completed!
2026-10-04 23:40:48,546 | INFO     | Duration: 47386.06 seconds
2026-10-04 23:40:48,546 | INFO     | Results saved to: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20261004_183059
2026-10-04 23:40:48,546 | INFO     | ============================================================

========================================
Role Selection Distribution:
  student_1: 12 (0.9%)
  teacher: 51 (3.9%)
  student_2: 14 (1.1%)
  student_3: 14 (1.1%)
  student_4: 9 (0.7%)
========================================


结果；
 python -c "import eval_accuracy as e; acc,n=e.get_acc('/home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20261004_183059/result.jsonl', dataset_type='gsm8k1', use_gold_patch=True); print(f'accuracy={acc:.4f}, n={n}')"
use dataset_type: gsm8k1
CSV written: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20261004_183059/logs_legacy/all_results_result.csv (1319 rows)
CSV written: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20261004_183059/logs_legacy/wrong_cases_result.csv (51 rows)
CSV written: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20261004_183059/logs_legacy/right_cases_result.csv (1268 rows)
Log written: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20261004_183059/logs_legacy/detailed_eval.log
unified table log: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20261004_183059/logs_legacy/new.log
All Results CSV: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20261004_183059/logs_legacy/all_results_result.csv
Wrong Cases CSV: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20261004_183059/logs_legacy/wrong_cases_result.csv
Right Cases CSV: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20261004_183059/logs_legacy/right_cases_result.csv
Detailed Log: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/gsm8k_close_20261004_183059/logs_legacy/detailed_eval.log
accuracy=0.9613, n=1319

------------------------------------------------------------------------------------------------
llama3.18B geography：
 python core/main.py --config core/config/llama31_geography.yaml

--------------------------------------------------------------------------------
2026-10-05 22:33:54,881 | INFO     |   student_1 complete, PPL: 2.3807
2026-10-05 22:33:54,982 | INFO     |   student_2 complete, PPL: 1.9382
2026-10-05 22:33:56,752 | INFO     |   student_3 complete, PPL: 1.7463
2026-10-05 22:33:56,752 | INFO     |   student_4 complete, PPL: 2.1238
2026-10-05 22:33:56,752 | INFO     | [Stage 3] Selection & Summary...
2026-10-05 22:33:56,806 | INFO     |   Selection result: Open Domain: Group A=3, Group B=2, Similarities=[student_4=0.696, student_1=0.662, student_2=0.529, student_3=0.360]
2026-10-05 22:33:56,806 | INFO     |   Best: teacher (PPL: 1.2611)
2026-10-05 22:33:56,807 | INFO     |   Worst: student_3 (PPL: 1.7463)
2026-10-05 22:33:57,067 | INFO     |     [+] Positive rule: IF a question is short and opinion-based, THEN answer in one plain sentence.
2026-10-05 22:33:57,616 | INFO     |     [-] Negative rule: **IF** a question asks to reflect the opposite opinion, **THEN** the answer must be a direct and concise statement of the opposite opinion, without elaboration or comparison.
2026-10-05 22:33:57,619 | INFO     | Shutting down thread pool...
2026-10-05 22:33:57,620 | INFO     | Thread pool shut down
2026-10-05 22:33:57,629 | INFO     |
============================================================
2026-10-05 22:33:57,629 | INFO     | Experiment Summary
2026-10-05 22:33:57,629 | INFO     | ============================================================
2026-10-05 22:33:57,629 | INFO     | Total questions: 5000
2026-10-05 22:33:57,629 | INFO     | Successfully processed: 5000
2026-10-05 22:33:57,629 | INFO     | Role distribution: {'teacher': 26, 'student_4': 20, 'student_1': 23, 'student_2': 13, 'student_3': 18}
2026-10-05 22:33:57,629 | INFO     | Rule base: 100 total rules
2026-10-05 22:33:57,630 | INFO     | PPL statistics: min=1.0017, max=2.4688, avg=1.2925
2026-10-05 22:34:12,587 | INFO     | Standard result file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/geography_open_20261005_165118/result.jsonl
2026-10-05 22:34:12,587 | INFO     | Detailed output file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/geography_open_20261005_165118/output.jsonl
2026-10-05 22:34:12,587 | INFO     | Rule base file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/geography_open_20261005_165118/rules.json
2026-10-05 22:34:12,587 | INFO     | Summary file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/geography_open_20261005_165118/summary.json
2026-10-05 22:34:12,587 | INFO     | ============================================================
2026-10-05 22:34:12,587 | INFO     | Experiment Completed!
2026-10-05 22:34:12,587 | INFO     | Duration: 49358.79 seconds
2026-10-05 22:34:12,587 | INFO     | Results saved to: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/geography_open_20261005_165118
2026-10-05 22:34:12,587 | INFO     | ============================================================

========================================
Role Selection Distribution:
  teacher: 26 (0.5%)
  student_4: 20 (0.4%)
  student_1: 23 (0.5%)
  student_2: 13 (0.3%)
  student_3: 18 (0.4%)
========================================


结果：
 ROUGE-Lsum=0.2839, n=5000

------------------------------------------------------------------------------------------------

Llama3.18B medicine数据集：
2026-10-08 11:03:11,651 | INFO     | Experiment Summary
2026-10-08 11:03:11,651 | INFO     | ============================================================
2026-10-08 11:03:11,651 | INFO     | Total questions: 5000
2026-10-08 11:03:11,651 | INFO     | Successfully processed: 5000
2026-10-08 11:03:11,651 | INFO     | Role distribution: {'student_2': 25, 'student_1': 13, 'teacher': 33, 'student_3': 16, 'student_4': 13}
2026-10-08 11:03:11,652 | INFO     | Rule base: 100 total rules
2026-10-08 11:03:11,652 | INFO     | PPL statistics: min=1.0047, max=1.5328, avg=1.1853
2026-10-08 11:03:27,851 | INFO     | Standard result file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/medicine_open_20261007_215545/result.jsonl
2026-10-08 11:03:27,851 | INFO     | Detailed output file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/medicine_open_20261007_215545/output.jsonl
2026-10-08 11:03:27,852 | INFO     | Rule base file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/medicine_open_20261007_215545/rules.json
2026-10-08 11:03:27,852 | INFO     | Summary file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/medicine_open_20261007_215545/summary.json
2026-10-08 11:03:27,852 | INFO     | ============================================================
2026-10-08 11:03:27,852 | INFO     | Experiment Completed!
2026-10-08 11:03:27,852 | INFO     | Duration: 76046.27 seconds
2026-10-08 11:03:27,852 | INFO     | Results saved to: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/medicine_open_20261007_215545
2026-10-08 11:03:27,852 | INFO     | ============================================================
========================================
Role Selection Distribution:
  student_2: 25 (0.5%)
  student_1: 13 (0.3%)
  teacher: 33 (0.7%)
  student_3: 16 (0.3%)
  student_4: 13 (0.3%)

结果：
Medicine     ROUGE-Lsum=0.2238, n=5000

------------------------------------------------------------------------------------------------
llama3.18B agriculture数据集：
 python core/main.py --config core/config/llama31_agriculture.yaml

============================================================
2026-10-07 11:31:01,447 | INFO     | Experiment Summary
2026-10-07 11:31:01,448 | INFO     | ============================================================
2026-10-07 11:31:01,448 | INFO     | Total questions: 5000
2026-10-07 11:31:01,448 | INFO     | Successfully processed: 5000
2026-10-07 11:31:01,448 | INFO     | Role distribution: {'teacher': 26, 'student_3': 22, 'student_2': 19, 'student_4': 18, 'student_1': 15}
2026-10-07 11:31:01,448 | INFO     | Rule base: 100 total rules
2026-10-07 11:31:01,448 | INFO     | PPL statistics: min=1.0001, max=2.3379, avg=1.1614
2026-10-07 11:31:11,795 | INFO     | Standard result file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/agriculture_open_20261006_190030/result.jsonl
2026-10-07 11:31:11,795 | INFO     | Detailed output file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/agriculture_open_20261006_190030/output.jsonl
2026-10-07 11:31:11,795 | INFO     | Rule base file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/agriculture_open_20261006_190030/rules.json
2026-10-07 11:31:11,795 | INFO     | Summary file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/agriculture_open_20261006_190030/summary.json
2026-10-07 11:31:11,795 | INFO     | ============================================================
2026-10-07 11:31:11,795 | INFO     | Experiment Completed!
2026-10-07 11:31:11,795 | INFO     | Duration: 88230.80 seconds
2026-10-07 11:31:11,795 | INFO     | Results saved to: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/agriculture_open_20261006_190030
2026-10-07 11:31:11,795 | INFO     | ============================================================

========================================
Role Selection Distribution:
  teacher: 26 (0.5%)
  student_3: 22 (0.4%)
  student_2: 19 (0.4%)
  student_4: 18 (0.4%)
  student_1: 15 (0.3%)
========================================

结果：
ROUGE-Lsum=0.1320, n=5000
------------------------------------------------------------------------------------------------

llama3.1 8B finance数据集：
============================================================
2026-10-09 03:03:30,592 | INFO     | Experiment Summary
2026-10-09 03:03:30,592 | INFO     | ============================================================
2026-10-09 03:03:30,592 | INFO     | Total questions: 5000
2026-10-09 03:03:30,592 | INFO     | Successfully processed: 5000
2026-10-09 03:03:30,592 | INFO     | Role distribution: {'student_3': 18, 'student_1': 13, 'student_2': 21, 'teacher': 33, 'student_4': 15}
2026-10-09 03:03:30,593 | INFO     | Rule base: 100 total rules
2026-10-09 03:03:30,593 | INFO     | PPL statistics: min=1.0007, max=2.3053, avg=1.3259
2026-10-09 03:03:46,553 | INFO     | Standard result file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/wealth_open_20261008_215521/result.jsonl
2026-10-09 03:03:46,553 | INFO     | Detailed output file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/wealth_open_20261008_215521/output.jsonl
2026-10-09 03:03:46,553 | INFO     | Rule base file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/wealth_open_20261008_215521/rules.json
2026-10-09 03:03:46,553 | INFO     | Summary file: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/wealth_open_20261008_215521/summary.json
2026-10-09 03:03:46,553 | INFO     | ============================================================
2026-10-09 03:03:46,553 | INFO     | Experiment Completed!
2026-10-09 03:03:46,553 | INFO     | Duration: 47289.21 seconds
2026-10-09 03:03:46,553 | INFO     | Results saved to: /home/lthpc/private/code/zcp/TTT/TF-TTCL/core/exp_res/wealth_open_20261008_215521
2026-10-09 03:03:46,553 | INFO     | ============================================================
========================================
Role Selection Distribution:
  student_3: 18 (0.4%)
  student_1: 13 (0.3%)
  student_2: 21 (0.4%)
  teacher: 33 (0.7%)
  student_4: 15 (0.3%)

结果：
ROUGE-Lsum=0.2788, n=5000

------------------------------------------------------------------------------------------------
