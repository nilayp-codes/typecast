| model | setup | runs | n | first-try | final | avg calls | field acc (norm) | coverage | e2e acc (norm) | skills recall (norm) | skills F1 (norm) | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| openrouter/qwen/qwen-2.5-7b-instruct | B | 3 | 30 | 82.2% (80.0%-83.3%) | 98.9% (96.7%-100.0%) | 1.20 (1.17-1.23) | 68.2% (66.0%-70.4%) | 98.3% (95.0%-100.0%) | 67.1% (62.9%-70.4%) | 44.3% (42.0%-46.9%) | 54.4% (51.9%-57.6%) | $0.0103 ($0.0101-$0.0105) |
| openrouter/qwen/qwen-2.5-7b-instruct | C | 3 | 30 | 98.9% (96.7%-100.0%) | 98.9% (96.7%-100.0%) | 1.00 (1.00-1.00) | 68.2% (66.5%-69.8%) | 98.3% (95.0%-100.0%) | 67.1% (65.0%-69.8%) | 45.9% (41.2%-50.0%) | 53.2% (48.9%-57.7%) | $0.0076 ($0.0076-$0.0076) |
| openrouter/qwen/qwen3-235b-a22b-2507 | B | 3 | 30 | 76.7% (73.3%-80.0%) | 88.9% (86.7%-93.3%) | 1.34 (1.33-1.37) | 85.6% (85.1%-86.0%) | 88.3% (85.0%-95.0%) | 75.6% (72.8%-80.9%) | 75.9% (74.7%-76.9%) | 70.4% (68.3%-71.6%) | $0.0211 ($0.0199-$0.0234) |
| openrouter/qwen/qwen3-235b-a22b-2507 | C | 3 | 30 | 98.9% (96.7%-100.0%) | 98.9% (96.7%-100.0%) | 1.00 (1.00-1.00) | 86.1% (85.6%-87.0%) | 98.3% (95.0%-100.0%) | 84.7% (82.8%-85.8%) | 71.5% (69.8%-74.8%) | 68.2% (65.7%-72.1%) | $0.0136 ($0.0134-$0.0140) |

Cells: mean (min-max) over the runs. Files:
- openrouter/qwen/qwen-2.5-7b-instruct B: 2026-11-04_openrouter_qwen_qwen-2.5-7b-instruct_B_2.json (08dc98), 2026-11-04_openrouter_qwen_qwen-2.5-7b-instruct_B_3.json (e8cd7f), 2026-11-04_openrouter_qwen_qwen-2.5-7b-instruct_B_4.json (5372bb)
- openrouter/qwen/qwen-2.5-7b-instruct C: 2026-11-04_openrouter_qwen_qwen-2.5-7b-instruct_C.json (32f272), 2026-11-04_openrouter_qwen_qwen-2.5-7b-instruct_C_2.json (e21235), 2026-11-04_openrouter_qwen_qwen-2.5-7b-instruct_C_3.json (40bafb)
- openrouter/qwen/qwen3-235b-a22b-2507 B: 2026-11-04_openrouter_qwen_qwen3-235b-a22b-2507_B_2.json (0d77f9), 2026-11-04_openrouter_qwen_qwen3-235b-a22b-2507_B_3.json (905ff2), 2026-11-04_openrouter_qwen_qwen3-235b-a22b-2507_B_4.json (69c270)
- openrouter/qwen/qwen3-235b-a22b-2507 C: 2026-11-04_openrouter_qwen_qwen3-235b-a22b-2507_C.json (0a87ab), 2026-11-04_openrouter_qwen_qwen3-235b-a22b-2507_C_2.json (5139ed), 2026-11-04_openrouter_qwen_qwen3-235b-a22b-2507_C_3.json (7a9535)
