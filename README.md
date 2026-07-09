# SmolVLA / SO-100 Practice Notes

Public support repository for SmolVLA, LeRobot datasets, and SO-100/SO100-style embodied AI experiments.

This repository is a public learning and engineering evidence entry for the SO101 / Pi0.5 / Evo-RL interview project line. It is not a packaged product and it does not contain private robot logs, credentials, or production deployment configuration.

## Portfolio / Interview Context

- Portfolio overview: https://notion.l2k.tech:28443/article/interview-portfolio
- P03 project page: https://notion.l2k.tech:28443/api/report-media/server-upload/notionnext-videos/interview-portfolio/20260708/project-homepages/p03-so101-evorl/index.html

This repository is the public support entry for the P03 SO101 / Evo-RL project line. It is suitable for discussing SmolVLA and LeRobot dataset practice; private robot safety gates, real hardware logs, dashboards, checkpoints, unpublished datasets, credentials, and unredacted camera data remain outside this public repository.

## What This Repo Contains

The repository records a practical path for testing SmolVLA-related workflows:

| Area | Files |
|---|---|
| Minimal model checks | `02-test_final_working.py`, `04-test_with_visualization.py`, `04-test_with_visualization_fixed.py`, `05-test_with_visualization_complete.py`, `08-test_with_denormalization.py` |
| Dataset notes | `README_DATASET.md`, `README_HOW_TO_USE_DATASET.md`, `SMOLVLA_DATASETS.md`, `13-数据集选择与使用经验.md` |
| SO-100 / SO100 practice | `10-SO100测试结果总结.md`, `10-test_so100_pickplace.py`, `02-在3060-8G上测试微调SO100/` |
| LIBERO setup notes | `05-LIBERO_SETUP_README.md`, `libero_test.log` |
| ROS2 integration planning | `16-ROS2集成完整方案.md`, `ROS2_INTEGRATION_CHECKLIST.md` |
| Visual evidence | `visualization_result.png`, `lerobot_aloha_sim_insertion_human_samples.png`, `lerobot_pusht_samples.png`, `01-阶段测试-二维数据测试/` |

## Quick Start

The quickest environment sanity check is the minimal working script:

```bash
conda activate smolvla
python 02-test_final_working.py
```

For dataset-oriented notes, start with:

```bash
python 01-simple_download.py
```

Then read:

- `QUICK_START.md`
- `README_DATASET.md`
- `README_HOW_TO_USE_DATASET.md`
- `SMOLVLA_DATASETS.md`

Some scripts and notes were created for a specific local GPU/workstation setup. Treat paths such as `/root/smolvla_project` as examples from that environment, not as a required public deployment path.

## Current Status

This repository is best read as an experiment log:

- SmolVLA input formatting, model loading, and inference calls were explored.
- LeRobot-style dataset layout and parquet/video separation were documented.
- SO-100/SO100 and LIBERO suitability were compared for practice and interview explanation.
- Several failure modes were kept intentionally, including dataset structure mismatch notes and LIBERO headless rendering issues.

The file `dataset_test_results.json` records failed structure checks for selected datasets. That is useful context: the repo preserves what did not work as well as what worked.

## Relationship To The Portfolio Project

This repo supports the P03 line:

> SO101 / Pi0.5 / Evo-RL training and evaluation loop.

Use this repository to explain:

- how SmolVLA and LeRobot datasets were evaluated before hardware integration,
- how dataset format, action shape, image input, and state vector assumptions were checked,
- what issues appear when moving from public datasets or simulators toward real robot evaluation,
- what belongs in a public learning repo versus a private hardware evidence repo.

Private robot safety gates, real hardware logs, internal paths, and non-public evaluation records are intentionally not published here.

## Public Boundary

This public repository should contain only sanitized learning material:

- No API keys, SSH keys, tokens, cookies, or service credentials.
- No private robot network addresses or live hardware connection settings.
- No unpublished dataset, checkpoint, or competition material.
- No private evaluation logs that identify local devices or operators.

If a script requires local paths, hardware, or datasets, treat it as a reproducibility note rather than a turnkey public demo.

## Suggested Reading Order

1. `QUICK_START.md`
2. `README_HOW_TO_USE_DATASET.md`
3. `SMOLVLA_DATASETS.md`
4. `10-SO100测试结果总结.md`
5. `16-ROS2集成完整方案.md`
6. `17-实践经验与技巧汇总.md`

## Notes For Interview Preparation

The strongest story here is not "a model benchmark score". The useful story is the engineering process:

1. inspect public dataset formats,
2. build the smallest inference check,
3. visualize model inputs and outputs,
4. document mismatch and rendering failures,
5. decide what is safe to move toward real robot evaluation.

That process connects this public repo to the private SO101/Evo-RL hardware evidence without exposing private runtime details.
