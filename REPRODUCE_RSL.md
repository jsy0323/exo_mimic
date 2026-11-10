# Windows / rsl 复现记录

源码：https://github.com/kaibohex/muscle-supermimic

- 主仓库：`5dad6f72234c021fe1df2ddb5c0661b406ea2ce8`
- MS-Human-700 子模块：`2d686957aefd5739cf4d2859a6acfd94c8f84400`
- 工作目录：`E:\exo_mimic`
- Conda 环境：`rsl`，Python 3.10.19
- GPU：RTX 5080 16GB，PyTorch 2.7.1+cu128
- mjlab 1.6.0、MuJoCo / MuJoCo-Warp 3.11.0、Warp 1.19.0、musclemimic-models 1.0.6、rsl-rl-lib 5.4.2

## 当前结果

当前默认任务为 **MFG-V11**，使用 `E:\exo_terrain\Env\MFG_Musculoskelet_V11\MFG_Raj2015_V11.xml`。沿用原环境肌肉模式：80 条下肢肌肉、26 个上身位置执行器，停用 14 个下肢位置执行器。外部 XML 和原始 NPZ 未修改。

已将 `train/1` 下的 120 条、364,487 帧 100Hz 动作转换至 `data/mfg_v11/`，划分为 109 条训练、11 条本地验证动作。训练副本按 XML 关节限位裁剪；速度按上游方式从重采样后的姿态计算。这是本地文件划分，不是论文 KIT 评测划分。

已有验证记录：**30 passed / 30 skipped**（跳过项依赖原仓库 KIT 数据）；MFG 真实步行动作的小规模 GPU 训练、仿真/解析项一致性、策略更新和检查点读写均通过。完整网络、512 环境、512 个想象样本及 32 步 rollout 完成一次完整更新，PyTorch 峰值显存约 2.68GB。`reproduction/mfg-profile-result.json` 保存该诊断结果。

此前长程训练日志仅到第 40 轮世界模型预训练；**尚未得到收敛策略**。调试检查点不能代表动作跟踪效果。本次 W&B 改动仅验证日志写入，没有启动训练。其余 KIT / MS-Human-700 命令保留在本文后部，供切换回上游模型时使用。

## YAML 训练参数

现在可集中修改 `configs/train_mfg_v11.yaml`，不需要改 Python 源码。该文件包括模型 XML、数据清单、环境/物理参数、网络结构、优化器参数、日志与 W&B 设置。

```bat
conda activate rsl
cd /d E:\exo_mimic

REM 只预览合并后的参数与命令，不启动训练、仿真或 W&B
python scripts\train_rsl.py --config configs\train_mfg_v11.yaml --dry-run

REM 确定配置后启动训练
python scripts\train_rsl.py --config configs\train_mfg_v11.yaml
```

常改参数的位置：

| YAML 字段 | 用途 | 当前配置 |
| --- | --- | --- |
| `model_xml`、`motion_file` | MFG XML 和已转换动作清单 | 指定的 V11 XML、`data/mfg_v11/train.txt` |
| `env.scene.num_envs` | 并行仿真环境数 | 512 |
| `agent.max_iterations` | 本次训练轮数；加载检查点时为追加轮数 | 16000 |
| `agent.lr` | 初始学习率 | 0.0002 |
| `agent.wm_only_iters` | 仅更新世界模型的前期轮数 | 500 |
| `agent.buffer_steps` | 每环境回放缓存长度 | 500 |
| `agent.wm_batch`、`agent.wm_batches` | 世界模型批次大小、每轮更新次数 | 2048、10 |
| `agent.pi_batch`、`agent.horizon` | 想象 rollout 批次与步数 | 512、32 |
| `agent.wm_hidden`、`enc_hidden`、`dec_hidden`、`critic_hidden` | 各网络隐藏层宽度（均在 `agent` 下） | 见 YAML |
| `agent.log_interval`、`agent.save_interval` | 曲线写入和检查点保存间隔 | 10、1000 |
| `agent.logger`、`agent.wandb_project`、`agent.run_name` | 日志类型、W&B 项目、运行名 | wandb、muscle_supermimic、mfg_v11_seed1 |
| `wandb.entity`、`wandb.mode` | W&B 团队/账号与在线/离线模式 | null、online |

例如修改学习率和训练轮数时，编辑 YAML 中现有的 `agent` 分组即可，不要新增第二个同名 `agent` 分组：

```yaml
agent:
  max_iterations: 20000
  lr: 0.0001
  pi_batch: 256
  horizon: 32
```

参数优先级为：**显式命令行参数 > YAML > 启动脚本默认值 / 任务默认值**。可临时覆盖 YAML，而不改文件：

```bat
python scripts\train_rsl.py --config configs\train_mfg_v11.yaml --num-envs 256 --iterations 20000 --wandb-mode offline
```

不指定 `--config` 时仍保留原来的命令行用法。YAML 内的相对路径统一相对于项目根目录 `E:\exo_mimic`；配置文件自身的 `--config` 路径相对于执行命令时的目录。网络层用 YAML 列表（如 `[2048, 2048, 2048]`），布尔值用 `true` / `false`，不要加引号。启动脚本会转换为 mjlab 接受的参数形式；训练器继续把实际环境与算法配置保存至运行目录的 `params/env.yaml` 和 `params/agent.yaml`。

`--dry-run` 检查配置结构、基本数值和文件路径并预览命令；其余 `env` / `agent` 字段名称与类型由 mjlab 在启动仿真前解析。提供的 MFG YAML 已通过原生解析器验证，不执行训练。

## MFG 训练与 W&B 曲线

`rsl` 已安装 `wandb`。在 Anaconda Prompt / CMD 中，首次使用先登录；API key 只在本机登录提示中输入：

```bat
conda activate rsl
cd /d E:\exo_mimic
python -m wandb login
python scripts\train_rsl.py --config configs\train_mfg_v11.yaml
```

启动脚本默认启用 W&B，同时持续写入本地 `train_log.csv`。运行开始时终端会打印 W&B run 链接，并保存到 `logs/rsl_rl/mfg_v11_tracking/<run>/wandb_url.txt`。打开该链接，在 Workspace 中查看曲线；横轴使用 `train/iteration`，默认每 10 轮写入一次，可用 `--log-interval 1` 调整。

| 曲线 | 含义 |
| --- | --- |
| `train/sim_rew` | 仿真中的平均归一化单步奖励 |
| `train/ep_ret_norm`、`train/ep_len` | 已结束回合的归一化回报与长度（控制步数） |
| `train/term_frac` | 已结束回合中失败终止的比例 |
| `train/wm_loss`、`train/wm_err8`、`train/wm_touch` | 世界模型损失、8 步误差、足部接触预测损失 |
| `train/pi_loss`、`train/v_loss`、`train/pi_gradnorm` | 策略损失、价值损失、策略梯度范数 |
| `train/imag_rew`、`train/imag_survival` | 想象 rollout 中的奖励和存活程度 |
| `train/rew_*` | 各奖励项对归一化单步奖励的贡献 |
| `train/it_per_s`、`train/gpu_mem_gb`、`train/lr` | 迭代速度、PyTorch 峰值显存（GB）和学习率 |

前 500 轮只训练世界模型，因此策略/价值相关指标为 0 属于预期行为。正常结束或 Ctrl+C 中断时会关闭并写完日志；W&B 结束不意味着自动保存额外模型检查点。检查点仍每 1000 轮及正常训练结束时保存。

如需写入团队项目，追加 `--wandb-entity <团队名称>`。不传时使用 `WANDB_ENTITY` 或当前登录账号。

离线记录、之后手动同步：

```bat
python scripts\train_rsl.py --config configs\train_mfg_v11.yaml --wandb-mode offline --run-name mfg_v11_offline
python -m wandb sync "logs\rsl_rl\mfg_v11_tracking\<run>\wandb\<offline-run目录>"
```

离线模式不会生成可立即查看的在线曲线；同步后才能在网页查看。`--wandb-mode` 不传时沿用 `WANDB_MODE`，环境变量也未设置时使用 W&B 默认在线模式。若只需本地 TensorBoard，可传 `--logger tensorboard`；`--smoke` 始终使用本地日志，不创建 W&B run。新日志选项不会补传历史 CSV。

## 本机适配

1. 将项目 Python 安装下限从 3.12 放宽至 3.10，保留现有 `rsl` 的 Python 和 CUDA PyTorch；已用测试及 GPU 训练检查验证。上游推荐环境仍是 Python 3.12。Warp 固定为 1.19.0，因为其提示 1.20 将移除 Python 3.10 支持。
2. 子模块仅在本地 Git 配置中使用 HTTPS 地址；不改上游 `.gitmodules`。
3. 将 `tests/test_motion.py` 的目录枚举期望值标准化，适配 Windows 路径分隔符。
4. 启动脚本为子进程设置 UTF-8，避免本机 CP932 终端输出帮助时报编码错误。
5. Windows 启动配置关闭 `torch.compile`，无需安装 Triton；MuJoCo-Warp 仍使用 GPU。性能与上游 Linux 编译模式存在差异。

环境安装前后清单分别保存于 `reproduction/rsl-before.txt` 和 `reproduction/rsl-installed.txt`。安装前 Conda 清单为 `reproduction/conda-before.txt`，这些清单不是完整环境备份。

## 使用

以下命令从项目根目录执行：

```powershell
conda activate rsl
$env:PYTHONUTF8 = '1'
$env:WARP_CACHE_PATH = "$PWD\logs\warp-cache"

# 在现有 rsl 环境重新安装时使用约束，保留 CUDA PyTorch
python -m pip install -e . pytest huggingface_hub -c reproduction/constraints-rsl.txt

# 原仓库测试；缺少动作数据时会跳过相关项
New-Item -ItemType Directory -Force logs | Out-Null
python -m pytest -q --basetemp=logs/pytest-local

# MFG 真实动作的 GPU 训练检查（会执行 4 次小网络更新）
python scripts/train_rsl.py --smoke
# 上游模型的合成参考 GPU 检查
python scripts/train_rsl.py --smoke --robot MyoFullBody
python scripts/train_rsl.py --smoke --robot MS-Human-700-Locked-Hands
python scripts/train_rsl.py --smoke --robot MS-Human-700
```

### 接入已有 KIT 数据

需要 MuscleMimic **已重定向至 MyoFullBody 的 KIT `.npz` 数据**，原始 AMASS/SMPL 数据不能直接使用。目录可指向数据根目录、`MyoFullBody`、`gmr` 或 `KIT`：

```powershell
python scripts/prepare_local_data.py 'D:\your-dataset-directory'
```

脚本校验上游划分的 972 条训练动作和 108 条测试动作，写入 `data/local_musclemimic/` 的绝对路径清单；不复制整个数据集。仅复制上游测试所需的一条 walking clip 至 `data/myofullbody/walk.npz`。

```powershell
# 真实动作上的小规模训练验证
python scripts/train_rsl.py --robot MyoFullBody --smoke --motion-file data/myofullbody/walk.npz

# MyoFullBody 正式训练：原始网络结构，缩小环境数/缓存/想象批次
python scripts/train_rsl.py --robot MyoFullBody --iterations 16000

# 若显存不足，可进一步降低规模
python scripts/train_rsl.py --robot MyoFullBody --num-envs 256 --policy-batch 256 --buffer-steps 256
```

默认 512 个环境、500 步缓存、512 个想象样本是针对本机的起始设置，尚未完成真实数据下的完整峰值显存验证。上游默认 4096 个环境超出本机显存预算；批次规模和编译模式变化后，训练时间及收敛指标需要重新实测。

日志和检查点写入 `logs/rsl_rl/myofullbody_tracking/<run>/`；启动脚本同样支持上述 W&B 参数。

### 评测与视频

用实际训练输出的检查点替换下面的路径：

```powershell
python -m muscle_supermimic.tasks.tracking.scripts.evaluate --task Mjlab-SuperMimic-Flat-MyoFullBody --checkpoint 'logs/rsl_rl/myofullbody_tracking/<run>/model_016000.pt' --motion-file data/local_musclemimic/kit_kinesis_test.txt --record logs/rollout.npz
python -m muscle_supermimic.tasks.tracking.scripts.render logs/rollout.npz logs/videos/
```

### MS-Human-700

先将同一数据重定向，再传入对应清单：

```powershell
python -m muscle_supermimic.tasks.tracking.scripts.retarget_myofullbody data/mshuman700_kit data/local_musclemimic/kit_kinesis_train.txt data/local_musclemimic/kit_kinesis_test.txt --workers 6
python scripts/train_rsl.py --robot MS-Human-700-Locked-Hands --motion-file data/mshuman700_kit/kit_kinesis_train.txt --num-envs 256 --policy-batch 256
python -m pytest -q --basetemp=logs/pytest-with-data
```

原始动作和重定向动作按上游说明留在本地，不加入版本管理。
