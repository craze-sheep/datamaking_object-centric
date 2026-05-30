# Goal: 分块调用 Claude Code 评审并迭代完善 ai_model

你是 `/home/lzy/project/slot-datamaking` 项目的主工程 AI。当前新模型已经实现于 `model/ai_model/`，你的任务不是重新设计模型，而是通过“分块调用 Claude Code 评审 -> 自主判断 -> 修复 -> 测试 -> 再评审”的闭环，把代码迭代到没有会阻碍训练和验证的明显问题。

## 访问限制

1. 只允许读取和修改：
   - `/home/lzy/project/slot-datamaking/model`
   - `/home/lzy/project/slot-datamaking/database`
2. `database/` 是原始数据，只能读取，不能修改。
3. 不要访问或修改项目中其他目录。
4. 所有评审日志、修复记录、测试结果、训练结果放在：
   - `model/review_logs/`
   - `model/runs/`
   - `model/ai_model/checkpoints/`

## 总目标

使用 Claude Code / `claudecode` 对当前代码进行严格评审，但必须分块调用，不要一次性把所有内容塞给 Claude Code。

Claude Code 是外部评审员，你是主工程师。你需要自己判断评审意见是否成立，修复必要问题，然后再次分块评审。重复该流程，直到最新一轮评审没有 Blocking 问题，也没有会导致真实数据训练失败、shape 错误、loss 错误、数据泄漏、GPU/OOM 失效、checkpoint/resume 失效的 Major 问题。

## 环境要求

1. 默认 Python 可能没有 `torch`，必须使用 conda 环境：

   ```bash
   conda run -n model ...
   ```

2. 先检查可用的 Claude Code 命令：

   ```bash
   command -v claudecode || command -v claude-code || command -v claude
   ```

3. 优先使用 GPU。训练和 smoke test 必须验证 CUDA 可用时走 GPU。
4. 如果 CUDA OOM，必须自动降低 batch size；如果 batch size 为 1 仍然 OOM，则降低模型规模或缩短训练配置继续验证。

## 分块评审原则

必须分块调用 Claude Code。每个块只关注一组文件和一类问题，保存独立评审日志。

建议分块如下：

1. 数据块
   - `model/dataset.py`
   - `model/ai_model/data_adapter.py`
   - 与真实 `database/` 样本读取、字段转换、mask、padding、view metadata 相关的代码

2. 模型编码与交互块
   - `model/ai_model/config.py`
   - `model/ai_model/encoder.py`
   - `model/ai_model/interaction.py`
   - `model/ai_model/temporal.py`

3. 解码、loss 与总模型块
   - `model/ai_model/decoder.py`
   - `model/ai_model/loss.py`
   - `model/ai_model/model.py`
   - `model/ai_model/__init__.py`

4. 训练、评估、checkpoint 块
   - `model/ai_model/train.py`
   - checkpoint 保存/加载
   - AMP、GPU、OOM 自适应、resume、日志、验证逻辑

5. 测试与实验报告块
   - `model/ai_model/EXPERIMENT_REPORT.md`
   - `model/tests/` 中相关测试，如存在
   - smoke test 命令、真实数据训练证明、结果记录

6. 汇总评审块
   - 在前面所有块评审与修复完成后，再让 Claude Code 做一次总评审。
   - 总评审只看是否还有跨模块 Blocking/Major 问题，不处理纯风格建议。

## 每轮评审日志命名

第 1 轮第 1 个块：

```text
model/review_logs/iter_01_block_01_data.md
model/review_logs/iter_01_block_02_encoder_interaction_temporal.md
model/review_logs/iter_01_block_03_decoder_loss_model.md
model/review_logs/iter_01_block_04_train_checkpoint.md
model/review_logs/iter_01_block_05_tests_report.md
model/review_logs/iter_01_block_06_summary.md
```

第 2 轮继续使用：

```text
model/review_logs/iter_02_block_01_data.md
...
```

你自己的判断和修复计划保存为：

```text
model/review_logs/fix_plan_iter_01.md
model/review_logs/fix_plan_iter_02.md
```

最终报告保存为：

```text
model/review_logs/final_review_report.md
```

## Claude Code 调用方式

根据本机实际可用命令选择 `claudecode`、`claude-code` 或 `claude`。如果命令支持从 stdin 接收 prompt，可以使用类似方式：

```bash
claudecode <<'EOF' > model/review_logs/iter_01_block_01_data.md
你是严格代码评审员。请只评审，不要修改代码。

项目路径：/home/lzy/project/slot-datamaking
访问范围：只关注 model/ 和 database/。

本次只评审数据块：
- model/dataset.py
- model/ai_model/data_adapter.py

请重点检查：
1. 是否正确读取真实 database 数据，而不是 mock/random 数据。
2. 是否正确使用 RGB、mask、object_static、object_dynamic、force_matrix、video metadata。
3. depth 可以暂时不用，但不要误用或破坏深度数据。
4. tensor shape、dtype、device 是否稳定。
5. valid_mask、padding object、static object 是否正确处理。
6. 是否存在未来信息泄漏。
7. 缺失文件、空 mask、物体数超过 max_objects、异常 JSON 等边界情况是否安全。

请按严重程度输出：
- Blocking
- Major
- Minor
- Suggestion

每个问题必须包含：
- 文件路径
- 具体位置或函数名
- 问题原因
- 影响
- 建议修复方式
EOF
```

如果 Claude Code CLI 的参数格式不同，先用 `--help` 查看用法，再等价调用。不要因为命令名或参数不同就跳过评审。

## 各分块评审 Prompt

### Block 01: 数据块

```text
你是严格代码评审员。请只评审，不要修改代码。

本次只评审数据块：
- model/dataset.py
- model/ai_model/data_adapter.py

请重点检查真实数据读取、字段使用、shape、dtype、valid_mask、padding object、static object、view/video metadata、异常样本处理、未来信息泄漏。

请按 Blocking / Major / Minor / Suggestion 输出问题。每个问题包含文件路径、位置、原因、影响、建议修复方式。
```

### Block 02: 编码、交互、时序块

```text
你是严格代码评审员。请只评审，不要修改代码。

本次只评审模型编码与交互块：
- model/ai_model/config.py
- model/ai_model/encoder.py
- model/ai_model/interaction.py
- model/ai_model/temporal.py

请重点检查模型是否合理使用 RGB、mask、物体属性、动态状态、力矩阵、view metadata；检查 shape 流、mask 流、static/dynamic object、padding object、时序预测是否存在未来信息泄漏；检查 GPU device/dtype 是否一致。

请按 Blocking / Major / Minor / Suggestion 输出问题。每个问题包含文件路径、位置、原因、影响、建议修复方式。
```

### Block 03: 解码、loss、总模型块

```text
你是严格代码评审员。请只评审，不要修改代码。

本次只评审解码、loss 与总模型块：
- model/ai_model/decoder.py
- model/ai_model/loss.py
- model/ai_model/model.py
- model/ai_model/__init__.py

请重点检查预测目标是否和 loss 匹配；RGB、mask、state、collision/force 等监督是否合理；loss 是否可能 NaN/Inf；mask 权重、padding object、static object 是否处理正确；总模型 forward 是否只用历史输入预测未来。

请按 Blocking / Major / Minor / Suggestion 输出问题。每个问题包含文件路径、位置、原因、影响、建议修复方式。
```

### Block 04: 训练、GPU、checkpoint 块

```text
你是严格代码评审员。请只评审，不要修改代码。

本次只评审训练、GPU、OOM、checkpoint 块：
- model/ai_model/train.py
- 与 checkpoint、resume、AMP、batch size 自适应、验证循环、日志相关的代码

请重点检查训练脚本是否能用真实 database 数据完成 forward、loss、backward、optimizer step、save checkpoint；是否优先使用 GPU；CUDA OOM 时是否自动降低 batch size；AMP 是否正确；resume 是否可靠；是否有 device/dtype 错误；是否会误用 mock 数据作为最终证明。

请按 Blocking / Major / Minor / Suggestion 输出问题。每个问题包含文件路径、位置、原因、影响、建议修复方式。
```

### Block 05: 测试与实验报告块

```text
你是严格代码评审员。请只评审，不要修改代码。

本次只评审测试与实验记录：
- model/ai_model/EXPERIMENT_REPORT.md
- model/tests/ 中与 ai_model 相关的测试，如果存在
- smoke test 命令和训练结果记录

请重点检查是否有真实数据 smoke train 证明；是否至少完成 forward + loss + backward + optimizer step；是否保存 checkpoint；报告是否如实说明使用了哪些字段、哪些字段没用、为什么；测试是否覆盖关键 shape、mask、loss、GPU/OOM、checkpoint。

请按 Blocking / Major / Minor / Suggestion 输出问题。每个问题包含文件路径、位置、原因、影响、建议修复方式。
```

### Block 06: 汇总评审

```text
你是严格代码评审员。请只评审，不要修改代码。

前面已经分别评审过数据、模型、loss、训练和测试。本次请做跨模块总评审，只关注是否还有会阻碍真实训练和验证的 Blocking/Major 问题。

请重点检查：
1. 数据流从 database 到 model 是否完整。
2. 模型是否真的使用真实数据字段。
3. 是否存在未来信息泄漏。
4. shape、mask、static/padding object 是否跨模块一致。
5. loss 和输出是否匹配。
6. GPU、OOM 自适应、AMP、checkpoint、resume 是否能真实工作。
7. smoke test 与报告是否足以证明模型能训练。

请只输出 Blocking 和 Major；如果没有，请明确写“未发现 Blocking/Major 问题”。
```

## 自主判断规则

1. Claude Code 的 Blocking 必须修。
2. Claude Code 的 Major 默认必须修，除非你能用代码事实、测试结果或真实训练结果证明它是误判。
3. Minor 可以修，也可以记录为后续优化。
4. Suggestion 不要求全部实现。
5. 不要为了迎合 Claude Code 做无意义重构。
6. 不要在纯风格建议上无限循环。
7. 如果不同分块或不同轮 Claude Code 意见冲突，以真实代码、真实测试和真实训练结果为准。

## 每轮修复后必须运行

至少运行：

```bash
conda run -n model python -m pytest model/tests -q
```

如果 `model/tests` 没有覆盖 `ai_model`，需要新增或运行 `ai_model` 自带测试/检查命令。

必须运行真实数据 smoke train，例如：

```bash
cd /home/lzy/project/slot-datamaking/model
conda run -n model python ai_model/train.py \
  --data-root /home/lzy/project/slot-datamaking/database \
  --epochs 1 \
  --batch-size 4 \
  --max-train-batches 2 \
  --max-val-batches 1 \
  --amp
```

如果参数名和实际 `train.py` 不一致，以实际脚本为准，但必须达到同等目标：

1. 读取真实 `database/` 样本。
2. 使用 GPU，如果 CUDA 可用。
3. 完成至少一个 forward。
4. 计算有限 loss，不出现 NaN/Inf。
5. 完成 backward 和 optimizer step。
6. 保存 checkpoint。
7. OOM 时自动降低 batch size。

## 结束条件

满足以下条件后可以停止迭代：

1. 最新一轮分块评审和汇总评审没有 Blocking。
2. 最新一轮没有会导致训练失败、数据错误、loss 错误、shape 错误、GPU/OOM 失效、checkpoint/resume 失效的 Major。
3. 本地测试通过，或明确说明无对应测试并补充了等价 smoke check。
4. 真实 database smoke train 成功。
5. checkpoint 成功保存且路径存在。
6. `model/review_logs/final_review_report.md` 已写明最终结论。

## 最终报告必须包含

在 `model/review_logs/final_review_report.md` 中写明：

1. 总共进行了几轮评审。
2. 每轮分了哪些块调用 Claude Code。
3. 每个块的 Claude Code 主要发现。
4. 你修复了哪些问题。
5. 哪些 Claude Code 建议没有采纳，为什么。
6. 最终测试命令和结果。
7. 最终 smoke train 命令和 loss 摘要。
8. 最终 device、batch size、AMP 状态。
9. checkpoint 路径。
10. 是否仍有 Minor/Suggestion 遗留。

## 禁止事项

1. 不要一次性调用 Claude Code 评审所有代码，必须分块。
2. 不要跳过 Claude Code 评审。
3. 不要只保存评审结果而不判断、不修复。
4. 不要盲目照搬 Claude Code 建议。
5. 不要用 mock/random batch 作为最终训练证明。
6. 不要修改 `database/` 原始数据。
7. 不要因为第一次失败或 OOM 就停止。
8. 不要把纯风格建议当作必须无限迭代的问题。
