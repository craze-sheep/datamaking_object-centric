# T7 时序预测模块设计

> 任务：T7 时序预测模块
> 上游：T6 交互模块
> 下游：T8 输出解码模块、T10 `model/models/temporal.py`
> 第一版目标：从历史 `Th=12` 帧 interaction-aware object tokens 预测未来 `Tp=12` 帧 object latent

---

## 1. 设计结论

第一版采用 object-time Transformer encoder-decoder：

- encoder memory：历史交互 token `[B,Th,N,D]` flatten 为 `[B,Th*N,D]`
- decoder query：learned future query `[Tp,N,D]` broadcast 为 `[B,Tp*N,D]`
- 输出：未来 object latent `[B,Tp,N,D]`

默认采用非自回归一次性预测全部未来 12 帧，不使用 teacher forcing。原因：训练稳定、接口简单、便于 T10 测试；长期预测/24帧以上再考虑自回归或 Mamba。

---

## 2. 输入输出接口

### 2.1 输入

来自 T6：

```python
temporal_module(
    inter_token: Tensor,  # [B,Th,N,D]
    valid_mask: Tensor,   # [B,N] bool
) -> dict
```

### 2.2 输出

```python
{
    'future_token': Tensor,       # [B,Tp,N,D]
    'memory': Tensor,             # [B,Th*N,D] encoder output, optional aux
    'future_query': Tensor,       # [B,Tp*N,D] decoder input, optional aux
    'token_padding_mask': Tensor, # [B,Th*N] bool, True means masked
    'valid_mask': Tensor,         # [B,N] bool, 透传给T8/T9
}
```

`future_token` 是传给 T8 的主输出。

---

## 3. 时序模型选型

### 3.1 第一版：Transformer encoder-decoder

选择理由：

- 当前 token 数短：`Th*N = 12*7 = 84`，`Tp*N = 84`，Transformer计算量可控。
- encoder-decoder结构天然适合“历史条件 -> 未来 query”。
- 与 SlotFormer/object-centric video prediction 思路一致，但使用监督 object tokens 和物理状态。
- PyTorch 原生 `nn.TransformerEncoder/Decoder` 或自写 block 都容易测试。

### 3.2 暂不使用 Mamba/SSM

Mamba优点是长序列线性复杂度，但第一版序列不长，安装/CUDA兼容和实现复杂度不划算。保留接口：只要未来模块输入输出仍为 `[B,Th,N,D] -> [B,Tp,N,D]`，可以替换内部实现。

### 3.3 暂不使用 ConvLSTM

ConvLSTM更适合 dense feature map，不适合当前 object token + graph interaction 表示。若回到纯RGB baseline，可在 SimVP/PredRNN 中对比，而不是主架构采用。

---

## 4. 输入组织与位置编码

### 4.1 Flatten object-time tokens

```python
x = inter_token.reshape(B, Th * N, D)  # order: t-major then object id
```

顺序约定：

```python
index = t * N + n
```

该顺序必须在 positional embedding 和 padding mask 中保持一致。

### 4.2 历史位置编码

历史 memory token 加：

```python
time_emb_hist: [Th,D]
object_emb:    [N,D]
x[b,t,n] += time_emb_hist[t] + object_emb[n]
```

如果未来启用 scene/view embedding，应在 T5 已注入，T7 不重复注入。

### 4.3 future query

使用 learned query：

```python
future_base_query: [Tp,N,D]
future_query = future_base_query + time_emb_future[:,None,:] + object_emb[None,:,:]
# future_base_query 是content query，time/object embedding 是显式位置偏置；二者有意同时保留
future_query = future_query.reshape(Tp*N,D).unsqueeze(0).expand(B,Tp*N,D)
```

future time embedding 与 history time embedding 可以共用同一个 `nn.Embedding(Th+Tp, D)`：history index `0..Th-1`，future output step `0..Tp-1` 对应绝对时间 index `Th..Th+Tp-1`。

---

## 5. Padding mask

### 5.1 history memory padding mask

PyTorch Transformer convention：`True` 表示 mask 掉。

```python
valid_time_obj = valid_mask[:, None, :].expand(B, Th, N)  # [B,Th,N]
memory_key_padding_mask = ~valid_time_obj.reshape(B, Th*N)  # [B,Th*N]
```

### 5.2 future query padding mask

Decoder query 本身可以包含 padding object query，但输出后必须置零：

```python
future_valid = valid_mask[:, None, :].expand(B,Tp,N)
future_token = future_token * future_valid[..., None]
```

如果使用 `tgt_key_padding_mask`，同样使用 `~future_valid.reshape(B,Tp*N)`。但第一版推荐不传 `tgt_key_padding_mask`，只在输出和loss处 mask padding object，避免 all-invalid 或大量 padding query 触发 decoder self-attention NaN。

---

## 6. 非自回归预测策略

第一版采用一次性预测：

```python
future_token = TransformerDecoder(future_query, memory)
```

实际 PyTorch 调用必须传入 padding mask：

```python
memory = encoder(
    x,
    src_key_padding_mask=memory_key_padding_mask,
)
future_flat = decoder(
    future_query,
    memory,
    memory_key_padding_mask=memory_key_padding_mask,
    tgt_mask=None,                 # 非自回归；future queries 可双向 self-attend
    tgt_key_padding_mask=None,     # v1不mask query，输出后置零更稳
)
```

不使用 teacher forcing，因为 decoder 不接收未来GT token，只接收 learned future query。这样训练和推理一致，避免 exposure bias。

优点：

- 简单稳定。
- 易于并行预测12帧。
- 输出固定 `[B,Tp,N,D]`，直接接 T8 多头解码。

局限：

- 长期一致性可能弱。
- 对 24/36 帧预测可能不如自回归 rollout。

后续扩展：

- autoregressive latent rollout：逐步把前一步 predicted token 作为下一步输入。
- scheduled sampling：训练时混合GT latent/pred latent；但当前没有GT future latent，需额外 encoder future frames，因此第一版不做。
- Mamba/SSM：替换 Transformer encoder-decoder。

---

## 7. Transformer 结构

默认配置：

```yaml
token_dim: 256
num_encoder_layers: 4
num_decoder_layers: 4
num_heads: 4
ffn_dim: 1024
dropout: 0.0 initially  # 第一版先跑通shape/loss；过拟合时训练阶段再调大
activation: gelu
norm_first: true
batch_first: true
history_length: 12
predict_length: 12
max_objects: 7
```

最小12GB配置：

```yaml
token_dim: 128
num_encoder_layers: 2
num_decoder_layers: 2
num_heads: 4
ffn_dim: 512
```

### 7.1 为什么 encoder-decoder 而不是 decoder-only

- encoder-decoder明确区分历史 memory 和未来 query。
- future query 可以直接对应 `[Tp,N]` 输出槽位。
- padding mask 更清晰：history mask 用于 memory，future mask 用于输出置零。

### 7.2 输出归一化

Decoder 输出后：

```python
future_token = LayerNorm(future_token)
future_token = future_token.reshape(B,Tp,N,D)
future_token *= valid_mask[:,None,:,None]
```

---

## 8. Multi-step loss 配合

T7 本身不计算 loss，但需保留未来时间维 `[Tp]`，让 T9 对不同预测步加权：

```python
future_step_weights: [Tp]
```

T7 不做时序权重衰减，也不读取该权重；`future_step_weights` 由 T9/loss config 定义。T7 只保证输出顺序 `future_token[:, t, :, :]` 对应真实未来帧 `batch[:, Th+t]`。

---

## 9. 接口给 T8

T8 接收：

```python
future_token: [B,Tp,N,D]
valid_mask: [B,N]
```

T7 输出 dict 必须透传 `valid_mask`，减少 T8 集成漏传风险。可选：T8 state head 可能需要最后一帧历史 state 用于静态物体 copy-last-state。该信息由整体模型从 encoder_out 透传，不由 T7 修改。

---

## 10. T10 实现建议

文件：`model/models/temporal.py`

建议类：

```python
@dataclass
class TemporalConfig:
    token_dim: int = 256
    history_length: int = 12
    predict_length: int = 12
    max_objects: int = 7
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    num_heads: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.0
    norm_first: bool = True
    batch_first: bool = True
```

T10 必须使用 `batch_first=True` 的 `TransformerEncoderLayer` / `TransformerDecoderLayer`，因为本文档所有 shape 都按 `[B,S,D]` 约定。

```python
class ObjectTimePositionalEmbedding(nn.Module): ...
class FutureQuery(nn.Module): ...
class TemporalPredictor(nn.Module): ...
```

组合关系：

```text
TemporalPredictor
  ├─ pos_embedding: ObjectTimePositionalEmbedding
  ├─ future_query: FutureQuery
  ├─ encoder: TransformerEncoder(batch_first=True)
  └─ decoder: TransformerDecoder(batch_first=True)
```

入口断言/规整：

```python
assert inter_token.ndim == 4
B, Th, N, D = inter_token.shape
assert Th == config.history_length
assert N == config.max_objects
assert D == config.token_dim
assert valid_mask.shape == (B, N)
valid_mask = valid_mask.bool().to(inter_token.device)
```

all-invalid guard：

```python
all_invalid = valid_mask.sum(dim=1) == 0
# 对 all-invalid 样本，临时 unmask object 0 的所有历史time，避免 attention softmax 全 -inf
safe_valid_mask = valid_mask.clone()
safe_valid_mask[all_invalid, 0] = True
# transformer 正常运行后，最终 future_token[all_invalid] = 0
```

第一版不传 `tgt_key_padding_mask`，只使用 `memory_key_padding_mask`，并在输出后按原始 `valid_mask` 置零。

---

## 11. 单元测试要求（T10）

测试文件：`model/tests/test_temporal.py`

必须覆盖：

1. `test_temporal_predictor_output_shape`
   - 输入 `[B,Th,N,D]`
   - 输出 `future_token [B,Tp,N,D]`

2. `test_padding_objects_output_zero`
   - valid_mask 中部分 False
   - 验证对应未来 token 全0

3. `test_memory_padding_mask_shape_and_values`
   - 验证 `token_padding_mask [B,Th*N]`
   - padding object 对所有 time 均为 True

4. `test_future_query_shape_and_order`
   - 验证 future_query `[B,Tp*N,D]`
   - 验证 index 顺序为 `t*N+n`

5. `test_gradients_flow_through_temporal_module`
   - `future_token.sum().backward()`
   - 验证 Transformer 和 future query 参数有梯度

6. `test_non_autoregressive_no_future_gt_required`
   - forward 只传 `inter_token` 和 `valid_mask`
   - 不需要未来GT token

7. `test_all_invalid_mask_outputs_zero_without_nan`
   - valid_mask 全 False
   - 输出全0，无 NaN

8. `test_minimal_config_shape`
   - token_dim=128、layers=2
   - 验证输出 shape 正确

---

## 12. 替代方案与放弃理由

| 方案 | 结论 | 理由 |
|------|------|------|
| 自回归 latent rollout | 后续可试 | 第一版训练/推理一致性优先；无GT future latent teacher forcing |
| Scheduled sampling | 第一版不用 | 需要future latent监督，复杂度高 |
| Mamba/SSM | 后续替换 | 当前序列短，Transformer更稳 |
| ConvLSTM | 不用于主架构 | 更适合feature map，不适合object token |
| 一帧一帧循环预测state | 第一版不用 | 慢且误差累积明显 |
| 直接线性外推 | baseline可试 | 表达能力不足，不能建模复杂交互 |

---

## 13. 已知风险与缓解

| 风险 | 缓解 |
|------|------|
| 非自回归长期一致性弱 | 第一版只预测12帧；T9可加时序权重；T11按 `t=0..11` 分帧记录 PSNR/state MSE 退化曲线，后续再决定是否自回归 |
| padding token被attention看到 | 使用 key_padding_mask；输出再次置零 |
| object id跨帧不一致 | T5/T3保证；T7依赖slot index一致 |
| learned future query 表达不足 | 加 future time/object embedding；后续可加入 last state conditioning |
| Transformer过拟合 | dropout可开，层数可降，先跑最小配置 |
| scene/view条件缺失 | T5已注入view；scene默认关闭，后续稳定scene_id后再开 |

---

## 14. 迭代记录

### 2026-05-28 OpenCode/子agent第一轮评审

结论：需要修改。

主要问题：
- PyTorch Transformer 需要显式 `batch_first=True`，否则文档 `[B,S,D]` shape 会错。
- decoder cross-attention 必须传入 `memory_key_padding_mask`，避免 attend 到 padding memory。
- all-invalid mask 可能导致 attention softmax 全 `-inf` 产生 NaN。
- 第一版 future query 不应传 `tgt_key_padding_mask`，输出后置零更稳。
- 需要补充输入 shape/config 断言和 `valid_mask` 透传。

修改结果：
- 配置和实现建议中加入 `batch_first=True`，要求 Encoder/Decoder layer 均使用 batch_first。
- 补充 encoder/decoder 调用伪代码，明确 `memory_key_padding_mask` 同时传给 encoder 和 decoder。
- 增加 all-invalid guard：临时 unmask object 0 防 NaN，最终输出仍按原始 valid_mask 置零。
- 明确 v1 不传 `tgt_key_padding_mask`，`tgt_mask=None`，future queries 可双向 self-attend。
- 增加入口断言、`valid_mask` 输出透传、multi-step loss归属说明。

### 2026-05-28 Claude Code最终评审

结论：PASS。

通过理由：
- T7七个子问题全覆盖：模型选型、自回归策略、teacher forcing、预测长度、位置编码、长期误差、multi-step loss。
- 与 `architecture.md` 和 `interaction_module.md` 的 shape、非自回归策略、valid_mask接口一致。
- OpenCode/子agent指出的 batch_first、decoder memory mask、all-invalid guard、future query mask、shape断言均已修复。

轻微建议（不阻塞）：
- T10实现 all-invalid guard 时注释其仅防NaN，最终输出仍置零。
- T10测试可验证 reshape 顺序与 `index=t*N+n` 一致。
- T9/config 中定义默认 `future_step_weights`，避免T9遗漏。

---

## 15. 最终状态

当前状态：T7时序预测模块设计已完成，OpenCode/子agent修订后，Claude Code最终评审通过。
