# vLLM PD Proxy Benchmark

一个纯客户端、可复现的异步压测工具，面向 OpenAI-compatible `chat/completions` 流式接口。默认配置连接：

```text
http://127.0.0.1:8192/v1/chat/completions
```

它不会读取或推断服务端 KV cache、NIXL 或 GPU 内部指标；所有延迟和吞吐均为客户端观测值。

## 1. 安装

建议在服务器新建环境，或直接在已有、能访问 Qwen tokenizer 缓存的环境中安装：

```bash
cd /home/gzw/pd_disaggregation/vllm_pd_benchmark
python -m pip install -r requirements.txt
```

工具不依赖 `datasets` 包，也不会联网下载测试数据。若 `tokenizer.path` 指向不存在的位置，Transformers 才会尝试按模型名获取 tokenizer。

## 2. 配置数据

修改 `config.json`。`request_count` 是每个已启用数据集的请求数；不足时 `sampling.repeat=true` 会循环取样。默认按配置中的数据集顺序逐个运行，每个阶段结束后才开始下一个数据集。

常见本地结构：

```text
/home/gzw/datasets/pd_benchmark/
├── sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json
├── gsm8k/test.jsonl
├── longbench/
│   ├── narrativeqa.jsonl
│   ├── qasper.jsonl
│   └── ... 共 21 个 JSONL
└── humaneval/test.jsonl
```

若 ShareGPT 文件名不同，先查看实际文件名再修改配置：

```bash
find /home/gzw/datasets/pd_benchmark/sharegpt -maxdepth 1 -type f
```

LongBench 的 `tasks` 可填写任意子集。删除 `tasks` 或设为 `[]` 会运行全部 21 个任务：

```json
{
  "name": "longbench",
  "enabled": true,
  "path": "/home/gzw/datasets/pd_benchmark/longbench",
  "tasks": []
}
```

四类数据的规范化方式：

- ShareGPT：读取 `conversations`，保留以 user 结尾的对话前缀；支持 `human/gpt` 和 `user/assistant` 角色。
- GSM8K：使用 `question` 字段，要求模型给出推理和最终答案。
- LongBench：组合 `context` 与 `input`，针对问答、摘要、分类、检索和代码任务使用相应提示模板。
- HumanEval：使用 `prompt` 字段，要求只续写代码。

## 3. 先做离线检查

离线检查会验证配置、读取数据、固定种子抽样、加载 tokenizer、计算输入 token，并保存请求 trace，但不会访问推理服务：

```bash
python benchmark.py --config config.json --dry-run
```

## 4. 运行压测

```bash
python -u benchmark.py --config config.json
```

可临时覆盖几个常用参数：

```bash
python -u benchmark.py --config config.json --requests 1000 --concurrency 32 --seed 20261102
```

若 warmup 失败，工具会给出警告并继续测量，因此正式实验前应检查最终 error 列。程序在有测量请求失败时仍保存全部结果，并以退出码 2 结束。

## 5. 负载模式

`load.mode` 支持：

- `closed_loop`：始终最多维持 `concurrency` 个在途请求；适合测系统饱和能力。`client_batch_size` 是客户端每次补充任务的最大数量。
- `burst`：每批提交 `client_batch_size` 个请求，等待整批完成后再提交下一批；实际在途数仍受 `concurrency` 限制。
- `poisson`：按固定 seed 生成指数分布到达间隔，平均速率由 `request_rate` 指定；超出服务能力时请求在客户端并发门限处排队。

`client_batch_size` 不是 vLLM 的内部 batch size，也不能强制服务端如何组批。

默认饱和实验使用 `separate_datasets=true`、`sampling.strategy=sequential`、`mode=burst`、`request_count=concurrency=client_batch_size=300`。每个数据集的 300 个请求一起提交到 Proxy，全部完成后再提交下一个数据集。客户端连接会一直保持到 Decode 结束；增大在途连接数后，前面的请求在 D 生成时，后面的请求才有机会继续进入 P。Proxy 负责传递 `kv_transfer_params`；实际 P/D 队列和动态批处理由服务端处理。是否同时打满两张卡仍需观察服务端指标，受 Proxy、KV 容量、GPU 和网络影响。

`max_output_tokens=2048` 配合 `ignore_eos=true` 尽量延长 Decode，但上下文仍需满足 `输入 token + 2048 <= max_model_len`。默认超时是 3600 秒；若队列更长，应相应提高。先以较小请求数试运行，再增加到 300。若只想生成到自然 EOS，把 `ignore_eos` 改回 `false`。

## 6. 指标定义

- TTFT：客户端开始发送请求，到 SSE 流中收到第一个非空生成内容的墙钟时间。
- E2E：客户端开始请求，到流式响应结束的墙钟时间。
- TPOT：输出至少 2 token 时，`(E2E - TTFT) / (output_tokens - 1)`。这是首 token 之后每个输出 token 的平均时间，不是逐 token inter-token latency。
- 输入 token：本地 tokenizer 对 `apply_chat_template(messages, add_generation_prompt=True)` 的计数。
- 输出 token：本地 tokenizer 对完整生成文本、不加 special token 的计数。
- 请求吞吐：成功请求数除以该统计组从最早开始到最晚完成的时间跨度。
- token 吞吐：成功请求的本地 token 数除以上述时间跨度。
- error rate：失败请求数除以全部测量请求数。

`summary.csv` 对 TTFT、TPOT、E2E、输入和输出 token 数给出 mean、p50、p90、p95、p99。流式 API 若返回 usage，`requests.csv` 还会保存 `server_prompt_tokens` 与 `server_completion_tokens`，但汇总统一使用指定的本地 tokenizer，以保持各请求口径一致。

## 7. 输出目录

每次运行新建一个带时间戳的目录：

```text
results/pd_benchmark_YYYYMMDD_HHMMSS_microseconds/
├── config.snapshot.json
├── run_metadata.json
├── requests.csv
├── summary.csv
└── timeseries.csv
```

- `requests.csv`：逐请求状态、错误、HTTP 状态、TTFT、TPOT、E2E、token 数、finish reason 和可选生成文本。
- `summary.csv`：overall、dataset、task 三层汇总。
- `timeseries.csv`：按时间桶记录开始/完成/活跃请求及吞吐；可关闭。
- `run_metadata.json`：环境、依赖版本、指标定义、warmup 结果，以及固定 seed 选中的完整样本 trace。
- `config.snapshot.json`：实际运行配置快照（包含命令行覆盖后的值）。

如生成内容可能包含敏感信息，将 `benchmark.save_output_text` 设为 `false`。

## 8. 推荐实验方式

先用 `--dry-run` 检查最大输入长度是否超过服务端 `--max-model-len`。正式对比不同 PD 配置时，保持数据 trace、seed、tokenizer、请求数、生成参数和负载模式不变；分别使用新的输出目录，不要只比较单次运行。


