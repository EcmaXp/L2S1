# Jev-type decision LoRA

The same pipeline supports all five local benchmark models: SmolLM2, Qwen3,
Gemma3, TinyLlama and Gemma4. A model profile pins its original Hugging Face
checkpoint, GGUF filename, Transformers loader and language attention adapter
targets. Each model gets its own native token export, discarded smoke run,
adapter, GGUF conversion and paired base/adapter evaluation. `--models all`
runs them sequentially and retains failures in the matrix summary.

The first real-weight pilot uses Gemma4 E2B. The other four profiles have CPU
architecture/gradient checks using tiny random models; those checks are not
pretrained-model, NF4, conversion or quality evidence for those models.

The [completed Gemma4 pilot](../benchmarks/jev-lora-20260927/README.md) improved raw label agreement from 54.30% to 57.85% and reduced soft KL from 3.7739 to 0.7544. Coverage fell from 92.75% to 48.35%, and correct accepted/all fell from 51.65% to 32.30%. Noul supplied the raw-accuracy gain; Choice and Score raw accuracy slightly declined. The existing rules regression also lost raw accuracy. The adapter remains an opt-in experiment.

## Ollaya reference and output

Reference: [ollaya-dev/ollaya at f9e2d11](https://github.com/ollaya-dev/ollaya/tree/f9e2d11fee1d01235878bfa6cfa1eb1e42bbbaea).
In particular, its [answer renderer](https://github.com/ollaya-dev/ollaya/blob/f9e2d11fee1d01235878bfa6cfa1eb1e42bbbaea/crates/ollaya-decision/src/answer.rs),
[API types](https://github.com/ollaya-dev/ollaya/blob/f9e2d11fee1d01235878bfa6cfa1eb1e42bbbaea/crates/ollaya-api/src/decide.rs),
and [Winnow family](https://github.com/ollaya-dev/ollaya/blob/f9e2d11fee1d01235878bfa6cfa1eb1e42bbbaea/docs/families/winnow.md)
inform the output contract and token-logit approach.

| Type | Native decision | Jev-shaped answer |
| --- | --- | --- |
| Choice | `choice` | `choice`, `probabilities`, `confidence` |
| Noul | `binary` | `noul = P(true)`; no confidence field |
| Score | zero-based `ordinal` | expected index in `score`, plus `legend`, `probabilities`, `confidence` |

Choice/Score confidence is `(K * max(p) - 1) / (K - 1)`, clamped to [0, 1].
It differs from L2S1's entropy confidence. Wire numbers use four decimal places;
winner selection and metrics use the original probabilities. A 300-case check
against compiled, unmodified Ollaya decision source matched all output JSONs.
This proves synthetic answer-rendering parity, not HTTP API or model parity.

`report_jev.py` writes one record per case to `jev-answers.jsonl`, with answers
keyed by question ID. `l2s1_policy` separately preserves native acceptance,
abstention reasons, candidate mass and entropy confidence. A raw Jev-shaped
answer is not permission to bypass an abstention. `usage.output_tokens` is zero:
the runtime reads candidate logits and the serializer constructs the answer.
This experiment adds no HTTP endpoint or shared multiquestion forward pass.

## Frozen first experiment

- Data: [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/tree/c76749ec58bd8c3d2ea706b31c333a9059c38f90),
  pinned revision `c76749ec58bd8c3d2ea706b31c333a9059c38f90`.
- Select 30 train cases from each of four workflows by a seeded ID hash:
  120 cases / 600 decisions. Reserve another 80 cases for development and leave
  1,000 unused. Evaluate the independent public test split: 400 cases / 2,000
  decisions. Exact canonical-state and ID overlap is rejected across all splits.
- Only state, instructions and criteria enter inference. Teacher probabilities
  supply soft targets; factors, gold labels and teacher metadata never enter prompts.
- This is a specialist trained on those four workflows. Test agreement measures
  agreement with synthetic teacher labels, not general decision correctness or a
  comparison with an unseen-workflow generalist. Near duplicates and pretraining
  exposure are not excluded.
- Export all cyclic answer-code assignments with the native renderer. Choose one
  assignment per training decision by a label-independent hash. Shared semantic
  labels, model identity, tokens and export source are validated before training.
- One epoch; LoRA rank 8, alpha 16, Q/V attention projections, dropout 0;
  microbatch 1, accumulation 12; AdamW learning rate 1e-4 decays to zero;
  soft-target candidate cross entropy plus 0.1 negative log candidate mass.
- NF4 frozen base, bf16 compute where supported (fp16 otherwise), fp32 adapters
  and normalization parameters. CUDA is required for this training path.
- A 12-example smoke adapter is discarded. Final training restarts from the base.
  Fixed final checkpoint; no test/development selection, threshold fitting or
  temperature fitting. These probabilities are not claimed calibrated.
- Re-evaluate both checkpoints in the same GGUF runtime and precision, with
  policy thresholds 0.8 top probability / 0.05 candidate mass. Report raw label
  accuracy, coverage, accepted accuracy, correct/all, hard and soft probability
  errors, per-type/per-workflow metrics, and full request latency.

## Run another model or the whole matrix

Use a Python environment with PyTorch, Transformers, PEFT, bitsandbytes,
huggingface_hub and pyarrow. The first run used PyTorch 2.14.0+cu130,
Transformers 5.17.0 and PEFT 0.21.0. Gemma4 requires a Transformers version that
supports its architecture. Existing local checkpoints can be reused offline.

Download the pinned dataset's `all/train-00000-of-00001.parquet` and
`all/test-00000-of-00001.parquet` to a source directory as `train.parquet` and
`test.parquet`. Retain the dataset card there as `README.md`.

```sh
python scripts/prepare_jev_data.py --source results/jev-source --output results/jev-data

# Explicit opt-in download; this does not train or change application defaults.
python scripts/run_jev_lora.py fetch --models all \
  --checkpoint-root results/jev-checkpoints --output results/jev-fetch

cargo build --release --features llama-cuda \
  --example export_decision_tokens --example evaluate_jsonl

python scripts/run_jev_lora.py run --models all \
  --checkpoint-root results/jev-checkpoints --data results/jev-data \
  --gguf-root models --output results/jev-matrix \
  --converter /path/to/matching/llama.cpp/convert_lora_to_gguf.py \
  --library-path /path/to/matching/llama.cpp/lib \
  --rules-fixture tests/fixtures/decision_benchmark.json
```

Use `--models gemma4` for one model, or list any subset. Checkpoints live under
`<checkpoint-root>/<pinned-revision>/`. Model weights remain local and are not
copied into Git. Fetch may require the user's existing Hugging Face access for
restricted checkpoints; no credentials are embedded or requested by the script.
Each output directory must be new. Inspect `<model>/run.json`, stage logs and
`summary.json` for failures; `status: ok` means completion, not improvement.
The optional rules fixture adds a paired regression run on the earlier 36
decisions. It is previously seen regression evidence, not another untouched test.

To add another supported causal model, supply `--profiles custom-models.json`
using the structure of `scripts/jev_model_profiles.json`. Pin the exact HF
revision and matching GGUF, choose the safe Auto loader and real adapter target
modules. Remote Python code is disabled. Unsupported token mappings, missing
adapter modules and failed GGUF conversions stop that model's run. Merely adding
a profile does not establish runtime support: execute smoke, conversion and
native evaluation before reporting a model as validated.

```sh
python -m unittest discover -s scripts -p test_jev_lora.py
# Run inside the training environment; uses tiny random CPU models, no downloads.
python -m unittest discover -s scripts -p test_jev_model_architectures.py
```
