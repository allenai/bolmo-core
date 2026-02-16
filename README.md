<div align="center">
  <br>
  <h1>Bolmo</h1>
  <h4>The first family of competitive fully open byte-level language models.</h4>
</div>

<img width="7711" height="4780" alt="bolmo_architecture" src="https://github.com/user-attachments/assets/a143aca6-4adf-4b57-b352-8bf93d51bb48" />

<p></p>

<p align="center">
  <a href="https://github.com/allenai/bolmo-core">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Repository-181717?logo=github"></a>
  <a href="https://huggingface.co/collections/allenai/bolmo">
    <img alt="HuggingFace Models" src="https://img.shields.io/badge/🤗-Models-yellow"></a>
  <a href="https://github.com/allenai/bolmo-core/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  <a href="https://discord.gg/sZq3jTNVNG">
    <img alt="Discord" src="https://img.shields.io/badge/Discord%20-%20blue?style=flat&logo=discord&label=Ai2&color=%235B65E9"></a>
</p>

---

**Bolmo** is the first fully-open byte-level language model achieving performance on the level of state-of-the-art subword-level language models. Unlike traditional language models that rely on subword tokenizers (like BPE or WordPiece), Bolmo operates directly on raw UTF-8 bytes, making it:

- **Free of subword tokenization**: No need for language-specific tokenizers or vocabulary management.
- **Universally applicable**: Works seamlessly across all languages, scripts, and domains.
- **Fully open**: Complete training code, model weights, data processing pipeline, and paper.
- **Competitive performance**: Comes close to matching (and in some cases exceeds) subword-based state-of-the-art models across a wide range of tasks.
- **Better character understanding**: Superior performance on tasks requiring character-level knowledge.

See our technical report for details: https://allenai.org/papers/bolmo.

This repository is a fork of [OLMo-core](https://github.com/allenai/OLMo-core) that implements the complete Bolmo architecture and training pipeline through **byteifying** - our approach to converting existing subword models to byte-level models, using <1% of the pretraining budget.

## Models

We release Bolmo models in two sizes:

| Model | Parameters | Base Model | HuggingFace |
|-------|-----------|------------|-------------|
| **Bolmo-7B** | 7.6B | Olmo 3 7B | [allenai/Bolmo-7B](https://huggingface.co/allenai/Bolmo-7B) |
| **Bolmo-1B** | 1.5B | OLMo 2 1B | [allenai/Bolmo-1B](https://huggingface.co/allenai/Bolmo-1B) |

Training data is available via HuggingFace at [allenai/bolmo_mix](https://huggingface.co/datasets/allenai/bolmo_mix).

## Installation

First install [PyTorch](https://pytorch.org) according to the instructions specific to your operating system and hardware.

#### Using UV (Recommended)

```bash
git clone https://github.com/allenai/bolmo-core.git
cd bolmo-core
uv venv --python 3.12.12
. .venv/bin/activate
uv sync --frozen --extra xlstm --extra wandb
```

This will install the precise version used for development of all packages (as saved in `uv.lock`). Installation should typically take less than five minutes.

#### Using pip

Alternatively, via pip:

```
pip install -e .[xlstm,wandb]
```

### Optional Dependencies

For full functionality, you may need:
- [flash-attn](https://github.com/Dao-AILab/flash-attention) for efficient attention
- [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) for optimized training
- [xlstm](https://github.com/NX-AI/xlstm) for xLSTM components (mLSTM layers used in the Bolmo local models)
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) for low-memory loss implementations

See the [OLMo-core documentation](https://olmo-core.readthedocs.io/) for complete installation details.

We tested Bolmo installation on Ubuntu 24.04 and Rocky Linux release 8.10.

## Demo

### Inference with HuggingFace

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"
bolmo = AutoModelForCausalLM.from_pretrained("allenai/Bolmo-7B", trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained("allenai/Bolmo-7B", trust_remote_code=True)

message = ["Language modeling is "]
input_ids = tokenizer(message, return_tensors="pt")["input_ids"].to(device)

# `max_new_tokens` is the amount of bytes to generate
response = bolmo.generate(input_ids, max_new_tokens=256, do_sample=True, temperature=0.1)
print(tokenizer.decode(response[0], skip_special_tokens=True))
```

This should quickly generate a completion on a standard GPU (usually <5min, including downloading the weights from the internet). It should generate something like:

```
Language modeling is a fundamental task in natural language processing (NLP), aiming to predict the probability of a sequence of words occurring in a given context. This prediction is crucial for various applications, including speech recognition, machine translation, and text generation. [...]
```

### HuggingFace checkpoints vs. olmo-core checkpoints

This codebase uses the olmo-core checkpoint format. Bolmo models can be converted from this format to the HuggingFace format via:

```bash
python3 src/examples/huggingface/convert_checkpoint_to_hf.py \
    -i /path/to/bolmo/checkpoint \
    -o /path/to/bolmo/checkpoint/in/hf/format \
    -s 65536 \ # max sequence length
    --dtype float32 \
    --skip-validation
```

Converting from HF format back to olmo-core is not implemented at the moment. However, we provide the original olmo-core checkpoints for Bolmo 1B and Bolmo 7B in the `olmo_core/` subdirectory on HF: [1B](https://huggingface.co/allenai/Bolmo-1B/tree/main/olmo_core), [7B](https://huggingface.co/allenai/Bolmo-7B/tree/main/olmo_core).

## Training

Bolmo training uses a two-stage byteifying procedure to convert existing subword models to byte-level:

### Stage 1: Subword-to-Byte Distillation
Quickly learn weights for local models while freezing the global model (9.8B tokens ≈ 43B bytes). Training scripts for this stage are available at `bolmo_scripts/launch_stage1_*`.

To run the Stage 1 training script, first prepare your training data (for example, the [bolmo_mix](https://huggingface.co/datasets/allenai/bolmo_mix) dataset) via the [Dolma toolkit](https://github.com/allenai/dolma), see [here](https://github.com/allenai/dolma/blob/main/docs/getting-started.md#step-4-tokenize-the-dataset) for details.

### Stage 2: End-to-End Training
Train the entire model to utilize byte-level information (39.3B tokens ≈ 173B bytes).  Training scripts for this stage are available at `bolmo_scripts/launch_stage2_*`.

To run the Stage 1 training script, first prepare your training data (for example, the [bolmo_mix](https://huggingface.co/datasets/allenai/bolmo_mix) dataset) via the [Dolma toolkit](https://github.com/allenai/dolma), see [here](https://github.com/allenai/dolma/blob/main/docs/getting-started.md#step-4-tokenize-the-dataset) for details.

## Post-Training via Task Arithmetic
Existing post-trained checkpoints can be byteified without additional training using Task Arithmetic:

```bash
python3 src/examples/bolmo/instructify.py \
    --output=/path/to/output/ \
    --checkpoint-dir=/path/to/bolmo/checkpoint \
    --base-checkpoint-dir=/path/to/base-olmo/checkpoint \
    --instruct-checkpoint-dir=/path/to/post-trained-olmo/checkpoint \
    --alpha=1.0
```


## Evaluation

### Bolmo 7B Results

Bolmo 7B matches or exceeds the performance of state-of-the-art byte-level models and comes close to the source Olmo 3 7B model:

| Category | Bolmo 7B | Olmo 3 7B | BLT 7B |
|----------|----------|-----------|---------|
| Character Understanding (CUTE) | 78.6 | 56.9 | 52.3 |
| Multilingual Char (EXECUTE) | 71.6 | 55.1 | 46.3 |
| Code | 41.0 | 40.1 | 31.6 |
| Math | 48.9 | 55.3 | 15.7 |
| MC Stem | 65.5 | 66.3 | 49.0 |
| MC Non-Stem | 75.8 | 77.7 | 56.6 |
| GenQA | 70.9 | 72.4 | 68.4 |

Full evaluation results available in the paper.

### Reproducing Evaluations

We use [olmes](https://github.com/allenai/olmes) for all evaluations.

You can evaluate Bolmo like any subword LLM, except for one Gotcha: for loglikelihood evals, we need to fold the boundary + byte logits (second half of the vocabulary) into the byte logits (first half of the vocabulary):

```python
probs = F.softmax(logits.float(), dim=-1)
probs[..., :self.vocab_size_bolmo] += probs[..., self.vocab_size_bolmo:self.vocab_size_bolmo*2]
logits = torch.log(probs)
logits[..., self.vocab_size_bolmo:self.vocab_size_bolmo*2] = -100_000
```

The fork of OLMES at http://github.com/bminixhofer/olmes implements this option. You can evaluate Bolmo like this for example:

```
olmes \
    --model allenai/Bolmo-1B \
    --model-args '{"max_length": 16384, "trust_remote_code": "true", "model_type": "hf_bolmo", "add_bos_token": "true"}' \
    --task bolmo1b \
    --batch-size 32 \
    --output-dir workspace
```

This will reproduce the evals for Bolmo 1B. See also [#6](https://github.com/allenai/bolmo-core/issues/6). The olmes task suites for Bolmo 1B and Bolmo 7B are defined here: [link](https://github.com/bminixhofer/olmes/blob/5fe45c51eeb7c867a6351926b7c470d622796ccc/oe_eval/configs/task_suites.py#L980-L1051).

## Citation

To cite Bolmo:

```bibtex
@misc{bolmo,
      title={Bolmo: Byteifying the Next Generation of Language Models}, 
      author={Benjamin Minixhofer and Tyler Murray and Tomasz Limisiewicz and Anna Korhonen and Luke Zettlemoyer and Noah A. Smith and Edoardo M. Ponti and Luca Soldaini and Valentin Hofmann},
      year={2025},
      eprint={2512.15586},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.15586}, 
}
```

For the underlying OLMo-core framework:

```bibtex
@misc{olmo20242olmo2furious,
  title={{2 OLMo 2 Furious}},
  author={{Team OLMo} and Pete Walsh and Luca Soldaini and Dirk Groeneveld and Kyle Lo and Shane Arora and Akshita Bhagia and Yuling Gu and Shengyi Huang and Matt Jordan and Nathan Lambert and Dustin Schwenk and Oyvind Tafjord and Taira Anderson and David Atkinson and Faeze Brahman and Christopher Clark and Pradeep Dasigi and Nouha Dziri and Michal Guerquin and Hamish Ivison and Pang Wei Koh and Jiacheng Liu and Saumya Malik and William Merrill and Lester James V. Miranda and Jacob Morrison and Tyler Murray and Crystal Nam and Valentina Pyatkin and Aman Rangapur and Michael Schmitz and Sam Skjonsberg and David Wadden and Christopher Wilhelm and Michael Wilson and Luke Zettlemoyer and Ali Farhadi and Noah A. Smith and Hannaneh Hajishirzi},
  year={2024},
  eprint={2501.00656},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2501.00656},
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
