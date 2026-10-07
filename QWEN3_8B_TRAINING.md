# Setting Up

1. Make sure you have cloned the https://github.com/allenai/bolmo-core repo and checked out the "qwen3" branch.

2. Create a venv and install dependencies:

```
uv venv --python 3.12.12
. .venv/bin/activate
uv sync --frozen --extra xlstm --extra wandb --extra eval # this will install torch==2.9.1

# install flash attention
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
# default datasets version seems to fail
uv pip install datasets==4.8.5 huggingface-hub==0.34.0
```

3. Convert the Qwen3-8B-Base checkpoint.

```bash
HF_CHECKPOINT=Qwen/Qwen3-8B-Base
OUTPUT_DIR=/data/benjamin/checkpoints/qwen3_8b_base_olmo_core
python3 src/examples/huggingface/convert_checkpoint_from_hf.py \
    --checkpoint-input-path $HF_CHECKPOINT \
    --output-dir $OUTPUT_DIR \
    --model-arch qwen3_8b \
    --tokenizer qwen3
```

It is crucial that this is `Qwen/Qwen3-8B-Base`. `Qwen/Qwen3-8B` IS NOT THE BASE MODEL (I was confused by this).

You should see something like:

```
2026-04-30 15:31:36.858 brev-uottiqy0e:0        __main__:380    INFO    Running OLMo core and HF models for validation...
2026-04-30 15:33:08.302 brev-uottiqy0e:0        __main__:252    INFO    Validation completed successful
```

4. Tokenize the training corpus:

```bash
# Assuming a working installation of the `dolma` tool (see https://github.com/allenai/dolma, I used commit 669f534823b08d266a8fff01f8a1c916a5a56576, the latest at the time of writing).
# Also need to download https://huggingface.co/datasets/allenai/bolmo_mix, for example:
# >> hf download --repo-type dataset allenai/bolmo_mix --local-dir bolmo_mix

# feel free to adjust --processes based on your setting (it's pretty slow with 32)
# should amount to ~160B tokens
dolma tokens \
    --documents "/path/to/bolmo_mix/**/*.zst" \
    --tokenizer.name_or_path "Qwen/Qwen3-8B" \
    --destination /path/to/bolmo_mix_qwen3 \
    --tokenizer.eos_token_id 151645 \
    --tokenizer.pad_token_id 151643 \
    --dtype uint32 \
    --processes 32
```

5. Create `data_sources_qwen3.txt` containing all numpy files in `/path/to/bolmo_mix_qwen3` separated by newlines. In my case:

```bash
(bolmo-repro) ~/benjamin/bolmo-core$ head data_sources_qwen3.txt 
/data/benjamin/preprocessed/bolmo_mix_qwen3/part-00-00000.npy
/data/benjamin/preprocessed/bolmo_mix_qwen3/part-00-00001.npy
/data/benjamin/preprocessed/bolmo_mix_qwen3/part-00-00002.npy
/data/benjamin/preprocessed/bolmo_mix_qwen3/part-00-00003.npy
/data/benjamin/preprocessed/bolmo_mix_qwen3/part-00-00004.npy
/data/benjamin/preprocessed/bolmo_mix_qwen3/part-01-00000.npy
/data/benjamin/preprocessed/bolmo_mix_qwen3/part-01-00001.npy
/data/benjamin/preprocessed/bolmo_mix_qwen3/part-01-00002.npy
/data/benjamin/preprocessed/bolmo_mix_qwen3/part-01-00003.npy
/data/benjamin/preprocessed/bolmo_mix_qwen3/part-01-00004.npy
(bolmo-repro) ~/benjamin/bolmo-core$ wc -l data_sources_qwen3.txt 
173 data_sources_qwen3.txt
```

Done!

# Training

## Stage 1

In Stage 1, we warm up the local encoder & decoder and keep the main model frozen.

```bash
# need to adjust to your setup, and probably to using Beaker
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export N_GPUS=8

NAME=stage1_bolmo_qwen3_8b
SEQUENCE_LENGTH=4096 \
DTYPE=float32 \
DATA_SOURCE=data_sources_qwen3.txt \
OLMO_ARCH=qwen3_8B \
OLMO_CKPT_PATH=/data/benjamin/checkpoints/qwen3_8b_base_olmo_core/model_and_optim \
TRAIN_MODE=stage_1 \
LOCAL_MODEL_STYLE="hnet:xlstm" \
ADD_HASH_EMBEDDINGS=false \
ADD_EXPANDED_EMBEDDINGS=true \
EMBEDDING_INIT_PATH="" \
SAVE_FOLDER=/data/benjamin/trained/$NAME \
torchrun --nproc-per-node=$N_GPUS src/examples/bolmo/train_stage1.py $NAME \
    train_module.bolmo_config.losses=[local_encoder,ce,local_decoder,boundary] \
    train_module.bolmo_config.loss_weights=[1,1,1,4] \
    train_module.bolmo_config.div_fn=kl \
    train_module.bolmo_config.binarization_temp=5.0 \
    train_module.bolmo_config.use_oracle_patch_reps=true \
    train_module.bolmo_config.teacher_force_boundaries=true \
    train_module.bolmo_config.encoder_loss_lookahead=4 \
    train_module.bolmo_config.encoder_loss_no_lookahead_weight=0.0 \
    train_module.bolmo_config.encoder_loss_lookahead_weights=[0.0,0.0,0.0,4.0] \
    train_module.bolmo_config.do_alm_debiasing=true \
    train_module.bolmo_config.merge_boundary_loss=false \
    train_module.optim.weight_decay=0.1 \
    train_module.max_grad_norm=0.5 \
    train_module.optim.lr=5e-4 \
    model.block.attention.use_flash=true \
    model.local_encoder.n_layers=1 \
    model.local_decoder.n_layers=4 \
    model.local_decoder.hnet_smooth=false \
    model.local_decoder.hnet_modulate=false \
    model.local_encoder.boundary_predictor_lookahead=1 \
    model.local_decoder.add_in_projection=true \
    model.local_decoder.add_norm_onto_residual=false \
    model.local_decoder.add_projected_patch_residuals=false \
    model.local_encoder.block_config.feed_forward.hidden_size=5504 \
    model.local_decoder.block_config.feed_forward.hidden_size=5504 \
    model.local_encoder.d_model=4096 \
    model.local_decoder.d_model=4096 \
    data_loader.global_batch_size=786432 \
    train_module.rank_microbatch_size=49152 \
    trainer.callbacks.checkpointer.ephemeral_save_interval=1000 \
    trainer.callbacks.checkpointer.save_interval=75000 \
    trainer.callbacks.downstream_evaluator.eval_interval=75000 \
    trainer.max_duration.value=75000
```

You will only need to adjust `OLMO_CKPT_PATH`, `SAVE_FOLDER` and potentially `rank_microbatch_size` so that `rank_microbatch_size * N_GPUS` <= `global_batch_size`.

On an H100, the `throughput/device/TPS` should be around or over 30k.

## Stage 2

In Stage 2, we train the entire model end-to-end.

```bash
# need to adjust to your setup, and probably to using Beaker
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export N_GPUS=8

NAME=stage2_bolmo_qwen3_8B
NUM_WORKERS=24 \
OLMO_ARCH=qwen3_8B \
SEQUENCE_LENGTH=4096 \
DATA_SOURCE=data_sources_qwen3.txt \
LOCAL_MODEL_STYLE="hnet:xlstm" \
ADD_HASH_EMBEDDINGS=false \
ADD_EXPANDED_EMBEDDINGS=true \
LR_SCHEDULE=linear_with_warmup \
STAGE1_CKPT_PATH=/data/benjamin/trained/stage1_bolmo_qwen3_8B/step75000/model_and_optim \
GLOBAL_MODEL_LEARNING_RATE=1.83e-5 \
SAVE_FOLDER=/data/benjamin/trained/$NAME \
torchrun --nproc-per-node=$N_GPUS src/examples/bolmo/train_stage2.py $NAME \
    train_module.optim.lr=3.66e-5 \
    data_loader.seed=1234 \
    data_loader.global_batch_size=1572864 \
    train_module.rank_microbatch_size=49152 \
    train_module.bolmo_config.losses=[ce,boundary] \
    train_module.bolmo_config.loss_weights=[1,4] \
    train_module.bolmo_config.teacher_force_boundaries=false \
    train_module.bolmo_config.do_alm_debiasing=false \
    train_module.bolmo_config.merge_boundary_loss=false \
    train_module.optim.weight_decay=0.1 \
    train_module.optim.betas=[0.9,0.95] \
    train_module.max_grad_norm=0.5 \
    model.block.attention.use_flash=true \
    model.local_encoder.n_layers=1 \
    model.local_decoder.n_layers=4 \
    model.local_decoder.hnet_smooth=false \
    model.local_decoder.hnet_modulate=false \
    model.local_encoder.boundary_predictor_lookahead=1 \
    model.local_decoder.add_in_projection=true \
    model.local_decoder.add_norm_onto_residual=false \
    model.local_decoder.add_projected_patch_residuals=false \
    model.local_encoder.block_config.feed_forward.hidden_size=5504 \
    model.local_decoder.block_config.feed_forward.hidden_size=5504 \
    model.local_encoder.d_model=4096 \
    model.local_decoder.d_model=4096 \
    trainer.callbacks.checkpointer.ephemeral_save_interval=1000 \
    trainer.callbacks.checkpointer.save_interval=30000 \
    trainer.callbacks.downstream_evaluator.eval_interval=150000 \
    trainer.max_duration.value=150000
```

You will only need to adjust `STAGE1_CKPT_PATH`, `SAVE_FOLDER` and potentially `rank_microbatch_size` so that `rank_microbatch_size * N_GPUS` <= `global_batch_size`.

On an H100, the `throughput/device/TPS` should be around or over 25k.