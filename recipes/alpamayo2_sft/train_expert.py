#!/usr/bin/env python
"""SFT the Alpamayo2-Super action expert on curated T4 windows.

`Alpamayo2Super.forward` is not the training entry point it looks like. It
computes the VLM's token cross-entropy and never calls the expert, while an
expert-enabled checkpoint sets `vlm.requires_grad_(False)` unless
`cotrain_expert_vlm` is on. Calling it as shipped therefore optimises nothing.
The released design is "train the 2B expert, freeze the 32B VLM", and this wires
that up: run the VLM to build the KV cache, hand the cache to the expert, and
take the expert's flow-matching loss.

No chain-of-thought. At inference the VLM writes a CoT and the expert reads the
cache that contains it; T4 carries no ground-truth CoT, so rather than train
against a prefix that inference will not reproduce, both sides are pinned to the
same CoT-free prompt.

Batch size is one per device by design, not by omission: `prepare_model_inputs`
refuses a batch, and a sample is already 24 images through a 32B backbone.
Throughput comes from gradient accumulation and from data parallelism.

Sharding is ZeRO-3, not `device_map="auto"`. The naive model-parallel map does not
work here at all: the expert appends to the VLM's KV cache, so a cache spread
across four cards makes `torch.cat` fail on a device mismatch. ZeRO-3 shards
storage while gathering each layer on the local device to compute, which keeps
the expert and the cache it reuses co-located -- and, unlike a layer map, it
makes the eighth GPU do useful work instead of merely holding weights.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from alpamayo2_super import helper
from alpamayo2_super.chat_template.conversation import build_conversation
from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super
from alpamayo2_super.models.utils import fuse_traj_tokens
from alpamayo2_super.t4.dataset import T4SFTDataset

TRAJ_KEYS = ("ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot")


#: Built once and reused. `helper.get_processor` goes through
#: `AutoProcessor.from_pretrained`, so calling it per sample re-reads the
#: preprocessor config from disk on every step. A plain module global rather than
#: `lru_cache` because the config object is not hashable.
_PROCESSOR: Any = None


def _processor(config: Any, tokenizer: Any) -> Any:
    """The checkpoint processor, built on first use."""
    global _PROCESSOR
    if _PROCESSOR is None:
        _PROCESSOR = helper.get_processor(tokenizer, config)
    return _PROCESSOR


def build_inputs(data: dict[str, Any], model: Alpamayo2Super) -> dict[str, Any]:
    """Tokenize one sample with a CoT-free prompt, ending where the expert takes over.

    `helper.create_messages` asks for chain-of-thought and then the trajectory.
    This asks for the trajectory alone, so the prefix the expert conditions on at
    training time is the prefix it gets at inference time.
    """
    config = model.config
    messages = build_conversation(
        data=data,
        num_tokens_per_history_traj=config.tokens_per_history_traj,
        num_tokens_per_future_traj=config.tokens_per_future_traj,
        components_order=["image", "traj_history", "prompt"],
        components_prompt=["traj_future"],
        generation_mode=True,
        include_camera_ids=config.include_camera_ids,
        camera_ids=data["camera_indices"],
        include_frame_nums=config.frame_label == "frame_num",
    )
    if messages[-1]["role"] == "assistant" and not messages[-1]["content"]:
        messages = messages[:-1]

    processor = _processor(config, model.tokenizer)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        add_vision_id=False, continue_final_message=False,
    )
    images = data["image_frames"].flatten(0, 1)
    images = (images.float() / 255.0) if images.dtype == torch.uint8 else images.float()
    tokenized = dict(
        processor(text=text, images=images, videos=None, padding=False,
                  return_tensors="pt", do_rescale=False)
    )
    return {"tokenized_data": tokenized, **{k: data[k] for k in TRAJ_KEYS}}


def training_step(model: Alpamayo2Super, batch: dict[str, Any]) -> torch.Tensor:
    """One expert flow-matching step over a single sample."""
    # Under a sharded device_map the embeddings and the expert sit on different
    # cards, and the trajectory is consumed by both: the history is scattered into
    # input_ids next to the embeddings, the future is the expert's target. Each
    # copy goes where its consumer is.
    embed_device = model.vlm.get_input_embeddings().weight.device
    expert_device = next(model.expert.parameters()).device
    tokenized = {k: (v.to(embed_device) if torch.is_tensor(v) else v)
                 for k, v in batch["tokenized_data"].items()}
    traj = {k: batch[k].to(expert_device) for k in TRAJ_KEYS}
    history = {k: batch[k].to(embed_device)
               for k in ("ego_history_xyz", "ego_history_rot")}

    # The history trajectory lives in the prompt as placeholder tokens; the future
    # does not, and must not -- the expert would then be reading its own target
    # out of the KV cache.
    tokenized["input_ids"] = fuse_traj_tokens(
        model.history_traj_tokenizer, model.future_traj_tokenizer,
        tokenized["input_ids"], history, model.config.traj_ids,
    )
    # The VLM is frozen, so nothing needs its activations for the backward pass.
    # Running it under no_grad is not an approximation -- the expert conditions on
    # the KV cache as a value -- and it is what makes the step fit at all: with
    # graph retention the 32B forward over 24 images alone exhausts an 80 GB card.
    with torch.no_grad():
        vlm_outputs = model.vlm(**tokenized, use_cache=True)
    return model.expert(traj, vlm_outputs,
                        cache_attention_mask=tokenized.get("attention_mask")).loss


def save_expert(model: Alpamayo2Super, engine: Any, path: Path, is_main: bool) -> None:
    """Write the expert weights, gathering ZeRO-3 shards first.

    Under stage 3 each rank holds a slice of every parameter, so `save_pretrained`
    called directly would write a file full of empty tensors.
    """
    if engine is None:
        if is_main:
            path.parent.mkdir(parents=True, exist_ok=True)
            model.expert.save_pretrained(path)
        return

    import deepspeed  # noqa: PLC0415

    params = list(model.expert.parameters())
    with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
        if is_main:
            path.parent.mkdir(parents=True, exist_ok=True)
            model.expert.save_pretrained(path, state_dict=model.expert.state_dict())
    if dist.is_initialized():
        dist.barrier()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--index", required=True)
    p.add_argument("--window-list", required=True)
    p.add_argument("--model-id", default=os.environ.get("ALPAMAYO2_SUPER_MODEL_ID",
                                                        "nvidia/Alpamayo2-Super"))
    p.add_argument("--decode", default="gpu", choices=("gpu", "cache"))
    p.add_argument("--device", default="cpu",
                   help='decode device for --decode gpu. "cpu" keeps all 80 GB for '
                        "the model: a step is tens of seconds and CPU decode is 2.3 s "
                        "a sample, so a few workers hide it completely, whereas GPU "
                        "decode competes with the weights for VRAM and OOMs NVDEC.")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--output-dir", required=True, help="working dir for checkpoints")
    p.add_argument("--final-dir", default=None,
                   help="NFS dir the finished weights are copied to")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--deepspeed", default=None, help="ZeRO config json")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    is_main = rank == 0

    def log(message: str) -> None:
        if is_main:
            print(message, flush=True)

    dataset = T4SFTDataset(
        args.index, cache_dir=args.cache_dir, window_list=args.window_list,
        decode=args.decode, device=args.device,
    )
    log(f"{dataset}")

    # 35.8 B in bf16 is 71.6 GB of weights; the expert's gradients and AdamW state
    # add ~24 GB more, so the model does not fit on one 80 GB card even with the
    # VLM frozen. device_map="auto" spreads the layers over the visible GPUs.
    # Loaded on CPU and left there: moving it to the GPU before deepspeed.initialize
    # materialises all 71.6 GB on every rank, which defeats stage 3 entirely (the
    # first run showed 71.6 GB/rank instead of ~9) and leaves NVDEC no room to
    # create a decoder context. ZeRO partitions and places it.
    model = Alpamayo2Super.from_pretrained(args.model_id, dtype=torch.bfloat16)
    if not args.deepspeed:
        model.to(device)
    if not model.config.enable_expert:
        raise SystemExit("this checkpoint has no expert; there is nothing to train")
    # from_pretrained already froze the VLM when cotrain is off; make it explicit
    # and verify, because a silently all-frozen model trains to no effect.
    model.vlm.requires_grad_(False)
    model.expert.requires_grad_(True)
    trainable = [q for q in model.parameters() if q.requires_grad]
    n_trainable = sum(q.numel() for q in trainable)
    if n_trainable == 0:
        raise SystemExit("no trainable parameters")
    log(f"trainable {n_trainable/1e9:.2f} B of "
        f"{sum(q.numel() for q in model.parameters())/1e9:.1f} B over {world_size} rank(s)")

    model.expert.train()
    # DeepSpeed refuses torch's AdamW when the optimizer is offloaded -- its CPU
    # kernel is what makes host-side updates affordable rather than a stall.
    offloads = False
    if args.deepspeed:
        cfg = json.loads(Path(args.deepspeed).read_text())
        offloads = "offload_optimizer" in cfg.get("zero_optimization", {})
    if offloads:
        from deepspeed.ops.adam import DeepSpeedCPUAdam  # noqa: PLC0415

        optimizer = DeepSpeedCPUAdam(trainable, lr=args.lr, weight_decay=0.01,
                                     betas=(0.9, 0.95))
    else:
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01,
                                      betas=(0.9, 0.95))

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, args.max_steps - args.warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)

    engine = None
    if args.deepspeed:
        import deepspeed  # noqa: PLC0415 - only needed on the sharded path

        config = json.loads(Path(args.deepspeed).read_text())
        config["gradient_accumulation_steps"] = args.grad_accum
        # The engine wraps the whole model so ZeRO sees every parameter, but only
        # the expert's are trainable, so only those get gradients and optimizer
        # state -- the frozen 32B is sharded storage and nothing more.
        engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model, model_parameters=trainable, optimizer=optimizer,
            lr_scheduler=scheduler, config=config,
        )

    # decode="gpu" uses CUDA, which a forked worker cannot; workers must spawn.
    loader_kwargs: dict[str, Any] = {"batch_size": None, "num_workers": args.num_workers}
    if world_size > 1:
        loader_kwargs["sampler"] = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
        )
    else:
        loader_kwargs["shuffle"] = True
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
        if args.decode == "gpu" and args.device != "cpu":
            # A forked worker cannot use CUDA; spawn is the price of GPU decode.
            loader_kwargs["multiprocessing_context"] = "spawn"
    loader = DataLoader(dataset, **loader_kwargs)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = micro = 0
    running = 0.0
    t_start = t_window = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    epoch = 0
    while step < args.max_steps:
        if world_size > 1 and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        epoch += 1
        for sample in loader:
            # T4SFTDataset already applied the task profile, so the sample is the
            # six-camera view; applying it again fails the seven-camera ring check.
            try:
                batch = build_inputs(sample, model)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = training_step(model, batch)
            except torch.OutOfMemoryError:
                # Peak residency is ~78 of 80 GB, so an unusually long sequence can
                # tip it. Drop that sample rather than the run; ranks stay in step
                # because the accumulation counter still advances below.
                log("OOM on one sample; skipping it")
                torch.cuda.empty_cache()
                loss = torch.zeros((), device=device, requires_grad=True)

            if engine is not None:
                engine.backward(loss)
                engine.step()          # a no-op until the accumulation boundary
            else:
                (loss / args.grad_accum).backward()
            running += loss.item()
            micro += 1

            if micro % args.grad_accum:
                continue
            if engine is None:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0:
                elapsed = time.perf_counter() - t_window
                lr_now = (optimizer.param_groups[0]["lr"] if engine is not None
                          else scheduler.get_last_lr()[0])
                log(f"step {step:6d}/{args.max_steps}  "
                    f"loss {running/(args.log_every*args.grad_accum):.4f}  "
                    f"lr {lr_now:.2e}  {elapsed/args.log_every:.2f} s/step  "
                    f"mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB/rank  "
                    f"samples/s {world_size*args.grad_accum*args.log_every/elapsed:.2f}")
                running = 0.0
                t_window = time.perf_counter()

            if step % args.save_every == 0 or step == args.max_steps:
                ckpt = out_dir / f"expert-step{step}"
                save_expert(model, engine, ckpt, is_main)
                log(f"saved {ckpt}")
            if step >= args.max_steps:
                break

    log(f"done in {(time.perf_counter()-t_start)/3600:.2f} h")

    # The working dir is node-local scratch; the finished weights are not.
    if args.final_dir:
        final = Path(args.final_dir)
        save_expert(model, engine, final / "expert-final", is_main)
        log(f"final weights -> {final / 'expert-final'}")
    if dist.is_initialized():
        dist.barrier()
    return 0


if __name__ == "__main__":
    sys.exit(main())
