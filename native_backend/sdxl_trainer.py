"""
SDXL LoRA training for Anima Studio's native backend - built on
Hugging Face's own `diffusers` + `peft` libraries rather than deriving
the diffusion math independently, since those are the well-established,
widely-used, already-correct primitives for this - the goal here is
sound new orchestration code, not reinventing SDXL's forward pass.

SDXL specifics this has to get right (the parts that are easy to get
subtly wrong, and why they're handled explicitly below):
  - TWO text encoders (CLIP ViT-L and OpenCLIP ViT-bigG), whose outputs
    are concatenated for cross-attention, plus a pooled embedding from
    the second encoder used for additional conditioning.
  - SDXL's UNet needs extra "micro-conditioning" (add_time_ids: original
    size, crop top-left, target size) alongside the usual timestep - a
    real, easy-to-forget requirement that's specific to SDXL (SD1.5 and
    most other UNets don't need this).
  - The SDXL VAE is prone to NaN/overflow in fp16, so it's kept in fp32
    for encode/decode even when the rest of training runs in fp16/bf16.
"""
from __future__ import annotations

from pathlib import Path

from . import dataset as ds


class TrainerError(RuntimeError):
    pass


def _get(args: dict, *path, default=None):
    cur = args
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_models(model_path: str, device: str, weight_dtype):
    """Loads the five SDXL components from a single checkpoint file, or
    a diffusers-format folder/repo if model_path isn't a single file.
    Returns a dict rather than a tuple so callers/tests can address
    pieces by name instead of positional order."""
    import torch
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    if not model_path:
        raise TrainerError("No base model path set (anima_args/general_args pretrained_model_name_or_path).")

    p = Path(model_path)
    if p.is_file():
        from diffusers import StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_single_file(str(p), torch_dtype=weight_dtype)
        unet, vae = pipe.unet, pipe.vae
        text_encoder_one, text_encoder_two = pipe.text_encoder, pipe.text_encoder_2
        tokenizer_one, tokenizer_two = pipe.tokenizer, pipe.tokenizer_2
        noise_scheduler = pipe.scheduler
    elif p.is_dir():
        unet = UNet2DConditionModel.from_pretrained(p, subfolder="unet", torch_dtype=weight_dtype)
        vae = AutoencoderKL.from_pretrained(p, subfolder="vae", torch_dtype=torch.float32)
        text_encoder_one = CLIPTextModel.from_pretrained(p, subfolder="text_encoder", torch_dtype=weight_dtype)
        text_encoder_two = CLIPTextModelWithProjection.from_pretrained(p, subfolder="text_encoder_2", torch_dtype=weight_dtype)
        tokenizer_one = CLIPTokenizer.from_pretrained(p, subfolder="tokenizer")
        tokenizer_two = CLIPTokenizer.from_pretrained(p, subfolder="tokenizer_2")
        noise_scheduler = DDPMScheduler.from_pretrained(p, subfolder="scheduler")
    else:
        raise TrainerError(f"Model path doesn't exist: {model_path}")

    vae = vae.to(device, dtype=torch.float32)  # kept fp32 - see module docstring
    unet = unet.to(device, dtype=weight_dtype)
    text_encoder_one = text_encoder_one.to(device, dtype=weight_dtype)
    text_encoder_two = text_encoder_two.to(device, dtype=weight_dtype)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    return {
        "unet": unet, "vae": vae,
        "text_encoder_one": text_encoder_one, "text_encoder_two": text_encoder_two,
        "tokenizer_one": tokenizer_one, "tokenizer_two": tokenizer_two,
        "noise_scheduler": noise_scheduler,
    }


def apply_lora(unet, network_dim: int, network_alpha: float):
    """LoRA only on the UNet's attention projection layers - the
    standard, minimal target set for SDXL LoRA (matches what every
    mainstream SDXL LoRA trainer targets by default; text-encoder LoRA
    is deliberately left out here as a scope decision, not an oversight)."""
    from peft import LoraConfig

    target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
    config = LoraConfig(r=network_dim, lora_alpha=network_alpha, target_modules=target_modules)
    unet.add_adapter(config)
    return unet


def encode_prompt(text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two, captions, device):
    """Returns (encoder_hidden_states, pooled_output) - the concatenated
    per-token embeddings from both encoders for cross-attention, plus
    text_encoder_two's pooled output, which SDXL needs as a separate
    conditioning signal (not just folded into the token embeddings)."""
    import torch

    tokens_one = tokenizer_one(captions, padding="max_length", max_length=tokenizer_one.model_max_length,
                                truncation=True, return_tensors="pt").input_ids.to(device)
    tokens_two = tokenizer_two(captions, padding="max_length", max_length=tokenizer_two.model_max_length,
                                truncation=True, return_tensors="pt").input_ids.to(device)

    out_one = text_encoder_one(tokens_one, output_hidden_states=True)
    out_two = text_encoder_two(tokens_two, output_hidden_states=True)
    # SDXL uses the second-to-last hidden state (matches diffusers' own
    # SDXL pipeline/training example) rather than the final layer output.
    hidden_one = out_one.hidden_states[-2]
    hidden_two = out_two.hidden_states[-2]
    pooled = out_two.text_embeds if hasattr(out_two, "text_embeds") else out_two[0]

    encoder_hidden_states = torch.cat([hidden_one, hidden_two], dim=-1)
    return encoder_hidden_states, pooled


def make_add_time_ids(original_size, crop_coords, target_size, device, dtype):
    import torch
    add_time_ids = list(original_size) + list(crop_coords) + list(target_size)
    return torch.tensor([add_time_ids], device=device, dtype=dtype)


def run(args: dict, dataset: dict, on_progress, should_stop) -> None:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.bfloat16 if _get(args, "general_args", "mixed_precision", default="fp16") == "bf16" else torch.float16
    if device == "cpu":
        weight_dtype = torch.float32  # fp16/bf16 matmul on CPU is either unsupported or pointlessly slow

    model_path = _get(args, "general_args", "pretrained_model_name_or_path") or _get(args, "anima_args", "pretrained_model_name_or_path")
    models = load_models(model_path, device, weight_dtype)
    unet = apply_lora(models["unet"], _get(args, "network_args", "network_dim", default=32),
                       _get(args, "network_args", "network_alpha", default=16.0))

    resolution = int(_get(args, "general_args", "resolution", default=1024))
    torch_dataset = ds.make_torch_dataset(dataset, resolution, tokenize_fn=None)
    batch_size = int(_get(args, "general_args", "batch_size", default=1))
    loader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)

    lr = float(_get(args, "optimizer_args", "unet_lr", default=_get(args, "optimizer_args", "learning_rate", default="1e-4")))
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    if not trainable_params:
        raise TrainerError("No trainable LoRA parameters found - apply_lora's target_modules may not match this UNet.")
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    max_steps = int(_get(args, "general_args", "max_train_steps", default=len(loader)))
    output_dir = Path(_get(args, "saving_args", "output_dir"))
    output_name = _get(args, "saving_args", "output_name", default="lora")
    save_every = int(_get(args, "saving_args", "save_every_n_steps", default=0)) or None

    step = 0
    unet.train()
    steps_per_epoch = max(1, len(loader))
    for pixel_values, captions in _cycle(loader):
        if should_stop() or step >= max_steps:
            break
        step += 1

        pixel_values = pixel_values.to(device, dtype=models["vae"].dtype)
        with torch.no_grad():
            latents = models["vae"].encode(pixel_values).latent_dist.sample()
            latents = latents * models["vae"].config.scaling_factor
        latents = latents.to(weight_dtype)

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, models["noise_scheduler"].config.num_train_timesteps, (bsz,), device=device).long()
        noisy_latents = models["noise_scheduler"].add_noise(latents, noise, timesteps)

        with torch.no_grad():
            encoder_hidden_states, pooled = encode_prompt(
                models["text_encoder_one"], models["text_encoder_two"],
                models["tokenizer_one"], models["tokenizer_two"], list(captions), device,
            )
        add_time_ids = make_add_time_ids((resolution, resolution), (0, 0), (resolution, resolution), device, weight_dtype)
        add_time_ids = add_time_ids.repeat(bsz, 1)

        model_pred = unet(
            noisy_latents, timesteps, encoder_hidden_states,
            added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time_ids},
        ).sample

        target = noise if models["noise_scheduler"].config.prediction_type == "epsilon" else \
            models["noise_scheduler"].get_velocity(latents, noise, timesteps)
        loss = F.mse_loss(model_pred.float(), target.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        on_progress(step=step, total_steps=max_steps, epoch=step // steps_per_epoch + 1, loss=float(loss.detach().cpu()))

        if save_every and step % save_every == 0:
            _save_lora(unet, output_dir, f"{output_name}-step{step:06d}")

    _save_lora(unet, output_dir, output_name)


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def _save_lora(unet, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    unet.save_lora_adapter(str(output_dir), weight_name=f"{name}.safetensors")
