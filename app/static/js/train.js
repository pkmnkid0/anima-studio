// ==========================================================================
// Train tab. Renders every setting from the reference project's schema,
// keeps a single `state` object in the UI's payload shape, and posts it
// to /api/train/build for a live TOML preview and to the backend for
// actual training.
// ==========================================================================

const Train = (() => {
  let schemaData = null;
  let statusTimer = null;

  const state = {
    train_mode: "lora",
    model_type: "anima",
    model: {
      dit_model: "", qwen3_model: "", vae_model: "", t5_tokenizer_path: "",
      qwen3_max_token_length: 512, t5_max_token_length: 512,
      timestep_sampling: "sigmoid", discrete_flow_shift: 3.0, sigmoid_scale: 1.0,
      vae_chunk_size: 0, vae_disable_cache: false, blocks_to_swap: 0,
      flash_attn: false, split_attn: false, unsloth_offload_checkpointing: false,
      sdxl_checkpoint: "", sdxl_vae: "",
    },
    general: {
      resolution_width: 1024, resolution_height_enabled: false, resolution_height: 1024,
      batch_size: 1, mixed_precision: "bf16", full_fp16: false, full_bf16: false, fp8_base: false,
      xformers: false, sdpa: true,
      seed: 42, prior_loss_weight: 1.0, max_data_loader_n_workers: 1,
      max_train_mode: "epochs", max_train_value: 10,
      gradient_checkpointing: true, gradient_accumulation_enabled: false, gradient_accumulation_steps: 1,
      cache_latents: true, cache_latents_to_disk: false,
      lowram: false, highvram: false, no_half_vae: false,
      keep_tokens_separator_enabled: false, keep_tokens_separator: "",
      training_comment_enabled: false, training_comment: "",
      protected_tags_file_enabled: false, protected_tags_file: "",
      vae: "",
    },
    network: {
      algo: "lora", preset: "",
      network_dim: 32, network_alpha: 16,
      conv_dim_enabled: false, conv_dim: 16, conv_alpha: 32,
      min_timestep: 0, max_timestep: 1000,
      train_target: "both",
      network_dropout_enabled: false, network_dropout: 0.1,
      rank_dropout_enabled: false, rank_dropout: 0.1,
      module_dropout_enabled: false, module_dropout: 0.1,
      cache_te_outputs: false, cache_te_to_disk: false,
      dylora_unit: 4,
      bypass_mode: false, use_tucker: false, train_norm: false, dora: false, rescaled: false,
      constraint_enabled: false, constraint: 0,
      lora_fa: false, ip_noise_gamma_enabled: false, ip_noise_gamma: 0.1,
      extra_network_args: [],
      block_weight: { enabled: false, down: "", mid: "", up: "" },
      block_dims: { enabled: false, values: "" },
      block_alphas: { enabled: false, values: "" },
      conv_block_dims: { enabled: false, values: "" },
      conv_block_alphas: { enabled: false, values: "" },
    },
    optimizer: {
      optimizer_type: "AdamW", learning_rate: "1e-4",
      unet_lr_enabled: false, unet_lr: "1e-4",
      te_lr_enabled: false, te_lr: "1e-4",
      lr_scheduler: "cosine", num_cycles: 1, poly_power: 1.0,
      min_lr: 0, gamma: 0.9, d_param: 0.9, decay_ratio: 0.1, decay_type: "1-sqrt",
      warmup_ratio_enabled: false, warmup_ratio: 0.05, zero_lr_warmup: false,
      scale_weight_norms_enabled: false, scale_weight_norms: 1.0,
      max_grad_norm: 1.0,
      min_snr_gamma_enabled: false, min_snr_gamma: 5.0,
      zero_terminal_snr: false, masked_loss: false,
      loss_type: "l2", huber_schedule: "snr", huber_param: 0.1,
      extra_optimizer_args: [{ name: "weight_decay", value: "0.1" }],
    },
    saving: {
      output_dir: "", output_name_enabled: false, output_name: "",
      save_precision: "fp16", save_model_as: "safetensors",
      resume_enabled: false, resume: "",
      save_only_last_enabled: false, save_last_mode: "epochs", save_last_value: 4,
      save_ratio_enabled: false, save_ratio: 1,
      save_tag_enabled: false, save_tag_dir: "",
      save_freq_enabled: true, save_freq_mode: "epochs", save_freq_value: 1,
      save_toml_enabled: false, save_toml_dir: "",
      save_state: false, save_last_state_enabled: false, save_last_state_mode: "epochs", save_last_state_value: 1,
    },
    bucket: { enabled: true, no_upscale: true, multires_training: false, min_reso: 256, max_reso: 2048, reso_steps: 64 },
    noise: { noise_offset_enabled: false, noise_offset: 0.1, pyramid_enabled: false, pyramid_iterations: 6, pyramid_discount: 0.3 },
    sample: { enabled: false, sampler: "ddim", freq_mode: "epochs", freq_value: 1, prompts_file: "" },
    logging: {
      enabled: false, log_with: "tensorboard", logging_dir: "",
      log_prefix_mode: "disabled", log_prefix: "", run_name_mode: "default", run_name: "",
      tracker_name_enabled: false, tracker_name: "", wandb_api_key: "",
    },
    edm_loss: {
      enabled: false, optimizer_type: "AdamW", learning_rate: "1e-2", optimizer_args: "",
      scheduler_enabled: false, warmup_percent: 0, constant_percent: 0,
      initial_weights: "", num_channels: 128,
    },
    experimental: {
      vae_batch_size_enabled: false, vae_batch_size: 1, vae_reflection: false,
      vae_custom_scale_enabled: false, vae_custom_scale: 0, vae_custom_shift_enabled: false, vae_custom_shift: 0,
      zero_cond_dropout: false, debiased_estimation_loss: false,
    },
    accelerate: { enabled: false, num_processes: 2, main_process_port: 29500 },
    extra_args: [],
    subsets: [],
  };

  // ------------------------------ generic field helpers ------------------------------

  function field(labelText, inputEl, opts = {}) {
    const label = h("label", {}, [labelText]);
    if (opts.hint) label.appendChild(h("span", { class: "hint-dot", title: opts.hint }, "?"));
    const cls = "field" + (opts.span ? ` span-${opts.span}` : "");
    return h("div", { class: cls }, [label, inputEl]);
  }

  function textInput(value, onChange, opts = {}) {
    const el = h("input", { type: "text", placeholder: opts.placeholder || "" });
    el.value = value ?? "";
    el.addEventListener("input", () => onChange(el.value));
    return el;
  }

  function pathInput(value, onChange) {
    const input = textInput(value, onChange, { placeholder: "path" });
    input.classList.add("mono");
    const browseBtn = h("button", { class: "btn small", type: "button", onclick: () => {
      FolderBrowser.open(input.value || null, (path) => { input.value = path; onChange(path); });
    } }, "…");
    return h("div", { class: "path-input" }, [input, browseBtn]);
  }

  function numberInput(value, onChange, opts = {}) {
    const el = h("input", { type: "number", step: opts.step ?? "any", min: opts.min, max: opts.max });
    el.value = value ?? "";
    el.addEventListener("input", () => onChange(el.value === "" ? "" : Number(el.value)));
    return el;
  }

  function selectInput(options, value, onChange) {
    // options: [{value,label}] or [{group,options:[{value,label}]}]
    const el = h("select", {});
    for (const opt of options) {
      if (opt.group) {
        const og = h("optgroup", { label: opt.group });
        for (const o of opt.options) og.appendChild(h("option", { value: o.value }, o.label ?? o.value));
        el.appendChild(og);
      } else {
        el.appendChild(h("option", { value: opt.value }, opt.label ?? opt.value));
      }
    }
    el.value = value;
    el.addEventListener("change", () => onChange(el.value));
    return el;
  }

  function toggleRow(checked, onChange, labelText, subText) {
    const input = h("input", { type: "checkbox" });
    input.checked = !!checked;
    input.addEventListener("change", () => onChange(input.checked));
    const label = h("label", { class: "toggle-row" }, [
      input, h("span", { class: "toggle" }),
      h("span", { class: "label" }, [labelText, subText ? h("small", {}, subText) : null]),
    ]);
    return label;
  }

  function checkboxRow(checked, onChange, labelText) {
    const input = h("input", { type: "checkbox" });
    input.checked = !!checked;
    input.addEventListener("change", () => onChange(input.checked));
    return h("label", { class: "checkbox-row" }, [input, labelText]);
  }

  function pillGroup(options, value, onChange) {
    const wrap = h("div", { class: "pill-group" });
    const btns = options.map((opt) => h("button", {
      type: "button",
      class: opt.value === value ? "active" : "",
      onclick: (e) => {
        wrap.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
        e.currentTarget.classList.add("active");
        onChange(opt.value);
      },
    }, opt.label));
    btns.forEach((b) => wrap.appendChild(b));
    return wrap;
  }

  // "checkbox that reveals a field below it" - the pervasive `_enabled` pattern.
  function optionalField(checked, onToggle, labelText, innerEl, opts = {}) {
    const body = h("div", { style: checked ? "" : "display:none", class: "stack" }, [innerEl]);
    const toggle = toggleRow(checked, (v) => {
      onToggle(v);
      body.style.display = v ? "" : "none";
    }, labelText, opts.subText);
    const cls = "field" + (opts.span ? ` span-${opts.span}` : "");
    return h("div", { class: cls }, [toggle, body]);
  }

  function keyValueList(items, opts = {}) {
    const list = h("div", { class: "key-value-list" });
    function renderRows() {
      list.innerHTML = "";
      items.forEach((item, idx) => {
        const nameInput = textInput(item.name, (v) => { item.name = v; }, { placeholder: "name" });
        const valueInput = textInput(item.value, (v) => { item.value = v; }, { placeholder: "value" });
        const row = h("div", { class: "key-value-row" }, [nameInput, valueInput]);
        if (opts.datasetFlag) {
          row.appendChild(checkboxRow(item.dataset, (v) => { item.dataset = v; }, "dataset"));
        }
        row.appendChild(h("button", {
          class: "btn small icon-only", type: "button",
          onclick: () => { items.splice(idx, 1); renderRows(); },
        }, "✕"));
        list.appendChild(row);
      });
    }
    renderRows();
    const addBtn = h("button", {
      class: "btn small", type: "button",
      onclick: () => { items.push({ name: "", value: "", dataset: false }); renderRows(); },
    }, "+ Add");
    return h("div", { class: "stack" }, [list, addBtn]);
  }

  // ------------------------------ section (collapsible card) ------------------------------

  function section(id, title, desc, bodyChildren, opts = {}) {
    const isOpen = opts.open ?? false;
    const sec = h("div", { class: `section${isOpen ? " open" : ""}`, id: `section-${id}` });
    const head = h("div", {
      class: "section-head",
      onclick: () => sec.classList.toggle("open"),
    }, [
      h("svg", { class: "chev", width: "10", height: "10", viewBox: "0 0 10 10", html: "<path d='M2 1l5 4-5 4' stroke='currentColor' stroke-width='1.6' fill='none'/>" }),
      h("span", { class: "title" }, title),
      h("span", { class: "desc" }, desc || ""),
    ]);
    if (opts.pillId) {
      const pill = h("span", { class: "toggle-pill", id: opts.pillId }, "off");
      head.appendChild(pill);
    }
    sec.appendChild(head);
    sec.appendChild(h("div", { class: "section-body" }, bodyChildren));
    return sec;
  }

  function setPill(pillId, on) {
    const el = document.getElementById(pillId);
    if (!el) return;
    el.textContent = on ? "on" : "off";
    el.classList.toggle("on", on);
  }

  // ------------------------------ Model (Anima) ------------------------------

  function buildModelSection() {
    const m = state.model;
    const typeToggle = h("div", { class: "field span-full" }, [
      h("label", {}, "Model type"),
      pillGroup(
        [{ value: "anima", label: "Anima (DiT)" }, { value: "sdxl", label: "SDXL" }],
        state.model_type || "anima",
        (v) => { state.model_type = v; renderAllSections(); refreshPreview(); },
      ),
    ]);

    if ((state.model_type || "anima") === "sdxl") {
      const grid = h("div", { class: "field-grid cols-3" }, [
        typeToggle,
        field("Base checkpoint", pathInput(m.sdxl_checkpoint, (v) => m.sdxl_checkpoint = v), { span: "full",
          hint: "An SDXL .safetensors checkpoint - bundles both CLIP text encoders already." }),
        field("External VAE", pathInput(m.sdxl_vae, (v) => m.sdxl_vae = v), { span: "full",
          hint: "Optional - leave blank to use the VAE baked into the checkpoint." }),
      ]);
      return section("model", "Model", "SDXL checkpoint, VAE", [grid], { open: true });
    }

    const grid = h("div", { class: "field-grid cols-3" }, [typeToggle]);

    grid.append(
      field("DiT model (Anima checkpoint)", pathInput(m.dit_model, (v) => m.dit_model = v), { span: "full" }),
      field("Qwen3 text encoder", pathInput(m.qwen3_model, (v) => m.qwen3_model = v), { span: "full" }),
      field("VAE", pathInput(m.vae_model, (v) => m.vae_model = v), { span: "full" }),
      field("T5 tokenizer path", pathInput(m.t5_tokenizer_path, (v) => m.t5_tokenizer_path = v),
        { hint: "Optional - leave blank to use the tokenizer bundled with the DiT model." }),
      field("Qwen3 max token length", numberInput(m.qwen3_max_token_length, (v) => m.qwen3_max_token_length = v, { step: 1, min: 1 })),
      field("T5 max token length", numberInput(m.t5_max_token_length, (v) => m.t5_max_token_length = v, { step: 1, min: 1 })),
    );

    const sigmoidScaleField = field("Sigmoid scale", numberInput(m.sigmoid_scale, (v) => m.sigmoid_scale = v, { step: 0.1 }));
    const flowShiftField = field("Discrete flow shift", numberInput(m.discrete_flow_shift, (v) => m.discrete_flow_shift = v, { step: 0.1 }));
    function syncTimestepFields() {
      sigmoidScaleField.style.display = ["sigmoid", "shift", "flux_shift"].includes(m.timestep_sampling) ? "" : "none";
      flowShiftField.style.display = ["sigma", "shift"].includes(m.timestep_sampling) ? "" : "none";
    }
    const samplingOptions = (schemaData.timestep_sampling || ["sigmoid", "shift", "sigma", "flux_shift", "uniform"])
      .map((v) => ({ value: v, label: v }));
    const samplingSelect = selectInput(samplingOptions, m.timestep_sampling, (v) => { m.timestep_sampling = v; syncTimestepFields(); });
    grid.append(field("Timestep sampling", samplingSelect), sigmoidScaleField, flowShiftField);
    syncTimestepFields();

    grid.append(
      field("VAE chunk size", numberInput(m.vae_chunk_size, (v) => m.vae_chunk_size = v, { step: 1, min: 0 }),
        { hint: "0 disables chunking." }),
      field("Blocks to swap", numberInput(m.blocks_to_swap, (v) => m.blocks_to_swap = v, { step: 1, min: 0 }),
        { hint: "Offloads N transformer blocks to CPU to reduce VRAM use. 0 disables it." }),
    );

    const toggles = h("div", { class: "field-grid cols-3" }, [
      toggleRow(m.vae_disable_cache, (v) => m.vae_disable_cache = v, "Disable VAE cache"),
      toggleRow(m.flash_attn, (v) => m.flash_attn = v, "Flash attention"),
      toggleRow(m.split_attn, (v) => m.split_attn = v, "Split attention", "Auto-enabled if xFormers is selected below."),
      toggleRow(m.unsloth_offload_checkpointing, (v) => m.unsloth_offload_checkpointing = v, "Unsloth offload checkpointing"),
    ]);

    return section("model", "Model", "DiT checkpoint, text encoder, VAE", [grid, toggles], { open: true });
  }

  // ------------------------------ General ------------------------------

  function buildGeneralSection() {
    const g = state.general;
    const grid = h("div", { class: "field-grid cols-3" });

    const heightField = field("Height", numberInput(g.resolution_height, (v) => g.resolution_height = v, { step: 64, min: 64 }));
    heightField.style.display = g.resolution_height_enabled ? "" : "none";
    grid.append(
      field("Resolution (width)", numberInput(g.resolution_width, (v) => g.resolution_width = v, { step: 64, min: 64 })),
      h("div", { class: "field" }, [
        h("label", {}, "\u00A0"),
        toggleRow(g.resolution_height_enabled, (v) => {
          g.resolution_height_enabled = v;
          heightField.style.display = v ? "" : "none";
        }, "Non-square"),
      ]),
      heightField,
      field("Batch size", numberInput(g.batch_size, (v) => g.batch_size = v, { step: 1, min: 1 })),
      field("Seed", numberInput(g.seed, (v) => g.seed = v, { step: 1 })),
      field("Prior loss weight", numberInput(g.prior_loss_weight, (v) => g.prior_loss_weight = v, { step: 0.1 }),
        { hint: "Only matters if you use regularization images." }),
      field("Dataloader workers", numberInput(g.max_data_loader_n_workers, (v) => g.max_data_loader_n_workers = v, { step: 1, min: 0 })),
    );

    const maxTrainField = field("Amount", numberInput(g.max_train_value, (v) => g.max_train_value = v, { step: 1, min: 1 }));
    grid.append(
      field("Train for", pillGroup([{ value: "epochs", label: "Epochs" }, { value: "steps", label: "Steps" }],
        g.max_train_mode, (v) => g.max_train_mode = v)),
      maxTrainField,
    );

    const precisionRow = h("div", { class: "field-grid cols-3" }, [
      field("Mixed precision", selectInput([{ value: "fp16" }, { value: "bf16" }, { value: "float" }], g.mixed_precision, (v) => g.mixed_precision = v)),
      toggleRow(g.full_fp16, (v) => g.full_fp16 = v, "Full fp16", "Overrides mixed precision."),
      toggleRow(g.full_bf16, (v) => g.full_bf16 = v, "Full bf16", "Overrides mixed precision."),
      (state.model_type || "anima") === "sdxl"
        ? toggleRow(g.fp8_base, (v) => g.fp8_base = v, "fp8 base weights", "Not supported for Anima - SDXL only.")
        : null,
    ]);

    const attnWrap = h("div", { class: "field" }, [
      h("label", {}, "Attention"),
      pillGroup(
        [{ value: "sdpa", label: "SDPA" }, { value: "xformers", label: "xFormers" }, { value: "none", label: "None" }],
        g.xformers ? "xformers" : g.sdpa ? "sdpa" : "none",
        (v) => { g.sdpa = v === "sdpa"; g.xformers = v === "xformers"; },
      ),
    ]);

    const toggles = h("div", { class: "field-grid cols-3" }, [
      toggleRow(g.gradient_checkpointing, (v) => g.gradient_checkpointing = v, "Gradient checkpointing"),
      toggleRow(g.cache_latents, (v) => { g.cache_latents = v; cacheDiskField.style.display = v ? "" : "none"; }, "Cache latents"),
      toggleRow(g.lowram, (v) => g.lowram = v, "Low RAM mode"),
      toggleRow(g.highvram, (v) => g.highvram = v, "High VRAM mode"),
      toggleRow(g.no_half_vae, (v) => g.no_half_vae = v, "No half VAE"),
    ]);
    var cacheDiskField = optionalField(g.cache_latents_to_disk, (v) => g.cache_latents_to_disk = v, "Cache latents to disk", h("div"));
    cacheDiskField.style.display = g.cache_latents ? "" : "none";
    toggles.appendChild(cacheDiskField);

    const gradAccumInner = numberInput(g.gradient_accumulation_steps, (v) => g.gradient_accumulation_steps = v, { step: 1, min: 1 });
    const advanced = h("div", { class: "field-grid cols-3" }, [
      optionalField(g.gradient_accumulation_enabled, (v) => g.gradient_accumulation_enabled = v, "Gradient accumulation", gradAccumInner),
      optionalField(g.keep_tokens_separator_enabled, (v) => g.keep_tokens_separator_enabled = v, "Keep-tokens separator",
        textInput(g.keep_tokens_separator, (v) => g.keep_tokens_separator = v, { placeholder: "|||" })),
      optionalField(g.training_comment_enabled, (v) => g.training_comment_enabled = v, "Training comment",
        textInput(g.training_comment, (v) => g.training_comment = v)),
      optionalField(g.protected_tags_file_enabled, (v) => g.protected_tags_file_enabled = v, "Protected tags file",
        pathInput(g.protected_tags_file, (v) => g.protected_tags_file = v)),
      field("External VAE override", pathInput(g.vae, (v) => g.vae = v),
        { hint: "Rarely needed for Anima - it already has its own VAE above." }),
    ]);

    return section("general", "General", "Resolution, precision, epochs", [grid, precisionRow, attnWrap, toggles, advanced], { open: true });
  }

  // ------------------------------ Subsets ------------------------------

  function subsetCard(subset, idx, onRemove, startOpen) {
    const card = h("div", { class: `subset-card${startOpen ? " open" : ""}` });
    const head = h("div", { class: "subset-head", onclick: (e) => {
      if (e.target.tagName !== "INPUT" && e.target.tagName !== "BUTTON") card.classList.toggle("open");
    } }, [
      h("svg", { class: "chev", width: "10", height: "10", viewBox: "0 0 10 10", html: "<path d='M2 1l5 4-5 4' stroke='currentColor' stroke-width='1.6' fill='none'/>" }),
      textInput(subset.name, (v) => subset.name = v),
      h("span", { style: "flex:1" }),
      h("button", { class: "btn small danger", type: "button", onclick: (e) => { e.stopPropagation(); onRemove(); } }, "Remove"),
    ]);

    const essentials = h("div", { class: "field-grid cols-3" }, [
      field("Image folder", pathInput(subset.image_dir, (v) => subset.image_dir = v), { span: "full" }),
      field("Repeats", numberInput(subset.num_repeats, (v) => subset.num_repeats = v, { step: 1, min: 1 })),
      field("Caption extension", selectInput([{ value: ".txt" }, { value: ".caption" }], subset.caption_extension, (v) => subset.caption_extension = v)),
      field("Keep first N tokens", numberInput(subset.keep_tokens, (v) => subset.keep_tokens = v, { step: 1, min: 0 })),
    ]);

    const flags = h("div", { class: "field-grid cols-3" }, [
      checkboxRow(subset.shuffle_caption, (v) => subset.shuffle_caption = v, "Shuffle caption tags"),
      checkboxRow(subset.flip_aug, (v) => subset.flip_aug = v, "Flip augment"),
      checkboxRow(subset.color_aug, (v) => subset.color_aug = v, "Color augment"),
      checkboxRow(subset.is_reg, (v) => subset.is_reg = v, "Regularization images"),
      checkboxRow(subset.is_val, (v) => subset.is_val = v, "Validation images"),
    ]);

    const cropPadding = numberInput(subset.random_crop_padding_percent, (v) => subset.random_crop_padding_percent = v, { step: 0.01, min: 0, max: 1 });
    const advancedPaths = h("div", { class: "field-grid cols-3" }, [
      field("Target/output folder", pathInput(subset.target_image_dir, (v) => subset.target_image_dir = v),
        { hint: "For control-net style paired datasets." }),
      field("Conditioning data folder", pathInput(subset.conditioning_data_dir, (v) => subset.conditioning_data_dir = v)),
      field("Protected tags file", pathInput(subset.protected_tags_file, (v) => subset.protected_tags_file = v)),
      optionalField(subset.random_crop, (v) => subset.random_crop = v, "Random crop", field("Padding %", cropPadding)),
    ]);

    const faceCropW = numberInput(subset.face_crop_width, (v) => subset.face_crop_width = v, { step: 0.1 });
    const faceCropH = numberInput(subset.face_crop_height, (v) => subset.face_crop_height = v, { step: 0.1 });
    const captionDropoutGroup = h("div", { class: "field-grid cols-3" }, [
      field("Dropout rate", numberInput(subset.caption_dropout_rate, (v) => subset.caption_dropout_rate = v, { step: 0.05, min: 0, max: 1 })),
      field("Dropout every N epochs", numberInput(subset.caption_dropout_every_n_epochs, (v) => subset.caption_dropout_every_n_epochs = v, { step: 1, min: 0 })),
      field("Per-tag dropout rate", numberInput(subset.caption_tag_dropout_rate, (v) => subset.caption_tag_dropout_rate = v, { step: 0.05, min: 0, max: 1 })),
    ]);
    const gammaMin = numberInput(subset.gamma_aug_min, (v) => subset.gamma_aug_min = v, { step: 0.01 });
    const gammaMax = numberInput(subset.gamma_aug_max, (v) => subset.gamma_aug_max = v, { step: 0.01 });
    const gammaRate = numberInput(subset.gamma_aug_rate, (v) => subset.gamma_aug_rate = v, { step: 0.05, min: 0, max: 1 });
    const gammaGroup = h("div", { class: "field-grid cols-3" }, [
      field("Min gamma", gammaMin), field("Max gamma", gammaMax), field("Rate", gammaRate),
    ]);
    const tokenWarmupMin = numberInput(subset.token_warmup_min, (v) => subset.token_warmup_min = v, { step: 1, min: 1 });
    const tokenWarmupStep = numberInput(subset.token_warmup_step, (v) => subset.token_warmup_step = v, { step: 1, min: 1 });
    const tokenWarmupGroup = h("div", { class: "field-grid cols-3" }, [
      field("Warmup min tokens", tokenWarmupMin), field("Warmup step", tokenWarmupStep),
    ]);

    const captionAdvanced = h("div", { class: "field-grid cols-3" }, [
      optionalField(!!subset.face_crop_enabled, (v) => subset.face_crop_enabled = v, "Face crop aug range",
        h("div", { class: "field-grid cols-2" }, [field("Width", faceCropW), field("Height", faceCropH)]), { span: "full" }),
      optionalField(!!subset.caption_dropout_enabled, (v) => subset.caption_dropout_enabled = v, "Caption dropout", captionDropoutGroup, { span: "full" }),
      optionalField(!!subset.gamma_aug_enabled, (v) => subset.gamma_aug_enabled = v, "Gamma augmentation", gammaGroup, { span: "full" }),
      optionalField(!!subset.token_warmup_enabled, (v) => subset.token_warmup_enabled = v, "Token warmup", tokenWarmupGroup, { span: "full" }),
    ]);

    const body = h("div", { class: "subset-body" }, [
      essentials,
      h("div", { class: "subgroup" }, [h("div", { class: "subgroup-title" }, "Augmentation & flags"), flags]),
      h("div", { class: "subgroup" }, [h("div", { class: "subgroup-title" }, "Paths & cropping"), advancedPaths]),
      h("div", { class: "subgroup" }, [h("div", { class: "subgroup-title" }, "Caption behavior"), captionAdvanced]),
    ]);

    card.append(head, body);
    return card;
  }

  function newSubset(name) {
    return {
      name: name || `subset_${state.subsets.length + 1}`, image_dir: "", target_image_dir: "", conditioning_data_dir: "",
      num_repeats: 1, caption_extension: ".txt", random_crop_padding_percent: 0.05,
      shuffle_caption: true, flip_aug: false, keep_tokens: 0, color_aug: false, random_crop: false,
      is_reg: false, is_val: false,
      face_crop_enabled: false, face_crop_width: 1, face_crop_height: 1,
      caption_dropout_enabled: false, caption_dropout_rate: 0, caption_dropout_every_n_epochs: 0, caption_tag_dropout_rate: 0,
      gamma_aug_enabled: false, gamma_aug_min: 0.95, gamma_aug_max: 1.05, gamma_aug_rate: 0.5,
      shuffle_caption_sigma_enabled: false, shuffle_caption_sigma: 0,
      token_warmup_enabled: false, token_warmup_min: 1, token_warmup_step: 1,
      protected_tags_file: "",
    };
  }

  function buildSubsetsSection() {
    const list = h("div");
    function renderList() {
      list.innerHTML = "";
      if (state.subsets.length === 0) {
        list.appendChild(h("div", { class: "empty-state" }, [h("p", {}, "No dataset folders added yet.")]));
      }
      state.subsets.forEach((subset, idx) => {
        list.appendChild(subsetCard(subset, idx, () => { state.subsets.splice(idx, 1); renderList(); }, true));
      });
    }
    renderList();

    function addFolder(path) {
      if (state.subsets.find((s) => s.image_dir === path)) return false;
      const name = path.replace(/[\\/]+$/, "").split(/[\\/]/).pop() || "dataset";
      state.subsets.push({ ...newSubset(name), image_dir: path });
      return true;
    }

    const addBtn = h("button", { class: "btn primary small", type: "button", onclick: () => {
      state.subsets.push(newSubset());
      renderList();
    } }, "+ Add dataset folder");

    const useActiveBtn = h("button", {
      class: "btn small", type: "button",
      title: "Use the folder you were just working with in Collect/Edit/Caption",
      onclick: () => {
        const active = ActiveDataset.get();
        if (!active) return toast("No dataset folder yet - collect or caption one first.", "info");
        if (!addFolder(active)) return toast("That folder is already added.", "info");
        renderList();
        toast(`Added "${active}".`, "success");
      },
    }, "Use current dataset");

    // Every subset already known to Collect/Caption/Edit (their side-rail
    // tabs) - so a whole dataset's worth of folders can be picked from or
    // added in one click, instead of browsing to each one by hand.
    const knownPicker = h("select", { style: "width:auto" }, [h("option", { value: "" }, "Pick a known subset…")]);
    const addKnownBtn = h("button", { class: "btn small", type: "button" }, "Add");
    const addAllBtn = h("button", {
      class: "btn small", type: "button",
      title: "Add every subset known to Collect/Caption/Edit that isn't already in this list",
      onclick: async () => {
        try {
          const { subsets } = await Api.listSubsets();
          const added = subsets.filter((s) => addFolder(s.path)).length;
          renderList();
          toast(added ? `Added ${added} subset${added === 1 ? "" : "s"}.` : "Nothing new to add - they're all already here.", added ? "success" : "info");
        } catch (err) {
          toast(err.message, "error");
        }
      },
    }, "Add all subsets");

    Api.listSubsets().then(({ subsets }) => {
      subsets.forEach((s) => {
        knownPicker.appendChild(h("option", { value: s.path }, `${s.name} (${s.image_count})`));
      });
    }).catch(() => {});

    addKnownBtn.addEventListener("click", () => {
      const path = knownPicker.value;
      if (!path) return;
      if (!addFolder(path)) { toast("That folder is already added.", "info"); return; }
      renderList();
      knownPicker.value = "";
    });

    return section("subsets", "Dataset folders", `${state.subsets.length || ""}`, [
      list,
      h("div", { class: "row", style: "flex-wrap:wrap;gap:8px" }, [addBtn, useActiveBtn, knownPicker, addKnownBtn, addAllBtn]),
    ], { open: true });
  }

  // ------------------------------ Bucketing ------------------------------

  function buildBucketSection() {
    const b = state.bucket;
    const inner = h("div", { class: "field-grid cols-3" }, [
      field("Min resolution", numberInput(b.min_reso, (v) => b.min_reso = v, { step: 64, min: 64 })),
      field("Max resolution", numberInput(b.max_reso, (v) => b.max_reso = v, { step: 64, min: 64 })),
      field("Resolution step", numberInput(b.reso_steps, (v) => b.reso_steps = v, { step: 8, min: 8 })),
      checkboxRow(b.no_upscale, (v) => b.no_upscale = v, "Don't upscale small images"),
      checkboxRow(b.multires_training, (v) => b.multires_training = v, "Multi-resolution training"),
    ]);
    inner.style.display = b.enabled ? "" : "none";
    const toggle = toggleRow(b.enabled, (v) => { b.enabled = v; inner.style.display = v ? "" : "none"; }, "Enable aspect-ratio bucketing");
    return section("bucket", "Bucketing", "Aspect-ratio buckets for mixed-size datasets", [toggle, inner], { open: true });
  }

  // ------------------------------ Network (LoRA / LyCORIS) ------------------------------

  function buildNetworkSection() {
    const n = state.network;
    const algoOptions = (schemaData.network_algos || []).map((a) => ({ value: a.value, label: a.label }));
    const grid = h("div", { class: "field-grid cols-3" });

    const lycorisGroup = h("div", { class: "field-grid cols-3" });
    const dyloraField = field("DyLoRA unit", numberInput(n.dylora_unit, (v) => n.dylora_unit = v, { step: 1, min: 1 }));
    const loraFaToggle = checkboxRow(n.lora_fa, (v) => n.lora_fa = v, "LoRA-FA (freeze A matrix)");

    function algoInfo(value) {
      return (schemaData.network_algos || []).find((a) => a.value === value) || { lycoris: false };
    }

    function syncAlgoGroups() {
      const info = algoInfo(n.algo);
      lycorisGroup.style.display = info.lycoris ? "" : "none";
      dyloraField.style.display = n.algo === "dylora" ? "" : "none";
      loraFaToggle.style.display = !info.lycoris ? "" : "none";
    }

    grid.append(
      field("Algorithm", selectInput(algoOptions, n.algo, (v) => { n.algo = v; syncAlgoGroups(); })),
      field("LyCORIS preset", textInput(n.preset, (v) => n.preset = v, { placeholder: "full / attn-mlp / …" })),
      field("Network dim (rank)", numberInput(n.network_dim, (v) => n.network_dim = v, { step: 1, min: 1 })),
      field("Network alpha", numberInput(n.network_alpha, (v) => n.network_alpha = v, { step: 1, min: 0.1 })),
      field("Min timestep", numberInput(n.min_timestep, (v) => n.min_timestep = v, { step: 10, min: 0, max: 1000 })),
      field("Max timestep", numberInput(n.max_timestep, (v) => n.max_timestep = v, { step: 10, min: 0, max: 1000 })),
      field("Train target", selectInput([
        { value: "both", label: "UNet + Text Encoder" }, { value: "unet", label: "UNet only" }, { value: "te", label: "Text encoder only" },
      ], n.train_target, (v) => n.train_target = v)),
    );

    const convDimEnabled = h("div", { class: "field-grid cols-2", style: n.conv_dim_enabled ? "" : "display:none" }, [
      field("Conv dim", numberInput(n.conv_dim, (v) => n.conv_dim = v, { step: 1, min: 1 })),
      field("Conv alpha", numberInput(n.conv_alpha, (v) => n.conv_alpha = v, { step: 1, min: 0.1 })),
    ]);
    lycorisGroup.append(
      optionalField(n.conv_dim_enabled, (v) => { n.conv_dim_enabled = v; convDimEnabled.style.display = v ? "" : "none"; },
        "Separate conv dim/alpha", h("div"), { span: "full" }),
      convDimEnabled,
      optionalField(n.rank_dropout_enabled, (v) => n.rank_dropout_enabled = v, "Rank dropout",
        numberInput(n.rank_dropout, (v) => n.rank_dropout = v, { step: 0.05, min: 0, max: 1 })),
      optionalField(n.module_dropout_enabled, (v) => n.module_dropout_enabled = v, "Module dropout",
        numberInput(n.module_dropout, (v) => n.module_dropout = v, { step: 0.05, min: 0, max: 1 })),
      optionalField(n.constraint_enabled, (v) => n.constraint_enabled = v, "Constraint",
        numberInput(n.constraint, (v) => n.constraint = v, { step: 0.1 })),
      checkboxRow(n.use_tucker, (v) => n.use_tucker = v, "Use Tucker decomposition"),
      checkboxRow(n.train_norm, (v) => n.train_norm = v, "Train norm layers"),
      checkboxRow(n.dora, (v) => n.dora = v, "DoRA weight decompose"),
      checkboxRow(n.rescaled, (v) => n.rescaled = v, "Rescaled"),
      checkboxRow(n.bypass_mode, (v) => n.bypass_mode = v, "Bypass mode"),
    );

    const dropoutGrid = h("div", { class: "field-grid cols-3" }, [
      optionalField(n.network_dropout_enabled, (v) => n.network_dropout_enabled = v, "Network dropout",
        numberInput(n.network_dropout, (v) => n.network_dropout = v, { step: 0.05, min: 0, max: 1 })),
      optionalField(n.ip_noise_gamma_enabled, (v) => n.ip_noise_gamma_enabled = v, "IP noise gamma",
        numberInput(n.ip_noise_gamma, (v) => n.ip_noise_gamma = v, { step: 0.01, min: 0 })),
      dyloraField,
    ]);

    const cacheTeInner = checkboxRow(n.cache_te_to_disk, (v) => n.cache_te_to_disk = v, "Also cache to disk");
    const cacheTeWrap = h("div", { style: n.cache_te_outputs ? "" : "display:none" }, [cacheTeInner]);
    const otherToggles = h("div", { class: "field-grid cols-3" }, [
      toggleRow(n.cache_te_outputs, (v) => { n.cache_te_outputs = v; cacheTeWrap.style.display = v ? "" : "none"; }, "Cache text encoder outputs"),
      cacheTeWrap,
      loraFaToggle,
    ]);

    dyloraField.style.display = n.algo === "dylora" ? "" : "none";
    loraFaToggle.style.display = !algoInfo(n.algo).lycoris ? "" : "none";

    // Block weights / block dims (advanced, simplified to comma-separated text)
    const blockWeightInner = h("div", { class: "field-grid cols-3" }, [
      field("Down (12 values)", textInput(n.block_weight.down, (v) => n.block_weight.down = v, { placeholder: "1,1,1,1,1,1,1,1,1,1,1,1" })),
      field("Mid (1 value)", textInput(n.block_weight.mid, (v) => n.block_weight.mid = v, { placeholder: "1" })),
      field("Up (12 values)", textInput(n.block_weight.up, (v) => n.block_weight.up = v, { placeholder: "1,1,1,1,1,1,1,1,1,1,1,1" })),
    ]);
    function simpleValuesField(obj, placeholder) {
      return field("Values", textInput(obj.values, (v) => obj.values = v, { placeholder }));
    }
    const blockExtras = h("div", { class: "field-grid cols-2" }, [
      optionalField(n.block_dims.enabled, (v) => n.block_dims.enabled = v, "Per-block dims",
        simpleValuesField(n.block_dims, "comma-separated, 1 per block")),
      optionalField(n.block_alphas.enabled, (v) => n.block_alphas.enabled = v, "Per-block alphas",
        simpleValuesField(n.block_alphas, "comma-separated, 1 per block")),
      optionalField(n.conv_block_dims.enabled, (v) => n.conv_block_dims.enabled = v, "Per-block conv dims",
        simpleValuesField(n.conv_block_dims, "comma-separated, 1 per block")),
      optionalField(n.conv_block_alphas.enabled, (v) => n.conv_block_alphas.enabled = v, "Per-block conv alphas",
        simpleValuesField(n.conv_block_alphas, "comma-separated, 1 per block")),
    ]);
    const blockSection = h("div", { class: "subgroup" }, [
      h("div", { class: "subgroup-title" }, "Block weighting (advanced)"),
      optionalField(n.block_weight.enabled, (v) => n.block_weight.enabled = v, "Per-block LR weights", blockWeightInner, { span: "full" }),
      blockExtras,
    ]);

    const extraArgsSection = h("div", { class: "subgroup" }, [
      h("div", { class: "subgroup-title" }, "Extra network_args (advanced / custom)"),
      keyValueList(n.extra_network_args),
    ]);

    syncAlgoGroups();

    return section("network", "Network (LoRA)", "Rank, algorithm, dropout", [
      grid, lycorisGroup, dropoutGrid, otherToggles, blockSection, extraArgsSection,
    ], { open: true });
  }

  // ------------------------------ Optimizer / Scheduler / Loss ------------------------------

  function buildOptimizerSection() {
    const o = state.optimizer;
    const optimizerOptions = [
      { group: "Recommended", options: (schemaData.optimizers?.recommended || []).map((v) => ({ value: v })) },
      { group: "All optimizers", options: (schemaData.optimizers?.all || []).map((v) => ({ value: v })) },
    ];

    const grid = h("div", { class: "field-grid cols-3" }, [
      field("Optimizer", selectInput(optimizerOptions, o.optimizer_type, (v) => o.optimizer_type = v)),
      field("Learning rate", textInput(o.learning_rate, (v) => o.learning_rate = v, { placeholder: "1e-4" }), { hint: "Scientific notation is fine, e.g. 1e-4" }),
      field("Max grad norm", numberInput(o.max_grad_norm, (v) => o.max_grad_norm = v, { step: 0.1, min: 0 })),
      optionalField(o.unet_lr_enabled, (v) => o.unet_lr_enabled = v, "Separate UNet LR",
        textInput(o.unet_lr, (v) => o.unet_lr = v, { placeholder: "1e-4" })),
      optionalField(o.te_lr_enabled, (v) => o.te_lr_enabled = v, "Separate text encoder LR",
        textInput(o.te_lr, (v) => o.te_lr = v, { placeholder: "1e-4" })),
    ]);

    const schedulerOptions = (schemaData.lr_schedulers || []).map((s) => ({ value: s.value, label: s.label }));
    const numCyclesField = field("Num cycles", numberInput(o.num_cycles, (v) => o.num_cycles = v, { step: 1, min: 1 }));
    const polyPowerField = field("Poly power", numberInput(o.poly_power, (v) => o.poly_power = v, { step: 0.1 }));
    const minLrField = field("Min LR", numberInput(o.min_lr, (v) => o.min_lr = v, { step: 0.00001 }));
    const gammaField = field("Gamma", numberInput(o.gamma, (v) => o.gamma = v, { step: 0.01, min: 0, max: 1 }));
    const dParamField = field("D", numberInput(o.d_param, (v) => o.d_param = v, { step: 0.01 }));
    const decayRatioField = field("Decay ratio", numberInput(o.decay_ratio, (v) => o.decay_ratio = v, { step: 0.05, min: 0, max: 1 }));
    const decayTypeField = field("Decay type", selectInput([{ value: "1-sqrt" }, { value: "cosine" }, { value: "linear" }], o.decay_type, (v) => o.decay_type = v));
    const schedulerExtras = h("div", { class: "field-grid cols-3" }, [
      numCyclesField, polyPowerField, minLrField, gammaField, dParamField, decayRatioField, decayTypeField,
    ]);

    function syncScheduler() {
      const info = (schemaData.lr_schedulers || []).find((s) => s.value === o.lr_scheduler) || { fields: [] };
      const show = new Set(info.fields || []);
      numCyclesField.style.display = show.has("num_cycles") ? "" : "none";
      polyPowerField.style.display = show.has("poly_power") ? "" : "none";
      minLrField.style.display = show.has("min_lr") ? "" : "none";
      gammaField.style.display = show.has("gamma") ? "" : "none";
      dParamField.style.display = show.has("d_param") ? "" : "none";
      decayRatioField.style.display = show.has("decay_ratio") ? "" : "none";
      decayTypeField.style.display = show.has("decay_type") ? "" : "none";
    }

    const zeroWarmupToggle = checkboxRow(o.zero_lr_warmup, (v) => o.zero_lr_warmup = v, "Warm up from zero");
    const zeroWarmupWrap = h("div", { style: o.warmup_ratio_enabled ? "" : "display:none" }, [zeroWarmupToggle]);
    const warmupInner = h("div", { class: "stack" }, [
      numberInput(o.warmup_ratio, (v) => o.warmup_ratio = v, { step: 0.01, min: 0, max: 1 }),
    ]);

    const schedulerRow = h("div", { class: "field-grid cols-3" }, [
      field("LR scheduler", selectInput(schedulerOptions, o.lr_scheduler, (v) => { o.lr_scheduler = v; syncScheduler(); })),
      optionalField(o.warmup_ratio_enabled, (v) => {
        o.warmup_ratio_enabled = v; zeroWarmupWrap.style.display = v ? "" : "none";
      }, "Warmup ratio", warmupInner),
      zeroWarmupWrap,
    ]);
    syncScheduler();

    const lossOptions = (schemaData.loss_types || []).map((l) => ({ value: l.value, label: l.label }));
    const huberParamField = field("Huber c", numberInput(o.huber_param, (v) => o.huber_param = v, { step: 0.01 }));
    const huberScheduleField = field("Huber schedule", selectInput(
      (schemaData.huber_schedules || ["snr", "exponential", "constant"]).map((v) => ({ value: v })),
      o.huber_schedule, (v) => o.huber_schedule = v,
    ));
    function syncLoss() {
      const info = (schemaData.loss_types || []).find((l) => l.value === o.loss_type) || { huber: false };
      huberParamField.style.display = info.huber ? "" : "none";
      huberScheduleField.style.display = info.huber ? "" : "none";
    }

    const otherRow = h("div", { class: "field-grid cols-3" }, [
      field("Loss type", selectInput(lossOptions, o.loss_type, (v) => { o.loss_type = v; syncLoss(); })),
      huberParamField, huberScheduleField,
      optionalField(o.scale_weight_norms_enabled, (v) => o.scale_weight_norms_enabled = v, "Scale weight norms",
        numberInput(o.scale_weight_norms, (v) => o.scale_weight_norms = v, { step: 0.1 })),
      optionalField(o.min_snr_gamma_enabled, (v) => o.min_snr_gamma_enabled = v, "Min-SNR gamma",
        numberInput(o.min_snr_gamma, (v) => o.min_snr_gamma = v, { step: 0.5 })),
      checkboxRow(o.zero_terminal_snr, (v) => o.zero_terminal_snr = v, "Zero terminal SNR"),
      checkboxRow(o.masked_loss, (v) => o.masked_loss = v, "Masked loss"),
    ]);
    syncLoss();

    const extraArgsSection = h("div", { class: "subgroup" }, [
      h("div", { class: "subgroup-title" }, "Extra optimizer_args (e.g. weight_decay, betas)"),
      keyValueList(o.extra_optimizer_args),
    ]);

    return section("optimizer", "Optimizer, scheduler & loss", `${o.optimizer_type} · ${o.lr_scheduler}`, [
      grid, schedulerRow, schedulerExtras, otherRow, extraArgsSection,
    ], { open: true });
  }

  // ------------------------------ Saving ------------------------------

  function buildSavingSection() {
    const s = state.saving;
    const grid = h("div", { class: "field-grid cols-3" }, [
      field("Output folder", pathInput(s.output_dir, (v) => s.output_dir = v), { span: "full" }),
      optionalField(s.output_name_enabled, (v) => s.output_name_enabled = v, "Custom output name",
        textInput(s.output_name, (v) => s.output_name = v, { placeholder: "my_anima_lora" })),
      field("Save precision", selectInput([{ value: "fp16" }, { value: "bf16" }, { value: "float" }], s.save_precision, (v) => s.save_precision = v)),
      field("Save format", selectInput([{ value: "safetensors" }, { value: "pt" }, { value: "ckpt" }], s.save_model_as, (v) => s.save_model_as = v)),
      optionalField(s.resume_enabled, (v) => s.resume_enabled = v, "Resume from state",
        pathInput(s.resume, (v) => s.resume = v)),
    ]);

    const freqValueField = field("Every", numberInput(s.save_freq_value, (v) => s.save_freq_value = v, { step: 1, min: 1 }));
    const freqModeField = pillGroup([{ value: "epochs", label: "Epochs" }, { value: "steps", label: "Steps" }], s.save_freq_mode, (v) => s.save_freq_mode = v);
    const freqExtras = h("div", { class: "field-grid cols-3", style: s.save_freq_enabled ? "" : "display:none" }, [freqModeField, freqValueField]);
    const freqRow = h("div", { class: "field-grid cols-3" }, [
      optionalField(s.save_freq_enabled, (v) => { s.save_freq_enabled = v; freqExtras.style.display = v ? "" : "none"; },
        "Save checkpoints periodically", h("div")),
      freqExtras,
    ]);

    const lastValueField = field("Keep last N", numberInput(s.save_last_value, (v) => s.save_last_value = v, { step: 1, min: 1 }));
    const lastModeField = pillGroup([{ value: "epochs", label: "Epochs" }, { value: "steps", label: "Steps" }], s.save_last_mode, (v) => s.save_last_mode = v);
    const lastExtras = h("div", { class: "field-grid cols-3", style: s.save_only_last_enabled ? "" : "display:none" }, [lastModeField, lastValueField]);
    const lastRow = h("div", { class: "field-grid cols-3" }, [
      optionalField(s.save_only_last_enabled, (v) => { s.save_only_last_enabled = v; lastExtras.style.display = v ? "" : "none"; },
        "Limit saved checkpoints", h("div")),
      lastExtras,
      optionalField(s.save_ratio_enabled, (v) => s.save_ratio_enabled = v, "Save ratio",
        numberInput(s.save_ratio, (v) => s.save_ratio = v, { step: 1, min: 1 })),
    ]);

    const otherRow = h("div", { class: "field-grid cols-3" }, [
      optionalField(s.save_tag_enabled, (v) => s.save_tag_enabled = v, "Write tag occurrence file",
        pathInput(s.save_tag_dir, (v) => s.save_tag_dir = v)),
      optionalField(s.save_toml_enabled, (v) => s.save_toml_enabled = v, "Save a copy of the config",
        pathInput(s.save_toml_dir, (v) => s.save_toml_dir = v)),
    ]);

    const saveStateInner = h("div", { class: "field-grid cols-3" }, [
      pillGroup([{ value: "epochs", label: "Epochs" }, { value: "steps", label: "Steps" }], s.save_last_state_mode, (v) => s.save_last_state_mode = v),
      numberInput(s.save_last_state_value, (v) => s.save_last_state_value = v, { step: 1, min: 1 }),
    ]);
    const saveStateRow = h("div", { class: "field-grid cols-2" }, [
      checkboxRow(s.save_state, (v) => { s.save_state = v; saveStateWrap.style.display = v ? "" : "none"; }, "Save optimizer/training state"),
    ]);
    var saveStateWrap = optionalField(s.save_last_state_enabled, (v) => s.save_last_state_enabled = v, "Limit saved states", saveStateInner);
    saveStateWrap.style.display = s.save_state ? "" : "none";

    return section("saving", "Saving", s.output_dir ? s.output_dir : "", [
      grid, freqRow, lastRow, otherRow, saveStateRow, saveStateWrap,
    ], { open: true });
  }

  // ------------------------------ Noise / Sample / Logging ------------------------------

  function buildNoiseSection() {
    const n = state.noise;
    const pyramidInner = h("div", { class: "field-grid cols-2" }, [
      field("Iterations", numberInput(n.pyramid_iterations, (v) => n.pyramid_iterations = v, { step: 1, min: 1 })),
      field("Discount", numberInput(n.pyramid_discount, (v) => n.pyramid_discount = v, { step: 0.01, min: 0, max: 1 })),
    ]);
    const grid = h("div", { class: "field-grid cols-2" }, [
      optionalField(n.noise_offset_enabled, (v) => n.noise_offset_enabled = v, "Noise offset",
        numberInput(n.noise_offset, (v) => n.noise_offset = v, { step: 0.01, min: 0 })),
      optionalField(n.pyramid_enabled, (v) => n.pyramid_enabled = v, "Pyramid (multi-res) noise", pyramidInner),
    ]);
    return section("noise", "Noise offset", "Optional regularization noise", [grid]);
  }

  function buildSampleSection() {
    const s = state.sample;
    const samplerOptions = (schemaData.samplers || ["ddim"]).map((v) => ({ value: v }));
    const body = h("div", { class: "field-grid cols-3" }, [
      field("Sampler", selectInput(samplerOptions, s.sampler, (v) => s.sampler = v)),
      field("Sample every", pillGroup([{ value: "epochs", label: "Epochs" }, { value: "steps", label: "Steps" }], s.freq_mode, (v) => s.freq_mode = v)),
      field("Amount", numberInput(s.freq_value, (v) => s.freq_value = v, { step: 1, min: 1 })),
      field("Prompts file", pathInput(s.prompts_file, (v) => s.prompts_file = v), { span: "full", hint: "A text file, one sample prompt per line." }),
    ]);
    body.style.display = s.enabled ? "" : "none";
    const toggle = toggleRow(s.enabled, (v) => { s.enabled = v; body.style.display = v ? "" : "none"; }, "Generate sample images during training");
    return section("sample", "Sample images", "Preview generations while training", [toggle, body]);
  }

  function buildLoggingSection() {
    const l = state.logging;
    const wandbField = field("Wandb API key", (() => {
      const el = h("input", { type: "password" });
      el.value = l.wandb_api_key;
      el.addEventListener("input", () => l.wandb_api_key = el.value);
      return el;
    })());
    const prefixManualField = field("Prefix text", textInput(l.log_prefix, (v) => l.log_prefix = v));
    const runNameManualField = field("Run name", textInput(l.run_name, (v) => l.run_name = v));

    function sync() {
      wandbField.style.display = ["wandb", "all"].includes(l.log_with) ? "" : "none";
      prefixManualField.style.display = l.log_prefix_mode === "manual" ? "" : "none";
      runNameManualField.style.display = l.run_name_mode === "manual" ? "" : "none";
    }

    const body = h("div", { class: "field-grid cols-3" }, [
      field("Log with", selectInput([{ value: "tensorboard" }, { value: "wandb" }, { value: "all" }], l.log_with, (v) => { l.log_with = v; sync(); })),
      field("Log directory", pathInput(l.logging_dir, (v) => l.logging_dir = v)),
      wandbField,
      field("Log prefix", selectInput([{ value: "disabled" }, { value: "default" }, { value: "manual" }], l.log_prefix_mode, (v) => { l.log_prefix_mode = v; sync(); })),
      prefixManualField,
      field("Run name", selectInput([{ value: "default" }, { value: "manual" }], l.run_name_mode, (v) => { l.run_name_mode = v; sync(); })),
      runNameManualField,
      optionalField(l.tracker_name_enabled, (v) => l.tracker_name_enabled = v, "Tracker project name",
        textInput(l.tracker_name, (v) => l.tracker_name = v)),
    ]);
    body.style.display = l.enabled ? "" : "none";
    sync();
    const toggle = toggleRow(l.enabled, (v) => { l.enabled = v; body.style.display = v ? "" : "none"; }, "Enable experiment logging");
    return section("logging", "Logging", "TensorBoard / Weights & Biases", [toggle, body]);
  }

  // ------------------------------ EDM² loss weighting ------------------------------

  function buildEdmSection() {
    const e = state.edm_loss;
    const schedulerInner = h("div", { class: "field-grid cols-2" }, [
      field("Warmup %", numberInput(e.warmup_percent, (v) => e.warmup_percent = v, { step: 0.05, min: 0, max: 1 })),
      field("Constant %", numberInput(e.constant_percent, (v) => e.constant_percent = v, { step: 0.05, min: 0, max: 1 })),
    ]);
    const optimizerOptions = [
      { group: "Recommended", options: (schemaData.optimizers?.recommended || []).map((v) => ({ value: v })) },
      { group: "All optimizers", options: (schemaData.optimizers?.all || []).map((v) => ({ value: v })) },
    ];
    const body = h("div", { class: "field-grid cols-3" }, [
      field("Optimizer", selectInput(optimizerOptions, e.optimizer_type, (v) => e.optimizer_type = v)),
      field("Learning rate", textInput(e.learning_rate, (v) => e.learning_rate = v)),
      field("Num channels", numberInput(e.num_channels, (v) => e.num_channels = v, { step: 1, min: 1 })),
      field("Optimizer args", textInput(e.optimizer_args, (v) => e.optimizer_args = v, { placeholder: "raw args string, optional" }), { span: "full" }),
      field("Initial weights", pathInput(e.initial_weights, (v) => e.initial_weights = v), { span: "full" }),
      optionalField(e.scheduler_enabled, (v) => e.scheduler_enabled = v, "LR scheduler for the loss weights", schedulerInner, { span: "full" }),
    ]);
    body.style.display = e.enabled ? "" : "none";
    const toggle = toggleRow(e.enabled, (v) => { e.enabled = v; body.style.display = v ? "" : "none"; }, "Enable EDM² loss weighting");
    return section("edm", "EDM\u00B2 loss weighting", "Learned per-timestep loss weights", [toggle, body]);
  }

  // ------------------------------ Experimental / Advanced ------------------------------

  function buildExperimentalSection() {
    const x = state.experimental;
    const body = h("div", { class: "field-grid cols-3" }, [
      optionalField(x.vae_batch_size_enabled, (v) => x.vae_batch_size_enabled = v, "VAE batch size",
        numberInput(x.vae_batch_size, (v) => x.vae_batch_size = v, { step: 1, min: 1 })),
      optionalField(x.vae_custom_scale_enabled, (v) => x.vae_custom_scale_enabled = v, "Custom VAE scale",
        numberInput(x.vae_custom_scale, (v) => x.vae_custom_scale = v, { step: 0.01 })),
      optionalField(x.vae_custom_shift_enabled, (v) => x.vae_custom_shift_enabled = v, "Custom VAE shift",
        numberInput(x.vae_custom_shift, (v) => x.vae_custom_shift = v, { step: 0.01 })),
      checkboxRow(x.vae_reflection, (v) => x.vae_reflection = v, "VAE reflection padding"),
      checkboxRow(x.zero_cond_dropout, (v) => x.zero_cond_dropout = v, "Zero conditioning dropout"),
      checkboxRow(x.debiased_estimation_loss, (v) => x.debiased_estimation_loss = v, "Debiased estimation loss"),
    ]);
    return section("experimental", "Experimental", "Mostly relevant to SDXL rectified-flow setups", [body]);
  }

  function buildAccelerateSection() {
    const a = state.accelerate;
    const body = h("div", { class: "field-grid cols-3" }, [
      field("Number of GPUs", numberInput(a.num_processes, (v) => a.num_processes = v, { step: 1, min: 1 })),
      field("Main process port", numberInput(a.main_process_port, (v) => a.main_process_port = v, { step: 1, min: 1024, max: 65535 })),
    ]);
    body.style.display = a.enabled ? "" : "none";
    const toggle = toggleRow(a.enabled, (v) => { a.enabled = v; body.style.display = v ? "" : "none"; }, "Multi-GPU (accelerate) training");
    return section("accelerate", "Multi-GPU", "Launch with accelerate across multiple GPUs", [toggle, body]);
  }

  function buildExtraArgsSection() {
    const body = h("div", { class: "stack" }, [
      h("p", { class: "card-hint", style: "margin:0" }, "Passed straight through to the backend - useful for flags this app doesn't have a dedicated field for yet."),
      keyValueList(state.extra_args, { datasetFlag: true }),
    ]);
    return section("extra", "Extra arguments", "Advanced / passthrough", [body]);
  }

  // ------------------------------ Orchestration ------------------------------

  function renderAllSections() {
    const panels = {
      dataset: [buildSubsetsSection(), buildBucketSection()],
      model: [buildModelSection(), buildNetworkSection()],
      training: [buildGeneralSection(), buildOptimizerSection(), buildNoiseSection(), buildEdmSection()],
      output: [buildSavingSection(), buildSampleSection(), buildLoggingSection()],
      advanced: [buildAccelerateSection(), buildExperimentalSection(), buildExtraArgsSection()],
    };
    for (const [name, sections] of Object.entries(panels)) {
      const container = document.getElementById(`train-panel-${name}`);
      container.innerHTML = "";
      container.append(...sections);
    }
  }

  function initSubtabs() {
    const bar = document.getElementById("train-subtabs");
    bar.addEventListener("click", (e) => {
      const btn = e.target.closest(".train-subtab");
      if (!btn) return;
      bar.querySelectorAll(".train-subtab").forEach((b) => b.classList.toggle("active", b === btn));
      document.querySelectorAll(".train-subtab-panel").forEach((p) => {
        p.style.display = p.id === `train-panel-${btn.dataset.subtab}` ? "block" : "none";
      });
    });
  }

  function currentUiPayload() {
    // Deep-clone and post-process the a la carte fields (comma lists -> arrays)
    const ui = JSON.parse(JSON.stringify(state));
    for (const subset of ui.subsets) {
      // nothing extra to post-process currently - kept for parity/extension
    }
    const bw = ui.network.block_weight;
    if (bw && bw.enabled) {
      bw.down = splitNumbers(bw.down);
      bw.mid = bw.mid === "" ? null : Number(bw.mid);
      bw.up = splitNumbers(bw.up);
    }
    for (const key of ["block_dims", "block_alphas", "conv_block_dims", "conv_block_alphas"]) {
      const g = ui.network[key];
      if (g && g.enabled) g.values = splitNumbers(g.values);
    }
    return ui;
  }

  function splitNumbers(text) {
    return (text || "").split(",").map((s) => s.trim()).filter((s) => s !== "").map(Number);
  }

  async function refreshPreview() {
    try {
      const data = await Api.trainBuild(currentUiPayload());
      document.getElementById("toml-preview").textContent = data.toml || "(empty)";
      renderWarnings(data.warnings || []);
      return data;
    } catch (err) {
      document.getElementById("toml-preview").textContent = `Error building config:\n${err.message}`;
    }
  }

  function renderWarnings(warnings) {
    const box = document.getElementById("train-warnings");
    const list = document.getElementById("train-warnings-list");
    if (!warnings.length) { box.style.display = "none"; return; }
    box.style.display = "block";
    list.innerHTML = "";
    warnings.forEach((w) => list.appendChild(h("li", {}, w)));
  }

  async function downloadToml() {
    const ui = currentUiPayload();
    const resp = await fetch("/api/train/download_toml", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(ui),
    });
    if (!resp.ok) { toast("Couldn't build the config file.", "error"); return; }
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = (ui.saving.output_name || "anima_lora_config") + ".toml";
    a.click();
    URL.revokeObjectURL(url);
  }

  async function loadTomlFile(file) {
    const text = await file.text();
    try {
      const data = await Api.parseToml(text);
      Object.keys(state).forEach((key) => delete state[key]);
      Object.assign(state, data.ui);
      renderAllSections();
      await refreshPreview();
      toast(`Loaded ${file.name}.`, "success");
    } catch (err) {
      toast(err.message, "error");
    }
  }

  function browseTomlFromDisk() {
    FileBrowser.open(null, "toml", async (path) => {
      try {
        const data = await Api.loadTomlFromDisk(path);
        Object.keys(state).forEach((key) => delete state[key]);
        Object.assign(state, data.ui);
        renderAllSections();
        await refreshPreview();
        toast(`Loaded ${path.split(/[\\/]/).pop()}.`, "success");
      } catch (err) {
        toast(err.message, "error");
      }
    });
  }

  function saveTomlToDisk() {
    FolderBrowser.open(null, async (dir) => {
      const filename = prompt("File name:", (state.saving?.output_name || "anima_lora_config") + ".toml");
      if (!filename) return;
      try {
        const data = await Api.saveTomlToDisk(currentUiPayload(), dir, filename);
        toast(`Saved to ${data.path}`, "success");
      } catch (err) {
        toast(err.message, "error");
      }
    });
  }

  // --- Presets ---

  async function refreshPresetList() {
    const data = await Api.listPresets();
    const select = document.getElementById("preset-select");
    select.innerHTML = "";
    select.appendChild(h("option", { value: "" }, "Load a preset…"));
    data.presets.forEach((p) => select.appendChild(h("option", { value: p.name }, p.name)));
  }

  async function savePreset() {
    const name = prompt("Preset name:");
    if (!name) return;
    await Api.savePreset(name, currentUiPayload());
    toast(`Saved preset "${name}".`, "success");
    await refreshPresetList();
  }

  async function loadPresetByName(name) {
    if (!name) return;
    const ui = await Api.loadPreset(name);
    Object.assign(state, ui);
    renderAllSections();
    await refreshPreview();
    toast(`Loaded preset "${name}".`, "success");
  }

  // --- Backend control ---

  function setConnDot(dotId, textId, ok, text) {
    const dot = document.getElementById(dotId);
    dot.classList.remove("ok", "bad");
    dot.classList.add(ok ? "ok" : "bad");
    document.getElementById(textId).textContent = text;
  }

  async function checkBackend() {
    const url = document.getElementById("backend-url").value.trim();
    try {
      const result = await Api.backendPing(url);
      if (result.reachable) {
        setConnDot("train-conn-dot", "train-conn-text", true, "Connected");
        setConnDot("backend-dot", "backend-status-text", true, "Backend online");
      } else {
        setConnDot("train-conn-dot", "train-conn-text", false, "Not reachable");
        setConnDot("backend-dot", "backend-status-text", false, "Backend offline");
      }
    } catch (err) {
      setConnDot("train-conn-dot", "train-conn-text", false, "Not reachable");
    }
  }

  async function startTraining() {
    const url = document.getElementById("backend-url").value.trim();
    const ui = currentUiPayload();
    const btn = document.getElementById("start-training-btn");
    btn.disabled = true;
    try {
      await Api.backendStart(ui, url);
      toast("Training started.", "success");
      document.getElementById("start-training-btn").style.display = "none";
      document.getElementById("stop-training-btn").style.display = "inline-flex";
      pollStatus(url);
    } catch (err) {
      toast(err.message, "error");
    } finally {
      btn.disabled = false;
    }
  }

  async function stopTraining() {
    const url = document.getElementById("backend-url").value.trim();
    try {
      await Api.backendStop(url);
      toast("Stop signal sent.", "info");
    } catch (err) {
      toast(err.message, "error");
    }
  }

  function pollStatus(url) {
    if (statusTimer) clearInterval(statusTimer);
    statusTimer = setInterval(async () => {
      try {
        const status = await Api.backendStatus(url);
        setConnDot("train-conn-dot", "train-conn-text", true, status.training ? "Training…" : (status.errored ? "Errored" : "Idle"));
        if (!status.training) {
          document.getElementById("start-training-btn").style.display = "inline-flex";
          document.getElementById("stop-training-btn").style.display = "none";
          clearInterval(statusTimer);
        }
      } catch {
        clearInterval(statusTimer);
      }
    }, 4000);
  }

  // --- Local backend launcher ---

  let launcherStatusTimer = null;

  function renderLauncherStatus(status) {
    const dot = document.getElementById("backend-launch-dot");
    dot.classList.remove("ok", "bad");
    dot.classList.add(status.running ? "ok" : "bad");
    document.getElementById("backend-launch-status-text").textContent =
      status.running ? `Running (${status.command})` : (status.error || "Not running");
    document.getElementById("backend-launch-start").style.display = status.running ? "none" : "inline-flex";
    document.getElementById("backend-launch-stop").style.display = status.running ? "inline-flex" : "none";
    const log = document.getElementById("backend-launch-log");
    if (status.log_tail && status.log_tail.length) {
      log.style.display = "block";
      log.textContent = status.log_tail.join("\n");
      log.scrollTop = log.scrollHeight;
    }
  }

  async function refreshLauncherStatus() {
    try {
      renderLauncherStatus(await Api.backendProcessStatus());
    } catch { /* non-critical */ }
  }

  function pollLauncherStatus() {
    if (launcherStatusTimer) clearInterval(launcherStatusTimer);
    launcherStatusTimer = setInterval(refreshLauncherStatus, 2500);
  }

  async function launchBackend() {
    const dir = document.getElementById("backend-launch-dir").value.trim();
    const command = document.getElementById("backend-launch-command").value.trim();
    if (!dir) return toast("Set the backend's folder first.", "error");
    const btn = document.getElementById("backend-launch-start");
    btn.disabled = true;
    try {
      renderLauncherStatus(await Api.backendProcessStart(dir, command));
      toast("Backend launching…", "success");
      pollLauncherStatus();
    } catch (err) {
      toast(err.message, "error");
    } finally {
      btn.disabled = false;
    }
  }

  async function stopLauncherBackend() {
    try {
      renderLauncherStatus(await Api.backendProcessStop());
      if (launcherStatusTimer) clearInterval(launcherStatusTimer);
    } catch (err) {
      toast(err.message, "error");
    }
  }

  async function saveLauncherSettingsFromForm() {
    const dir = document.getElementById("backend-launch-dir").value.trim();
    const command = document.getElementById("backend-launch-command").value.trim();
    const autostart = document.getElementById("backend-launch-autostart").checked;
    try {
      await Api.backendProcessSaveSettings(dir, command, autostart);
    } catch { /* saved opportunistically - not worth interrupting the user over */ }
  }

  async function initLauncher() {
    try {
      const settings = await Api.backendProcessSettings();
      document.getElementById("backend-launch-dir").value = settings.dir || "";
      document.getElementById("backend-launch-command").value = settings.command || "python backend.py";
      document.getElementById("backend-launch-autostart").checked = !!settings.autostart;
    } catch { /* first run - defaults are fine */ }
    await refreshLauncherStatus();
    pollLauncherStatus();

    document.getElementById("backend-launch-browse").addEventListener("click", () => {
      FolderBrowser.open(document.getElementById("backend-launch-dir").value || null, (path) => {
        document.getElementById("backend-launch-dir").value = path;
        saveLauncherSettingsFromForm();
      });
    });
    document.getElementById("backend-launch-start").addEventListener("click", launchBackend);
    document.getElementById("backend-launch-stop").addEventListener("click", stopLauncherBackend);
    document.getElementById("backend-launch-autostart").addEventListener("change", saveLauncherSettingsFromForm);
    document.getElementById("backend-launch-command").addEventListener("change", saveLauncherSettingsFromForm);
    document.getElementById("backend-launch-dir").addEventListener("change", saveLauncherSettingsFromForm);
  }

  // --- In-app backend installer (clone + run installer, streamed) ---

  let setupStatusTimer = null;

  function renderSetupStatus(status) {
    const dot = document.getElementById("backend-setup-dot");
    dot.classList.remove("ok", "bad");
    const busy = status.stage === "installing";

    // status.installed reflects the venv actually being on disk right
    // now - not just what this process's own install run did - so a
    // previous session's completed install (or the auto-install that
    // may have started before this tab was even opened) shows correctly
    // instead of looking like "Not started".
    let label;
    if (busy) {
      label = "Running the installer…";
    } else if (status.stage === "error") {
      dot.classList.add("bad");
      label = status.error || "Something went wrong";
    } else if (status.installed) {
      dot.classList.add("ok");
      label = "Installed" + (status.stage === "done" ? " - see the log for next steps" : "");
    } else {
      label = "Not started";
    }
    document.getElementById("backend-setup-status-text").textContent = label;
    document.getElementById("backend-setup-start").disabled = busy;

    const log = document.getElementById("backend-setup-log");
    if (status.log_tail && status.log_tail.length) {
      log.style.display = "block";
      log.textContent = status.log_tail.join("\n");
      log.scrollTop = log.scrollHeight;
    }
    document.getElementById("backend-setup-input-row").style.display =
      status.stage === "installing" && status.waiting_for_input ? "flex" : "none";

    if (!busy && setupStatusTimer) {
      clearInterval(setupStatusTimer);
      setupStatusTimer = null;
    }
  }

  async function refreshSetupStatus() {
    try {
      renderSetupStatus(await Api.backendSetupStatus());
    } catch { /* non-critical */ }
  }

  function pollSetupStatus() {
    if (setupStatusTimer) clearInterval(setupStatusTimer);
    setupStatusTimer = setInterval(refreshSetupStatus, 1500);
  }

  async function startBackendSetup() {
    try {
      renderSetupStatus(await Api.backendSetupStart());
      toast("Installing backend dependencies - watch the log below.", "success");
      pollSetupStatus();
    } catch (err) {
      toast(err.message, "error");
    }
  }

  async function sendSetupInput() {
    const input = document.getElementById("backend-setup-input");
    const text = input.value;
    if (!text) return;
    try {
      renderSetupStatus(await Api.backendSetupInput(text));
      input.value = "";
    } catch (err) {
      toast(err.message, "error");
    }
  }

  async function initSetup() {
    const s = await Api.backendSetupStatus().catch(() => null);
    if (s) {
      renderSetupStatus(s);
      document.getElementById("backend-setup-target-label").textContent =
        s.bundled ? `Installing into: ${s.bundled_dir}` : "Bundled backend folder not found.";
      document.getElementById("backend-setup-start").disabled = !s.bundled;
      if (s.stage === "installing") pollSetupStatus();
    }

    document.getElementById("backend-setup-start").addEventListener("click", startBackendSetup);
    document.getElementById("backend-setup-send").addEventListener("click", sendSetupInput);
    document.getElementById("backend-setup-input").addEventListener("keydown", (e) => {
      if (e.key === "Enter") sendSetupInput();
    });
  }

  async function init() {
    schemaData = await Api.trainSchema();
    renderAllSections();
    initSubtabs();
    await refreshPreview();
    await refreshPresetList();

    document.getElementById("preview-refresh-btn").addEventListener("click", refreshPreview);
    document.getElementById("download-toml-btn").addEventListener("click", downloadToml);
    document.getElementById("load-toml-btn").addEventListener("click", () => {
      document.getElementById("load-toml-file").click();
    });
    document.getElementById("load-toml-file").addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (file) loadTomlFile(file);
      e.target.value = ""; // allow re-selecting the same file later
    });
    document.getElementById("load-toml-disk-btn").addEventListener("click", browseTomlFromDisk);
    document.getElementById("save-toml-disk-btn").addEventListener("click", saveTomlToDisk);
    document.getElementById("preset-save-btn").addEventListener("click", savePreset);
    document.getElementById("preset-select").addEventListener("change", (e) => loadPresetByName(e.target.value));
    document.getElementById("backend-check-btn").addEventListener("click", checkBackend);
    document.getElementById("start-training-btn").addEventListener("click", startTraining);
    document.getElementById("stop-training-btn").addEventListener("click", stopTraining);
    await initSetup();
    await initLauncher();
  }

  return {
    init,
    schemaData: () => schemaData,
    state,
    field, textInput, pathInput, numberInput, selectInput, toggleRow, checkboxRow,
    pillGroup, optionalField, keyValueList, section, setPill,
    buildModelSection, buildGeneralSection, buildSubsetsSection, buildBucketSection,
    buildNetworkSection, buildOptimizerSection, buildSavingSection,
    buildNoiseSection, buildSampleSection, buildLoggingSection, buildEdmSection,
    buildExperimentalSection, buildAccelerateSection, buildExtraArgsSection,
    refreshPreview,
  };
})();
