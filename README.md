# Anima Studio

**A local all-in-one workspace for preparing and training LoRAs for Anima and SDXL.**

Anima Studio combines dataset collection, image editing, captioning, tagging, training configuration, and monitoring in one web application.

> ⚠️ **WORK IN PROGRESS**
>
> Anima Studio is experimental and under active development. Anima training is currently supported through `LoRA_Easy_Training_Scripts_Backend`, but complete production training with real Anima weights has not yet been fully validated.

---

## ✨ Features

### 📥 Dataset Collection

Collect training images from multiple sources:

* Danbooru
* Gelbooru
* Safebooru
* e621
* Rule34
* Konachan
* yande.re
* Web pages / image URLs
* Local files

Includes search, tag autocomplete, previews, pagination, bulk downloads, and local imports.

### 🖼️ Image Editing

Prepare images before training:

* Crop and resize
* **Upscale cropped regions**
* Remove unwanted areas
* Save edited copies
* Zoom images
* Detect duplicates
* Delete unwanted images

Duplicate detection uses perceptual dHash.

### 🏷️ Captioning & Tagging

* Edit individual or multiple captions
* WD14 automatic tagging
* Tag frequency analysis
* Remove or rename tags globally
* Search captions and filenames

### 🧠 LoRA Training

Anima Studio supports separate training paths for **Anima** and **SDXL**.

#### Anima LoRA Training

Anima training currently uses:

**`LoRA_Easy_Training_Scripts_Backend`**

The interface provides:

* Training configuration
* `.toml` import/export
* Dataset validation
* Backend connection
* Training start/stop controls
* Training monitoring

A **Diffusion Pipe-based Anima training backend is planned** as a future alternative.

#### SDXL LoRA Training

SDXL uses a **native `diffusers` + `peft` training backend** included with Anima Studio.

Tested components include:

* LoRA injection
* Forward/backward passes
* Multiple training steps
* Loss calculation
* LoRA saving

---

## 🧠 Training Backends

Anima Studio is designed to support multiple training backends.

```text
                    Anima Studio
                         │
                   Training UI
                         │
              ┌──────────┴──────────┐
              │                     │
            Anima                  SDXL
              │                     │
              ▼                     ▼
 LoRA_Easy_Training_Scripts    Native diffusers
        Backend                   + PEFT
              │
              │
        Future option
              ▼
       Diffusion Pipe
       Anima Backend
```

### Current

```text
Anima → LoRA_Easy_Training_Scripts_Backend
SDXL  → Native diffusers + PEFT
```

### Planned

```text
Anima → Diffusion Pipe
```

The Diffusion Pipe implementation is intended to provide an additional Anima training option rather than immediately replacing the existing backend.

---

# 📊 Training Monitor

Monitor:

* Training status
* Progress
* Samples
* Checkpoints
* Loss
* Learning rate

TensorBoard can optionally be used for training graphs.

---

# 🚀 Installation

## Windows

Run:

```bat
run.bat
```

Then open:

```text
http://127.0.0.1:7864
```

### Manual Installation

```bash
pip install -r requirements.txt
python run.py
```

---

# 🧠 Anima Training Backend

The current Anima backend is included in:

```text
training_backend/
```

Install it with:

```bat
setup_backend.bat
```

Or use **Set up backend** from the Train tab.

Start it with:

```text
training_backend/run.bat
```

The default backend address is:

```text
http://127.0.0.1:8000
```

Then use **Check connection** in the Train tab.

---

# 🏋️ Workflow

```text
Collect → Edit → Caption → Train → Monitor
```

1. **Collect** — Find and download images.
2. **Edit** — Crop, upscale, clean, and resize.
3. **Caption** — Edit captions or use WD14.
4. **Train** — Configure Anima or SDXL LoRA training.
5. **Monitor** — Track training, samples, loss, and checkpoints.

---

# 📁 Project Structure

```text
anima-studio/
├── app/
├── native_backend/
├── training_backend/
├── run.py
├── run.bat
├── run.sh
├── setup_backend.bat
├── setup_backend.sh
├── requirements.txt
└── THIRD_PARTY_NOTICES.md
```

---

# ⚙️ Optional Components

### WD14 Tagging

Requires:

```text
onnxruntime
numpy
```

The tagging model is downloaded when needed.

### TensorBoard

```bash
pip install tensorboard
```

---

# ⚠️ Known Limitations

* Anima training has not been fully validated with real Anima weights.
* Anima progress reporting is currently limited.
* The Diffusion Pipe Anima backend is planned but not yet implemented.
* Flux training is not currently supported.
* Native training currently focuses on LoRA rather than full model fine-tuning.
* GPU, CUDA, and PyTorch compatibility varies by system.
* VRAM requirements depend on the model, resolution, batch size, optimizer, and other settings.

When reporting an issue, include:

```text
GPU:
Operating System:
Python version:
PyTorch version:
CUDA version:
Training model:
Training backend:
Training settings:
Error message:
```

---

# 🗺️ Roadmap

* [ ] Improve Anima training stability
* [ ] Validate complete Anima training runs
* [ ] Develop Diffusion Pipe Anima backend
* [ ] Improve SDXL training features
* [ ] Improve training progress reporting
* [ ] Improve VRAM optimization
* [ ] Add more training presets
* [ ] Improve dataset validation
* [ ] Expand captioning tools
* [ ] Improve error handling
* [ ] Add more model architectures

---

# 📜 Third-Party Software

Anima Studio includes third-party training code, including components from:

* `LoRA_Easy_Training_Scripts_Backend`
* `sd-scripts`

See `THIRD_PARTY_NOTICES.md` and the included license files for details.

---

# 🔒 Privacy

Anima Studio is designed to run locally. Your datasets and generated files are not automatically uploaded.

Network access is used when you explicitly request external resources such as image downloads, booru searches, or model/tagger downloads.

---

# 🤝 Contributing

Bug reports, testing, suggestions, and contributions are welcome.

Please use GitHub Issues to report problems or request features.

---

# ⭐ Project Status

**Anima Studio is an experimental open-source project under active development.**

The goal is to simplify the entire LoRA workflow—from dataset collection and preparation to training and monitoring—in one application.

If you find it useful, consider ⭐ starring the repository and reporting issues or suggestions.

## License

See the included license files and `THIRD_PARTY_NOTICES.md` for licensing information.
