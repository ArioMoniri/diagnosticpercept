# Running on Colab Enterprise (Vertex AI)

The free Colab tier disconnects after ~12 h idle or ~24 h total. The
bridge-loop architecture in this repo wants the session alive for the full
H6 + H7 + sycophancy sweep (~3–5 h) plus any iteration. Colab Enterprise
on Google Cloud runs the notebook on a Vertex AI Workbench runtime with
**configurable idle shutdown (default 24 h, up to weeks)** and a dedicated
A100 / H100 GPU — no random disconnects.

This guide covers two run modes:

  A. **Interactive** — open the notebook in Colab Enterprise UI, run it
     yourself, keep the bridge cell running so I can drive subsequent
     tasks via `git push`. *Recommended for ongoing iteration.*
  B. **Batch** — submit the notebook via the Vertex AI Notebook Executor
     so it runs to completion in the background and writes results to a
     GCS bucket. *Recommended for full unattended replication runs.*

---

## A. Interactive mode

### A1. Create a runtime template (one-time)

Console → **Vertex AI → Colab Enterprise → Runtime templates → Create**.

Pick **ONE** of these three configs based on quota:

| field | 🥇 H100 80GB | 🥈 A100 80GB | (fallback) A100 40GB |
|---|---|---|---|
| Display name | `diagnostic-percept-h100` | `diagnostic-percept-a100-80g` | `diagnostic-percept-a100-40g` |
| Machine type | **`a3-highgpu-1g`** | **`a2-ultragpu-1g`** | `a2-highgpu-1g` |
| GPU | NVIDIA H100 80 GB ×1 | NVIDIA A100 80 GB ×1 | NVIDIA A100 40 GB ×1 |
| Wall time (full sweep) | ~1.6 hr | ~1.8 hr | ~2.4 hr |
| Approx cost/hr | ~$11.06 | ~$5.05 | ~$3.67 |
| H1 backward fits Qwen3-32B? | ✅ bf16 | ✅ NF4 | ❌ (auto-drops to 14B/8B) |
| Disk | 300 GB pd-ssd | 300 GB pd-ssd | 200 GB pd-balanced |
| Idle shutdown | 4 h | 4 h | 4 h |
| Internet egress | ON | ON | ON |
| Env vars | `GH_TOKEN=ghp_...` | `GH_TOKEN=ghp_...` | `GH_TOKEN=ghp_...` |

Note: **do not pick `a2-highgpu-1g` again** (it's the 40 GB A100 — works
but the notebook will auto-drop to Qwen3-14B because the 32B backward is
too tight). The notebook auto-detects GPU memory and:
- ≥ 70 GB → uses Qwen3-32B at bf16 (best capability)
- 24–70 GB → uses Qwen3-14B (NF4 ~7 GB, bf16 ~28 GB; safe for H1 backward)
- 12–24 GB → uses Qwen3-8B
- < 12 GB → uses Qwen3-4B

so you don't have to set `MODEL_OVERRIDE` manually.

### A1a. Verify the idle-shutdown slider before saving

Vertex AI Colab Enterprise has *two* idle settings — both must be set:

| setting | where | recommended |
|---|---|---|
| **Runtime idle shutdown** | Runtime template → "Auto-shutdown" or "Idle shutdown" slider | **4 h** (or longer) |
| **Notebook idle shutdown** | Runtime template → "Notebook idle timeout" (if shown) | **4 h** |

The defaults are often **10 minutes** — far too short for the full sweep
and for the install + kernel-restart cycle. If you see your VM
power off after a few minutes (`audit: op=disconnect` → `reboot:
Power down` in the runtime logs), bump these sliders up.

### A1b. Check quota first

Console search → **Quotas** → filter by:
- `NVIDIA_H100_GPUS` (option A)
- `NVIDIA_A100_80GB_GPUS` (option B)

If 0 in your region, click *Request adjustment* — usually approved
within minutes for these GPU families.

### A2. Create a runtime from the template

Runtime templates → click the template → **Create runtime**. Wait ~2 min
for it to come up green.

### A3. Import the notebook from GitHub

Notebooks → **My notebooks → Import → URL**:

```
https://github.com/ArioMoniri/diagnosticpercept/blob/main/notebooks/diagnostic_percept.ipynb
```

(Or paste the raw URL — both work in Colab Enterprise's importer.)

### A4. Connect notebook → runtime

In the notebook view, the top-right has a runtime dropdown. Pick the
runtime you created in A2. The kernel will connect.

### A5. Add secrets

Colab Enterprise has no `google.colab.userdata` API. Use one of:

- **Env var** in the runtime template: under *Environment variables* add
  `GH_TOKEN=ghp_xxx`. This is the recommended path because the bridge
  needs to push back; setting it in the template means every new runtime
  inherits it. For HF gated models, also add `HF_TOKEN=hf_xxx`.
- **At the top of the notebook**, add a cell:
  ```python
  import os
  os.environ['GH_TOKEN'] = 'ghp_xxxxxxxxxxxxxxxx'
  # os.environ['HF_TOKEN'] = 'hf_...'   # only if you set a gated MODEL_OVERRIDE
  ```
  Run this cell once. Don't commit it. (Or use the bridge to push in a
  separate cell that reads from a GCS bucket.)

### A6. Run all + leave the bridge cell running

`Runtime → Run all`. The last cell (`bridge — authenticate + start the
poll loop`) will keep running and pick up tasks I push from the local
side. With the 24 h idle shutdown you can leave it overnight.

### A7. To stop

Notebook menu → Disconnect → runtime can be deleted or kept warm. Idle
shutdown will reap it automatically.

---

## B. Batch mode (Vertex AI Notebook Executor)

For a one-shot full sweep where you don't need the bridge:

```bash
# One-time install
pip install google-cloud-aiplatform

# Submit
python scripts/submit_to_vertex.py \
  --project-id YOUR_PROJECT \
  --region us-central1 \
  --gcs-output gs://YOUR_BUCKET/diagnostic_percept_runs/ \
  --machine-type a2-highgpu-1g \
  --accelerator-type NVIDIA_TESLA_A100 \
  --accelerator-count 1
```

The script under `scripts/submit_to_vertex.py` walks the entire notebook
to completion on a fresh runtime, then uploads `results/run.json` and
all `results/h6/*.csv` to your GCS bucket. No bridge involvement.

---

## Differences from free Colab

| | free Colab | Colab Enterprise |
|---|---|---|
| Idle shutdown | ~12 h | configurable, default 24 h |
| Max session | ~24 h | configurable, up to weeks |
| GPU | T4 (often), L4 (sometimes), A100 (rarely) | guaranteed (A100 / H100 / L4 by quota) |
| Secrets API | `google.colab.userdata` | runtime env vars, GCS, Secret Manager |
| Cost | free tier or Colab Pro flat | per-minute runtime cost (~$3.7/h for a2-highgpu-1g) |
| Bridge fit | bridge stops on disconnect | bridge runs as long as runtime is up |

---

## Cost-control tip

`a2-highgpu-1g` is ~$3.67/hour on-demand in `us-central1`. The full H6
DEEP run (1273 q × 3 conditions) plus H7 + H8 + sycophancy is ~5 h on
A100 → ~$18. Set idle shutdown to **2 h** for short iteration sessions;
push it to 24 h only when you intend to leave it overnight.
