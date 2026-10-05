# V5 Roadmap: Agent-Collaborative IntronForge

**Date:** 2026-10-04  
**Source:** `studio/project.yaml`, `studio/stories-v5.md`, `tools/harness.py`, `tools/step_ladder.py`, a code-review discipline.

## 1. Where we are

V3+V4 are complete and reviewed. The public IntronForge repo is blank-slate clean. The current backlogs are:

| id | status | one-liner |
|---|---|---|
| `pnginfo-consolidate` | backlog | drop N images of one character → one profile + variation lines |
| `experiment-harness` | backlog | agent-assisted settings zeroing with measured receipts |
| `provider-tams-civitai` | backlog | multi-backend generation with credit guardrails |
| `dataset-mode-hardening` | backlog | null-safe loading, batch save guards, wildcard lines |
| `anima-lane-adapter` | backlog | drive ComfyUI Anima checkpoint lane from the same profiles |

The forward direction: **"experiment-harness" is the V5 opener** — the feature that turns "agent-friendly" into "agent genuinely useful."

## 2. V5 goal (one sentence)

IntronForge becomes a collaborative bench between the user and any agent harness: the agent proposes, runs, and measures controlled experiments; the user reviews the receipts and accepts or rejects the recommendation.

## 3. Sequencing

1. **`experiment-harness`** — build it first. It uses only the existing local Forge provider, costs nothing, and productizes the session's own investigation method.
2. **`pnginfo-consolidate`** — merge it next because it feeds the dataset/LoRA training lane and reuses the same ingest parser.
3. **`provider-tams-civitai`** — come back only after the spending guardrails are reviewed by the user. Money is a user-controlled gate, never an agent default.
4. **`dataset-mode-hardening`** — harden batch mode after LoRA dataset needs are clear.
5. **`anima-lane-adapter`** — last; it is a new engine adapter, not a settings refinement.

## 4. `experiment-harness` specification

### 4.1 Problem

Finding the right denoise / steps / CFG / LoRA weight for a new character currently means manual ladder runs in `tools/harness.py` and `tools/step_ladder.py`. The tools work, but they are CLI-only, ad-hoc, and the results live in `results/` as loose files. An agent cannot replay, compare, or recommend from them without human help.

### 4.2 Desired outcome

A first-class `/api/experiment` endpoint and UI that:
- accepts a base profile and one or more experiment axes,
- runs every combination sequentially with the same seed (unless seed is the axis),
- crops the face from each render,
- computes CIELAB drift and texture energy against the baseline,
- persists a results table under `experiments/<id>/`,
- links each render back to the normal outputs tree via sidecar tickets,
- lets the agent read the table and propose settings,
- requires the user to accept before writing anything into a saved profile.

### 4.3 API contract (draft)

```json
POST /api/experiment
{
  "character": "example-character",
  "name": "office-portrait",
  "axes": {
    "detailer.tabs[0].denoise": [0.3, 0.4, 0.5, 0.6]
  },
  "fixed_seed": 3940102728,
  "metrics": ["lab_drift", "texture_energy", "ssim_vs_baseline"],
  "baseline": "current"   // or a named profile
}
```

Response:
```jsonn
{
  "experiment_id": "exp_20261004_143022",
  "url": "/experiments/exp_20261004_143022",
  "runs": 4,
  "status": "running",
  "results": []
}
```

`GET /api/experiment/{id}` returns status + metrics table.  
`POST /api/experiment/{id}/apply` writes the winning run's settings into the profile, **only if the user calls it**.

### 4.4 Metrics to port from existing tools

From `tools/harness.py`:
- CIELAB ΔE inside the changed mask (dE_mean, dE_p95).
- L/a/b means before and after.
- Changed-pixel fraction.

From `tools/step_ladder.py`:
- Face detection with `face_yolov8n.pt`.
- Texture energy = mean |Laplacian| on the L channel inside the face box.
- Lab skin tone inside the face box.
- Pairwise delta between neighboring parameter values.

### 4.5 Modules to touch

- `forge_client.py`: expose a `run_experiment_matrix(profile, axes, fixed_seed)` helper that yields `(overrides, image, info)` per cell.
- `experiments.py` (new): `Experiment` class, matrix expansion, face crop, metric compute, JSON persistence.
- `app.py`: add `POST /api/experiment`, `GET /api/experiment/{id}`, `POST /api/experiment/{id}/apply`.
- `static/index.html`: experiment builder UI (axes picker, results table, accept/reject).
- `universal.json`: keep untouched; experiment inherits the same universal pins as normal generation.

### 4.6 Acceptance criteria

01. A four-value detailer-denoise ladder runs unattended and returns a metrics table.
02. Each run writes a normal ticketed render into `outputs/<character>/<name>/` with the experiment id in the sidecar.
03. Re-running an experiment with the same signature returns the cached results without re-rendering.
04. The agent can read `/api/experiment/{id}` and propose settings, but `/api/experiment/{id}/apply` requires an explicit user POST.
05. The feature works with only local Forge; no credit-backed providers are invoked.

### 4.7 Out of scope for `experiment-harness`

- Multi-backend runs (TAMS/CivitAI) — that is `provider-tams-civitai`.
- Automatic profile overwrite — the agent proposes, the user accepts.
- Prompt-rewrite experiments — keep this to numeric/sampler axes first.

## 5. Open questions for the user

1. Should `experiment-harness` reuse the existing `tools/harness.py` + `step_ladder.py` code directly, or be a clean rewrite inside `profile_deck/`?
2. Which axes matter most for the first version? Denoise, steps, CFG, LoRA weight, sampler, upscaler, seed?
3. Do you want the UI first, or the API + agent skill first?

## 6. Status

`experiment-harness` shipped, reviewed, and smoke-tested 2026-10-04. The review's
5 required fixes and all suggestions are applied. The live denoise-ladder receipt is
`profile_deck/experiments/<id>/`; full QA notes on the board
(`studio/project.yaml`, story `experiment-harness`).

## 7. Next action

`pnginfo-consolidate` per the sequencing in section 3.
