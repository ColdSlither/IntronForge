# PRD: Profile Deck (story deck-mvp)

One paragraph: a local web app that tunnels into Forge Neo's REST API and
gives character profiles with every generation setting baked in, so renders
are consistent across sessions and batch-able for the Anima LoRA datasets.
Tensor.Art convenience, Forge-level control, profiles as plain JSON.

## Module specs

- `forge_client.py` — payload builder + API caller. Owns the ADetailerArgs
  defaults, the auto-strip rule (detailer prompt = base prompt minus
  `<lora:...>` tokens unless the tab overrides), hires config, and the
  sd_vae path quirk of this Forge build.
- `app.py` — FastAPI, bound to 127.0.0.1:7877. Profile CRUD, single
  generate, batch generate, output serving. Every render writes the image
  plus a sidecar JSON ticket with every effective setting.
- `static/index.html` — single page, vanilla JS. Profiles grouped by
  character, editor form, generate, batch textarea, per-profile gallery.
- `profiles/<character>/<name>.json` — the profiles. Seeded: claire and
  romie from the vault tickets.

## Profile schema (v1)

base: prompt, negative_prompt, checkpoint, vae, width, height, sampler,
scheduler, steps, cfg_scale, distilled_cfg_scale, seed.
hires: null or {enabled, upscaler, scale, steps, denoise, cfg}.
detailer: null or {strip_lora_from_prompt, tabs: [{enabled, model,
confidence, denoise, padding, prompt_override}]} in pass order.

## Acceptance criteria

01. `GET /api/profiles` lists profiles grouped by character; claire and
    romie are present out of the box.
02. Generating claire/office-portrait from the UI returns one image within
    120 seconds and writes it under
    `profile_deck/outputs/claire/office-portrait/` with a sidecar JSON
    ticket containing every effective setting, including the hires and
    detailer blocks.
03. The detailer pass prompts in the sidecar ticket have no
    `<lora:...>` tokens while the base prompt keeps them.
04. Editing any field in the profile editor and saving persists to the
    profile's JSON file on disk.
05. Batch mode with three variation lines produces three images into the
    same output folder in one submission, no further input.
06. The server binds 127.0.0.1 only and rejects profile keys outside
    `[a-z0-9._-]` so no path escape exists.

## Out of scope (backlog stories)

- vault-sync: write render tickets back to the Obsidian vault.
- dataset-mode-hardening: wildcard line syntax, resume, seed reports.
- anima-lane-adapter: same profiles driving the ComfyUI engine for the
  Anima checkpoint lane.
