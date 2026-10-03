# IntronForge

I like using Forge and Auto1111, but there are too many options and
dials, too many extensions to get lost in. And no matter what, even when
an image generated the way I liked, the finish still left something to
be desired, and I know it is in all of the settings. I needed a UX that
worked for me, and I had a ton of ZCode credits. Thus, IntronForge. It
is still a work in progress, but so far it is basic and gets to the
point of what I need.

Profile-per-character front end for Stable Diffusion Forge. Each
character gets a profile carrying render ratios; every render writes a
sidecar ticket beside the image, so settings are never lore. Built
against Forge Classic Neo with ADetailer, Pony XL checkpoints.

## Run

1. Launch Forge with the REST API on (required, the Gradio API cannot
   carry ADetailer args):
   `bash /path/to/forge/webui.sh --api`
2. `bash start-deck.sh`
3. Open `http://127.0.0.1:7877`.

## Point it at your Forge

Three spots carry install paths. Search for yours and replace:

- `profile_deck/lora_store.py`: `LORA_DIR`, `FORGE_CONFIG`
- `profile_deck/forge_client.py`: Forge base URL
- `start-deck.sh`: venv python path, host, port

The CivitAI key is read at runtime from Forge's own `config.json`
(`custom_api_key`). It is never stored in this repo.

## Layout

- `profile_deck/`: the app. `universal.json` holds the shared ratios
  (profiles carry null and inherit). `profiles/` ships as an empty
  skeleton, add your own character JSONs and they just work. CivitAI
  enrichment caches to a local `lora_cache.json` at runtime (never committed).
- `tools/`: `harness.py` drives `/sdapi/v1/txt2img` one variant per run
  with the same seed; `step_ladder.py` runs denoise ladders with
  face-crop metrics.
- V2 features in the deck: PNG ingest to profile drafts, prompt
  translator across model lanes, danbooru tag assist, LoRA visual
  browser with CivitAI enrichment, TensorArt-style shell layout.
- `studio/`: product notes and next stories.

## Ticket discipline

Historical tickets stay verbatim. New facts arrive as dated notes, data
stays next to the code, and every number carries a receipt. That is the
whole method.

MIT licensed. Built hard, shared freely.
