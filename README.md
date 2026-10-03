# IntronForge

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
  (profiles carry null and inherit). `profiles/` ships two example
  profiles. `lora_cache.json` caches CivitAI enrichment by file hash.
- `tools/`: `harness.py` drives `/sdapi/v1/txt2img` one variant per run
  with the same seed; `step_ladder.py` runs denoise ladders with
  face-crop metrics.
- `studio/`: product notes and next stories.

## Ticket discipline

Historical tickets stay verbatim. New facts arrive as dated notes, data
stays next to the code, and every number carries a receipt. That is the
whole method.

MIT licensed. Built hard, shared freely.
