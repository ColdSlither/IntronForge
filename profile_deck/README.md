# Profile Deck

Character profiles with every generation setting baked in, one click to
Forge Neo. The skin tone fix from 2026-10-03 is built in: detailer prompts
are automatically stripped of `<lora:...>` tokens unless a tab overrides.

## Run

```
/path/to/forge/venv/bin/python -m uvicorn app:app \
    --host 127.0.0.1 --port 7877
```

Then open http://127.0.0.1:7877 (Forge must be running on 7860).

## Where things live

- `profiles/<character>/<name>.json` — the profiles. Edit in the UI or by hand.
- `outputs/<character>/<name>/` — every render plus a sidecar JSON ticket
  holding every effective setting (the banked-ticket discipline, automated).

## Rules baked into the client

- Universal ratios: steps, cfg, distilled cfg, hires scale/steps/denoise/cfg
  and detailer conf/denoise/pad live in `universal.json`. Profiles store
  null there and inherit, so one change covers every character. **Lock
  ratios** pins the current values across characters and restarts;
  **Unlock ratios** lets profiles populate their own values again.
- The profile JSON is canonical: selecting a character always populates it.
  Form edits ride along as ephemeral run state and never touch disk; only
  the Save profile button writes. A random render's actual seed is in its
  sidecar ticket, so you can bank winners deliberately.
- Locked profiles (Lock profile) refuse overwrites; saving creates a
  timestamped variant with `variant_of` lineage.
- LoRAs: profiles carry a `loras` list rendered as `<lora:name:weight>`
  tokens on the base prompt only (detailer passes auto-strip them). The
  selector searches installed models; **fetch info** enriches a LoRA from
  CivitAI by hash using the API key from Forge's own config.json (reads
  only, cached in `lora_cache.json`, previews in `lora_previews/`).
- Detailer prompts = base prompt minus lora tokens (auto), per-tab override wins.
- Detailer tabs run in listed order: put body before face.
- A VAE override, if ever set, must be a file PATH in this Forge build.
- Batch lines are appended to the base prompt; seed modes: increment, fixed, random.

## Brand (IntronForge)

Exact palette: Void #050705 · Hull #0A0E0A · Plate #131A13 · Line #243426 ·
Phosphor #3AF26A · Hazard #FFB347. Mint #d9f5e0 carries body text (from the
banner lettering). Hazard stripe across the top edge, phosphor block cursor
in the wordmark. Site: intronforge.com.

## Credits

LoRA training in IntronForge runs on [AcademiaSD LoRAlab-TrainerStudio](https://github.com/AcademiaSD/AcademiaSD_LoRAlab-TrainerStudio)
(MIT License), driven as a separate pinned component through its pipeline
scripts. See `THIRD_PARTY_NOTICES.md` at the project root for the full
attribution and scope.
