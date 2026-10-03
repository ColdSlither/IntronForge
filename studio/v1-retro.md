# Version 1 Retrospective (2026-10-03)

What v1 is: a working, measured, personal generation studio on Forge Neo
plus a profile deck, and a paper trail that documents why every setting
is what it is. Consider everything below publishable material.

## What v1 proved (with receipts)

1. ADetailer skin tone drift on the Pony lane: cause found, fix
   ticketed, measured in CIELAB. The detailer prompt inherits the base
   prompt's lora token, and that darkens repainted faces.
   - Vault: AI Art Stack - ADetailer Skin Tone Ticket
   - Numbers + crops: `results/` in this workspace
2. "Overbaking" decomposed: DPM++ 2M convergence gloss is the gloss;
   Euler a never converges, so steps are a look selector, not a quality
   dial. Face detail is flat from 14 to 24 steps.
   - Data: `results/ladder/ladder.json`, `face_strip.png`
3. Forge Neo's Gradio API cannot carry ADetailer args (state-bound);
   the REST API can, and now runs with --api.
4. The Profile Deck: profiles as JSON, settings baked in, batch mode,
   automatic lora-strip, sidecar tickets on every render.
   - `profile_deck/`, PRD in `studio/prd-profile-deck.md`

## The debt v1 carries (v2 must pay)

- Paths, ports, and one GPU are hardcoded. No config file.
- One engine adapter (Forge REST). ComfyUI lane untouched.
- The UI is utilitarian. Fine for the user, not for strangers.
- No installer, no docs beyond this folder.

## First field verdict (2026-10-03, after first real session)

- Two local character lanes validated through the deck: both came
  out consistent with baked settings, seed variation only.
- User verdict: "working with the API is more effective" than UI work.
  Signals that v2 should lean further into profiles-as-data with a thin
  UI, and that the Gradio-style cockpit is the wrong destination.

## v2 candidate shapes (decision deliberately NOT made yet)

- Productize the deck for the character-LoRA crowd (engine adapters,
  config, packaging), or
- Content-first: the X account documents the build; the tool is the
  story's spine, or both, sequenced.

Sequencing note: prove the workflow on one full character dataset first,
then abstract. Building for strangers before the owner's workflow is
stable is how tools die.
