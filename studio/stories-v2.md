# V2 Story Plots (2026-10-03, not started)

Three chained features the user plotted after V1 field sessions. Ordering
is deliberate: ingest feeds the translator, the translator feeds the tag
assist. Each is one session of build at minimal rigor.

## Archive census (2026-10-03, Thoth AI Art shared)

- 3,434 PNGs across the four folders. Metadata-rich: ~1,607 (47 percent).
  - 1,096 carry A1111/Forge infotext (Forge v1.9.3 era: Style Selector
    keys, old ADetailer keys incl. "2nd" tabs, Clip skip, VAE).
  - 493 carry ComfyUI graph JSON (prompt/workflow chunks).
  - ~1,827 PNGs plus all jpg/webp/tif have no recoverable settings.
- Base families: overwhelmingly Pony (1,069 score-tag detections;
  meichidarkmix, ponyDiffusionV6XL, atomixPony3DXL, magicalpony
  checkpoints; EMS-named Civitai downloads mostly Pony-era).
- The two Workflows folders are identical copies of each other.
- The local archive already groups renders by character folder:
  multi-image consolidation gets grouping
  for free from directory structure plus prompt similarity.
- Story amendment: pnginfo-ingest needs TWO parsers, Forge infotext AND
  ComfyUI graph JSON (walk KSampler/checkpoint/lora nodes). The old
  ADetailer keys map into the deck's detailer tabs.
- Scope refinement (user, 2026-10-03): extract ONLY what the character
  prompt needs, and normalize ComfyUI graph data INTO Forge-compliant
  form (the deck speaks Forge; ComfyUI is just another source dialect).

## Story: lora-visual-browser (user: "put it on the V2 list", 2026-10-03)

Goal: replace the deck's text-search LoRA selector with the Forge
extra-networks visual: card grid with previews, folder filter tags
(Characters/, Style/, My Loras/...), search, click card to add.

- The preview wall is the real feature: Forge shows "NO PREVIEW" for most
  of the 127 loras because the files lack preview images. The deck's
  by-hash enrichment (already live) can mass-fill previews, trigger
  words, and source links from CivitAI in the background — a grid that
  is more complete than Forge's own.
- Batch enrich button with progress; previews cached in lora_previews;
  folder tags derived from the LoRA directory structure.
- Acceptance sketch: 01. grid shows previews for every enriched lora.
  02. folder tags filter the grid. 03. click card adds to profile with
  default weight; slider inline on the profile's lora slots.

## Story: tensorart-shell (user reference screenshots, 2026-10-03)

Goal: restructure the deck's UI toward the Tensor.Art layout, keeping the
current guts ("akin because I like what you've done").

- Two-panel shell: LEFT = the profile editor as stacked sectioned cards
  (model card with checkpoint thumbnail; LoRA slots with inline weight
  sliders and add/remove; setup card with resolution presets and seed;
  HD-fix card with magnification chips like Tensor.Art's 1x/1.5x/2x/3x;
  detailer card per tab). RIGHT = gallery/canvas with a prompt bar on
  top and the render log beneath.
- The prompt lives in the top bar (Tensor.Art style), not buried in a
  fieldset. A1111-compatible parameter import/export stays (the deck's
  sidecar tickets already are that).
- Acceptance sketch: 01. no horizontal scrolling; editor and gallery
  visible simultaneously. 02. every numeric field still binds to the
  same ephemeral-run + save-semantics as V1. 03. profile switching
  repopulates exactly as V1 rules require.

## Story: pretty-pass (V3, tail end)

Goal: the polish round per the studio method — visual design system
(spacing, type, hover states), skeleton loading states, toasts instead
of status text, responsive layout. Explicitly LAST, after the workflows
stabilize; prettying a moving target wastes the pass.

## Story: pnginfo-ingest

Goal: drop archived PNGs into the deck and get a draft character profile.

- Scope 1 (single image): drag-drop or file-pick a PNG, parse its
  generation metadata (Forge/A1111 infotext), show a review form, save as
  a new unlocked profile. Universal-ratio fields become null (inherit);
  prompt, negative, checkpoint, sampler, scheduler, size, hires config,
  seed map from the PNG.
- Scope 2 (multi-image consolidation): select N images of one character,
  the deck extracts the stable token core (identity) vs the varying tail
  (pose/outfit/expression), and drafts ONE profile plus a variation list
  that doubles as batch lines. This is the LoRA-dataset fast path.
- Notes: parsing the Forge infotext is already solved in the reverse
  direction (sidecar tickets); ADetailer params in old PNGs map into
  detailer tabs where present.
- Acceptance sketch: 01. drop one archived PNG, get a working profile
  that generates. 02. drop ten PNGs of one character, get one profile + variation
  lines that reproduce the set's variety.

## Story: prompt-translator

Goal: take parsed metadata (especially the character prompt) and rewrite
it to the user's forward lane: NIXES_v5.5.43 + incase_style_v3_ponyxl 0.8
(declared 2026-10-03 as THE base checkpoint and style lora moving
forward). Output-lane tables for SDXL/Illustrious are demoted to optional
future scope; the target is always the NIXES+Incase grammar.

- Mechanism: detect the SOURCE base from the prompt's shape (score_9
  prefix = Pony; masterwork/best_quality = Illustrious; raw = SDXL),
  then normalize to the NIXES+Incase ticket: score scaffold kept,
  source_cartoon lane, character-count tags normalized, old style LoRA
  tokens (Expressive_H, g0th1cPXL, Smooth_Anime, etc.) stripped because
  Incase carries the style now, character LoRAs preserved only if they
  map to the vault LoRA registry.
- Archive note: old PNGs carry incase-ilff-v3-4 (an older Incase
  variant) — normalize all incase variants to incase_style_v3_ponyxl.
- Hard part said out loud: the mechanical 80 percent is a rules table;
  the tail (artist/style tokens that do not transfer) is judgment. Ship
  with a side-by-side diff preview and apply-on-confirm, never silent.
- Acceptance sketch: 01. any archived prompt in, NIXES+Incase prompt
  out with score scaffold and no legacy style loras. 02. character-count
  tags survive. 03. nothing applies without a preview step.

## Story: danbooru-assist

Goal: autocomplete/autocorrect in the prompt fields: type "bikini armor",
get `bikini_armor`; type "green ballroom gown", get suggestions from the
real tag space.

- Data source decision (pre-made): local Danbooru tag database via the
  community tagcomplete CSV (tag, category, aliases, count) bundled next
  to the deck. No network dependency, no rate limits, aliases give the
  autocorrect behavior for free.
- Behavior: underscore conversion on space, prefix + substring matching
  ranked by tag count, alias hits resolved, category-aware (character /
  copyright / general / meta shown differently). Insert on click or Tab.
- Hard part said out loud: multi-word phrases ("green ballroom gown") do
  not exist as single tags; the assist should propose the nearest real
  tags (green_dress, ballroom, green_themed) rather than one wrong tag.
- Acceptance sketch: 01. typing a known tag's alias offers the canonical
  tag. 02. suggestions ranked by popularity. 03. works offline.

## Chaining

pnginfo-ingest produces raw prompts from the archive -> prompt-translator
normalizes them per lane -> danbooru-assist cleans the vocabulary while
editing. Together they turn the archived PNG pile into validated profiles
per character, which is the on-ramp to the Anima LoRA datasets.
