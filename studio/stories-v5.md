# V5 Story Plots: Agent-Collaborative IntronForge (2026-10-04, in progress)

North star (user-declared): IntronForge as a COLLABORATIVE tool between
user and agent. Any harness integrates; or it runs as an external skill.
The agent assists in zeroing down the settings that best fit the user.
Generation can flow through multiple backends — local Forge, CivitAI
(credits), Tensor.art TAMS (credits) — when the user has balance.

Existing foundation: the REST API already self-documents
(/openapi.json, 40 paths; /docs; /redoc verified 200). The whole deck
was itself agent-driven this session — V5 productizes that.

## Story: agent-api (stable harness surface)

Goal: any harness (ZCode, Claude Code, Cursor, scripts) can discover
and drive IntronForge from the API alone.

- Done free by FastAPI: /openapi.json, /docs, /redoc.
- Do: title rebrand (Profile Deck -> IntronForge), consistent error
  contract ({detail} everywhere — audited in review), and a
  short AGENTS.md-style API intro (auth model: localhost trust; no keys
  unless the user enables them).
- Acceptance sketch: 01. /openapi.json lists every route. 02. An agent
  with only the schema performs list-profiles -> generate -> read
  gallery without human help.

## Story: intronforge-skill (external skill package)

Goal: a portable skill any agent harness can install: SKILL.md teaching
the workflows + a thin `ifapi` CLI wrapper. Workflows: profile list/
open, generate + batch, prompt lint/translate/sections, library curation
(rate/tag/dupes), LoRA import, experiments. Publish via the public repo
(blank-slate discipline, the maintainer's publish flow).

- Acceptance sketch: 01. `ifapi profiles` / `generate` / `library` run
  from a fresh shell with no setup beyond curl+python. 02. SKILL.md
  alone is enough for a cold agent to render an image. 03. No secrets in
  the skill; the CivitAI key stays server-side.

## Story: experiment-harness (agent-assisted settings zeroing)

Goal: productize the session's investigation method (skin-tone ladder,
steps ladder) as a first-class feature. The agent runs controlled
experiments; the user gets a measured recommendation with receipts.

- POST /api/experiment: axes (denoise/steps/cfg/lora-weight/sampler/
  upscaler/seed), values, base profile; runs sequentially, same seed;
  auto face-crop; metrics (CIELAB drift vs baseline, texture energy)
  ported from tools/harness.py + step_ladder.py into the deck.
- Results table persisted per experiment (experiments/); sidecar
  tickets link renders to the experiment.
- The AGENT reads results and proposes settings; the USER accepts —
  collaborative by construction.
- Acceptance sketch: 01. a 4-value denoise ladder runs unattended and
  returns a metrics table. 02. each run is a normal ticketed render.
  03. re-running an experiment id is idempotent (cache by signature).

## Story: provider-tams + provider-civitai (multi-backend generation)

Goal: generation through Tensor.art TAMS and CivitAI backends when the
user has credits. forge_client refactors into a provider interface:
generate(profile, overrides) -> image + ticket; tickets record provider
and cost.

- TAMS shape (from tams-docs.tensor.art): REST, SIGNED requests (not
  bearer — signature generation page at build time), create-job/poll
  model, models + resource endpoints, workflow templates, credit
  billing. CivitAI: member text-to-image API, buzz/credits.
- Cost guardrails are the core of this story, mirroring the studio's
  own rule (nothing gets spent on the agent's initiative): providers
  DISABLED by default; keys added by the user; every credit run needs
  explicit user confirmation (per-run, not blanket); a dry-run estimate
  endpoint; agent may PROPOSE a cloud run, never start one.
- Prompt dialect: profiles are Pony lane; provider translation
  reuses the translator's rule-table pattern per target backend.
- Acceptance sketch: 01. provider registry with forge always available.
  02. TAMS job create/poll round-trip against a test account. 03. a
  credit run without user confirmation is refused by design. 04. ticket
  shows provider + estimated cost.

## Story: library-physical-folders (disk-real organization, DRAFTED not built)

Goal: real directories on disk for curated groups — the physical model deferred
when virtual folders shipped (2026-10-04, library-curation-qol). Primary
driver: LoRA dataset prep (gather N renders of one character into one training
folder) plus real backup and export structure.

Background: virtual folders live in the curation index; renders never move.
That is the safe curation layer. This story adds the disk layer beneath it,
schema-preserving by design.

Design (to confirm at build time):

- Physical organization stays two-level: `outputs/<character>/<profile>/`.
  "Creating a folder" = creating a new character or profile node. No free-form
  nesting: scan/key/thumb/route layers all assume exactly two levels and the
  full review hardened them; nested folders would reopen that surface.
- Move = per-render migration exactly like profile rename: png + `.json`
  ticket + `.intronforge.json` sidecar move together, library index keys
  migrate, cached thumbs invalidate, curation (rating/tags/virtual folders/
  wd14/phash) follows the key. Batch moves loop the checked selection into
  the target node with a receipt.
- Export-for-dataset = copy, not move: selection to an arbitrary disk path
  (default under a `datasets/` root), optional caption `.txt` per image
  generated from the sidecar ticket prompt (LoRA trainers consume this
  directly; ties into the pnginfo-consolidate dataset lane).
- Virtual folders keep working unchanged on top; a moved render keeps its
  virtual memberships.

Acceptance sketch:

01. Create a node, move N checked renders: pngs + both sidecars + index keys
    + thumbs land under the new node, source dir is empty of them, the library
    shows them under the new node, and an automated check finds no orphaned
    keys or thumbs anywhere.
02. Export a selection to a dataset folder: files copied, originals untouched,
    caption `.txt` files written from sidecar prompts when requested.
03. Collisions refuse with 409; moving renders of a locked profile is allowed
    (renders are not the locked artifact) and noted in the move receipt.
04. Re-running an interrupted move skips already-migrated files (idempotent).

Risks to respect at build time: filename collisions on move (same-second
stamps plus shared seeds), watcher interplay during bulk moves, the outputs
listing, and prompt-stats path scans.

## Story: intron-depot (Obsidian-like knowledge base, v1 SHIPPED 2026-10-04)

Goal: IntronDepot, an Obsidian-like knowledge base inside IntronForge, reachable
from the main page, linking profiles, characters, AND non-character subjects
(logos, backgrounds, styles) so every generation endeavor has a home for its
prompts, guides, and settings. Named by the user 2026-10-04.

Why: prompt knowledge lives in three places today: saved CivitAI guide pages
(HTML folders in the project root), the snippet library, and the vault.
Reviewing a guide means leaving the tool. The depot brings guides, prompts, and
recipes into the deck, categorized and searchable, and gives agents a
first-class way to read the same knowledge (the agent-friendly north star).

Design principles (to confirm at build time):

- Markdown files on disk under `profile_deck/depot/`, one note per file:
  frontmatter (type, tags) + wiki-link syntax `[[...]]`. Plain markdown means
  the folder opens directly as an Obsidian vault (github.com/obsidianmd was
  the user's reference): links and tags resolve there too, no lock-in, and the
  existing vault-sync flow can pick the folder up.
- Subjects are broader than characters: a note can link to a profile
  (`[[<character>/<profile>]]`), a character (`[[<character>]]`), or any free-form
  subject (`[[logos]]`, `[[backgrounds]]`, `[[style:<your-style-lora>]]`). Backlinks are
  derived from note links, never written into profile JSON; profiles stay
  clean and machine-owned.
- Note types: guide (ingested or hand-written), prompt (insertable into the
  prompt bar through the existing chip/addWord path), and recipe (frontmatter
  carrying a hires block and/or ADetailer tabs block in profile schema).
- Recipes apply like experiment cells: explicit user action, confirm dialog,
  locked profiles refuse, apply writes a receipt. Never automatic.
- Categorization is user tags + note type; no forced taxonomy.

Acceptance sketch:

01. Depot panel from the main page: search, tag filter, note reader; wiki-links
    to profiles, characters, and subjects are clickable (opens the profile in
    the editor or filters the relevant view).
02. One saved CivitAI guide folder ingests into a categorized, readable note
    without leaving the deck.
03. A prompt note inserts into the prompt bar on click; a recipe note applies
    its hires/detailer block to the open profile behind an explicit confirm;
    locked profiles refuse with 409.
04. Opening `profile_deck/depot/` as an Obsidian vault resolves the same links
    and tags: a compatibility proof, not a rebuild of Obsidian.
05. Agent surface: `/api/depot` list/read/search endpoints in the OpenAPI
    schema; the ifapi skill gains a depot workflow (agent reads the guide,
    then generates).
06. Snippet bridge decided and documented: snippets either migrate into depot
    notes or stay separate with a stated reason.

Risks: scope creep toward rebuilding Obsidian (resist: frontmatter, wiki-links,
and tags only), HTML guide conversion quality, taxonomy drift, and keeping the
depot folder compatible with the vault sync.

## Story: prompt-studio (dynamic lane-aware prompt editors, DRAFTED not built)

Goal: turn the positive and negative prompt windows into dynamic editors that
are to AI art prompting what a word processor is to office work. The editor
knows the model language in use (SDXL, Pony, Illustrious first; Anima, Flux,
and Klien later) and actively helps: autocomplete, suggestions, structure
awareness, live linting, and eventually AI-assisted formatting. Declared by
the user 2026-10-04 under the widened north star (all-in-one AI art
production studio).

What exists today (the organs, unassembled):

- danbooru-assist: local 183k-tag autocomplete with alias chains under both
  prompt fields.
- tag-mode-editor: token chips with weights, BREAK, line structure.
- redundancy-linter: server-side scoring with a receipt.
- prompt-translator: lane normalization with a receipt diff.
- section-composer: the authoritative 6-section format.
- snippet-library and trigger-word chips.

Design (to confirm at build time):

- One editor surface for both fields, replacing the plain textareas; the
  current textareas remain the data model (prompt is the single source of
  truth; the editor is a view over it, same rule as tag mode).
- Lane model: SDXL base, Pony (score/source vocab, BREAK semantics),
  Illustrious (masterwork vocab); lane declared per profile or detected from
  prompt content, surfaced in the editor chrome. Anima, Flux, and Klien
  (spelling to confirm) join later as rule packs, not code changes.
- Token-level rendering: quality tokens, source tags, weights, BREAK, and
  lora tokens each get distinct treatment; unknown tokens get a hint.
- Autocomplete moves inside the editor with lane-aware ranking (lane vocab
  first, then the danbooru dump).
- Live inline lint (duplicate, conflict, subsumption) as underlines with
  one-click fix through the existing server-side linter.
- AI assistance (RESOLVED 2026-10-04 by user declaration): the provider IS the
  agent harness. No LLM provider, key, or budget exists inside IntronForge;
  the connection is Hermes Agent, the reference harness (its gateway v0.21.5
  runs locally, api_server healthy on http://127.0.0.1:8642),
  The deck side of the connection is an assist mailbox that stays
  harness-agnostic per the north star:

  01. An "ask agent" action in the editor chrome POSTs /api/assist
      {kind: format-for-lane, prompt, negative_prompt, lane, note}; the
      request is persisted and simply waits. The deck never calls out and
      cannot spend.
  02. Any harness polls GET /api/assist?status=open (the ifapi skill gains
      an assist workflow), does the work under its OWN provider and money
      rules, and POSTs /api/assist/{id}/propose {prompt, negative_prompt,
      rationale, proposed_by}. The provider gate lives in the harness: the
      harness side governs the spend, never the deck.
  03. The editor shows the proposal as a diff receipt (translator pattern);
      Apply writes the textareas locally, Save profile persists as always;
      accept/decline closes the request with a paper trail. The user-accept
      gate stays in the deck, the spend gate stays in the harness: agent
      proposes, user accepts, by construction.
  04. Optional later convenience, explicitly deferred: a thin push to Hermes
      (drop a paste, or POST the gateway api_server) so the harness side learns
      of a request without polling. Coupling to Hermes internals is exactly
      why it is not part of the first cut.

Acceptance sketch:

01. Editing in the new surface round-trips token-identically with the banked
    the banked example prompts (the tag-mode standard).
02. Lane is detected or declared and shown; Pony vs Illustrious vs plain SDXL
    vocab rank correctly in autocomplete for both fields.
03. Duplicates and conflicts appear as inline underlines while typing; fix
    applies through the existing lint apply endpoint with a receipt.
04. A cold user can compose a lane-correct prompt start to finish without
    leaving the editor or memorizing the vocab (suggestions carry the load).
05. AI assist, when enabled and invoked explicitly, returns a diff receipt
    for the target lane; apply is confirm-gated; with assist disabled the
    editor loses nothing.
06. Negative field gets full parity: lane-aware vocab, lint, autocomplete.

Phasing: v1 (editor shell, lanes, autocomplete, live lint) SHIPPED 2026-10-04.
v2 (assist mailbox) SHIPPED same day: POST/GET /api/assist, propose and
resolve endpoints, an "ask agent" button with a 3s poll and a diff receipt
(apply writes the textareas, Save persists), reload adoption of the newest
unfinished request, and an ifapi skill workflow for the harness side. The
Hermes-side polling workflow is the harness side to wire, not IntronForge code.

Risks: editor complexity creeping into the single-source-of-truth prompt
model (keep the view/viewmodel discipline), browser performance on very
long prompts, and the AI-assist provider decision (key, cost, local).

## Story: lora-training-lane (dataset-to-LoRA loop with LoRAlab-TrainerStudio, DRAFTED not built)

Goal: close the studio loop from curated renders to trained LoRAs. The
trainer is AcademiaSD LoRAlab-TrainerStudio (MIT), user-identified 2026-10-04
as aligned with IntronForge training goals: a local Flask web UI
(127.0.0.1:4990 launcher, one trainer at a time on port 5000), its own
trainers on PyTorch/Diffusers/PEFT with 4-bit NF4 loading, and — decisive
for us — first-class SDXL trainers with Pony and Illustrious presets, an
Anima trainer (4GB), and FLUX.2 Klein. (Klein confirms the "Klien" from the
prompt-studio lane list: real model, right name.) Its VLM captioning with a
danbooru-tag mode overlaps our wd14 lane; ours curates before training, its
dataset manager captions inside training.

Division of labor (IntronForge curates, LoRAlab trains):

- FEEDER: the library-physical-folders export becomes the dataset hand-off —
  checked renders into a folder with PNGs plus caption `.txt` from sidecar
  prompts (or wd14 tags), with rating/tag/duplicate filters applied before
  export so the dataset is curated by definition.
- HAND-OFF v1: the deck exports to a folder LoRAlab's dataset manager points
  at. No API coupling: it has no REST API, its UI is the interface.
- RETURN: LoRAlab exports trained LoRAs to Forge/A1111 folders; the deck's
  lora scan, CivitAI enrichment, and trigger-word chips pick them up
  unchanged.
- OPTIONAL v2: a Depot recipe/guide carrying the LoRAlab preset that matches
  each character's banked settings, plus a launcher health-check link in the
  deck sidebar.

Acceptance sketch:

01. From the library, export a rating-filtered selection of one character as
    a dataset folder with captions; train it in LoRAlab; the trained LoRA
    appears in the deck's lora list with trigger words.
02. Export is repeatable and documented: same filters produce the same
    dataset.
03. No new spend paths: training is local; nothing cloud; the GPU is shared
    with Forge by user management.

Out of scope: driving LoRAlab programmatically (no API exists), GPU
scheduling between Forge and the trainer, and cloud templates.

## Story: trainer-integration (LoRAlab features inside IntronForge, v1 SHIPPED 2026-10-04)

Goal: bring LoRA training INTO IntronForge per user direction 2026-10-04
("integrate those features in IntronForge and properly credit them on
github"), replacing the hand-off model. LoRAlab TrainerStudio is MIT and its
pipeline is file-driven and headless-capable: per-model scripts
(`0_caption_<m>.py`, `1_pre_cache_<m>.py`, `2_train_lora_<m>.py`) each read a
`train_settings_<m>.json` and run under their own venv (Python 3.13, PyTorch
CUDA 13); the Flask launcher only orchestrates them.

Architecture (bridge orchestration, not dependency merge):

- LoRAlab stays a separate component at a pinned path
  (`~/AcademiaSD_LoRAlab-TrainerStudio`, commit-pinned) with its own venv.
  The deck never imports its code into the Forge venv (torch/diffusers
  versions conflict by design); it DRIVES the scripts as subprocesses.
- Backend: `POST /api/train/start` {dataset (a feeder export), model (sdxl
  first), settings} writes a project dir + settings JSON, then runs the
  0/1/2 chain via their venv python in a background thread;
  `GET /api/train/status` streams step/loss/log tail; outputs land in the
  Forge Lora dir where the existing scan + enrichment + trigger-word chips
  register them automatically.
- Training presets live in Depot recipe notes carrying train_settings JSON,
  so per-character presets are banked, searchable knowledge.
- UI: a Training panel on the experiment-harness pattern (start, poll,
  receipts, live log tail, preview thumbnails when present).
- v1 lane: SDXL with the Pony/Illustrious presets. Anima and FLUX.2 Klein
  join when their lanes open.

Credit (the GitHub obligation, non-negotiable):

- A THIRD_PARTY_NOTICES.md plus a README Credits section in BOTH the local
  repo and the public blank-slate publish: AcademiaSD LoRAlab-TrainerStudio,
  MIT License, repository URL, and precisely what IntronForge uses of it
  (the caption/pre-cache/train pipeline and presets). Their LICENSE file is
  preserved. The public publish flow (blank-slate) must carry it.
- If their scripts are ever modified in place, changes are documented in the
  notice and offered upstream.

Acceptance sketch:

01. From the deck: pick the exported dataset and a Pony preset, start;
    polling shows steps and loss; the finished LoRA appears in the deck's
    lora list with a receipt linking dataset, settings, and credit.
02. The LoRAlab launcher remains fully usable alongside; no lockout.
03. The public repo carries the credit section visibly on GitHub.
04. No new spend paths: training is local; one GPU means warning when Forge
    is mid-render before starting a train.

Risks: upstream script drift (pin the commit, update deliberately), settings
JSON schema differences per model, progress-parsing brittleness against
their console stream format, VRAM contention with a live Forge.

## Sequencing rationale

agent-api is already half-done (free wins landing now); the skill makes
every harness integration one copy-paste; experiment-harness is the
collaborative differentiator (agent + receipts); providers come last —
they touch money and need the guardrails reviewed before anything runs.
