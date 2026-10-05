---
name: intronforge
description: Drive IntronForge, the local character-profile image studio over Forge. Use for generating/iterating character renders, batch dataset runs, prompt linting/translation/sections, library curation, LoRA imports, and settings experiments. Requires the deck running on 127.0.0.1:7877 (Forge on 7860 with --api).
---

# IntronForge — agent operations

A profile = a character's full generation recipe (prompt, negative,
checkpoint, loras with weights, hires, detailer passes). Everything is
JSON over HTTP at http://127.0.0.1:7877. Full schema: GET /openapi.json
(Swagger at /docs).

## Core loop

1. `GET /api/profiles` — map of character -> profiles.
2. `POST /api/generate` — body `{"character", "name", "overrides"?,
   "profile"?}`. Optional inline `profile` = ephemeral edits (never
   saved unless the user saves). Every render writes a PNG + sidecar
   JSON ticket (every effective setting) into
   profile_deck/outputs/<character>/<profile>/.
3. To bank a change the user likes: edit the profile JSON via
   `PUT /api/profile/<char>/<name>`, or say so and let the user save
   from the UI. Locked profiles refuse overwrites (409) — that is the
   user's banked ticket; propose a variant instead.

## Prompt tools

- `POST /api/lint` `{prompt, negative_prompt, protected}` — redundancy
  findings; apply selected ones via `/api/lint/apply`.
- `POST /api/translate` — normalize any prompt to the configured lane
  lane; returns a change receipt, never applies silently.
- `POST /api/sections/parse|compose` — the six-section BREAK format.

## Library (all renders)

`GET /api/library?character=&name=&min_rating=&favorites=&tag=&seed=`
Filter `tag` matches user tags AND wd14 auto-tags. Rate:
`POST /api/library/rate`. Dupes: `/api/library/duplicates`. Watcher
auto-registers new renders (thumbs + phash) within seconds.

## LoRAs

`GET /api/loras` (127 installed, enriched previews/trigger words).
Import: `POST /api/loras/import` `{"ref": "<civitai url or id>"}` then
poll `/api/loras/import/status` for progress. The CivitAI key lives in
Forge's config — never handle it.

## Experiments (settings zeroing)

`POST /api/experiment` `{"character", "name", "axes": {"detailer.tabs.0.denoise":
[0.3,0.4,0.5,0.6]}, "fixed_seed"?}` runs a same-seed ladder in the background
(one render per value plus a baseline) and persists a metrics table
(CIELAB dE in the changed mask, texture energy, pairwise deltas) under
profile_deck/experiments/. Poll `GET /api/experiment/{id}` until status done.
Re-running the same signature returns cached results without re-rendering.
Applying a winning cell to the profile is the user's call:
`POST /api/experiment/{id}/apply` (refuses locked profiles).

## Assist mailbox (agent connection)

The user files a request from the editor ("Ask agent"); the harness answers.
The deck never calls out and never spends; your provider rules govern your
side.

1. `GET /api/assist?status=open` — pending requests (kind format-for-lane or
   general, with prompt, negative, lane, note).
2. Do the work (format the prompt for the lane, honor the note), then
   `POST /api/assist/{id}/propose` `{"prompt", "negative_prompt",
   "rationale", "proposed_by": "<you>"}`.
3. The user applies or declines in the UI (`/resolve` is theirs). Do not
   write profiles directly to deliver an assist; the receipt is the channel.

## Depot (knowledge base)

`GET /api/depot?q=&tag=&type=` — search notes (markdown, Obsidian-compatible,
under profile_deck/depot/). `GET /api/depot/{id}` reads one; links are
[[wiki-links]] to profiles (`example-character/office-portrait`), characters, or
subjects (`logos`); `GET /api/depot/backlinks/{target}` finds notes linked
to a target. Note types: guide, prompt (body inserts into the prompt bar),
recipe (hires/detailer_tabs blocks; the user applies via
`POST /api/depot/{id}/apply-recipe`, locked profiles refuse). Create/update
notes with `POST /api/depot`. Useful for reading the user's prompt guides
before generating instead of guessing.

## Rules of engagement

- Renders cost GPU minutes; batch only on the user's ask.
- Nothing cloud/credit-based without explicit user confirmation (V5
  provider guardrail).
- Locked profile 409 = deliberate; do not unlock to work around.
- Report from tickets and sidecars, not from memory.
