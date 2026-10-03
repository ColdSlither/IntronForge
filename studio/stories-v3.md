# V3 Story Plots: Image Manager Lane (2026-10-03, not started)

User reference: cksdnfas/ComfyUI-Image-Manager v3.0.0 ("something similar
for V3"). That extension is an image library over a generation stack:
thumbnail browsing, folder watching, ratings and favorites, perceptual
hash similarity search, prompt statistics, WD14 auto-tagging, MCP server.

The deck already owns half of this: sidecar tickets on every render (the
metadata layer), the outputs tree organized by character and profile, the
ingest parser (Forge infotext plus ComfyUI graphs), and the lightbox.
V3 adds the library: a global view, curation state, and search over the
whole archive. Skipped from the reference on purpose: MCP server (the
deck is already agent-driven over REST) and multi-user auth (localhost
personal tool).

Ordering is deliberate: library-view is the foundation everything else
indexes into; the tagger is last because it is the only story that needs
a model download.

## Story: library-view

Goal: one grid over every render in outputs/, not the per-profile gallery
V1 has. Filters by character, profile, date, later by rating and tag.
Click opens the existing lightbox with the sidecar ticket beside it.

- Backend: GET /api/library with pagination and filters; scans the
  outputs tree, merges sidecar tickets.
- Thumbnails: first view generates a 256px cache thumb per image into a
  thumbs directory; the grid serves cache, never originals. Sized for
  the archive: 3000+ renders (census) must browse smoothly.
- Acceptance sketch: 01. grid lists every render across characters with
  character and profile labels. 02. filters work. 03. lightbox opens
  full resolution with the sidecar ticket rendered beside it. 04. first
  browse of 3000 images completes thumbnail caching without stalling
  the grid.

## Story: ratings-tags

Goal: curation state per image: star rating, favorite flag, free tags.

- Storage decision (pre-made): a separate library index keyed by image
  path, NOT sidecar amendment. Sidecar tickets are write-once render
  records; mutating them breaks the audit trail invariant.
- Rating and tag UI in the lightbox; filters integrate with
  library-view.
- Acceptance sketch: 01. a rating persists across restarts. 02. filter
  by rating and tag works. 03. the render sidecars are byte-identical
  after rating (the invariant holds).

## Story: similarity-search

Goal: perceptual hash (pHash) per image in the library index; find
similar from any image; near-duplicate detection with a hamming-distance
threshold.

- This is the dataset curation tool: before an Anima LoRA training run,
  report near-duplicate groups so the dataset keeps one copy of each
  look instead of twenty.
- Acceptance sketch: 01. index builds over the full archive. 02. find
  similar returns visually related renders. 03. near-dupe report groups
  duplicates with paths.

## Story: folder-watcher

Goal: watch Forge's output directory (and optionally the archive) and
auto-register new renders: thumbnail, hash, sidecar parse, into the
library. The reference extension's folder-watching analog.

- Acceptance sketch: 01. a new render appears in the library within
  seconds of generation. 02. interrupted watching recovers on restart
  without duplicating entries.

## Story: prompt-stats

Goal: aggregate over sidecar tickets: tag frequencies per character and
profile, positive and negative separation, prompt-token co-occurrence.
Feeds the danbooru-assist (V2) with the user's real usage data instead
of a generic corpus.

- Acceptance sketch: 01. stats endpoint and view over all sidecars.
  02. top tags per character reported. 03. filters compose with the
  library filters.

## Story: wd14-tagger

Goal: WD14 tagger integration (local ONNX model, roughly a 400MB
download) to auto-tag archived images with booru tags into the library
index. The heaviest story and the only one needing a model install.

- Synergy: WD14 tags images; danbooru-assist writes prompts. Closed
  loop for dataset building.
- Acceptance sketch: 01. tagger runs locally on a test image. 02.
  batch-tagging a character folder populates the index. 03. tags feed
  the library filters.
