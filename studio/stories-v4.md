# V4 Story Plots: Prompt Formatting (2026-10-03, not started)

Focus (user): prompt formatting. Three sources merged into one spec:

- User's own vault format: AI Art Stack - Prompt Section Database (five
  fixed BREAK sections: Quality, Character, Shot, Clothing, Light; one
  job per section; 7052 rules: color every garment, one lens per shot,
  size every bust).
- PromptRedundancyAssistant (v74199506-cyber): redundancy scoring (20
  exact dupes, 12 semantic overlap, 3 repeated words, 12 color
  contradictions, 6 generic-subsumed), model profiles, trigger-word
  protection, apply-on-confirm, undo.
- AIDE-TOOLS 5 features (civitai article 16934, saved HTML in project
  root): duplicate tag highlighting, drag-and-drop tag reorder,
  click-to-edit tags with weight arrows, Break button, resizable prompt
  fields. Plus (from the same article's wider release): tagset presets.

## Story: tag-mode-editor

Goal: the prompt fields gain a tag view. Tokens parse into chips;
click a chip to edit text or nudge its weight with arrow controls;
duplicates highlight in matching colors; drag to reorder; a BREAK
button inserts section dividers. Text view stays available (chips and
text are two views of one value; every edit writes back to the text).

- Acceptance sketch: 01. chips round-trip losslessly (weights, parens,
  BREAK). 02. duplicates share a highlight color. 03. drag reorders and
  the text view reflects it. 04. Tab between chips, Esc cancels an edit.

## Story: redundancy-linter

Goal: the PRA scoring adapted to the deck. A linter button (and a
passive badge) scores the current prompt: exact duplicates, semantic
overlap pairs from a curated table (masterpiece/best quality, size
pairs, redundant lighting terms), repeated words above threshold,
contradictory colors (hair/eye), generic tags subsumed by specific
ones present. Receipt panel with apply-on-confirm per item, trigger
words from the profile's loras protected, negative prompt handled
conservatively (dupes only).

- Acceptance sketch: 01. seeded example prompt scores with the right
  breakdown. 02. trigger words never flagged. 03. applying one item
  leaves the rest of the prompt untouched. 04. undo restores the prior
  text.

## Story: section-composer  (AUTHORITATIVE SCHEMA, user-declared 2026-10-03)

Goal: the user's prompt format as structure. Source: Dorian_White, "Pony
Prompts for Dummies" (civitai, saved HTML in project root), adapted by
the user; matches the banked example prompts.

Six logical sections (user's numbering):
1. Quality and Style
2. Keystone, Composition, Posture, Gesture, Action, Supporting details
3. Body, Hair, Face, Clothes and Details  (Primary Character)
4. Secondary Character (optional; own BREAK when present)
5. Setting
5. Lighting Detail (user's addition; article keeps light in Setting)

Compose mapping (matches banked prompts): sections 1+2 share the first
BREAK block, section 3 is the second block, section 4 gets its own
BREAK, sections 5+6 share the final block. One example character: 3 BREAK blocks;
a second character: same.

- Acceptance sketch: 01. an example banked prompt parses into the six
  sections with nothing lost and recomposes to the same text. 02. same
  for a second character. 03. a messy unsectioned prompt still parses (best-effort,
  unmatched lines keep adjacency). 04. editing one section rewrites
  only that section's lines.

## Story: snippet-library

Goal: named reusable section blocks. Outfits, lighting setups, shot
grammar, and full section templates stored as snippets (global or
per-character), inserted into a section slot with one click. Fed
initially from the vault libraries (Camera Angles, Lighting and Pose
Prompt Library) and the user's own banked prompts.

- Acceptance sketch: 01. save a Clothing block as a named snippet.
  02. insert it into another character's Clothing slot. 03. snippets
  persist across restarts.

Order rationale: tag-mode-editor is the substrate (chips make the
linter's highlighting and the composer's per-section scoping natural);
linter second; composer third; snippets last (they compose with
everything).
