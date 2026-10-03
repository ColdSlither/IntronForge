# Studio: Adetailer Project

Adapted from ColdSlither's Claude-Code-Game-Studios at **minimal rigor**, per that
repo's own finding: heavier process bought traceability, not a better product.

## Roles

| Studio role | Filled by |
|---|---|
| Producer | the coordinating session (this agent), owns scope and the board |
| Architect | `code-architect` subagent on request |
| Engineer | the coordinating session |
| Reviewer | `code-reviewer` subagent, gates every story before close |
| QA | observed runs; a story cannot close until the artifact was run and the output looked at |

## Pipeline (minimal)

brief -> PRD with numbered acceptance criteria -> build -> QA evidence ->
review pass -> story closes. Anything bigger than one session becomes stories
in the PRD backlog instead of process.

## Rules

- Collaborative, not autonomous: scope changes come back to the user as options.
- Every setting that reaches the engine is recorded in a sidecar ticket.
- Nothing closes on "it ran". It closes on "I looked at what it made".
