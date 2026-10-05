# Third-Party Notices

## AcademiaSD LoRAlab-TrainerStudio

IntronForge's LoRA training capability is built on
[AcademiaSD LoRAlab-TrainerStudio](https://github.com/AcademiaSD/AcademiaSD_LoRAlab-TrainerStudio)
by AcademiaSD, used under the **MIT License** (their LICENSE file is preserved
in the installed copy and linked from their repository).

What IntronForge uses:

- The file-driven training pipeline scripts for SDXL
  (`scripts/1_pre_cache_sdxl.py`, `scripts/2_train_lora_sdxl.py`), run as
  subprocesses under LoRAlab's own virtual environment. IntronForge writes
  their `settings/*.json` configuration files (backing up and restoring any
  existing ones around each run) and captures their console output.
- The model presets (SDXL Base, Pony Diffusion V6 XL, Illustrious XL,
  Juggernaut XI, and the custom-checkpoint path).
- Nothing is redistributed inside this repository: LoRAlab is installed
  separately by the user and referenced at a pinned commit.

Pinned commit at integration time: `49e89e8e13790a7a0539093d660a23317d86f73c`.

Any local modifications to their scripts, if ever made, will be documented
here and offered upstream.

Their underlying model dependencies keep their own licenses (SDXL Open
RAIL++-M; check their repository for per-model terms).
