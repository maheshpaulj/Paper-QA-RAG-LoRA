"""Rename a saved LoRA adapter's keys so sentence-transformers actually loads it.

    python -m scripts.fix_adapter_keys [models/minilm-lora]

sentence-transformers saves the adapter with peft's PeftModel naming:

    base_model.model.encoder.layer.0.attention.self.query.lora_A.weight

but loads it back into a plain BertModel, which expects:

    encoder.layer.0.attention.self.query.lora_A.default.weight

Keys that don't match are silently **re-initialised** rather than erroring, and a
fresh LoRA has lora_B = 0 -- so the adapter contributes nothing and the model is
byte-for-byte the base model. It looks like it loaded. It did nothing.

Idempotent: run it again and it's a no-op. Verify with a cosine check against the
base model -- 1.0 means the adapter is still dead.
"""
import shutil
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

from config import LORA_MODEL_DIR

PREFIX = "base_model.model."


def fix_key(key):
    key = key.replace(PREFIX, "", 1)
    for lora in ("lora_A", "lora_B"):
        key = key.replace(f".{lora}.weight", f".{lora}.default.weight")
    return key


def main(model_dir):
    path = Path(model_dir) / "adapter_model.safetensors"
    if not path.exists():
        raise SystemExit(f"No adapter at {path}")

    sd = load_file(str(path))
    if not any(k.startswith(PREFIX) for k in sd):
        print(f"{path.name}: keys already look right, nothing to do")
        return

    backup = path.with_suffix(".safetensors.bak")
    if not backup.exists():
        shutil.copy(path, backup)

    fixed = {fix_key(k): v for k, v in sd.items()}
    if len(fixed) != len(sd):
        raise SystemExit("key collision while renaming -- aborting, adapter untouched")

    save_file(fixed, str(path), metadata={"format": "pt"})
    print(f"rewrote {len(fixed)} tensors in {path}")
    print(f"  {next(iter(sd))}\n  -> {next(iter(fixed))}")
    print(f"backup: {backup.name}")
    print("\nNow verify it isn't a no-op:")
    print("  python -m scripts.check_lora")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else LORA_MODEL_DIR)
