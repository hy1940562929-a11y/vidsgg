"""Build real cached CLIP text anchors from an EXISTING local checkpoint.

Requires the OpenAI `clip` package to already exist in the execution environment.
This script does not install packages, download a model, or access datasets.
Input JSON: a list of foreground predicate names in native class-ID order 1..P.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicates", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", action="append",
                        help='Repeatable template containing {predicate}.')
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error("--checkpoint must be a local CLIP checkpoint file, not a model name")
    if args.output.exists():
        parser.error("Refusing to overwrite an existing anchor file")
    names = json.loads(args.predicates.read_text(encoding="utf-8"))
    if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n.strip() for n in names):
        parser.error("Predicate JSON must be a nonempty list of foreground names")
    if len(set(names)) != len(names):
        parser.error("Duplicate predicate names")
    templates = args.template or ["a subject {predicate} an object"]
    if any("{predicate}" not in item for item in templates):
        parser.error("Every template must contain {predicate}")
    try:
        import clip
    except ImportError as error:
        raise SystemExit("OpenAI clip is not installed here. Run in your existing CLIP environment; no automatic installation is performed.") from error
    model, _ = clip.load(str(args.checkpoint.resolve()), device="cpu", jit=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.inference_mode():
        all_templates = []
        for template in templates:
            phrases = [template.format(predicate=name.replace("_", " ")) for name in names]
            encoded = model.encode_text(clip.tokenize(phrases)).float()
            all_templates.append(F.normalize(encoded, dim=-1))
        embeddings = F.normalize(torch.stack(all_templates).mean(0), dim=-1)
    digest = hashlib.sha256()
    with args.checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    payload = {"embeddings": embeddings.cpu(), "predicates": names,
               "templates": templates, "checkpoint_sha256": digest.hexdigest(),
               "text_encoder": "OpenAI CLIP; local checkpoint", "synthetic": False,
               "class_id_convention": "foreground IDs 1..P; row 0 is ID 1"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "shape": list(embeddings.shape),
                      "synthetic": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
