"""Audit a local runtime image against a clean source checkout before publishing.

No credentials or evaluator-private files are read. Checks every layer's paths,
application-layer credential patterns, labels, and exact installed source bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any


def docker_json(docker: str, *args: str) -> Any:
    return json.loads(subprocess.check_output([docker, *args], text=True))


def audit(docker: str, image: str, source: Path, revision: str) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("revision must be a full lowercase commit SHA")
    config = docker_json(docker, "image", "inspect", image)[0]["Config"]
    labels = config.get("Labels") or {}
    if labels.get("org.opencontainers.image.source") != "https://github.com/Aryabhatta-0/EMBERBench":
        raise ValueError("incorrect OCI source label")
    if labels.get("org.opencontainers.image.revision") != revision or config["User"] != "ember":
        raise ValueError("incorrect source revision or runtime user")
    if any(re.match(r"(?:.*API_KEY|.*TOKEN|.*PASSWORD|.*SECRET|AWS_.*)=", v, re.I) for v in config["Env"]):
        raise ValueError("credential-like environment variable found; values suppressed")
    package = source / "src" / "emberbench"
    expected = {
        p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in package.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    }
    if not expected:
        raise ValueError("source checkout contains no package files")
    inspect_code = """
import hashlib,json,pathlib,emberbench
out={}
for tag,root in [('source',pathlib.Path('/opt/emberbench/src/emberbench')),('installed',pathlib.Path(emberbench.__file__).parent)]:
 out[tag]={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
print(json.dumps(out))
"""
    actual = docker_json(docker, "run", "--rm", "--network", "none", "--entrypoint", "python", image, "-c", inspect_code)
    if actual != {"source": expected, "installed": expected}:
        raise ValueError("runtime or installed package differs from committed source")
    forbidden = {"emberbench-private", ".aws", ".ssh", ".docker", ".emberbench-backup-keys", "batches", "environment-pack"}
    credential = re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{36,}|AKIA[A-Z0-9]{16}")
    layers = app_files = 0
    with tempfile.TemporaryDirectory(prefix="emberbench-image-audit-") as temp:
        archive = Path(temp) / "image.tar"
        subprocess.run([docker, "image", "save", "-o", str(archive), image], check=True)
        with tarfile.open(archive) as saved:
            manifest_file = saved.extractfile("manifest.json")
            if manifest_file is None:
                raise ValueError("missing image manifest")
            manifest = json.load(manifest_file)
            for name in manifest[0]["Layers"]:
                layers += 1
                layer_file = saved.extractfile(name)
                if layer_file is None:
                    raise ValueError("missing image layer")
                with tarfile.open(fileobj=layer_file, mode="r|*") as layer:
                    for member in layer:
                        path = member.name.removeprefix("./")
                        parts = set(path.lower().split("/"))
                        if forbidden & parts or path.lower().endswith((".aesgcm", ".zip.key")):
                            raise ValueError("forbidden artifact path in image layer; contents suppressed")
                        if not member.isfile() or not path.startswith("opt/emberbench/"):
                            continue
                        app_files += 1
                        if path.endswith((".json", ".jsonl", ".key", ".env", ".7z", ".zip", ".tar", ".whl")) or parts & {"scenarios", "tests", "recipe", "batches"}:
                            raise ValueError("evaluation or private asset in application layer; contents suppressed")
                        content = layer.extractfile(member)
                        if content is None or credential.search(content.read()):
                            raise ValueError("credential pattern or unreadable application file; contents suppressed")
    return {"source_commit": revision, "source_files_verified": len(expected), "layers_scanned": layers,
            "application_files_scanned": app_files, "forbidden_artifacts": 0, "credential_patterns": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--image", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.docker, args.image, args.source, args.revision), indent=2))


if __name__ == "__main__":
    main()
