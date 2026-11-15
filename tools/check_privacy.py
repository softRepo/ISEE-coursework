"""Check publication content without printing sensitive matched values.

Use --tree before Git initialization, --tracked before publishing, and
--staged in a pre-commit hook. Git modes inspect index blobs, including
files added with git add --force, rather than trusting .gitignore.
Only the exact manually audited reports, presentation and privacy-processed
photos are allowed as binary assets.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXACT_PATHS = {
    ".gitignore", ".gitattributes", "README.md", 
    "tools/check_privacy.py", ".githooks/pre-commit", ".github/workflows/privacy.yml",
    "人工智能/README.md", "电子工程训练（甲）/README.md",
}
FACE_PREFIX = "人工智能/人脸表情识别/"
LAB_PREFIX = "人工智能/AILAB/"
LAB_TEXT_PATHS = {'AI_Lab1/README.md',
 'AI_Lab1/eight_puzzle.py',
 'AI_Lab1/modified_8_puzzle.py',
 'AI_Lab1/requirements.txt',
 'AI_Lab1/test.py',
 'AI_Lab2/README.md',
 'AI_Lab2/alpha_beta.py',
 'AI_Lab2/compare.py',
 'AI_Lab2/debug.py',
 'AI_Lab2/requirements.txt',
 'AI_Lab3/README.md',
 'AI_Lab3/baseline/bayes.py',
 'AI_Lab3/baseline/lenet.py',
 'AI_Lab3/report/preamble.tex',
 'AI_Lab3/report/report.tex',
 'AI_Lab3/requirements.txt',
 'AI_Lab3/results/bayes_bernoulli_results.csv',
 'AI_Lab3/results/bayes_bernoulli_results.json',
 'AI_Lab3/results/bayes_pca_results.csv',
 'AI_Lab3/results/bayes_pca_results.json',
 'AI_Lab3/results/lenet_base_metrics.json',
 'AI_Lab3/results/lenet_plus_metrics.json',
 'AI_Lab3/src/bayes_bernoulli.py',
 'AI_Lab3/src/bayes_pca.py',
 'AI_Lab3/src/common_fashion.py',
 'AI_Lab3/src/lenet_base.py',
 'AI_Lab3/src/lenet_plus.py',
 'README.md'}
PATTERNS = {
    "email address": re.compile(r"(?i)[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}"),
    "mainland mobile number": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "identity number": re.compile(r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"),
    "student identifier": re.compile(r"(?i)(?:学\s*号|student[_ -]?(?:id|number))\s*[:：=]\s*[\"']?[a-z0-9]{6,18}"),
    "absolute Windows path": re.compile(r"(?i)(?<![a-z])[a-z]:[\\/][^\s\"'<>]+"),
    "personal Windows path": re.compile(r"(?i)(?<![a-z])[a-z]:[\\/](?:users|documents and settings)[\\/][^\s\"'<>]+"),
    "personal Unix path": re.compile(r"/(?:Users|home)/[a-zA-Z0-9_.-]+/"),
    "GitHub credential": re.compile(r"(?:gh[pousr]_[a-zA-Z0-9]{30,}|github_pat_[a-zA-Z0-9_]{30,})"),
    "cloud credential": re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}|AIza[0-9A-Za-z_-]{30,}"),
    "OpenAI credential": re.compile(r"sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{30,}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "credential assignment": re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\b\s*[:=]\s*[\"'][a-zA-Z0-9_+/=.-]{12,}[\"']"),
    "URL with credentials": re.compile(r"https?://[^\s/@]+:[^\s/@]+@"),
}
MAX_TEXT_SIZE = 2 * 1024 * 1024
MAX_APPROVED_BINARY_SIZE = 64 * 1024 * 1024
# Updating these files requires another content and metadata review before
# replacing a digest. Other binary files and modified bytes remain blocked.
APPROVED_BINARIES = {'人工智能/AILAB/AI_Lab1/report/report.pdf': 'f4f7672b5114fd2f8a7a11188c57a41d7d014f3c907394274e35192deb80caae',
 '人工智能/AILAB/AI_Lab2/report/report.pdf': 'af1fed4e814f0492572f3c2c25ebf31dae13ae67a0b2d573da633e5a9f4e7da5',
 '人工智能/AILAB/AI_Lab3/report/report.pdf': '5bd51467cd7c48b5d770382fb905ef92de0a2ce2106a1bc38a5d39dd289a1eab',
 '人工智能/人脸表情识别/docs/presentation.pptx': '6afcb22bff59b2b90f9ddf645ad20f93d7faa88179eba2029f01fc6b3624d5cf',
 '人工智能/人脸表情识别/docs/report.pdf': '990692cfd2f3d5017e72dcdf6d12fdf345b8cf76d18fd904606e478538266af7',
 '电子工程训练（甲）/基础实验/基础部分实验报告.pdf': '0f8dc3dbce39236cef5309d9d13f059fccb63c0787298752035626e0ce17e955',
 '电子工程训练（甲）/小车/最终照片/01-上视图.png': '903ca3d1c2e11591b36b4d9d9122f064c0985a47531ac5b1107706152bd1afb7',
 '电子工程训练（甲）/小车/最终照片/02-底部结构.png': 'c49b190c5a090ecc91de3eaa4ea8d409154ce15b9768c015f09cb4b65bf0cc55',
 '电子工程训练（甲）/小车/最终照片/03-正面结构.png': '089b2c53887d5420782dc87579285486265b6fb341bd17c06adbcadfde3f062a',
 '电子工程训练（甲）/小车/最终照片/04-侧面全貌.png': '9e94ff91ac407ae74c5e65ff32417f0c25c2ce974a890f7299aa06c9811a90dc',
 '电子工程训练（甲）/智能插座/智能插座测试报告.pdf': 'cc68b01f1f50afe471369fb935009f045cf77b1f8fc1267458d983c06140264d',
 '电子工程训练（甲）/智能插座/照片/01-PCB元件面.png': 'd656f0356ce88f56b4a2343c88d4978e30d213ccfa09947dc26c7e0f87793c6d',
 '电子工程训练（甲）/智能插座/照片/02-PCB焊接面.png': '30db4649fa16854093f4c6153cf5b9907bc6b3c220af32ca95ba5539e46e1177'}


def allowed_path(name: str) -> bool:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        return False
    if name in EXACT_PATHS or name in APPROVED_BINARIES:
        return True
    if name.startswith(LAB_PREFIX):
        return name[len(LAB_PREFIX):] in LAB_TEXT_PATHS
    if not name.startswith(FACE_PREFIX):
        return False
    path = PurePosixPath(name[len(FACE_PREFIX):])
    if str(path) in {"README.md", "requirements.txt", "docs/report.tex", "models/detectors/README.md"}:
        return True
    if path.suffix == ".py" and len(path.parts) == 1:
        return True
    if path.suffix == ".py" and len(path.parts) == 2 and path.parts[0] in {"models", "utils"}:
        return True
    return len(path.parts) == 3 and path.parts[:2] == ("docs", "results") and path.suffix in {".txt", ".csv", ".md"}



def inspect_content(name: str, data: bytes, mode: str = "100644") -> list[str]:
    findings = []
    if not allowed_path(name):
        findings.append("file outside publication allowlist")
    if mode not in {"100644", "100755"}:
        findings.append("symlink or submodule is not allowed")
    limit = MAX_APPROVED_BINARY_SIZE if name in APPROVED_BINARIES else MAX_TEXT_SIZE
    if len(data) > limit:
        return findings + ["oversized publication file"]
    if name in APPROVED_BINARIES:
        if hashlib.sha256(data).hexdigest() != APPROVED_BINARIES[name]:
            findings.append("binary document differs from the manually audited version")
        if name.endswith(".pdf") and (not data.startswith(b"%PDF-") or not data.rstrip().endswith(b"%%EOF")):
            findings.append("invalid report PDF envelope")
        if name.endswith(".pptx") and not data.startswith(b"PK\x03\x04"):
            findings.append("invalid presentation envelope")
        if name.endswith(".png") and (not data.startswith(b"\x89PNG\r\n\x1a\n") or not data.endswith(b"IEND\xaeB`\x82")):
            findings.append("invalid approved PNG envelope")
        return findings
    if b"\0" in data:
        return findings + ["binary file"]
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return findings + ["file is not UTF-8 text"]
    for label, pattern in PATTERNS.items():
        if pattern.search(name):
            findings.append("filename: " + label)
        for line_number, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                findings.append(f"line {line_number}: {label}")
    return findings


def git(*args: str) -> bytes:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True)
    if result.returncode:
        raise RuntimeError("Git inspection failed; initialize and stage the publication tree first.")
    return result.stdout


def index_files():
    for entry in git("ls-files", "--stage", "-z").split(b"\0"):
        if not entry:
            continue
        header, raw_name = entry.split(b"\t", 1)
        mode, object_id, stage = header.decode("ascii").split()
        if stage != "0":
            raise RuntimeError("Resolve Git index conflicts before publishing.")
        name = raw_name.decode("utf-8")
        if mode == "160000":
            yield name, b"", mode
        else:
            yield name, git("cat-file", "blob", object_id), mode


def tree_files():
    # Skip only Git internals and Python bytecode created by local verification.
    def walk(directory: Path):
        for path in sorted(directory.iterdir()):
            if path.name in {".git", "__pycache__"}:
                continue
            name = path.relative_to(ROOT).as_posix()
            if path.is_symlink():
                yield name, b"", "120000"
            elif path.is_dir():
                yield from walk(path)
            else:
                yield name, path.read_bytes(), "100644"
    yield from walk(ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--tree", action="store_true", help="Inspect publication files on disk")
    modes.add_argument("--tracked", action="store_true", help="Inspect all Git index blobs")
    modes.add_argument("--staged", action="store_true", help="Inspect all Git index blobs before committing")
    args = parser.parse_args()
    count = 0
    failed = False
    try:
        for name, data, mode in tree_files() if args.tree else index_files():
            count += 1
            findings = inspect_content(name, data, mode)
            if findings:
                failed = True
                # Do not print filenames: they may themselves contain private data.
                print(f"File #{count}: " + "; ".join(findings), file=sys.stderr)
        if count == 0:
            raise RuntimeError("No publication files found; refusing an empty check.")
    except (OSError, RuntimeError, UnicodeError) as exc:
        print(f"Privacy check could not complete: {type(exc).__name__}", file=sys.stderr)
        return 2
    print(f"Checked {count} files; " + ("review required." if failed else "no configured privacy patterns found."))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
