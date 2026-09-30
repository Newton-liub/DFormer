#!/usr/bin/env python3
"""Maintain the external paper bundle library and its MMFR indexes.

The library is intentionally operated as whole directories.  This program never
edits bundle contents; legacy archive inputs are read-only and are copied only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


LIBRARY_ROOT = Path(r"D:\0Project\origin\论文")
LEGACY_ROOT = Path(r"D:\0Project\DFormer-archive-20260922\doc\paper")
REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_JSON = REPO_ROOT / "MMFR" / "03_reference" / "PAPER_LIBRARY_INDEX.json"
INDEX_MD = REPO_ROOT / "MMFR" / "03_reference" / "PAPER_LIBRARY_INDEX.md"
PAPER_INDEX_MD = REPO_ROOT / "MMFR" / "03_reference" / "paper-index.md"
DUPLICATES_DIR_NAME = "论文_duplicates_review"
SCHEMA_VERSION = 1
MAX_FOLDER_LENGTH = 110
EXCLUDED_BODY_SUFFIXES = ("_formula", "_图表汇总")
UNVERIFIED_MANUAL_IDS = {"RE042", "RE053", "RE188", "RE447"}
MANUAL_ID_RE = re.compile(r"\b(?:PR|RE|AI)\d{3}\b|\bMoSA\b", re.IGNORECASE)
DOI_RE = re.compile(r"(?<![A-Z0-9])10\.\d{4,9}/[^\s<>\]\[{}\"']+", re.IGNORECASE)
ARXIV_RE = re.compile(
    r"(?:arxiv(?:\.org/(?:abs|pdf)/)?\s*[:/]?\s*)"
    r"((?:\d{4}\.\d{4,5}|[a-z][a-z.-]*/\d{7})(?:v\d+)?)",
    re.IGNORECASE,
)
GENERIC_FOLDER_NAMES = {
    "paper", "papers", "doc", "docs", "document", "documents", "pdf",
    "supplement", "supplemental", "supplementary", "supplementary material",
    "补充材料", "补充材料2", "附件", "论文", "文献", "word", "formula",
    "figure", "figures", "table", "tables",
}
GENERIC_HEADINGS = {
    "abstract", "摘要", "contents", "table of contents", "目录", "references",
    "reference", "参考文献", "title", "paper title", "论文标题",
}
WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


class LibraryError(RuntimeError):
    """An input or index condition that must stop the operation safely."""


@dataclass
class Bundle:
    source_path: Path
    target_path: Path
    title: str
    title_source: str
    normalized_title: str
    doi: Optional[str]
    arxiv: Optional[str]
    body_markdown: Optional[str]
    body_sha256: Optional[str]
    resource_counts: dict[str, Any]
    ambiguous_body: bool = False
    warnings: list[str] = field(default_factory=list)
    seed_manual_ids: list[str] = field(default_factory=list)
    seed_unverified_ids: list[str] = field(default_factory=list)
    previous_folder_names: list[str] = field(default_factory=list)

    def record(self, library_root: Path) -> dict[str, Any]:
        try:
            bundle_path = self.target_path.resolve(strict=False).relative_to(library_root.resolve(strict=False)).as_posix()
        except ValueError:
            # External source paths are used only for identity comparisons; final index entries must be in-root.
            bundle_path = str(self.target_path.resolve(strict=False))
        return {
            "bundle_path": bundle_path,
            "previous_folder_names": list(self.previous_folder_names),
            "folder": bundle_path,
            "entry_md": (bundle_path.rstrip("/\\") + "/" + self.body_markdown) if self.body_markdown else None,
            "md_sha256": self.body_sha256,
            "bundle_signature": {"md_sha256": self.body_sha256, **self.resource_counts},
            "content_capabilities": {
                "full_text": bool(self.body_sha256),
                "figures": self.resource_counts.get("figures", 0) > 0,
                "tables_structured": self.resource_counts.get("tables_structured", 0) > 0,
                "tables_rendered": self.resource_counts.get("tables_rendered", 0) > 0,
                "formulas_latex": self.resource_counts.get("formulas_latex", 0) > 0,
                "formulas_image": self.resource_counts.get("formulas_image", 0) > 0,
                "word_summary": self.resource_counts.get("word_summary", 0) > 0,
            },
            "year": None,
            "authors": [],
            "title": self.title,
            "title_source": self.title_source,
            "normalized_title": self.normalized_title,
            "doi": self.doi,
            "arxiv": self.arxiv,
            "body_markdown": self.body_markdown,
            "body_sha256": self.body_sha256,
            "resource_counts": copy.deepcopy(self.resource_counts),
            "ambiguous_body": self.ambiguous_body,
            "human": {
                "manual_ids": list(self.seed_manual_ids),
                "unverified_manual_ids": list(self.seed_unverified_ids),
            },
        }


@dataclass
class FsAction:
    kind: str
    source: Path
    destination: Path


@dataclass
class Plan:
    bundles: list[Bundle] = field(default_factory=list)
    actions: list[FsAction] = field(default_factory=list)
    duplicate_records: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    bootstrap_complete: bool = False


def _path_key(path: Path) -> str:
    return os.path.normcase(str(path.resolve(strict=False))).replace("/", "\\")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def _normalise_title(title: str) -> str:
    value = unicodedata.normalize("NFKC", title).casefold()
    return "".join(ch for ch in value if ch.isalnum())


def _is_trustworthy_title(title: str) -> bool:
    cleaned = re.sub(r"[`*_~]", "", title).strip().strip("# ")
    folded = unicodedata.normalize("NFKC", cleaned).casefold()
    if not cleaned or folded in GENERIC_HEADINGS or len(cleaned) > 300:
        return False
    if len(cleaned) < 3 or not any(ch.isalnum() for ch in cleaned):
        return False
    if re.fullmatch(r"(?:PR|RE|AI)\d{3}|MoSA", cleaned, re.IGNORECASE):
        return False
    return True


def _extract_h1(text: str) -> Optional[str]:
    for line in text.splitlines()[:80]:
        match = re.match(r"^\s*#\s+(?!#)(.*?)\s*#*\s*$", line)
        if not match:
            continue
        heading = match.group(1).strip()
        heading = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", heading)
        heading = re.sub(r"<[^>]+>", "", heading)
        heading = re.sub(r"[`*_~]", "", heading).strip()
        if _is_trustworthy_title(heading):
            return heading
    return None


def _folder_title(folder: Path, library_root: Path) -> str:
    candidates = [folder.name]
    parent = folder.parent
    while parent != folder and _inside(parent, library_root) and parent != library_root:
        candidates.append(parent.name)
        parent = parent.parent
    candidate = next(
        (name for name in candidates if unicodedata.normalize("NFKC", name).casefold() not in GENERIC_FOLDER_NAMES),
        folder.name,
    )
    # Common archive convention: "Author - 2025 - Title" or "Author - Title".
    stripped = re.sub(r"^.+?\s+-\s+(?:(?:19|20)\d{2}\s+-\s+)?(.+)$", r"\1", candidate).strip()
    if stripped:
        candidate = stripped
    candidate = candidate.replace("_", " ").strip()
    return candidate or folder.name or "Untitled bundle"


def _extract_identifiers(text: str) -> tuple[Optional[str], Optional[str], list[str]]:
    opening: list[str] = []
    for line in text.splitlines()[:45]:
        if re.match(r"^\s*#{1,3}\s*(?:abstract|introduction|1[. ]+introduction)\b", line, re.IGNORECASE):
            break
        opening.append(line)
    header = "\n".join(opening)
    dois = {match.group(0).rstrip(".,;:)\"'").casefold() for match in DOI_RE.finditer(header)}
    arxivs = {re.sub(r"v\d+$", "", match.group(1).casefold().removesuffix(".pdf"))
              for match in ARXIV_RE.finditer(header)}
    warnings: list[str] = []
    if len(dois) > 1:
        warnings.append("multiple DOI candidates in paper header; DOI unresolved")
    if len(arxivs) > 1:
        warnings.append("multiple arXiv candidates in paper header; arXiv unresolved")
    return next(iter(dois)) if len(dois) == 1 else None, next(iter(arxivs)) if len(arxivs) == 1 else None, warnings


def _eligible_markdown(folder: Path) -> list[Path]:
    result = []
    try:
        for child in folder.iterdir():
            if not child.is_file() or child.suffix.casefold() != ".md":
                continue
            stem = child.stem.casefold()
            if stem.endswith(EXCLUDED_BODY_SUFFIXES):
                continue
            result.append(child)
    except OSError as exc:
        raise LibraryError(f"cannot list bundle directory {folder}: {exc}") from exc
    return sorted(result, key=lambda item: item.name.casefold())


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        raise LibraryError(f"cannot read bundle markdown {path}: {exc}") from exc


def _resource_counts(folder: Path, body_path: Optional[Path]) -> dict[str, Any]:
    counts = {"figures": 0, "tables_structured": 0, "tables_rendered": 0,
              "formulas_latex": 0, "formulas_image": 0, "word_summary": 0}
    for area, suffix, key in (("Figure", ".jpg", "figures"),
                              ("Tables", ".xlsx", "tables_structured"),
                              ("Tables", ".jpg", "tables_rendered"),
                              ("Formula", ".jpg", "formulas_image"),
                              ("Formula", ".md", "formulas_latex"),
                              ("Word", ".md", "word_summary")):
        source = folder / area
        if source.is_dir():
            counts[key] += sum(1 for item in source.iterdir()
                               if item.is_file() and item.suffix.casefold() == suffix
                               and (key != "formulas_latex" or item.stem.casefold().endswith("_formula")))
    counts["total"] = sum(counts.values())
    return counts


def _analyse_bundle(folder: Path, library_root: Path, source_kind: str = "library") -> Optional[Bundle]:
    markdown = _eligible_markdown(folder)
    if not markdown:
        return None

    title_infos: list[tuple[Path, str, Optional[str]]] = []
    for item in markdown:
        text = _read_text(item)
        title_infos.append((item, text, _extract_h1(text)))

    ambiguous = False
    selected: Optional[Path] = None
    selected_text = ""
    warnings: list[str] = []
    if len(title_infos) == 1:
        selected, selected_text, _ = title_infos[0]
    else:
        folder_key = _normalise_title(folder.name.replace("_", " "))
        exact_stem = [entry for entry in title_infos if _normalise_title(entry[0].stem.replace("_", " ")) == folder_key]
        trusted = [entry for entry in title_infos if entry[2] is not None]
        choice = exact_stem if len(exact_stem) == 1 else (trusted if len(trusted) == 1 else [])
        if choice:
            selected, selected_text, _ = choice[0]
            warnings.append("multiple eligible top-level Markdown files; one was selected by a unique title cue")
        else:
            ambiguous = True
            warnings.append("multiple eligible top-level Markdown files; body identity is ambiguous")
            selected, selected_text, _ = max(title_infos, key=lambda entry: (len(entry[1]), entry[0].name.casefold()))

    h1 = _extract_h1(selected_text) if selected is not None else None
    if h1 and not ambiguous:
        title = h1
        title_source = "h1"
    else:
        title = _folder_title(folder, library_root)
        title_source = "folder"

    body_sha: Optional[str] = None
    if selected is not None and not ambiguous:
        try:
            body_sha = hashlib.sha256(selected.read_bytes()).hexdigest()
        except OSError as exc:
            raise LibraryError(f"cannot hash body Markdown {selected}: {exc}") from exc
    doi, arxiv, id_warnings = _extract_identifiers(selected_text)
    warnings.extend(id_warnings)
    body_rel = selected.relative_to(folder).as_posix() if selected is not None else None
    counts = _resource_counts(folder, selected if not ambiguous else None)
    return Bundle(
        source_path=folder.resolve(),
        target_path=folder.resolve(),
        title=title,
        title_source=title_source,
        normalized_title=_normalise_title(title),
        doi=doi,
        arxiv=arxiv,
        body_markdown=body_rel,
        body_sha256=body_sha,
        resource_counts=counts,
        ambiguous_body=ambiguous,
        warnings=warnings,
    )


def _scan_bundles(root: Path, source_kind: str = "library") -> list[Bundle]:
    if not root.is_dir():
        raise LibraryError(f"bundle source directory does not exist: {root}")
    result: list[Bundle] = []
    for current, dirs, _files in os.walk(root, followlinks=False):
        dirs[:] = sorted(
            (name for name in dirs if name not in {".git", "__pycache__", "论文_duplicates_review"}),
            key=str.casefold,
        )
        folder = Path(current)
        if "supplemental" in folder.name.casefold() or "supplementary" in folder.name.casefold():
            dirs[:] = []  # a supplement is not a standalone canonical paper
            continue
        bundle = _analyse_bundle(folder, root, source_kind)
        if bundle is not None:
            result.append(bundle)
            dirs[:] = []
    return sorted(result, key=lambda item: str(item.source_path).casefold())


def _relation(left: dict[str, Any], right: dict[str, Any]) -> str:
    """Classify verified identifier/title matches; uncertain versions remain warnings."""
    left_doi, right_doi = left.get("doi"), right.get("doi")
    left_arxiv, right_arxiv = left.get("arxiv"), right.get("arxiv")
    if left_doi and right_doi and left_doi.casefold() != right_doi.casefold():
        return "possible" if left.get("normalized_title") == right.get("normalized_title") else "none"
    if left_arxiv and right_arxiv and left_arxiv.casefold() != right_arxiv.casefold():
        return "possible" if left.get("normalized_title") == right.get("normalized_title") else "none"
    same_id = bool((left_doi and right_doi and left_doi.casefold() == right_doi.casefold())
                   or (left_arxiv and right_arxiv and left_arxiv.casefold() == right_arxiv.casefold()))
    if same_id and not left.get("ambiguous_body") and not right.get("ambiguous_body"):
        return "strong"
    if left.get("normalized_title") and left["normalized_title"] == right.get("normalized_title"):
        if (left.get("title_source") == right.get("title_source") == "h1"
                and not left.get("ambiguous_body") and not right.get("ambiguous_body")):
            return "title"
        return "possible"
    import difflib
    similarity = difflib.SequenceMatcher(None, left.get("normalized_title", ""),
                                         right.get("normalized_title", "")).ratio()
    return "possible" if similarity >= 0.94 and len(left.get("normalized_title", "")) >= 25 else "none"


def _is_exact(left: dict[str, Any], right: dict[str, Any]) -> bool:
    relation = _relation(left, right)
    if relation not in {"strong", "title"}:
        return False
    body_sha = left.get("body_sha256")
    if not body_sha or body_sha != right.get("body_sha256"):
        return False
    return left.get("resource_counts") == right.get("resource_counts")


def _is_alternate(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if _relation(left, right) not in {"strong", "title"}:
        return False
    if left.get("body_sha256") == right.get("body_sha256") and left.get("resource_counts") == right.get("resource_counts"):
        return False
    return True


def _safe_component(title: str, max_length: int = MAX_FOLDER_LENGTH) -> str:
    value = unicodedata.normalize("NFKC", title)
    value = value.replace(":", " -").replace("/", "-").replace("\\", "-")
    value = re.sub(r'[<>"|?*\x00-\x1f]', "", value)
    value = re.sub(r"\s+", " ", value).strip().rstrip(".") or "Untitled bundle"
    if value.split(".", 1)[0].upper() in WINDOWS_RESERVED_NAMES:
        value = "_" + value
    return value[:max_length].rstrip(" .") or "Untitled bundle"


def _allocate_destination(root: Path, title: str, bundle: Bundle, reserved: set[str], *, suffix_hint: str = "") -> Path:
    candidate = root / _safe_component(title)
    key = _path_key(candidate)
    if key in reserved or candidate.exists():
        raise LibraryError(
            f"physical title collision requires human version/disambiguation choice: {bundle.source_path} -> {candidate}"
        )
    reserved.add(key)
    return candidate


def _unique_review_destination(review_root: Path, bundle: Bundle, reserved: set[str]) -> Path:
    base = _safe_component(bundle.title)
    hint = (bundle.body_sha256 or hashlib.sha256(str(bundle.source_path).encode("utf-8")).hexdigest())[:8]
    suffix = f"__{hint}"
    name = f"{base[: max(1, MAX_FOLDER_LENGTH - len(suffix))].rstrip(' .') or 'bundle'}{suffix}"
    candidate = review_root / name
    counter = 2
    while _path_key(candidate) in reserved or candidate.exists():
        extra = f"_{counter}"
        candidate = review_root / f"{name[:MAX_FOLDER_LENGTH - len(extra)]}{extra}"
        counter += 1
    reserved.add(_path_key(candidate))
    return candidate


def _relative_bundle_dir(md_path: str, library_root: Path, legacy_root: Path) -> Optional[Path]:
    normalized = md_path.replace("\\", "/").strip().lstrip("./")
    parts = [part for part in normalized.split("/") if part]
    if len(parts) < 2:
        return None
    if parts[0].casefold() not in {"dformer-doc-paper", "mmfr-附件"}:
        return None
    # paper-index.md explicitly records both archive groups relative to the CURRENT external library.
    return (library_root / Path(*parts).parent).resolve(strict=False)


def _parse_seed_mappings(index_md: Path, library_root: Path, legacy_root: Path) -> dict[str, dict[str, list[str]]]:
    if not index_md.is_file():
        return {}
    try:
        lines = index_md.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except OSError as exc:
        raise LibraryError(f"cannot read mapping source {index_md}: {exc}") from exc
    mappings: dict[str, dict[str, list[str]]] = {}
    for line in lines:
        if not line.lstrip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 5 or set(cells[0]) <= {"-", ":", " "}:
            continue
        id_cell = cells[0]
        path_cell = cells[4]
        manual_ids = []
        for match in MANUAL_ID_RE.finditer(id_cell):
            value = match.group(0)
            canonical = "MoSA" if value.casefold() == "mosa" else value.upper()
            if canonical not in manual_ids:
                manual_ids.append(canonical)
        md_paths = re.findall(r"`([^`]+?\.md)`", path_cell, flags=re.IGNORECASE)
        if not md_paths:
            continue
        for md in md_paths:
            folder = _relative_bundle_dir(md, library_root, legacy_root)
            if folder is None:
                continue
            key = _path_key(folder)
            entry = mappings.setdefault(key, {"manual_ids": [], "unverified_manual_ids": []})
            for code in manual_ids:
                target = "unverified_manual_ids" if code in UNVERIFIED_MANUAL_IDS else "manual_ids"
                if code not in entry[target]:
                    entry[target].append(code)
    return mappings


def _attach_seed_mapping(bundle: Bundle, mappings: dict[str, dict[str, list[str]]]) -> None:
    values = mappings.get(_path_key(bundle.source_path))
    if values:
        bundle.seed_manual_ids = list(values["manual_ids"])
        bundle.seed_unverified_ids = list(values["unverified_manual_ids"])


def _validate_human(value: Any, where: str) -> None:
    if not isinstance(value, dict):
        raise LibraryError(f"cannot safely preserve human field at {where}: expected a JSON object")
    for key in ("manual_ids", "unverified_manual_ids"):
        if key in value and (not isinstance(value[key], list) or any(not isinstance(item, str) for item in value[key])):
            raise LibraryError(f"cannot safely parse human.{key} at {where}; index left unchanged")


def _validate_bundle_record(record: Any, where: str) -> None:
    if not isinstance(record, dict):
        raise LibraryError(f"invalid bundle record at {where}")
    if not isinstance(record.get("bundle_path"), str):
        raise LibraryError(f"missing bundle_path at {where}")
    _validate_human(record.get("human"), where)


def _external_record(record: dict[str, Any], library_id: str) -> dict[str, Any]:
    auto = {key: copy.deepcopy(value) for key, value in record.items()
            if key not in {"lib_id", "human", "alternate_bundles", "status"}}
    auto["library_id"] = library_id
    result = {"auto": auto, "human": copy.deepcopy(record.get("human", {}))}
    if "status" in record:
        result["status"] = record["status"]
    if "alternate_bundles" in record:
        result["alternate_bundles"] = [_external_record(alt, library_id)
                                        for alt in record["alternate_bundles"]]
    return result


def _internal_record(record: dict[str, Any]) -> dict[str, Any]:
    if "auto" not in record:
        return record  # support existing flat index without discarding its human-owned metadata
    auto = record.get("auto")
    if not isinstance(auto, dict):
        raise LibraryError("auto metadata must be a JSON object; index left unchanged")
    result = copy.deepcopy(auto)
    result["lib_id"] = auto.get("library_id")
    result["human"] = copy.deepcopy(record.get("human"))
    if "status" in record:
        result["status"] = record["status"]
    if "alternate_bundles" in record:
        if not isinstance(record["alternate_bundles"], list):
            raise LibraryError("alternate_bundles must be a list; index left unchanged")
        result["alternate_bundles"] = [_internal_record(alt) for alt in record["alternate_bundles"]]
    return result


def _load_index(path: Path, library_root: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LibraryError(f"cannot parse existing JSON index {path}: {exc}; no files will be changed") from exc
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise LibraryError(f"unsupported or invalid paper index schema in {path}; no files will be changed")
    papers = data.get("papers")
    if not isinstance(papers, list):
        raise LibraryError(f"existing index has no valid papers list: {path}")
    data["papers"] = papers = [_internal_record(item) if isinstance(item, dict) else item for item in papers]
    seen_lib_ids: set[str] = set()
    seen_paths: set[str] = set()
    max_number = 0
    for position, item in enumerate(papers):
        where = f"{path}:papers[{position}]"
        if not isinstance(item, dict):
            raise LibraryError(f"invalid paper record at {where}")
        lib_id = item.get("lib_id")
        if not isinstance(lib_id, str) or not re.fullmatch(r"LIB\d{6,}", lib_id):
            raise LibraryError(f"invalid stable LIB identifier at {where}")
        if lib_id in seen_lib_ids:
            raise LibraryError(f"duplicate LIB identifier {lib_id} in existing index")
        seen_lib_ids.add(lib_id)
        max_number = max(max_number, int(lib_id[3:]))
        _validate_bundle_record(item, where)
        bundles = [item]
        alternates = item.get("alternate_bundles", [])
        if not isinstance(alternates, list):
            raise LibraryError(f"alternate_bundles must be a list at {where}")
        bundles.extend(alternates)
        for alternate_index, bundle_record in enumerate(bundles[1:], start=1):
            _validate_bundle_record(bundle_record, f"{where}.alternate_bundles[{alternate_index - 1}]")
        for bundle_record in bundles:
            key = _stored_path_key(bundle_record.get("bundle_path", ""), library_root)
            if key in seen_paths:
                raise LibraryError(f"bundle path is recorded more than once in existing index: {bundle_record['bundle_path']}")
            seen_paths.add(key)
    duplicate_review = data.get("duplicate_review", [])
    if not isinstance(duplicate_review, list) or any(not isinstance(item, dict) for item in duplicate_review):
        raise LibraryError("existing duplicate_review field is invalid; index left unchanged")
    counter = data.get("next_lib_number", max_number + 1)
    if not isinstance(counter, int) or isinstance(counter, bool) or counter < 1:
        raise LibraryError("existing next_lib_number is invalid; index left unchanged")
    data["next_lib_number"] = max(counter, max_number + 1)
    return data


def _stored_path_key(value: str, library_root: Path) -> str:
    path = Path(value.replace("/", os.sep).replace("\\", os.sep))
    if not path.is_absolute():
        path = library_root / path
    return _path_key(path)


def _apply_auto_fields(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(old)
    for key, value in new.items():
        if key == "human":
            continue
        if key == "previous_folder_names":
            updated[key] = list(dict.fromkeys([*old.get(key, []), *value]))
        else:
            updated[key] = copy.deepcopy(value)
    return updated


def _duplicate_entry(bundle_record: dict[str, Any], canonical_record: dict[str, Any], *, status: str,
                     review_path: Optional[Path] = None, original_path: Optional[Path] = None) -> dict[str, Any]:
    return {
        "original_path": str((original_path or Path(bundle_record["bundle_path"])).resolve(strict=False)),
        "canonical_path": str(Path(canonical_record["bundle_path"]).resolve(strict=False)),
        "review_path": str(review_path.resolve(strict=False)) if review_path else None,
        "sha256": bundle_record.get("body_sha256"),
        "resource_counts": copy.deepcopy(bundle_record.get("resource_counts", {})),
        "status": status,
        "title": bundle_record.get("title", ""),
    }


def _entry_key(item: dict[str, Any]) -> tuple[str, str]:
    return (str(item.get("original_path", "")).casefold(), str(item.get("sha256", "")))


def _record_manual_conflicts(papers: list[dict[str, Any]], existing_count: int = 0) -> list[str]:
    owners: dict[str, tuple[int, str, str]] = {}
    warnings: list[str] = []
    for index, paper in enumerate(papers):
        lib_id = paper["lib_id"]
        title = paper.get("title", "")
        for bundle in [paper, *paper.get("alternate_bundles", [])]:
            for manual_id in list(bundle.get("human", {}).get("manual_ids", [])):
                previous = owners.get(manual_id)
                if previous is None:
                    owners[manual_id] = (index, lib_id, title)
                elif previous[0] != index:
                    message = (f"[MANUAL ID CONFLICT] {manual_id} is already assigned to: "
                               f"{previous[1]} / {previous[2]}; New candidate: "
                               f"{lib_id} / {title}. No change was made to the old mapping.")
                    if index < existing_count:
                        raise LibraryError(message + " Existing index left unchanged pending human correction.")
                    bundle["human"]["manual_ids"].remove(manual_id)
                    warnings.append(message)
    return warnings


def _all_bundle_records(paper: dict[str, Any]) -> list[dict[str, Any]]:
    return [paper, *paper.get("alternate_bundles", [])]


def _sync_index(bundles: list[Bundle], library_root: Path, old_index: Optional[dict[str, Any]],
                duplicate_records: list[dict[str, Any]], *, bootstrap_complete: bool) -> tuple[dict[str, Any], list[str]]:
    papers = copy.deepcopy(old_index.get("papers", [])) if old_index else []
    duplicate_review = copy.deepcopy(old_index.get("duplicate_review", [])) if old_index else []
    warnings: list[str] = []
    next_number = old_index.get("next_lib_number", 1) if old_index else 1
    old_bootstrap_complete = bool(old_index and old_index.get("bootstrap_complete"))

    by_path: dict[str, tuple[int, Optional[int]]] = {}
    for paper_index, paper in enumerate(papers):
        for bundle_index, record in enumerate(_all_bundle_records(paper)):
            by_path[_stored_path_key(record["bundle_path"], library_root)] = (paper_index, None if bundle_index == 0 else bundle_index - 1)

    present_papers: set[int] = set()
    new_papers: list[dict[str, Any]] = []
    duplicate_keys = {_entry_key(item) for item in duplicate_review}
    for item in duplicate_records:
        key = _entry_key(item)
        if key not in duplicate_keys:
            duplicate_review.append(item)
            duplicate_keys.add(key)

    def create_paper(bundle_record: dict[str, Any]) -> dict[str, Any]:
        nonlocal next_number
        lib_id = f"LIB{next_number:06d}"
        next_number += 1
        paper = copy.deepcopy(bundle_record)
        paper["lib_id"] = lib_id
        paper["alternate_bundles"] = []
        paper["status"] = "present"
        return paper

    for bundle in bundles:
        bundle_record = bundle.record(library_root)
        for warning in bundle.warnings:
            warnings.append(f"{bundle.target_path}: {warning}")
        path_key = _stored_path_key(bundle_record["bundle_path"], library_root)
        matched = by_path.get(path_key)
        if matched is not None:
            paper_index, alternate_index = matched
            paper = papers[paper_index]
            old_bundle = paper if alternate_index is None else paper["alternate_bundles"][alternate_index]
            old_human = old_bundle.get("human", {})
            incoming_human = bundle_record.get("human", {})
            for key in ("manual_ids", "unverified_manual_ids"):
                old_values = old_human.get(key, [])
                incoming_values = incoming_human.get(key, [])
                if any(value not in old_values for value in incoming_values):
                    warnings.append(
                        f"manual mapping conflict at {bundle_record['bundle_path']} ({key}); existing human field preserved"
                    )
            if alternate_index is None:
                papers[paper_index] = _apply_auto_fields(paper, bundle_record)
            else:
                updated_alt = _apply_auto_fields(old_bundle, bundle_record)
                paper["alternate_bundles"][alternate_index] = updated_alt
            papers[paper_index]["status"] = "present"
            present_papers.add(paper_index)
            continue

        candidates: list[tuple[str, dict[str, Any], int, Optional[int]]] = []
        for paper_index, paper in enumerate(papers):
            for bundle_index, old_bundle in enumerate(_all_bundle_records(paper)):
                relation = _relation(bundle_record, old_bundle)
                if relation != "none":
                    candidates.append((relation, old_bundle, paper_index, None if bundle_index == 0 else bundle_index - 1))
        for paper_index, paper in enumerate(new_papers):
            for bundle_index, old_bundle in enumerate(_all_bundle_records(paper)):
                relation = _relation(bundle_record, old_bundle)
                if relation != "none":
                    candidates.append((relation, old_bundle, -paper_index - 1, None if bundle_index == 0 else bundle_index - 1))

        candidate_owners = {candidate[2] for candidate in candidates}
        exact = [candidate for candidate in candidates if _is_exact(bundle_record, candidate[1])]
        alternates = [candidate for candidate in candidates if _is_alternate(bundle_record, candidate[1])]
        if len(candidate_owners) == 1 and len(exact) == 1:
            _relation_name, canonical_record, paper_index, _alt_index = exact[0]
            review_record = {
                "original_path": str(bundle.source_path.resolve(strict=False)),
                "canonical_path": str((library_root / canonical_record["bundle_path"].replace("/", os.sep)).resolve(strict=False)),
                "review_path": None,
                "sha256": bundle_record.get("body_sha256"),
                "resource_counts": copy.deepcopy(bundle_record.get("resource_counts", {})),
                "status": "detected_not_moved",
                "title": bundle_record.get("title", ""),
            }
            key = _entry_key(review_record)
            if key not in duplicate_keys:
                duplicate_review.append(review_record)
                duplicate_keys.add(key)
            if paper_index >= 0:
                present_papers.add(paper_index)
            continue

        if len(candidate_owners) == 1 and len(alternates) == 1:
            _relation_name, _old_record, paper_index, _old_alt = alternates[0]
            if paper_index >= 0:
                paper = papers[paper_index]
                paper.setdefault("alternate_bundles", []).append(bundle_record)
                for code in bundle_record.get("human", {}).get("manual_ids", []):
                    if code not in paper["human"].setdefault("manual_ids", []):
                        paper["human"]["manual_ids"].append(code)
                paper["status"] = "present"
                present_papers.add(paper_index)
            else:
                target = new_papers[-paper_index - 1]
                target.setdefault("alternate_bundles", []).append(bundle_record)
                for code in bundle_record.get("human", {}).get("manual_ids", []):
                    if code not in target["human"].setdefault("manual_ids", []):
                        target["human"]["manual_ids"].append(code)
                target["status"] = "present"
            continue

        if candidates:
            warnings.append(
                f"possible identity/version match for {bundle.target_path}; kept as a separate LIB entry (no merge)"
            )
        paper = create_paper(bundle_record)
        new_papers.append(paper)

    # Old entries remain stable; absence is recorded rather than deleting or renumbering them.
    for index, paper in enumerate(papers):
        if index not in present_papers:
            paper["status"] = "missing"
    existing_count = len(papers)
    papers.extend(new_papers)
    warnings.extend(_record_manual_conflicts(papers, existing_count))
    id_by_path = {_stored_path_key(record["bundle_path"], library_root): paper["lib_id"]
                  for paper in papers for record in _all_bundle_records(paper)}
    for review in duplicate_review:
        canonical = review.get("canonical_path")
        if canonical:
            review["canonical_lib_id"] = id_by_path.get(_stored_path_key(canonical, library_root))
    data = {
        "schema_version": SCHEMA_VERSION,
        "library_root": str(library_root.resolve(strict=False)),
        "next_lib_number": max(next_number, max((int(item["lib_id"][3:]) + 1 for item in papers), default=1)),
        "bootstrap_complete": bool(bootstrap_complete or old_bootstrap_complete),
        "papers": papers,
        "duplicate_review": duplicate_review,
    }
    return data, warnings


def _review_log(bundle: Bundle, canonical: Bundle, review_path: Optional[Path], status: str) -> dict[str, Any]:
    return {
        "original_path": str(bundle.source_path.resolve(strict=False)),
        "canonical_path": str(canonical.target_path.resolve(strict=False)),
        "review_path": str(review_path.resolve(strict=False)) if review_path else None,
        "sha256": bundle.body_sha256,
        "resource_counts": copy.deepcopy(bundle.resource_counts),
        "status": status,
        "title": bundle.title,
    }


def _plan_body_rename(bundle: Bundle, plan: Plan) -> None:
    """Rename only the identified top-level body, after any whole-bundle move/copy."""
    if bundle.ambiguous_body or not bundle.body_markdown:
        return
    desired = _safe_component(bundle.target_path.name, max_length=200) + ".md"
    if bundle.body_markdown == desired:
        return
    old = bundle.target_path / bundle.body_markdown
    new = bundle.target_path / desired
    if new.exists() and _path_key(old) != _path_key(new):
        plan.warnings.append(f"body name conflict; kept unchanged: {old}")
        return
    plan.actions.append(FsAction("rename_file", old, new))
    bundle.body_markdown = desired


def _bootstrap_plan(root: Path, legacy_root: Path, mappings: dict[str, dict[str, list[str]]]) -> Plan:
    if not root.is_dir():
        raise LibraryError(f"main paper library does not exist: {root}")
    plan = Plan(bootstrap_complete=True)
    main = _scan_bundles(root)
    if legacy_root.is_dir():
        legacy = _scan_bundles(legacy_root, "legacy")
    else:
        legacy = []
        plan.warnings.append(f"known read-only legacy source is unavailable; skipped: {legacy_root}")
    for bundle in [*main, *legacy]:
        _attach_seed_mapping(bundle, mappings)
        for warning in bundle.warnings:
            plan.warnings.append(f"{bundle.source_path}: {warning}")

    source_paths = {_path_key(bundle.source_path) for bundle in main}
    reserved: set[str] = set()
    for child in root.iterdir():
        if _path_key(child) not in source_paths:
            reserved.add(_path_key(child))
    retained: list[Bundle] = []
    duplicate_sources: list[tuple[Bundle, Bundle]] = []
    for bundle in main:
        exact_match = next((item for item in retained if _is_exact(bundle.record(root), item.record(root))), None)
        if exact_match:
            for code in bundle.seed_manual_ids:
                if code not in exact_match.seed_manual_ids:
                    exact_match.seed_manual_ids.append(code)
            for code in bundle.seed_unverified_ids:
                if code not in exact_match.seed_unverified_ids:
                    exact_match.seed_unverified_ids.append(code)
            duplicate_sources.append((bundle, exact_match))
            continue
        alternate = next((item for item in retained if _is_alternate(bundle.record(root), item.record(root))), None)
        if alternate:
            bundle.target_path = bundle.source_path
            retained.append(bundle)
            plan.warnings.append(f"same paper / different extraction retained for human choice: {bundle.source_path} vs {alternate.source_path}")
            continue
        if bundle.source_path.parent == root and bundle.source_path.name == _safe_component(bundle.title):
            destination = bundle.source_path
            reserved.add(_path_key(destination))
        else:
            destination = _allocate_destination(root, bundle.title, bundle, reserved)
        if _path_key(destination) == _path_key(bundle.source_path):
            bundle.target_path = bundle.source_path
        else:
            bundle.target_path = destination
            bundle.previous_folder_names.append(bundle.source_path.name)
            plan.actions.append(FsAction("move", bundle.source_path, destination))
        retained.append(bundle)

    review_root = root.parent / DUPLICATES_DIR_NAME
    review_reserved = {_path_key(child) for child in review_root.iterdir()} if review_root.is_dir() else set()
    for duplicate, canonical in duplicate_sources:
        review_destination = _unique_review_destination(review_root, duplicate, review_reserved)
        plan.actions.append(FsAction("move", duplicate.source_path, review_destination))
        plan.duplicate_records.append(
            _review_log(duplicate, canonical, review_destination, "moved_to_review")
        )
        plan.warnings.append(
            f"main-library exact duplicate (md_sha256={duplicate.body_sha256}, counts={duplicate.resource_counts}) will be moved as a whole bundle: {duplicate.source_path} -> {review_destination}"
        )

    accepted = list(retained)
    external_alternates: list[Bundle] = []
    for bundle in legacy:
        exact = next((item for item in accepted if _is_exact(bundle.record(root), item.record(root))), None)
        if exact:
            for code in bundle.seed_manual_ids:
                if code not in exact.seed_manual_ids:
                    exact.seed_manual_ids.append(code)
            plan.skipped.append(f"legacy exact duplicate left read-only and not copied: {bundle.source_path}")
            plan.duplicate_records.append(
                _review_log(bundle, exact, None, "external_exact_not_copied")
            )
            continue
        different = next((item for item in accepted if _is_alternate(bundle.record(root), item.record(root))), None)
        if different:
            external_alternates.append(bundle)
            plan.warnings.append(f"legacy same paper / different extraction left read-only for human choice: {bundle.source_path}")
            continue
        possible = [item for item in accepted if _relation(bundle.record(root), item.record(root)) == "possible"]
        if possible:
            plan.warnings.append(
                f"legacy bundle has only a possible identity/version match; left read-only and not copied: {bundle.source_path}"
            )
            continue
        destination = _allocate_destination(root, bundle.title, bundle, reserved)
        bundle.target_path = destination
        plan.actions.append(FsAction("copy", bundle.source_path, destination))
        accepted.append(bundle)

    plan.bundles = retained + external_alternates + [bundle for bundle in legacy if any(
        action.kind == "copy" and _path_key(action.source) == _path_key(bundle.source_path)
        for action in plan.actions
    )]
    for bundle in plan.bundles:
        if bundle.target_path.parent == root:
            _plan_body_rename(bundle, plan)
    return plan


def _import_plan(paths: list[Path], root: Path, mappings: dict[str, dict[str, list[str]]]) -> Plan:
    plan = Plan()
    if root.exists() and not root.is_dir():
        raise LibraryError(f"library root is not a directory: {root}")
    existing = _scan_bundles(root) if root.is_dir() else []
    reserved = {_path_key(child) for child in root.iterdir()} if root.is_dir() else set()
    discovered: list[Bundle] = []
    for given in paths:
        source = given.expanduser().resolve(strict=False)
        if not source.exists() or not source.is_dir():
            raise LibraryError(f"input must be an existing bundle or parent directory: {given}")
        found = _scan_bundles(source)
        if not found:
            raise LibraryError(f"no bundle with an eligible top-level Markdown body found under: {given}")
        for bundle in found:
            _attach_seed_mapping(bundle, mappings)
            if not any(_path_key(bundle.source_path) == _path_key(item.source_path) for item in discovered):
                discovered.append(bundle)
    for bundle in discovered:
        if _inside(bundle.source_path, root):
            bundle.target_path = bundle.source_path
            if not any(_path_key(bundle.source_path) == _path_key(item.source_path) for item in existing):
                existing.append(bundle)
            plan.skipped.append(f"already inside library; not copied: {bundle.source_path}")
            continue
        match = next((item for item in existing if _is_exact(bundle.record(root), item.record(root))), None)
        if match:
            plan.skipped.append(f"exact bundle already exists; source retained and not copied: {bundle.source_path}")
            continue
        different = next((item for item in existing if _is_alternate(bundle.record(root), item.record(root))), None)
        if different:
            plan.warnings.append(f"same paper / different extraction retained for human choice: {bundle.source_path}")
            existing.append(bundle)
            continue
        possible = [item for item in existing if _relation(bundle.record(root), item.record(root)) == "possible"]
        if possible:
            plan.warnings.append(f"possible duplicate, source left unchanged: {bundle.source_path}")
            continue
        destination = _allocate_destination(root, bundle.title, bundle, reserved)
        bundle.target_path = destination
        bundle.previous_folder_names.append(bundle.source_path.name)
        plan.actions.append(FsAction("copy", bundle.source_path, destination))
        existing.append(bundle)
        _plan_body_rename(bundle, plan)
    unique: dict[str, Bundle] = {}
    for bundle in existing:
        unique.setdefault(_path_key(bundle.target_path), bundle)
    plan.bundles = list(unique.values())
    return plan


def _refresh_plan(root: Path, mappings: dict[str, dict[str, list[str]]]) -> Plan:
    bundles = _scan_bundles(root)
    for bundle in bundles:
        _attach_seed_mapping(bundle, mappings)
    return Plan(bundles=bundles)


def _apply_actions(actions: list[FsAction], root: Path) -> None:
    seen_destinations: set[str] = set()
    for action in actions:
        if action.kind in {"move", "copy"} and not action.source.is_dir():
            raise LibraryError(f"planned source bundle disappeared: {action.source}")
        key = _path_key(action.destination)
        if key in seen_destinations:
            raise LibraryError(f"two planned operations target the same path: {action.destination}")
        seen_destinations.add(key)
        if action.destination.exists() and _path_key(action.source) != key:
            raise LibraryError(f"destination already exists; refusing to overwrite: {action.destination}")
    if actions:
        root.mkdir(parents=True, exist_ok=True)
    for action in actions:
        if action.kind == "rename_file":
            if not action.source.is_file() or action.destination.exists():
                raise LibraryError(f"body rename source missing or destination occupied: {action.source} -> {action.destination}")
            action.source.rename(action.destination)
            continue
        action.destination.parent.mkdir(parents=True, exist_ok=True)
        if action.kind == "move":
            shutil.move(str(action.source), str(action.destination))
        elif action.kind == "copy":
            shutil.copytree(action.source, action.destination, symlinks=True)
        else:
            raise LibraryError(f"unknown filesystem action: {action.kind}")


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temp_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def _render_markdown(index: dict[str, Any]) -> str:
    lines = [
        "# Paper Library Index", "",
        "> 由 `human/paper_library.py` 生成；JSON 是机器事实源。论文实体不进入 Git。",
        "> 正文读入口 Markdown；表格数值优先 Tables/*.xlsx，图读 Figure/*.jpg，公式优先 Formula/*_formula.md 并核对图片；Word 仅辅助。", "",
        f"- 主库（本机）：`{index['library_root']}`",
        f"- Canonical records：{len(index['papers'])}", "",
        "| Preferred ID / LIB | 标题 | 人工编号 | 年份 / DOI / arXiv | 一级目录 / 正文入口 | 图 / 结构表 / 渲染表 / LaTeX / 公式图 | 标签 / 角色 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for paper in index["papers"]:
        human = paper.get("human", {})
        manual = human.get("manual_ids", [])
        preferred = human.get("preferred_id") or (manual[0] if manual else paper["lib_id"])
        identity = "; ".join(str(value) for value in
                             (paper.get("year"), paper.get("doi"), paper.get("arxiv")) if value) or "—"
        flags = paper.get("content_capabilities", {})
        caps = " / ".join("Y" if flags.get(key) else "—" for key in
                          ("figures", "tables_structured", "tables_rendered", "formulas_latex", "formulas_image"))
        tags = human.get("tags", [])
        roles = human.get("project_roles", [])
        notes = ", ".join(str(x) for x in [*(tags if isinstance(tags, list) else [tags]),
                                           *(roles if isinstance(roles, list) else [roles])] if x) or "—"
        safe = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {safe(preferred)} / {paper['lib_id']} | {safe(paper['title'])} | "
                     f"{safe(', '.join(manual) or '—')} | {safe(identity)} | "
                     f"`{safe(paper['folder'])}` / `{safe(paper['entry_md'])}` | {caps} | {safe(notes)} |")
    lines.extend(["", "## 待人工确认的其他抽取", ""])
    for paper in index["papers"]:
        for alternate in paper.get("alternate_bundles", []):
            lines.append(f"- {paper['lib_id']} / {paper['title']}: `{alternate['bundle_path']}`（不自动合并或选优）")
    if not any(paper.get("alternate_bundles") for paper in index["papers"]):
        lines.append("无。")
    lines.extend(["", "## Exact bundle duplicate review", ""])
    for item in index.get("duplicate_review", []):
        lines.append(f"- {item.get('canonical_lib_id') or '待定位'}: `{item['original_path']}` → "
                     f"`{item.get('review_path') or '原处保留'}`；正文 SHA-256 `{item.get('sha256') or '未知'}`。")
    if not index.get("duplicate_review"):
        lines.append("无。")
    lines.append("")
    return "\n".join(lines)


def _write_indexes(index: dict[str, Any]) -> None:
    public = {**index, "papers": [_external_record(paper, paper["lib_id"]) for paper in index["papers"]]}
    json_text = json.dumps(public, ensure_ascii=False, indent=2) + "\n"
    md_text = _render_markdown(index)
    # Write both temporary files before replacing either index to catch serialization/disk errors early.
    INDEX_JSON.parent.mkdir(parents=True, exist_ok=True)
    INDEX_MD.parent.mkdir(parents=True, exist_ok=True)
    temp_files: list[tuple[str, Path]] = []
    try:
        for path, text in ((INDEX_JSON, json_text), (INDEX_MD, md_text)):
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent,
                                             prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
                temp_files.append((handle.name, path))
        for temp_name, destination in temp_files:
            os.replace(temp_name, destination)
    finally:
        for temp_name, _destination in temp_files:
            if os.path.exists(temp_name):
                os.unlink(temp_name)


def _print_plan(plan: Plan, *, dry_run: bool, root: Path) -> None:
    prefix = "WOULD" if dry_run else "WILL"
    for action in plan.actions:
        print(f"{prefix} {action.kind}: {action.source} -> {action.destination}")
    for item in plan.skipped:
        print(f"SKIP: {item}")
    for warning in plan.warnings:
        print(f"WARNING: {warning}")
    if not plan.actions:
        print("No bundle copy or move is planned.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan and maintain whole paper bundles. With no paths, scan the default library and update both indexes. "
            "Use --bootstrap once to normalize the main library and copy missing read-only legacy bundles."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("paths", nargs="*", type=Path,
                        help="bundle directory or parent directory containing bundles to import")
    parser.add_argument("--bootstrap", action="store_true",
                        help="first-time consolidation from the main library and read-only legacy archive")
    parser.add_argument("--dry-run", action="store_true", help="show planned copies, moves and index updates only")
    parser.add_argument("--root", type=Path, default=LIBRARY_ROOT,
                        help="external paper bundle library root")
    parser.add_argument("--legacy-root", type=Path, default=LEGACY_ROOT,
                        help="read-only archive source used by --bootstrap")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.bootstrap and args.paths:
        parser.error("--bootstrap cannot be combined with explicit import paths")
    root = args.root.expanduser().resolve(strict=False)
    legacy_root = args.legacy_root.expanduser().resolve(strict=False)
    try:
        old_index = _load_index(INDEX_JSON, root)
        if args.bootstrap and old_index and old_index.get("bootstrap_complete"):
            raise LibraryError("bootstrap is already marked complete; use the normal scan/import mode")
        seed_mappings = _parse_seed_mappings(PAPER_INDEX_MD, root, legacy_root)
        if args.bootstrap:
            plan = _bootstrap_plan(root, legacy_root, seed_mappings)
        elif args.paths:
            plan = _import_plan(args.paths, root, seed_mappings)
        else:
            plan = _refresh_plan(root, seed_mappings)

        index, sync_warnings = _sync_index(
            plan.bundles,
            root,
            old_index,
            plan.duplicate_records,
            bootstrap_complete=plan.bootstrap_complete,
        )
        plan.warnings.extend(sync_warnings)
        _print_plan(plan, dry_run=args.dry_run, root=root)
        if args.dry_run:
            print(f"WOULD update {INDEX_JSON} and {INDEX_MD}; indexed bundles: {len(plan.bundles)}")
            return 0
        _apply_actions(plan.actions, root)
        _write_indexes(index)
        print(
            f"Completed: {len(plan.bundles)} bundle(s) scanned/imported; "
            f"{len(plan.actions)} whole-bundle operation(s); "
            f"{len(index['papers'])} LIB record(s)."
        )
        manual_ids = {code for paper in index["papers"] for code in paper.get("human", {}).get("manual_ids", [])}
        alternates = sum(len(paper.get("alternate_bundles", [])) for paper in index["papers"])
        print(f"Canonical papers: {len(index['papers'])}; alternate extractions: {alternates}; "
              f"confirmed manual IDs: {len(manual_ids)}; exact duplicate records: {len(index['duplicate_review'])}")
        print(f"Updated: {INDEX_JSON}")
        print(f"Updated: {INDEX_MD}")
        return 0
    except (LibraryError, OSError, shutil.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
