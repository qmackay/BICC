##FULLY AI GENERATED

#!/usr/bin/env python3
"""Normalize tiepoint folder/file orientation using a core priority order.

Dry-run by default. Use --apply to make changes.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

def load_core_order(root: Path) -> list[str]:
    """Load list_sites from parameters.yml in the root directory."""
    params_path = root / "parameters.yml"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing parameters file: {params_path}")

    for line in params_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("list_sites:"):
            continue

        value = stripped.split(":", 1)[1].strip()
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
            raise ValueError(f"Invalid list_sites format in {params_path}")
        return parsed

    raise ValueError(f"Could not find list_sites in {params_path}")


def canonical_pair(core_a: str, core_b: str, core_rank: dict[str, int]) -> tuple[str, str]:
    if core_rank[core_a] <= core_rank[core_b]:
        return core_a, core_b
    return core_b, core_a


def swap_depth_columns(file_path: Path, apply: bool) -> bool:
    """Swap first two columns in data rows. Returns True if changed."""
    lines = file_path.read_text(encoding="utf-8").splitlines(keepends=True)
    changed = False
    output: list[str] = []
    seen_header = False

    for line in lines:
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            output.append(line)
            continue

        if not seen_header and stripped.lower().startswith("depth1") and "depth2" in stripped.lower():
            seen_header = True
            output.append(line)
            continue

        if seen_header:
            newline = "\n" if line.endswith("\n") else ""
            data = line.rstrip("\n")
            parts = data.split("\t")
            if len(parts) >= 2:
                parts[0], parts[1] = parts[1], parts[0]
                swapped = "\t".join(parts) + newline
                output.append(swapped)
                if swapped != line:
                    changed = True
                continue

            ws_parts = data.strip().split()
            if len(ws_parts) >= 2:
                ws_parts[0], ws_parts[1] = ws_parts[1], ws_parts[0]
                swapped = "\t".join(ws_parts) + newline
                output.append(swapped)
                if swapped != line:
                    changed = True
                continue

        output.append(line)

    if changed and apply:
        file_path.write_text("".join(output), encoding="utf-8")

    return changed


def rename_with_collision_check(src: Path, dst: Path, apply: bool) -> None:
    if src == dst:
        return
    if dst.exists():
        raise FileExistsError(f"Cannot rename {src} -> {dst}: destination exists.")
    if apply:
        src.rename(dst)


def process_folder(folder: Path, apply: bool, core_rank: dict[str, int]) -> tuple[int, int, int]:
    """Returns (folders_changed, files_renamed, files_rewritten)."""
    folder_name = folder.name
    if "-" not in folder_name:
        return 0, 0, 0

    parts = folder_name.split("-")
    if len(parts) != 2:
        return 0, 0, 0

    core_a, core_b = parts
    if core_a not in core_rank or core_b not in core_rank:
        return 0, 0, 0

    first, second = canonical_pair(core_a, core_b, core_rank)
    if (first, second) == (core_a, core_b):
        return 0, 0, 0

    old_pair = f"{core_a}-{core_b}"
    new_pair = f"{first}-{second}"
    files_renamed = 0
    files_rewritten = 0

    for txt_file in sorted(folder.glob("*.txt")):
        changed = swap_depth_columns(txt_file, apply=apply)
        if changed:
            files_rewritten += 1

        new_name = txt_file.name.replace(old_pair, new_pair)
        if new_name != txt_file.name:
            rename_with_collision_check(txt_file, txt_file.with_name(new_name), apply=apply)
            files_renamed += 1

    rename_with_collision_check(folder, folder.with_name(new_pair), apply=apply)
    return 1, files_renamed, files_rewritten


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Normalize tiepoint folder/file orientation by core priority."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=str(script_dir),
        help="Root folder containing core-pair subfolders (default: script directory).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes. Without this flag, script runs in dry-run mode.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    core_order = load_core_order(root)
    core_rank = {core: i for i, core in enumerate(core_order)}

    folders_changed = 0
    files_renamed = 0
    files_rewritten = 0

    print(f"{'APPLY' if args.apply else 'DRY-RUN'} mode")
    print(f"Root: {root}")
    print(f"core order: {core_order}")

    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        changed_folders, renamed, rewritten = process_folder(
            folder, apply=args.apply, core_rank=core_rank
        )
        if changed_folders:
            old_name = folder.name
            core_a, core_b = old_name.split("-")
            first, second = canonical_pair(core_a, core_b, core_rank)
            print(f"flip folder: {old_name} -> {first}-{second}")
            folders_changed += changed_folders
            files_renamed += renamed
            files_rewritten += rewritten

    print("---")
    print(f"folders flipped: {folders_changed}")
    print(f"files renamed: {files_renamed}")
    print(f"files depth-swapped: {files_rewritten}")


if __name__ == "__main__":
    main()
