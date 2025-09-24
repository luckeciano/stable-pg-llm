#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

def _is_int_like(x):
    if isinstance(x, int):
        return True
    if isinstance(x, float):
        return float(x).is_integer()
    if isinstance(x, str):
        try:
            int(x)
            return True
        except Exception:
            return False
    return False

def _to_int(x) -> int:
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    if isinstance(x, str):
        return int(x.strip())
    raise ValueError("not int-like")

def _compile_matcher(patterns: List[str], case_sensitive: bool):
    if not patterns:
        return None
    compiled = []
    flags = 0 if case_sensitive else re.IGNORECASE
    for p in patterns:
        if p.startswith("re:"):
            try:
                compiled.append(("re", re.compile(p[3:], flags)))
            except re.error as e:
                print(f"[WARN] invalid regex '{p}': {e}", file=sys.stderr)
        else:
            compiled.append(("sub", p if case_sensitive else p.lower()))
    def _match(text: str) -> bool:
        t = text if case_sensitive else text.lower()
        for kind, obj in compiled:
            if kind == "re":
                if obj.search(text):
                    return True
            else:
                if obj in t:
                    return True
        return False
    return _match

def _any_dir_matches(root: str, dirpath: str, matcher) -> bool:
    if matcher is None:
        return True
    rel = os.path.relpath(dirpath, root)
    if rel == ".":
        return False
    parts = rel.split(os.sep)
    for i, part in enumerate(parts, 1):
        sub = os.path.join(*parts[:i])
        if matcher(part) or matcher(sub):
            return True
    return matcher(rel)

def _collect_files(root: str, prefix: str,
                   dir_include: List[str], dir_exclude: List[str],
                   dir_case_sensitive: bool) -> List[str]:
    out = []
    inc_match = _compile_matcher(dir_include, dir_case_sensitive)
    exc_match = _compile_matcher(dir_exclude, dir_case_sensitive)
    for dirpath, _, filenames in os.walk(root):
        if exc_match and _any_dir_matches(root, dirpath, exc_match):
            continue
        if inc_match and not _any_dir_matches(root, dirpath, inc_match):
            continue
        for fn in filenames:
            if os.path.basename(fn).startswith(prefix):
                out.append(os.path.join(dirpath, fn))
    return out

def _load_freq_dict(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("JSON is not an object/dict")
    freq: Dict[str, int] = {}
    for k, v in data.items():
        if not isinstance(k, str):
            k = str(k)
        if _is_int_like(v):
            try:
                freq[k] = _to_int(v)
            except Exception:
                print(f"[WARN] {path}: value for key '{k}' not convertible to int -> skipped", file=sys.stderr)
        else:
            print(f"[WARN] {path}: value for key '{k}' is not int-like -> skipped", file=sys.stderr)
    return freq

def _merge_dicts(paths: List[str]) -> Tuple[Dict[str, int], List[str]]:
    merged = defaultdict(int)
    loaded_ok = []
    for p in paths:
        try:
            d = _load_freq_dict(p)
            for k, v in d.items():
                merged[k] += v
            loaded_ok.append(p)
        except Exception as e:
            print(f"[SKIP] {p}: {e}", file=sys.stderr)
    # sort by value desc, then key asc
    items = sorted(merged.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    return dict(items), loaded_ok

def _compute_norm_items(masked: Dict[str, int], allfreq: Dict[str, int], min_total_freq: int) -> List[Tuple[str, float, float, int, int]]:
    """
    Build a list of (key, masked_over_all_ratio, total_ratio_in_all, masked_count, all_count)
    for keys in both dicts, all_count>0, and all_count >= min_total_freq.

    - masked_over_all_ratio = masked_count / all_count
    - total_ratio_in_all    = all_count / sum(all_counts)

    Sorted by ratio ↓, masked ↓, all ↓ (tie-break key asc).
    """
    total_all = sum(allfreq.values())
    if total_all == 0:
        print("[WARN] norm: sum of 'all' counts is zero; total ratios will be 0.", file=sys.stderr)

    items: List[Tuple[str, float, float, int, int]] = []
    for k, mv in masked.items():
        if k not in allfreq:
            print(f"[WARN] norm: key present only in masked: '{k}'", file=sys.stderr)
            continue
        denom = allfreq[k]
        if denom == 0:
            print(f"[WARN] norm: zero denominator for key '{k}' -> skipped", file=sys.stderr)
            continue
        if denom < min_total_freq:
            continue  # filter by TOTAL count
        ratio = float(mv) / float(denom)
        total_ratio = (float(denom) / float(total_all)) if total_all > 0 else 0.0
        items.append((k, ratio, total_ratio, mv, denom))

    # stable multi-key sort: key asc, then all desc, masked desc, ratio desc
    items.sort(key=lambda t: t[0])                       # key asc
    items.sort(key=lambda t: t[4], reverse=True)         # all_count desc
    items.sort(key=lambda t: t[3], reverse=True)         # masked_count desc
    items.sort(key=lambda t: t[1], reverse=True)         # ratio desc
    return items

def _find_experiment_directories(root: str) -> List[str]:
    """Find experiment directories that contain mask_data files."""
    experiment_dirs = []
    for item in os.listdir(root):
        item_path = os.path.join(root, item)
        if os.path.isdir(item_path):
            # Check if this directory contains mask_data files
            has_mask_data = False
            for root_dir, _, files in os.walk(item_path):
                for file in files:
                    if file.startswith("masked_token_freq") or file.startswith("all_token_freq"):
                        has_mask_data = True
                        break
                if has_mask_data:
                    break
            if has_mask_data:
                experiment_dirs.append(item_path)
    return sorted(experiment_dirs)

def _process_experiment_directory(exp_dir: str, out_dir: str, args) -> Tuple[Dict[str, int], Dict[str, int], List[str]]:
    """Process a single experiment directory and return merged data."""
    exp_name = os.path.basename(exp_dir)
    exp_out_dir = os.path.join(out_dir, exp_name)
    os.makedirs(exp_out_dir, exist_ok=True)
    
    print(f"[INFO] Processing experiment directory: {exp_name}")
    
    # Collect files for this experiment
    all_files = _collect_files(exp_dir, args.all_prefix, args.dir_include, args.dir_exclude, args.dir_case_sensitive)
    masked_files = _collect_files(exp_dir, args.masked_prefix, args.dir_include, args.dir_exclude, args.dir_case_sensitive)
    
    if not all_files and not masked_files:
        print(f"[INFO] No files found in {exp_dir}", file=sys.stderr)
        return {}, {}, []
    
    # Merge files for this experiment
    all_merged, all_ok = _merge_dicts(all_files)
    masked_merged, masked_ok = _merge_dicts(masked_files)
    
    # Compute normalized items
    norm_items = _compute_norm_items(masked_merged, all_merged, args.min_freq)
    
    # Write outputs for this experiment
    indent = 2 if args.pretty else None
    all_out = os.path.join(exp_out_dir, f"{args.all_prefix}_merged.json")
    masked_out = os.path.join(exp_out_dir, f"{args.masked_prefix}_merged.json")
    norm_out = os.path.join(exp_out_dir, "norm_freqs.json")
    
    if all_ok:
        with open(all_out, "w", encoding="utf-8") as f:
            json.dump(all_merged, f, ensure_ascii=False, indent=indent)
        print(f"[DONE] Wrote {all_out} from {len(all_ok)} files")
    else:
        print(f"[SKIP] No '{args.all_prefix}' files merged in {exp_name}.", file=sys.stderr)
    
    if masked_ok:
        with open(masked_out, "w", encoding="utf-8") as f:
            json.dump(masked_merged, f, ensure_ascii=False, indent=indent)
        print(f"[DONE] Wrote {masked_out} from {len(masked_ok)} files")
    else:
        print(f"[SKIP] No '{args.masked_prefix}' files merged in {exp_name}.", file=sys.stderr)
    
    # Write normalized file for this experiment
    with open(norm_out, "w", encoding="utf-8") as f:
        f.write("{\n")
        for i, (k, ratio, total_ratio, mv, denom) in enumerate(norm_items):
            key_json = json.dumps(k, ensure_ascii=False)
            val_json = f"[{ratio}, {total_ratio}, {mv}, {denom}]"
            trailing = "," if i < len(norm_items) - 1 else ""
            f.write(f"  {key_json}: {val_json}{trailing}\n")
        f.write("}\n")
    print(f"[DONE] Wrote {norm_out} with {len(norm_items)} normalized entries")
    
    return all_merged, masked_merged, norm_items

def main():
    ap = argparse.ArgumentParser(
        description="Merge token-frequency JSONs under multiple experiment directories; output merged dicts and normalized ratios for each experiment and globally."
    )
    ap.add_argument("--artifact-dir", required=True, help="Path to the root folder containing experiment directories.")
    ap.add_argument("--out", default=None, help="Output directory (default: same as --artifact-dir).")
    ap.add_argument("--all-prefix", default="all_token_freq", help="Prefix for 'all token freq' files (default: all_token_freq).")
    ap.add_argument("--masked-prefix", default="masked_token_freq", help="Prefix for 'masked token freq' files (default: masked_token_freq).")
    ap.add_argument("--dir-include", action="append", default=[],
                    help="Folder include filter (repeatable). Substring or regex if prefixed with 're:'. If provided, only matching folders are considered.")
    ap.add_argument("--dir-exclude", action="append", default=[],
                    help="Folder exclude filter (repeatable). Substring or regex if prefixed with 're:'. Matching folders are skipped.")
    ap.add_argument("--dir-case-sensitive", action="store_true", help="Case-sensitive folder matching (default false).")
    ap.add_argument("--min-freq", type=int, default=0,
                    help="Minimum TOTAL frequency (all_count) required to include a token in normalized output (default 0).")
    ap.add_argument("--pretty", action="store_true",
                    help="Pretty-print merged JSONs (not used for norm file, which is one-key-per-line).")
    ap.add_argument("--skip-individual", action="store_true",
                    help="Skip processing individual experiment directories, only create global merge.")
    ap.add_argument("--skip-global", action="store_true",
                    help="Skip creating global merge file, only process individual experiments.")
    args = ap.parse_args()

    root = os.path.abspath(args.artifact_dir)
    out_dir = os.path.abspath(args.out or root)
    os.makedirs(out_dir, exist_ok=True)

    # Find experiment directories
    experiment_dirs = _find_experiment_directories(root)
    
    if not experiment_dirs:
        print(f"[ERROR] No experiment directories found in {root}", file=sys.stderr)
        return 1
    
    print(f"[INFO] Found {len(experiment_dirs)} experiment directories: {[os.path.basename(d) for d in experiment_dirs]}")
    
    # Global data to accumulate across all experiments
    global_all_merged = defaultdict(int)
    global_masked_merged = defaultdict(int)
    global_norm_items = []
    
    # Process each experiment directory
    if not args.skip_individual:
        for exp_dir in experiment_dirs:
            all_merged, masked_merged, norm_items = _process_experiment_directory(exp_dir, out_dir, args)
            
            # Accumulate for global merge
            for k, v in all_merged.items():
                global_all_merged[k] += v
            for k, v in masked_merged.items():
                global_masked_merged[k] += v
    
    # Create global merge file
    if not args.skip_global:
        print(f"[INFO] Creating global merge across all {len(experiment_dirs)} experiments...")
        
        # Convert defaultdicts to regular dicts and sort
        global_all_dict = dict(sorted(global_all_merged.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))
        global_masked_dict = dict(sorted(global_masked_merged.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))
        
        # Compute global normalized items
        global_norm_items = _compute_norm_items(global_masked_dict, global_all_dict, args.min_freq)
        
        # Write global outputs
        indent = 2 if args.pretty else None
        global_all_out = os.path.join(out_dir, f"global_{args.all_prefix}_merged.json")
        global_masked_out = os.path.join(out_dir, f"global_{args.masked_prefix}_merged.json")
        global_norm_out = os.path.join(out_dir, "global_norm_freqs.json")
        
        if global_all_dict:
            with open(global_all_out, "w", encoding="utf-8") as f:
                json.dump(global_all_dict, f, ensure_ascii=False, indent=indent)
            print(f"[DONE] Wrote global {global_all_out} from all experiments")
        else:
            print(f"[SKIP] No global '{args.all_prefix}' data merged.", file=sys.stderr)
        
        if global_masked_dict:
            with open(global_masked_out, "w", encoding="utf-8") as f:
                json.dump(global_masked_dict, f, ensure_ascii=False, indent=indent)
            print(f"[DONE] Wrote global {global_masked_out} from all experiments")
        else:
            print(f"[SKIP] No global '{args.masked_prefix}' data merged.", file=sys.stderr)
        
        # Write global normalized file
        with open(global_norm_out, "w", encoding="utf-8") as f:
            f.write("{\n")
            for i, (k, ratio, total_ratio, mv, denom) in enumerate(global_norm_items):
                key_json = json.dumps(k, ensure_ascii=False)
                val_json = f"[{ratio}, {total_ratio}, {mv}, {denom}]"
                trailing = "," if i < len(global_norm_items) - 1 else ""
                f.write(f"  {key_json}: {val_json}{trailing}\n")
            f.write("}\n")
        print(f"[DONE] Wrote global {global_norm_out} with {len(global_norm_items)} normalized entries")
    
    print(f"[INFO] Processing complete!")
    return 0

if __name__ == "__main__":
    main()
