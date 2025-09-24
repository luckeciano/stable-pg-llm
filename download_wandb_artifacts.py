#!/usr/bin/env python3
import argparse
import os
import sys
from typing import List, Dict

def _safe(s: str) -> str:
    # make a filesystem-friendly string
    return "".join(c if c.isalnum() or c in ("-", "_", ".", ":") else "_" for c in s)

def _artifact_type_name(artifact) -> str:
    # W&B sometimes returns a type object; normalize to plain name.
    t = getattr(artifact, "type", None)
    if t is None:
        return ""
    return getattr(t, "name", t)

def _run_sort_key(run) -> float:
    """
    Robust timestamp for sorting runs (newest first).
    Tries several attribute names and falls back to created_at; returns a float.
    """
    for attr in ("updated_at", "updatedAt", "finished_at", "finishedAt",
                 "created_at", "createdAt"):
        v = getattr(run, attr, None)
        if v is not None:
            try:
                return float(v.timestamp())
            except Exception:
                pass
    return -1.0  # unknown

def _find_runs(api, entity: str, project: str, run_name: str):
    """
    Return ALL runs that match:
      1) display_name == run_name
      2) name == run_name
      3) (if resolvable) run id == run_name
    Deduplicates by run.id and sorts by most-recent based on a robust timestamp.
    """
    path = f"{entity}/{project}"
    found: Dict[str, object] = {}

    # 1) display_name
    try:
        for r in api.runs(path, filters={"display_name": run_name}):
            found[r.id] = r
    except Exception as e:
        print(f"[WARN] listing runs by display_name failed: {e}", file=sys.stderr)

    # 2) name
    try:
        for r in api.runs(path, filters={"name": run_name}):
            found[r.id] = r
    except Exception as e:
        print(f"[WARN] listing runs by name failed: {e}", file=sys.stderr)

    # 3) direct id
    try:
        r = api.run(f"{entity}/{project}/{run_name}")
        found[r.id] = r
    except Exception:
        pass

    runs = list(found.values())
    if not runs:
        print(f"[WARN] No runs found for '{run_name}' in {entity}/{project}.", file=sys.stderr)
        return []

    runs.sort(key=_run_sort_key, reverse=True)
    return runs

def _download_for_run(run, out_dir: str, file_filter: str, case_sensitive: bool, dry_run: bool=False) -> List[str]:
    """
    Download matching files from matching artifacts for a single run into:
      <out_dir>/<safe_run_label>__<run_id>/<artifact_name:version>/
    Only files whose name/path contains `file_filter` are saved.
    Returns list of local paths actually downloaded.
    """
    run_label = (getattr(run, "display_name", None) or getattr(run, "name", None) or run.id)
    run_dir = os.path.join(out_dir, f"{_safe(run_label)}__{run.id}")
    os.makedirs(run_dir, exist_ok=True)

    try:
        arts = list(run.logged_artifacts())
    except Exception as e:
        print(f"[SKIP] run {run.id} – cannot list artifacts: {e}", file=sys.stderr)
        return []

    if not arts:
        print(f"[SKIP] run {run.id} – no artifacts.", file=sys.stderr)
        return []

    # Filter: type == "mask_data" and base name startswith "masking_stats"
    def is_target(a) -> bool:
        a_type = _artifact_type_name(a)
        base_name = (a.name or "").split(":")[0]  # strip :vN
        return a_type == "mask_data" and base_name.startswith("masking_stats")

    targets = [a for a in arts if is_target(a)]
    if not targets:
        print(f"[SKIP] run {run.id} – no matching 'mask_data' artifacts with prefix 'masking_stats'.", file=sys.stderr)
        return []

    # Deterministic ordering: by base name then version number
    def sort_key(a):
        base = (a.name or "").split(":")[0]
        ver = getattr(a, "version", "")  # e.g., 'v12'
        try:
            vnum = int(ver[1:]) if ver.startswith("v") else -1
        except Exception:
            vnum = -1
        return (base, vnum)

    targets.sort(key=sort_key)

    # prepare matcher
    filt = file_filter if case_sensitive else file_filter.lower()
    def _match(path: str) -> bool:
        # match on basename or full path; case-insensitive by default
        p = path if case_sensitive else path.lower()
        base = os.path.basename(path)
        b = base if case_sensitive else base.lower()
        return (filt in p) or (filt in b)

    downloaded_paths: List[str] = []
    print(f"[RUN] {run.id}  name='{run_label}'  artifacts={len(targets)}")

    for art in targets:
        name = art.name or "artifact"
        atype = _artifact_type_name(art)

        # list files in artifact
        try:
            entries = list(getattr(art, "manifest").entries.keys())
        except Exception as e:
            print(f"  - {name} (type={atype})  [SKIP] cannot read manifest: {e}", file=sys.stderr)
            continue

        # apply filename filter
        matching = [p for p in entries if _match(p)]
        if not matching:
            print(f"  - {name} (type={atype})  [SKIP] no files matching '{file_filter}'")
            continue

        print(f"  - {name} (type={atype})  files={len(matching)} matching '{file_filter}'")

        # destination per artifact (use artifact name including :vN to avoid collisions)
        art_dir = os.path.join(run_dir, _safe(name))
        os.makedirs(art_dir, exist_ok=True)

        if dry_run:
            for p in matching:
                print(f"      * {p}")
            continue

        # download only the matching files
        for relpath in matching:
            try:
                ref = art.get_path(relpath)  # get a reference to a specific file
                local_path = ref.download(root=art_dir)  # returns the local file path
                downloaded_paths.append(local_path)
                print(f"      ✓ {relpath}")
            except Exception as e:
                print(f"      [SKIP] {relpath}: {e}", file=sys.stderr)

    if not downloaded_paths and not dry_run:
        print(f"[SKIP] run {run.id} – nothing downloaded.", file=sys.stderr)
    return downloaded_paths

def download_masking_stats(entity: str, project: str, run_name: str, out_dir: str,
                           file_filter: str = "token_freq", case_sensitive: bool = False,
                           dry_run: bool=False) -> List[str]:
    """
    Find ALL runs that match run_name and download matching files from their masking_stats artifacts.
    Returns a flat list of downloaded file paths (or directories when W&B returns folders).
    """
    import wandb
    api = wandb.Api(overrides={"entity": entity, "project": project})

    runs = _find_runs(api, entity, project, run_name)
    if not runs:
        return []

    os.makedirs(out_dir, exist_ok=True)

    all_downloaded: List[str] = []
    for run in runs:
        try:
            paths = _download_for_run(run, out_dir, file_filter=file_filter,
                                      case_sensitive=case_sensitive, dry_run=dry_run)
            all_downloaded.extend(paths)
        except Exception as e:
            print(f"[SKIP] run {getattr(run, 'id', '?')} – unexpected error: {e}", file=sys.stderr)
            continue

    return all_downloaded

def main():
    p = argparse.ArgumentParser(
        description="Download W&B 'mask_data' artifacts (prefix 'masking_stats') and filter files by substring."
    )
    p.add_argument("--wandb_entity", required=True, help="W&B entity (org/user)")
    p.add_argument("--wandb_project", required=True, help="W&B project")
    p.add_argument("--run_name", required=True, help="Run display name, run name, or run id")
    p.add_argument("--out", default="./wandb_artifacts", help="Output directory (default: ./wandb_artifacts)")
    p.add_argument("--file_filter", default="token_freq", help="Substring to match in filenames (default: 'token_freq')")
    p.add_argument("--case_sensitive", action="store_true", help="Make filename matching case-sensitive")
    p.add_argument("--dry-run", action="store_true", help="List runs, artifacts, and matching files without downloading")
    args = p.parse_args()

    try:
        paths = download_masking_stats(
            entity=args.wandb_entity,
            project=args.wandb_project,
            run_name=args.run_name,
            out_dir=args.out,
            file_filter=args.file_filter,
            case_sensitive=args.case_sensitive,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            if paths:
                print("\nDownloaded files:")
                for pth in paths:
                    print("  ", pth)
            else:
                print("\nNo files were downloaded.")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
