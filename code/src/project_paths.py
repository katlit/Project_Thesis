from pathlib import Path


def find_unique_dir(names, search_roots):
    """Find one directory by name below a small set of Drive roots."""
    names_folded = {name.casefold() for name in names}
    direct = [Path(root) / name for root in search_roots for name in names]
    matches = [path for path in direct if path.is_dir()]
    if not matches:
        for root in map(Path, search_roots):
            if root.is_dir():
                matches.extend(
                    path for path in root.rglob("*")
                    if path.is_dir() and path.name.casefold() in names_folded
                )
    unique = list(dict.fromkeys(path.resolve() for path in matches))
    if len(unique) != 1:
        raise FileNotFoundError(f"Expected one of {names}; found {len(unique)}: {unique}")
    return unique[0]


def discover_dataset_roots(project_root):
    """Return the IndustrialInventory and 3DRealCar/HQ200 roots."""
    project_root = Path(project_root)
    roots = [project_root / "data", project_root]
    industrial = find_unique_dir(["IndustrialInventory"], roots)
    hq200 = find_unique_dir(["3DrealCarHQ200", "HQ200"], roots)
    if (hq200 / "3DrealCarHQ200").is_dir():
        hq200 = hq200 / "3DrealCarHQ200"
    return industrial, hq200

