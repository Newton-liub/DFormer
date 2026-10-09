"""CPU-only asset/environment inspection and optional SwanLab preparation.

No model construction, GPU operation, download or training is performed here.
Useful on both the local machine and the existing no-GPU cloud instance.
"""
import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace


def inspect_assets(data_root):
    data_root = Path(data_root)
    report = {"python": sys.version.split()[0], "dataset_root": str(data_root)}
    existing_parent = data_root
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    usage = shutil.disk_usage(existing_parent)
    report["disk_gib"] = {"total": round(usage.total / 2**30, 2), "free": round(usage.free / 2**30, 2)}
    report["packages"] = {}
    for name in ("torch", "mmcv", "timm", "swanlab"):
        try:
            package = importlib.import_module(name)
            report["packages"][name] = getattr(package, "__version__", "unknown")
        except ImportError as error:
            report["packages"][name] = str(error)
    report["datasets"] = {}
    for name, label_folder in (("SUNRGBD", "labels"), ("NYUDepthv2", "Label")):
        root = data_root / name
        asset = {"exists": root.is_dir()}
        if root.is_dir():
            asset["folders"] = {folder: sum(1 for p in (root / folder).iterdir() if p.is_file()) if (root / folder).is_dir() else None for folder in ("RGB", "Depth", label_folder)}
            asset["splits"] = {split: len((root / split).read_text(encoding="utf-8").splitlines()) if (root / split).is_file() else None for split in ("train.txt", "test.txt")}
        report["datasets"][name] = asset
    downloads = data_root / "downloads"
    archives = list(data_root.glob("*.zip")) + (list(downloads.glob("*.zip")) if downloads.is_dir() else [])
    report["local_archives"] = [{"path": str(p), "bytes": p.stat().st_size} for p in archives if "SUN" in p.name.upper() or "NYU" in p.name.upper()]
    pretrained = Path("checkpoints/pretrained/DFormerv2_Small_pretrained.pth")
    report["pretrained"] = {"exists": pretrained.is_file(), "bytes": pretrained.stat().st_size if pretrained.is_file() else None}
    report["swanlab_login_store_exists"] = (Path.home() / ".swanlab" / ".netrc").is_file()
    return report


def monitoring_check(mode):
    from copy import deepcopy
    from local_configs.research.DFormerv2_S_SUNRGBD import C
    from research.tracking import start_tracking
    config = deepcopy(C)
    config.experiment_name = "odg-cpu-preparation"
    config.swanlab_mode = mode
    config.log_dir = "outputs/odg-preparation/monitor"
    config.swanlab_log_dir = config.log_dir + "/swanlab"
    config.swanlab_enabled = True
    config.geometry_mode = "odg"
    args = SimpleNamespace(cpu_preparation=True, geometry_mode="odg")
    run = start_tracking(config, args, SimpleNamespace(distributed=False))
    try:
        run.log({"preparation/cpu_only": 1, "preparation/gpu_training_performed": 0}, step=0)
        public_url = getattr(run, "public_url", None)
        if callable(public_url):
            public_url = public_url()
        return {"mode": mode, "public_url": str(public_url) if public_url else None, "log_dir": config.swanlab_log_dir}
    finally:
        run.finish()


def transfer_probe(archive, output):
    """Copy only 8 MiB of the existing archive for a short transfer measurement."""
    archive, output = Path(archive), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("rb") as source, output.open("xb") as target:
        target.write(source.read(8 * 2**20))
    return {"probe": str(output), "probe_bytes": output.stat().st_size,
            "archive_bytes": archive.stat().st_size}


def unpack_sun(archive, data_root):
    """Extract one inspected SUN archive to an absent target, without overwrites."""
    import zipfile
    root = Path(data_root).resolve()
    target = root / "SUNRGBD"
    if target.exists():
        raise FileExistsError("SUNRGBD already exists; reuse/check it instead of overwriting")
    with zipfile.ZipFile(archive) as package:
        members = package.infolist()
        for item in members:
            destination = (root / item.filename).resolve()
            if not destination.is_relative_to(target) or "\\\\" in item.filename:
                raise ValueError("Unexpected archive member: " + item.filename)
        required = sum(item.file_size for item in members)
        if shutil.disk_usage(root).free < required + 2**30:
            raise RuntimeError("Insufficient space for extraction plus 1 GiB reserve")
        package.extractall(root)
    return {"extracted_to": str(target), "uncompressed_bytes": required}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--monitor", choices=("online", "offline", "disabled"))
    parser.add_argument("--transfer-probe", metavar="SUN_ARCHIVE")
    parser.add_argument("--probe-output", default="outputs/odg-preparation/tmp/odg-transfer-probe-20261010.bin")
    parser.add_argument("--unpack-sun", metavar="SUN_ARCHIVE")
    parser.add_argument("--discard-probe", default=False, action="store_true")
    args = parser.parse_args()
    if args.transfer_probe:
        print(json.dumps(transfer_probe(args.transfer_probe, args.probe_output), indent=2))
    if args.unpack_sun:
        print(json.dumps(unpack_sun(args.unpack_sun, args.data_root), indent=2))
    if args.discard_probe:
        probe = Path(args.probe_output)
        if probe.name != "odg-transfer-probe-20261010.bin":
            raise ValueError("Only the named transfer probe may be discarded")
        probe.unlink(missing_ok=True)
    print(json.dumps(inspect_assets(args.data_root), ensure_ascii=False, indent=2))
    if args.monitor:
        print(json.dumps({"monitoring": monitoring_check(args.monitor)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
