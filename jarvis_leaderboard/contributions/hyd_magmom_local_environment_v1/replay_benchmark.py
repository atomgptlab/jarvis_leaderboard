"""Reproduce frozen benchmark predictions without reading benchmark targets."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.request import urlopen
import zipfile

from predict_frozen import feature_matrix, predict

RAW_NAME = "jdft_3d-12-12-2022.json"
RAW_URL = "https://ndownloader.figshare.com/files/38521619"
RAW_SHA = "72da38713e509195220771b83b07cbd2d21f5d0d5cc203e1067dd597465b0b35"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def raw_database(raw_json, cache):
    cache.mkdir(parents=True, exist_ok=True)
    path = raw_json or cache / RAW_NAME
    if not path.exists():
        if raw_json:
            raise FileNotFoundError(path)
        archive_path = cache / "raw-source.zip"
        with urlopen(RAW_URL, timeout=120) as response, archive_path.open("wb") as out:
            while block := response.read(1024 * 1024):
                out.write(block)
        with zipfile.ZipFile(archive_path) as archive:
            path.write_bytes(archive.read(RAW_NAME))
    if digest(path) != RAW_SHA:
        raise ValueError("Raw structure database differs from the pinned training source")
    return path


def write_structures(ids, raw_path, destination):
    if not isinstance(ids, list) or len(ids) != len(set(ids)) or not all(isinstance(jid, str) for jid in ids):
        raise ValueError("Expected unique ordered IDs only")
    wanted = set(ids)
    from prepare_data import clean_atoms
    by_id = {record["jid"]: clean_atoms(record) for record in json.loads(raw_path.read_text())
             if record["jid"] in wanted}
    if set(by_id) != wanted:
        raise ValueError("Some requested structures are absent from the pinned database")
    with destination.open("w") as out:
        for jid in ids:
            out.write(json.dumps({"id": jid, "atoms": by_id[jid]}, separators=(",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True, help="Ordered IDs only; no target values")
    parser.add_argument("--raw-json", type=Path)
    parser.add_argument("--cache", type=Path, default=Path(".replay-cache"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_path = raw_database(args.raw_json, args.cache)
    ids = json.loads(args.ids.read_text())
    # The raw database contains computed properties, but only atoms are retained.
    structures = args.cache / "inference-structures.jsonl"
    write_structures(ids, raw_path, structures)
    selection = json.loads(args.selection.read_text())
    started = time.monotonic()
    predicted_ids, x, names = feature_matrix(structures, selection["components"])
    if predicted_ids.tolist() != ids:
        raise ValueError("Inference row order changed")
    probabilities, predictions = predict(args.selection, predicted_ids, x, names)
    import csv
    csv_path = args.output.with_suffix("")
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "prediction"])
        writer.writerows(zip(ids, predictions.tolist()))
    info = zipfile.ZipInfo(csv_path.name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(args.output, "w") as archive:
        archive.writestr(info, csv_path.read_bytes(), compresslevel=9)
    report = {"rows": len(ids), "inference_seconds": time.monotonic()-started,
              "raw_sha256": RAW_SHA, "ids_sha256": digest(args.ids),
              "selection_sha256": digest(args.selection), "csv_sha256": digest(csv_path),
              "zip_sha256": digest(args.output), "benchmark_labels_read": False}
    args.output.with_name("replay-evidence.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
