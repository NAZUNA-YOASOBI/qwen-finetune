from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def extract_hr(root: Path) -> None:
    archive_path = root / 'RSGPT-Simbench' / 'RSVQA-HR' / 'raw' / 'Images.tar'
    out_dir = root / 'RSGPT-Simbench' / 'RSVQA-HR' / 'images'
    out_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    skipped = 0
    total = 0
    with tarfile.open(archive_path, 'r') as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = str(member.name)
            if not name.startswith('Data/') or not name.endswith('.tif'):
                continue
            total += 1
            out_path = out_dir / Path(name).name
            if out_path.is_file():
                skipped += 1
                continue
            src = tf.extractfile(member)
            if src is None:
                continue
            out_path.write_bytes(src.read())
            extracted += 1
            if extracted % 500 == 0:
                print(f'[HR] extracted={extracted} skipped={skipped} total_seen={total}', flush=True)
    print(f'[HR] done extracted={extracted} skipped={skipped} total_tif_members={total}', flush=True)


def extract_lr(root: Path) -> None:
    archive_path = root / 'RSGPT-Simbench' / 'RSVQA-LR' / 'raw' / 'Images_LR.zip'
    out_dir = root / 'RSGPT-Simbench' / 'RSVQA-LR' / 'images'
    out_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    skipped = 0
    total = 0
    with zipfile.ZipFile(archive_path, 'r') as zf:
        for name in zf.namelist():
            if not name.startswith('Images_LR/') or not name.endswith('.tif'):
                continue
            total += 1
            out_path = out_dir / Path(name).name
            if out_path.is_file():
                skipped += 1
                continue
            out_path.write_bytes(zf.read(name))
            extracted += 1
    print(f'[LR] done extracted={extracted} skipped={skipped} total_tif_members={total}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Extract RSVQA images into flat image directories.')
    parser.add_argument('--datasets', type=str, default='all', choices=['all', 'hr', 'lr'])
    args = parser.parse_args()

    root = project_root()
    if args.datasets in {'all', 'hr'}:
        extract_hr(root)
    if args.datasets in {'all', 'lr'}:
        extract_lr(root)


if __name__ == '__main__':
    main()
