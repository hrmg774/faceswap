# utils/rename_file.py

from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import time

TZ_HANOI = timezone(timedelta(hours=7))


def _is_formatted(stem: str, prefix: str) -> bool:
    return bool(re.match(rf'^{re.escape(prefix)}_\d{{8}}_\d{{6}}_\d{{2}}$', stem))


def _unique_path(folder: Path, prefix: str, time_str: str, ext: str) -> Path:
    counter = 1
    while True:
        candidate = folder / f'{prefix}_{time_str}_{counter:02d}{ext}'
        if not candidate.exists():
            return candidate
        counter += 1


def _pending_files(folder: Path) -> list:
    prefix = folder.name
    return sorted(
        [f for f in folder.iterdir()
         if f.is_file()
         and not _is_formatted(f.stem, prefix)],
        key=lambda x: x.stat().st_mtime,
    )


def rename_by_current_time(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        return
    prefix = folder.name
    for f in _pending_files(folder):
        time_str = datetime.now(TZ_HANOI).strftime('%Y%m%d_%H%M%S')
        f.rename(_unique_path(folder, prefix, time_str, f.suffix.lower()))
        time.sleep(0.01)


def rename_by_history(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        return
    prefix = folder.name

    files = [(f, f.stat().st_mtime)
             for f in sorted(folder.iterdir(), key=lambda x: x.stat().st_mtime)
             if f.is_file()]
    if not files:
        return

    # Pass 1: rename to temp to free all existing name slots
    temp_files = []
    for i, (f, mtime) in enumerate(files):
        tmp = folder / f'__tmp_{i:04d}{f.suffix}'
        f.rename(tmp)
        temp_files.append((tmp, mtime))

    # Pass 2: rename to final names, counter resets from _01 per timestamp
    counters = {}
    for tmp, mtime in temp_files:
        dt = datetime.fromtimestamp(mtime, tz=TZ_HANOI)
        time_str = dt.strftime('%Y%m%d_%H%M%S')
        counters[time_str] = counters.get(time_str, 0) + 1
        final = folder / \
            f'{prefix}_{time_str}_{counters[time_str]:02d}{tmp.suffix.lower()}'
        tmp.rename(final)
