# utils/delete_file.py

from pathlib import Path


def clear_all_files(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        return
    for f in folder.glob('**/*'):
        if f.is_file():
            f.unlink()
