# utils/delete_file.py

from pathlib import Path


def clear_all_files(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        return

    # Quét tất cả các cấp, chỉ lấy file
    files = [f for f in folder.glob("**/*") if f.is_file()]

    for f in files:
        f.unlink()

    print(f"Deleted all files in folder '{folder.name}' successfully!")
