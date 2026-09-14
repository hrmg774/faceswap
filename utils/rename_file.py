# utils/rename_file.py

from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import time

TZ_HANOI = timezone(timedelta(hours=7))


def rename_file(old_name, new_name):
    old_name = Path(old_name)
    new_name = Path(new_name)

    if old_name.exists():
        old_name.rename(new_name)
        print(f"Rename {old_name.name} to {new_name.name} successfully!")
        return True
    else:
        print(f"File {old_name.name} not found!")
        return False


def rename_by_current_time(folder_path):
    # Doi ten theo gio hien tai luc bam chay code
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder not found: {folder_path}")
        return

    prefix = folder.name
    
    # Sắp xếp file theo thời gian chỉnh sửa (mtime) từ cũ nhất đến mới nhất
    # File nào được cho vào folder trước sẽ đứng trước
    all_files = sorted(
        [f for f in folder.iterdir() if f.is_file()],
        key=lambda x: x.stat().st_mtime
    )

    pattern = rf"^{re.escape(prefix)}_\d{{8}}_\d{{6}}_\d{{2}}$"

    files_to_rename = []
    for f in all_files:
        if re.match(pattern, f.stem):
            continue
        files_to_rename.append(f)

    if not files_to_rename:
        print(f"All files in '{prefix}' are already formatted!")
        return

    print(
        f"Found {len(all_files)} files total. Renaming {len(files_to_rename)} files by current time..."
    )

    for file_goc in files_to_rename:
        dt_now = datetime.now(TZ_HANOI)
        time_str = dt_now.strftime("%Y%m%d_%H%M%S")
        duoi_file = file_goc.suffix.lower()

        counter = 1
        while True:
            stem_candidate = f"{prefix}_{time_str}_{counter:02d}"
            
            if not any(folder.glob(f"{stem_candidate}.*")):
                new_name_str = f"{folder}/{stem_candidate}{duoi_file}"
                break
            counter += 1

        rename_file(str(file_goc), new_name_str)
        time.sleep(0.01)

    print("Successfully synced new files to current rename time!")


def rename_by_history(folder_path):
    # Doi ten dua theo lich su file luc tao hoac tai ve
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder not found: {folder_path}")
        return

    prefix = folder.name
    
    # Sắp xếp file theo thời gian chỉnh sửa (mtime) từ cũ nhất đến mới nhất
    # File nào được cho vào folder trước sẽ đứng trước
    all_files = sorted(
        [f for f in folder.iterdir() if f.is_file()],
        key=lambda x: x.stat().st_mtime
    )

    pattern = rf"^{re.escape(prefix)}_\d{{8}}_\d{{6}}_\d{{2}}$"

    files_to_rename = []
    for f in all_files:
        if re.match(pattern, f.stem):
            continue
        files_to_rename.append(f)

    if not files_to_rename:
        print(f"All files in '{prefix}' are already formatted!")
        return

    print(
        f"Found {len(all_files)} files total. Renaming {len(files_to_rename)} files by file age..."
    )

    for file_goc in files_to_rename:
        timestamp = file_goc.stat().st_mtime
        dt_vn = datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone(
            TZ_HANOI
        )

        time_str = dt_vn.strftime("%Y%m%d_%H%M%S")
        duoi_file = file_goc.suffix.lower()

        counter = 1
        while True:
            stem_candidate = f"{prefix}_{time_str}_{counter:02d}"
            
            if not any(folder.glob(f"{stem_candidate}.*")):
                new_name_str = f"{folder}/{stem_candidate}{duoi_file}"
                break
            counter += 1

        rename_file(str(file_goc), new_name_str)

    print(f"Synced all new files in '{prefix}' successfully!")
