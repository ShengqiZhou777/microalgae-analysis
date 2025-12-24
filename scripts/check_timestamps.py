
import os
import glob
from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime
import pandas as pd
import pathlib

def get_exif_timestamp(image_path):
    try:
        image = Image.open(image_path)
        exifdata = image.getexif()
        # Iterate over all EXIF data fields
        for tag_id in exifdata:
            # get the tag name, instead of human unreadable tag id
            tag = TAGS.get(tag_id, tag_id)
            if tag == 'DateTimeOriginal':
                return exifdata.get(tag_id)
            if tag == 'DateTime':
                return exifdata.get(tag_id)
        
        # If no basic EXIF, try _getexif for older PIL versions or specific formats
        if hasattr(image, '_getexif'):
             exif = image._getexif()
             if exif:
                 for tag_id, value in exif.items():
                     tag = TAGS.get(tag_id, tag_id)
                     if tag == 'DateTimeOriginal':
                         return value
                     if tag == 'DateTime':
                         return value
    except Exception as e:
        pass
    return None

def parse_time(time_str):
    if not time_str:
        return None
    try:
        return datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
    except ValueError:
        return None

def check_folder(folder_path):
    files = sorted(list(glob.glob(os.path.join(folder_path, "*.jpg"))))
    if not files:
        print(f"No jpg files in {folder_path}")
        return None
    
    data = []
    for f in files:
        fname = os.path.basename(f)
        ts_str = get_exif_timestamp(f)
        ts = parse_time(ts_str)
        # Fallback to mtime if no EXIF? User asked for shooting time, so maybe not. 
        # But if EXIF is missing, we should note it.
        data.append({'filename': fname, 'timestamp': ts})
    
    df = pd.DataFrame(data)
    
    # Check if sorted
    df_with_time = df.dropna(subset=['timestamp'])
    
    if len(df_with_time) == 0:
        print(f"No valid EXIF timestamps in {folder_path}")
        return None

    msg = f"Folder: {folder_path} (Count: {len(files)}, With Time: {len(df_with_time)})"
    
    # Check monotonicity
    # We assume filenames are numbers or sortable strings that imply order.
    # The list `files` is already sorted by filename.
    # So we just check if timestamps are monotonic.
    
    is_monotonic = df_with_time['timestamp'].is_monotonic_increasing
    
    start_time = df_with_time['timestamp'].min()
    end_time = df_with_time['timestamp'].max()
    
    print(f"{msg}")
    print(f"  Time Range: {start_time} to {end_time}")
    print(f"  Filename sort matches Time sort: {is_monotonic}")
    
    if not is_monotonic:
        # Find violations
        df_with_time['prev_time'] = df_with_time['timestamp'].shift(1)
        violations = df_with_time[df_with_time['timestamp'] < df_with_time['prev_time']]
        print(f"  Violations found: {len(violations)}")
        print(violations.head())
        
    return {
        'folder': folder_path,
        'start': start_time,
        'end': end_time,
        'monotonic': is_monotonic
    }

def main():
    base_dir = "data/TIMECOURSE"
    # Find all 'images' directories
    images_dirs = []
    for root, dirs, files in os.walk(base_dir):
        if 'images' in dirs:
            images_dirs.append(os.path.join(root, 'images'))
        if os.path.basename(root) == 'images':
             # Already in an images dir, though walk should cover parent
             pass

    results = []
    for d in sorted(images_dirs):
        res = check_folder(d)
        if res:
            results.append(res)
            
    print("\n--- Group Correspondence Summary ---")
    results.sort(key=lambda x: x['start'] if x['start'] else datetime.min)
    
    for r in results:
        print(f"{r['folder']}: {r['start']} -> {r['end']}")

if __name__ == "__main__":
    main()
