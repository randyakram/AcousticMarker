import re
import pathlib
import pandas as pd

def parse_patient_and_site(filename, fallback_prefix):

    m = re.match(r"(?:Copy of\s+)+(.+?)_([A-Za-z]+)_.*?_(\d+)(?:_\d+)?\.wav$", filename, flags=re.IGNORECASE)
    if m:
        name, site, segment = m.groups()
        patient_id = name.strip().replace(' ', '_')
        return f"{fallback_prefix}_{patient_id}", site.lower(), int(segment)

    m = re.match(r"Copy of ([A-Za-z]+)_(\d+)\.wav$", filename, flags=re.IGNORECASE)
    if m:
        site, segment = m.groups()
        return f"{fallback_prefix}_70", site.lower(), int(segment)

    m = re.match(r"patient[ _](\d+)_.*?_([A-Za-z]+)_(\d+)\.wav$", filename, flags=re.IGNORECASE)
    if m:
        num, site, segment = m.groups()
        return f"{fallback_prefix}_{num}", site.lower(), int(segment)

    m = re.match(r"patient[ _](\d+)_.*?_([A-Za-z]+)\.wav$", filename, flags=re.IGNORECASE)
    if m:
        num, site = m.groups()
        return f"{fallback_prefix}_{num}", site.lower(), None

    m = re.match(r"Copy of ([A-Za-z]+)[ _]+(\d+)_+(\d+)\.wav$", filename, flags=re.IGNORECASE)
    if m:
        site, num, segment = m.groups()
        return f"{fallback_prefix}_{num}", site.lower(), int(segment)

    m = re.match(r"Copy of ([A-Za-z]+)[ _]+(\d+)_?\.wav$", filename, flags=re.IGNORECASE)
    if m:
        site, num = m.groups()
        return f"{fallback_prefix}_{num}", site.lower(), None

    m = re.match(r"Copy of ([A-Za-z]+)[ _]\((\d+)\)(?:_(\d+))?\.wav$", filename, flags=re.IGNORECASE)
    if m:
        site, num, segment = m.groups()
        return f"{fallback_prefix}_{num}", site.lower(), (int(segment) if segment else None)

    return f"{fallback_prefix}_{pathlib.Path(filename).stem}", "unknown", None

def load_filepaths_from_folders(data_dir, class_folders, classes):
    rows, unmatched = [], []
    for folder_name, label in class_folders.items():
        folder = pathlib.Path(data_dir) / folder_name
        if not folder.exists():
            print(f"Warning: folder not found: {folder}")
            continue
        class_name = classes[label]
        for f in sorted(folder.glob("*.wav")):
            patient_id, site, segment = parse_patient_and_site(f.name, fallback_prefix=class_name)
            if site == "unknown":
                unmatched.append(f.name)
            rows.append({
                "filepath": str(f), "label": label, "class_name": class_name,
                "patient_id": patient_id, "site": site, "segment": segment,
            })
    if unmatched:
        print(f"Note: {len(unmatched)} file(s) didn't match any known naming pattern "
              "and were each treated as their own patient:")
        for u in unmatched[:20]:
            print("  -", u)
        if len(unmatched) > 20:
            print(f"  ... and {len(unmatched) - 20} more")
    return pd.DataFrame(rows)