"""Prepare the Kaggle Leaf Classification dataset for training.

Source layout (as downloaded):
    images/images/<id>.jpg          # 1,584 leaf images
    train.csv/train.csv             # id -> species for 990 labelled images
    test.csv/test.csv               # 594 unlabelled images

Output layout:
    dataset/<species>/<id>.jpg          # full labelled dataset
    dataset_split/train/<species>/...   # 70 %
    dataset_split/validation/<species>/ # 20 %
    dataset_split/test/<species>/       # 10 %
    model/class_names.json              # ordered class labels

The old 5-class sample dataset (Mango, Neem, ...) is moved to
`legacy_dataset/` instead of being deleted.
"""

import csv
import json
import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, 'images', 'images')
TRAIN_CSV = os.path.join(BASE_DIR, 'train.csv', 'train.csv')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
SPLIT_DIR = os.path.join(BASE_DIR, 'dataset_split')
LEGACY_DIR = os.path.join(BASE_DIR, 'legacy_dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

SPLIT_RATIOS = {'train': 0.7, 'validation': 0.2, 'test': 0.1}


def find_train_csv():
    """Support both `train.csv` as a file and as an extracted directory."""
    candidates = [
        TRAIN_CSV,
        os.path.join(BASE_DIR, 'train.csv'),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "Could not find train.csv. Expected train.csv/train.csv or train.csv"
    )


def load_labels(csv_path):
    with open(csv_path, newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    labelled = []
    for row in rows:
        image_id = str(row.get('id', '')).strip()
        species = str(row.get('species', '')).strip()
        if not image_id or not species:
            continue
        labelled.append((image_id, species))
    if not labelled:
        raise ValueError(f'No labelled rows found in {csv_path}')
    return labelled


def archive_legacy_dataset(species):
    """Move non-Kaggle folders out of dataset/ so nothing is deleted."""
    if not os.path.isdir(DATASET_DIR):
        return
    existing = [d for d in os.listdir(DATASET_DIR)
                if os.path.isdir(os.path.join(DATASET_DIR, d))]
    legacy = [d for d in existing if d not in species]
    if not legacy:
        return
    os.makedirs(LEGACY_DIR, exist_ok=True)
    for name in legacy:
        src = os.path.join(DATASET_DIR, name)
        dst = os.path.join(LEGACY_DIR, name)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.move(src, dst)
        print(f'Archived legacy class folder: {name} -> legacy_dataset/{name}')


def split_ids(ids):
    """Deterministic per-class split: 7 train / 2 validation / 1 test for 10 images."""
    ids = sorted(ids, key=lambda value: int(value))
    count = len(ids)
    n_train = max(1, int(round(count * SPLIT_RATIOS['train'])))
    n_validation = max(1, int(round(count * SPLIT_RATIOS['validation'])))
    if n_train + n_validation >= count:
        n_train = max(1, count - 2)
        n_validation = 1
    n_test = count - n_train - n_validation

    return {
        'train': ids[:n_train],
        'validation': ids[n_train:n_train + n_validation],
        'test': ids[n_train + n_validation:n_train + n_validation + n_test],
    }


def copy_image(image_id, class_name, destination_root):
    src = os.path.join(IMAGES_DIR, f'{image_id}.jpg')
    if not os.path.isfile(src):
        raise FileNotFoundError(f'Missing image for id {image_id}: {src}')
    class_dir = os.path.join(destination_root, class_name)
    os.makedirs(class_dir, exist_ok=True)
    dst = os.path.join(class_dir, f'{image_id}.jpg')
    if not os.path.isfile(dst):
        shutil.copy2(src, dst)


def main():
    csv_path = find_train_csv()
    labelled = load_labels(csv_path)
    species = sorted({name for _, name in labelled})
    labelled_ids = {image_id for image_id, _ in labelled}

    print(f'Labels file: {csv_path}')
    print(f'Labelled images: {len(labelled)} | classes: {len(species)}')

    archive_legacy_dataset(set(species))

    # Full labelled dataset, grouped by class.
    for image_id, class_name in labelled:
        copy_image(image_id, class_name, DATASET_DIR)

    # Train / validation / test split.
    if os.path.isdir(SPLIT_DIR):
        shutil.rmtree(SPLIT_DIR)

    summary = {'train': 0, 'validation': 0, 'test': 0}
    by_class = {}
    for class_name in species:
        ids = [image_id for image_id, name in labelled if name == class_name]
        parts = split_ids(ids)
        by_class[class_name] = {key: len(value) for key, value in parts.items()}
        for split_name, split_list in parts.items():
            for image_id in split_list:
                copy_image(image_id, class_name, os.path.join(SPLIT_DIR, split_name))
            summary[split_name] += len(split_list)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, 'class_names.json'), 'w', encoding='utf-8') as handle:
        json.dump(species, handle, indent=2)
    with open(os.path.join(SPLIT_DIR, 'split_summary.json'), 'w', encoding='utf-8') as handle:
        json.dump({'counts': summary, 'per_class': by_class}, handle, indent=2)

    unlabelled = sorted(
        name[:-4] for name in os.listdir(IMAGES_DIR)
        if name.lower().endswith('.jpg') and name[:-4] not in labelled_ids
    )
    print(f'Split counts: {summary}')
    print(f'Unlabelled images kept for manual testing: {len(unlabelled)}')
    print(f'Class names saved to model/class_names.json')


if __name__ == '__main__':
    main()
