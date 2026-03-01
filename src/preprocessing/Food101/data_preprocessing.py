import pathlib
from sklearn.model_selection import train_test_split

base_path = pathlib.Path(__file__).resolve().parent.parent.parent.parent

def get_image_paths_and_labels(base_path):
    images_folder = base_path / "data" / "processed" / "Food101" / "images"
    image_paths = []
    labels = []

    for dir in images_folder.iterdir():
        if dir.is_dir():
            dir_name = dir.name
            for image in dir.iterdir():
                image_paths.append(image)
                labels.append(dir_name)
    return image_paths, labels

def create_class_mapping(labels):
    classes = sorted(list(set(labels)))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return class_to_idx

def train_test_val_split(image_paths, labels):
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels

def mapping_return():
    image_paths, labels = get_image_paths_and_labels(base_path)
    mappings = create_class_mapping(labels)
    return mappings

def splits_return():
    image_paths, labels = get_image_paths_and_labels(base_path)
    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = train_test_val_split(image_paths, labels)
    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels