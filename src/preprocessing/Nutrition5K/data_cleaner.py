import numpy as np
import pandas as pd

def load_metadata(project_root):
    metadata_folder = project_root / "data" / "raw" / "Nutrition5K" / "metadata"
    column_names = ['dish_id', 'calories', 'mass', 'fat', 'carb', 'protein']

    df_cafe1 = pd.read_csv(
        metadata_folder / 'nutrition5k_dataset_metadata_dish_metadata_cafe1.csv',
        sep=',', on_bad_lines='warn', header=None, usecols=range(6), names=column_names
    )
    df_cafe2 = pd.read_csv(
        metadata_folder / 'nutrition5k_dataset_metadata_dish_metadata_cafe2.csv',
        sep=',', on_bad_lines='warn', header=None, usecols=range(6), names=column_names
    )

    df = pd.concat([df_cafe1, df_cafe2], ignore_index=True)
    df = df.drop_duplicates().reset_index(drop=True)
    return df

def add_images_paths(df, project_root):
    images_folder = project_root / "data" / "raw" / "Nutrition5K" / "realsense_overhead"

    def path_validator(dish_id):
        full_rgb_path = images_folder / str(dish_id) / "rgb.png"
        full_depth_path = images_folder / str(dish_id) / "depth_color.png"
        if full_rgb_path.exists() and full_depth_path.exists():
            return pd.Series([str(full_rgb_path), str(full_depth_path)])
        else:
            return pd.Series([np.nan, np.nan])

    df[['image_rgb_path', 'image_depth_path']] = df['dish_id'].apply(path_validator)
    return df

def df_cleaner(df):
    df = df.dropna(subset=['image_rgb_path']).copy()
    df = df[df['calories'] > 0].copy()
    return df

def df_test_train_split(df, project_root):
    rgb_train_ids_path = project_root / "data" / "raw" / "Nutrition5K" / "splits" / "rgb_train_ids.txt"
    if rgb_train_ids_path.exists():
        lines = [line.strip() for line in rgb_train_ids_path.read_text(encoding='utf-8').splitlines()]
        train_ids_set = set(lines)
    else:
        print("Warning: rgb_train_ids.txt not found. Using random split.")
        train_ids_set = set()

    def is_it_test_or_train(dish_id):
        return "train" if str(dish_id) in train_ids_set else "test"

    df['split'] = df['dish_id'].apply(is_it_test_or_train)
    return df

def get_processed_data(project_root):
    df = load_metadata(project_root)
    df = add_images_paths(df, project_root)
    df = df_cleaner(df)
    df = df_test_train_split(df, project_root)
    return df