import pathlib
import cv2
import torch
from ultralytics import YOLO
from src.preprocessing.Nutrition5K.data_cleaner import get_processed_data

def create_model():
    model = YOLO('yolov8n-seg.pt')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model

def crop_image(project_root, model):
    output_folder = project_root / "data" / "processed" / "Nutrition5K"
    (output_folder / "rgb_crops").mkdir(parents=True, exist_ok=True)
    (output_folder / "depth_crops").mkdir(parents=True, exist_ok=True)
    (output_folder / "metadata").mkdir(parents=True, exist_ok=True)

    df = get_processed_data(project_root)
    df_1 = get_processed_data(project_root)
    print(f"Found {len(df)} records to process.")

    for idx, row in df.iterrows():
        rgb_img = cv2.imread(str(row["image_rgb_path"]))
        depth_img = cv2.imread(str(row["image_depth_path"]))

        if rgb_img is None or depth_img is None:
            continue

        h, w, _ = rgb_img.shape

        margin = 0.1
        x1_m, y1_m = int(w * margin), int(h * margin)
        x2_m, y2_m = int(w * (1 - margin)), int(h * (1 - margin))

        rgb_crop = rgb_img[y1_m:y2_m, x1_m:x2_m]
        depth_crop = depth_img[y1_m:y2_m, x1_m:x2_m]

        results = model.predict(rgb_img, classes=[45, 61], conf=0.2, verbose=False)

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            best_idx = torch.argmax(results[0].boxes.conf).item()
            best_box = results[0].boxes.xyxy[best_idx].tolist()

            x1, y1, x2, y2 = best_box
            width, height = x2 - x1, y2 - y1

            if height >= 400 and width >= 400:
                pw, ph = int(width * 0.1), int(height * 0.1)
                x1_c, y1_c = max(0, int(x1 - pw)), max(0, int(y1 - ph))
                x2_c, y2_c = min(w, int(x2 + pw)), min(h, int(y2 + ph))

                rgb_crop = rgb_img[y1_c:y2_c, x1_c:x2_c]
                depth_crop = depth_img[y1_c:y2_c, x1_c:x2_c]

        unique_name = f"{row['dish_id']}_{idx}"
        output_rgb_path = output_folder / "rgb_crops" / f"{unique_name}.jpg"
        output_depth_path = output_folder / "depth_crops" / f"{unique_name}.png"
        # output_rgb_path_1 = f"/content/rgb_crops/{unique_name}.jpg"
        # output_depth_path_1 = f"/content/depth_crops/{unique_name}.png"

        cv2.imwrite(str(output_rgb_path), rgb_crop)
        cv2.imwrite(str(output_depth_path), depth_crop)

        # cv2.imwrite(str(output_rgb_path_1), rgb_crop)
        # cv2.imwrite(str(output_depth_path_1), depth_crop)

        df.at[idx, 'image_rgb_path'] = str(output_rgb_path)
        df.at[idx, 'image_depth_path'] = str(output_depth_path)

        # df_1.at[idx, 'image_rgb_path'] = str(output_rgb_path_1)
        # df_1.at[idx, 'image_depth_path'] = str(output_depth_path_1)

        if idx % 100 == 0:
            print(f"Processed: {idx}/{len(df)}...")

    output_csv_path = output_folder / "metadata" / "metadata_clean.csv"
    # output_csv_path_1 = output_folder / "metadata" / "metadata_clean_1.csv"
    df.to_csv(output_csv_path, index=False)
    # df_1.to_csv(output_csv_path_1, index=False)
    print(f"CSV file saved in: {output_csv_path}")

if __name__ == "__main__":
    project_root = pathlib.Path(__file__).resolve().parent.parent.parent.parent
    yolo_model = create_model()
    crop_image(project_root, yolo_model)