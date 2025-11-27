import glob
import torch
import os
from PIL import Image

from models.model import Model
from data.datamodule import val_transforms, class_names

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

inference_model = Model().to(DEVICE)
inference_model.load_state_dict(torch.load("models/model.pt", map_location=DEVICE))
inference_model.eval()

def predict_image(image_path):
    # load image
    img = Image.open(image_path).convert("RGB")
    
    # use SAME preprocessing as validation
    img_tensor = val_transforms(img)          # [C, H, W]
    img_tensor = img_tensor.unsqueeze(0)      # type: ignore # [1, C, H, W]
    img_tensor = img_tensor.to(DEVICE)

    with torch.no_grad():
        outputs = inference_model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    pred_class = class_names[pred_idx.item()] # type: ignore
    return pred_class, conf.item()

def predict_folder(folder_path):
    image_paths = []
    # add more extensions if needed
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))

    print(f"Found {len(image_paths)} images in {folder_path}")

    for img_path in image_paths:
        pred_class, conf = predict_image(img_path)

        print(f"Image: {os.path.basename(img_path)} --> Predicted: {pred_class} (Confidence: {conf:.4f})")

predict_folder("sample")