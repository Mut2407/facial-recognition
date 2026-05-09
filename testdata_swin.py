import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm


# --- Config---
MODEL_PATH = "swin_best_model.pth"
IMAGE_TO_TEST = "upload/angry-test.jpg"
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(CLASS_NAMES)


# --- Load model ---
def load_model(model_path, num_classes):
    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False)
    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file model '{model_path}'.")
        sys.exit(1)
    except RuntimeError as e:
        print(f"Lỗi khi tải model: {e}")
        sys.exit(1)

    model = model.to(DEVICE)
    model.eval()
    return model


# --- PREPROCESSING ---
def get_test_transform(img_size):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


# --- PREDICTION FUNCTION ---
def predict_image(model, image_path, transform, class_names, device):
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file ảnh '{image_path}'.")
        return
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_batch)

    logits = output.mean(dim=[2, 3])
    probabilities = torch.softmax(logits[0], dim=0)

    predicted_idx = logits.argmax(dim=1).item()
    predicted_class = class_names[predicted_idx]
    confidence = probabilities[predicted_idx].item()

    print(f"\n--- Result: {image_path} ---")
    print(f"Emotion: {predicted_class}")
    print(f"Accuracy: {confidence * 100:.2f}%")


# --- main ---
if __name__ == "__main__":

    model = load_model(MODEL_PATH, NUM_CLASSES)
    test_transform = get_test_transform(IMG_SIZE)
    predict_image(model, IMAGE_TO_TEST, test_transform, CLASS_NAMES, DEVICE)
