import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_PATH = "upload/fear-test.jpg"
MODEL_PATH = "swin_final_model.pth"
NUM_CLASSES = 7

# === Load model ===
model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False)
in_features = model.head.in_features
model.head = nn.Sequential(
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, NUM_CLASSES)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# === Preprocess image ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

image = Image.open(IMG_PATH).convert("RGB")
img_tensor = transform(image).unsqueeze(0).to(DEVICE)

# === Predict ===
with torch.no_grad():
    outputs = model(img_tensor)
    if outputs.ndim > 2:  # Just in case
        outputs = torch.mean(outputs, dim=[2, 3])
    pred = outputs.argmax(1).item()

labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear',
               3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

probabilities = F.softmax(outputs, dim=1)
confidence = probabilities[0][pred].item()

print(f"\n--- Result: {IMG_PATH} ---")
print(f"Emotion: {labels_dict[pred]}")
print(f"Accuracy: {confidence * 100:.2f}%")
