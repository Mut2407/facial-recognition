import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import cv2
import numpy as np

# --- Config---
MODEL_PATH = "swin_best_model.pth"
HAAR_CASCADE_PATH = "haarcascade_frontalface_default.xml"
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
NUM_CLASSES = len(labels_dict)

def load_model(model_path, num_classes):
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False)
    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file model '{model_path}'.")
        print("Vui lòng tải file 'swin_best_model.pth' và đặt vào cùng thư mục.")
        sys.exit(1)
    except RuntimeError as e:
        print(f"Lỗi khi tải model: {e}")
        sys.exit(1)

    model = model.to(DEVICE)
    model.eval()
    return model

def get_test_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    
    print(f"Đang tải model lên thiết bị {DEVICE}...")
    model = load_model(MODEL_PATH, NUM_CLASSES)
    
    test_transform = get_test_transform(IMG_SIZE)

    if not os.path.exists(HAAR_CASCADE_PATH):
        print(f"Lỗi: Không tìm thấy file '{HAAR_CASCADE_PATH}'.")
        print("Vui lòng tải file và đặt vào cùng thư mục.")
        sys.exit(1)
        
    faceDetect = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

    video = cv2.VideoCapture(0)
    print("Đã khởi động webcam. Nhấn 'q' để thoát.")

    while True:
        ret, frame = video.read()
        if not ret:
            print("Lỗi: Không thể đọc frame từ webcam.")
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = faceDetect.detectMultiScale(gray, 1.3, 3)

        for (x, y, w, h) in faces:
            

            sub_face_img_color = frame[y:y+h, x:x+w]
            image_pil = Image.fromarray(cv2.cvtColor(sub_face_img_color, cv2.COLOR_BGR2RGB))
            input_tensor = test_transform(image_pil)
            input_batch = input_tensor.unsqueeze(0).to(DEVICE)

            # --- Prediction ---
            with torch.no_grad():
                output = model(input_batch) 
            
            output = output.mean(dim=1)
            probabilities = torch.softmax(output[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
            label = predicted_idx.item()
            conf_score = confidence.item()

            result_text = f"{labels_dict[label]} ({conf_score * 100:.2f}%)"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
            cv2.putText(frame, result_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Frame", frame)
        
        # Nhấn 'q' để thoát
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    print("Đã đóng webcam.")