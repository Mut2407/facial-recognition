import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from main_model import HiFuse_Small as create_model
from facenet_pytorch import MTCNN
import glob  # <-- THÊM THƯ VIỆN NÀY

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using device: {device}")

    # --- Cấu hình thư mục ---
    input_folder = "test_images"  # <-- THAY ĐỔI: Đặt tên thư mục chứa ảnh của bạn ở đây
    output_folder = "results"     # <-- THÊM: Thư mục để lưu kết quả
    os.makedirs(output_folder, exist_ok=True) # Tạo thư mục nếu chưa có
    # --- Kết thúc cấu hình ---

    num_classes = 7
    img_size = 224
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # --- Lấy danh sách tất cả các ảnh ---
    image_paths = []
    # Tìm tất cả các loại ảnh phổ biến
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
    
    if not image_paths:
        print(f"⚠️ Không tìm thấy ảnh nào trong thư mục: '{input_folder}'")
        return
    print(f"🔍 Tìm thấy {len(image_paths)} ảnh. Bắt đầu xử lý...")
    # --- Kết thúc lấy danh sách ---

    # --- Tải model và các file cần thiết (chỉ tải 1 lần) ---
    mtcnn = MTCNN(keep_all=False, device=device)

    json_path = './class_indices.json'
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    model = create_model(num_classes=num_classes).to(device)
    model_weight_path = "model_weight/best_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    # --- Kết thúc tải model ---

    # --- Bắt đầu vòng lặp xử lý từng ảnh ---
    for img_path in image_paths:
        try:
            print(f"\nProcessing: {os.path.basename(img_path)}")
            # Mở ảnh
            img = Image.open(img_path).convert("RGB")

            # Phát hiện khuôn mặt
            face = mtcnn(img)
            if face is None:
                print(f"⚠️ Không tìm thấy khuôn mặt trong ảnh {os.path.basename(img_path)}, sử dụng toàn bộ ảnh.")
                face = data_transform(img).unsqueeze(0).to(device)
            else:
                face = torch.nn.functional.interpolate(face.unsqueeze(0), size=(224, 224), mode='bilinear').to(device)

            # Dự đoán
            with torch.no_grad():
                output = torch.squeeze(model(face)).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()

            # Lấy kết quả
            prediction_text = class_indict[str(predict_cla)]
            probability = predict[predict_cla].item()

            print(f"    -> Kết quả: {prediction_text}, Độ chính xác: {probability:.3f}")

            # --- Lưu ảnh kết quả (thay vì plt.show()) ---
            plt.figure(figsize=(8, 8)) # Tạo một figure mới
            plt.imshow(img)
            plt.title(f"{prediction_text} ({probability:.3f})", fontsize=16)
            plt.axis('off') # Ẩn trục tọa độ
            
            # Tạo đường dẫn lưu file
            output_filename = os.path.basename(img_path)
            save_path = os.path.join(output_folder, output_filename)
            
            plt.savefig(save_path, bbox_inches='tight')
            plt.close() # Đóng figure lại để giải phóng bộ nhớ
            # --- Kết thúc lưu ảnh ---

        except Exception as e:
            # Nếu có lỗi (ví dụ file ảnh hỏng), in ra lỗi và tiếp tục
            print(f"❌ Lỗi khi xử lý file {img_path}: {e}")
    
    print(f"\n✅ Xử lý hoàn tất! Tất cả kết quả đã được lưu vào thư mục '{output_folder}'.")

if __name__ == '__main__':
    main()