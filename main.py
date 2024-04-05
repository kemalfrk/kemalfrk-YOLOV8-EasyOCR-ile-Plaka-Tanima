#ultralytics reposunun klonlanması ve indirilmesi
!git clone https://github.com/ultralytics/ultralytics
!pip install ultralytics



import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cv2
import easyocr
from ultralytics import YOLO

# EasyOCR Reader'ı başlatıyoruz
reader = easyocr.Reader(['tr'], gpu=True)

# YOLO modelini yüklüyoruz
model = YOLO('C:/Users/kemal/Desktop/plaka_tanıma/runs/detect/train13/weights/best.pt') 

# Görüntülerin bulunduğu klasör yolu
image_folder = "test_images"

# Plakaları saklamak için bir klasör yoksa oluştur
if not os.path.exists("plaka_resimleri/cikti"):
    os.makedirs("plaka_resimleri/cikti")

# Görüntülerin listesini al
image_files = os.listdir(image_folder)

# Her bir görüntüyü işle
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)

    # Görüntüyü YOLO ile işliyoruz
    results = model(image_path)

    
    for result in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, class_id = map(int, result)
        y1=y1-20
        
        
        if class_id== 0:  # Plaka sınıfı 
            # Plaka bölgesini kırp
            plate_crop = image[y1:y2, x1:x2]
            # Görüntüyü griye çevir , medianblur ekle ve  eşik değeri uygula
            gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            blurred_plate = cv2.medianBlur(gray_plate, 3)
            # thresholding uygula
            _, thresh_plate = cv2.threshold(blurred_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            
            kernel = np.ones((1,1),np.uint8) 
            eroded_plate = cv2.erode(thresh_plate, kernel, iterations=2)
            
            
            
            # EasyOCR ile metni oku ve sonuçları işle
            ocr_results= reader.readtext(eroded_plate)
            for (bbox, text, prob) in ocr_results:
                
                # Bounding box koordinatlarını ayarlıyoruz
                (top_left, top_right, bottom_right, bottom_left) = bbox
                top_left = (int(top_left[0]), int(top_left[1]))
                bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

                # Metni yazdır (plakanın üzerine)
                cv2.putText(eroded_plate, text, (top_left[0], top_left[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)

                # İşlenmiş görüntüyü kaydet
                plate_save_path = os.path.join("plaka_resimleri/cikti", f"{image_file}_adatifthresh.jpg")
                cv2.imwrite(plate_save_path, eroded_plate)