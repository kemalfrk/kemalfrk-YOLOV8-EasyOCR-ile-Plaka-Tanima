import cv2
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from easyocr import Reader  

VIDEOS_DIR = os.path.join('.', 'videos')

video_path = os.path.join(VIDEOS_DIR, '9.mp4')
video_path_out = '{}_tespit_ocr.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('.', 'runs', 'detect', 'train13', 'weights', 'best.pt')

# custom modelimizi yüklüyoruz
model=YOLO(model_path)

threshold = 0.5

# EasyOCR kütüphanesini başlatıyoruz
reader = Reader(['tr'], gpu=True)  

while ret:

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if score > threshold:
            
            plate_region = frame[y1:y2, x1:x2]

            # .readtext() ile metni okuyoruz
            text = reader.readtext(plate_region)

            
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            if text:
                cv2.putText(frame, text[0][1], (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()
    
cap.release()
out.release()
cv2.destroyAllWindows()