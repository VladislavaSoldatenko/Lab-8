import cv2
import numpy as np

def track_marker_with_fly():
    cap = cv2.VideoCapture(0)
    down_points = (640, 480)
    fly_img = cv2.imread('fly64.png', cv2.IMREAD_UNCHANGED)  # Загружаем с альфа-каналом
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, down_points, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        ret, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            
            # Проверка положения метки
            if x < 50 and y < 50:  # Левый верхний угол
                color = (255, 0, 0)  # Синий
            elif x > down_points[0] - 50 and y > down_points[1] - 50:  # Правый нижний угол
                color = (0, 0, 255)  # Красный
            else:
                color = (0, 255, 0)  # Зеленый
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Центр метки
            center_x = x + w//2
            center_y = y + h//2
            
            # Наложение изображения мухи
            if fly_img is not None:
                # Размеры изображения мухи
                fly_h, fly_w = fly_img.shape[:2]
                
                # Позиция для наложения (центр мухи совпадает с центром метки)
                x1 = center_x - fly_w//2
                y1 = center_y - fly_h//2
                x2 = x1 + fly_w
                y2 = y1 + fly_h
                
                # Проверка, чтобы не выйти за границы кадра
                if x1 >= 0 and y1 >= 0 and x2 <= frame.shape[1] and y2 <= frame.shape[0]:
                    # Если есть альфа-канал
                    if fly_img.shape[2] == 4:
                        alpha = fly_img[:, :, 3] / 255.0
                        for c in range(0, 3):
                            frame[y1:y2, x1:x2, c] = (
                                alpha * fly_img[:, :, c] + 
                                (1 - alpha) * frame[y1:y2, x1:x2, c]
                            )
                    else:
                        frame[y1:y2, x1:x2] = fly_img

        cv2.imshow('Marker Tracking with Fly', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    track_marker_with_fly()