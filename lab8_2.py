import cv2
import numpy as np

def track_marker():
    cap = cv2.VideoCapture(0)
    down_points = (640, 480)
    
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
            
            # Отображение координат центра метки
            center_x = x + w//2
            center_y = y + h//2
            cv2.putText(frame, f"({center_x}, {center_y})", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('Marker Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    track_marker()