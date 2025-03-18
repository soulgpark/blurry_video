import cv2
import numpy as np

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

recording = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))
    
    blurred_frame = cv2.GaussianBlur(frame, (25, 25), 0) 
    mask = np.zeros_like(frame, dtype=np.uint8) 

    for (x, y, w, h) in faces:
        mask[y:y+h, x:x+w] = frame[y:y+h, x:x+w]

    processed_frame = np.where(mask > 0, mask, blurred_frame)

    if recording:
        cv2.circle(processed_frame, (50, 50), 10, (0, 0, 255), -1)

    cv2.imshow('Camera', processed_frame)
    
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break
    elif key == 32: 
        recording = not recording
        if recording: 
            if out is None: 
                out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
        else:  
            if out is not None:
                out.release()
                out = None
    
    if recording and out is not None:
        out.write(processed_frame) 

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()