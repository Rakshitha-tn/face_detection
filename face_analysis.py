import cv2
from deepface import DeepFace
import os

# ——— Paths to your downloaded models ———
MODEL_DIR   = "models"
PROTO_PATH  = os.path.join(MODEL_DIR, "deploy_gender.prototxt")
MODEL_PATH  = os.path.join(MODEL_DIR, "gender_net.caffemodel")

# ——— Load the DNN gender classifier ———
gender_net = cv2.dnn.readNet(PROTO_PATH, MODEL_PATH)
GENDER_LIST = ['Male', 'Female']
MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# ——— Load Haar cascade for face detection ———
cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
if cascade.empty():
    raise IOError("Failed to load Haar cascade.")

# ——— Open the webcam ———
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

cv2.namedWindow("Face Analysis", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Face Analysis", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # — Gender via Adience DNN with RGB swap —
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227,227), MEAN_VALUES, swapRB=True)
        gender_net.setInput(blob)
        preds = gender_net.forward()[0]
        male_score, female_score = float(preds[0]), float(preds[1])
        gender = 'Female' if female_score > male_score else 'Male'

        # — Age & Emotion via DeepFace —
        try:
            res = DeepFace.analyze(face_img,
                                   actions=['age','emotion'],
                                   enforce_detection=False)
            data    = res[0] if isinstance(res, list) else res
            age     = data.get('age', 'N/A')
            emotion = data.get('dominant_emotion', 'N/A')
        except Exception:
            age, emotion = 'N/A', 'N/A'

        # Overlay results per face
        cv2.putText(frame, f'Age: {age}',      (x, y-40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f'Gender: {gender}',(x, y-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),2)
        cv2.putText(frame, f'Emotion: {emotion}',(x, y-60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255),2)

        # Optional: show raw DNN scores
        cv2.putText(frame, f'M:{male_score:.2f}', (x+w-80, y+h+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
        cv2.putText(frame, f'F:{female_score:.2f}', (x+w-80, y+h+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)

    cv2.imshow("Face Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
