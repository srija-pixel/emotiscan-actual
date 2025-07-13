import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        print("Emotion:", emotion)

        cv2.putText(frame, f"Emotion: {emotion}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except Exception as e:
        print("No face or error:", e)

    cv2.imshow("EmotiScan Live", frame)

    if cv2.waitKey(1) & 0xFF == 27: 
     break



cap.release()
cv2.destroyAllWindows()
