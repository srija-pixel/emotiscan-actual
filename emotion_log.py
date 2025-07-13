import cv2
import csv
from datetime import datetime
from deepface import DeepFace

# Initialize webcam
cap = cv2.VideoCapture(0)

# CSV file setup
log_filename = "emotion_log.csv"
with open(log_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Emotion"])

print("🟢 Logging emotions... Press Q to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
        emotion = result[0]['dominant_emotion']

        # Get current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Log into CSV
        with open(log_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, emotion])

        # Show emotion on screen
        cv2.putText(frame, f"Emotion: {emotion}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except Exception as e:
        print("⚠️ No face or error:", e)

    cv2.imshow("EmotiScan Logger", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Emotion logging stopped.")
