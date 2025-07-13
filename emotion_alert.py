import cv2
from deepface import DeepFace
from plyer import notification
from twilio.rest import Client

from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")


# List of distress emotions
distress_emotions = ["sad", "angry", "fear", "disgust"]

cap = cv2.VideoCapture(0)
print("Press Q to quit")

last_alert_time = None
# Twilio setup

client = Client(account_sid, auth_token)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
        emotion = result[0]['dominant_emotion']

        # Show emotion on screen
        cv2.putText(frame, f"Emotion: {emotion}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show popup alert if emotion is distressing
        if emotion in distress_emotions:
            current_time = datetime.now()
            # Send WhatsApp alert
        # Send WhatsApp alert
            client.messages.create(
    from_='whatsapp:+14155238886',  # Twilio sandbox number
    to='whatsapp:+918420897550',    # Your caregiver's number
    body=f"🚨 EmotiScan Alert: {emotion.upper()} detected at {datetime.now().strftime('%H:%M:%S')}"
)
     
            if not last_alert_time or (current_time - last_alert_time).seconds > 10:
                # Notify every 10 seconds max
                notification.notify(
                    title="⚠️ EmotiScan Alert!",
                    message=f"Distress emotion detected: {emotion}",
                    timeout=5  # stays for 5 seconds
                )
                last_alert_time = current_time

        print(f"Detected: {emotion}")

    except Exception as e:
        print("Error:", e)

    cv2.imshow("EmotiScan Alert System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
