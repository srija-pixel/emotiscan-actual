from deepface import DeepFace

result = DeepFace.analyze(img_path="test_image.jpg", actions=["emotion"])

# Print full emotion breakdown
print("Detailed Emotion Scores:")
print(result[0]["emotion"])

# Print top detected emotion
print("Dominant Emotion:", result[0]['dominant_emotion'])
