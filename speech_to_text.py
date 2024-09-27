import speech_recognition as sr

# Path to the uploaded audio file
audio_file_path = 'E:\\Project\\TTS\\Dataset\\wav28.wav'

# Initialize recognizer
recognizer = sr.Recognizer()

# Open the audio file
with sr.AudioFile(audio_file_path) as source:
    # Adjust for ambient noise and record the audio
    recognizer.adjust_for_ambient_noise(source)
    audio_data = recognizer.record(source)

# Perform speech recognition
try:
    transcript = recognizer.recognize_google(audio_data)
    transcript
except sr.UnknownValueError:
    transcript = "Could not understand the audio"
except sr.RequestError:
    transcript = "Could not request results; check your network connection"

print(transcript)