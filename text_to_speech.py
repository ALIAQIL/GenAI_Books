import os
from gtts import gTTS


file_path = "GenAIbooks/resume/atomic_habits_1_flash.md"

def speak_file(path):
    try:

        if not os.path.exists(path):
            return

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()


        clean_text = text.replace("*", "").replace("#", "")

        tts = gTTS(text=clean_text, lang='en', slow=False)
        
        audio_file = "resume_audio.mp3"
        tts.save(audio_file)

        os.system(f"mpg123 '{audio_file}'")
        
    except Exception as e:
        print(f"Error: {e}")

speak_file(file_path)