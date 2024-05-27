import sounddevice as sd
from scipy.io.wavfile import write

def record_audio(duration=5, fs=44100):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()
    write("..\Send Data\output.wav", fs, recording)


record_audio()  # Record for 5 seconds
