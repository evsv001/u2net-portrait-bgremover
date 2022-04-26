import pyaudio
# pip install pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 16000
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "Oldboy.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("开始录音,请说话......")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("录音结束,请闭嘴!")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()


import torch
import torchaudio
import matplotlib.pyplot as plt

filename = "Oldboy.wav"
waveform,sample_rate = torchaudio.load(filename)
print("Shape of waveform:{}".format(waveform.size())) #音频大小
print("sample rate of waveform:{}".format(sample_rate))#采样率
plt.figure()
plt.plot(waveform.t().numpy())
plt.show()

# import soundfile as sf
 
# data, samplerate = sf.read('Oldboy.wav')
# sf.write('new_file.ogg', data, samplerate)