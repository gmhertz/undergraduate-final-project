# Print the total duration of the war files
import wave
import contextlib
import os

total_duration = 0
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".wav"):
        with contextlib.closing(wave.open(filename, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            total_duration = total_duration + duration

print(total_duration)