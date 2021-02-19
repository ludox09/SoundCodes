#!/usr/bin/env python3
import numpy as np
import simpleaudio as sa

f1 = 440
f2 = 523
f3 = 660
fs = 44100  # 44100 samples per second
seconds = 3  # Note duration of 3 seconds

# Generate array with seconds*sample_rate steps, ranging between 0 and seconds
t = np.linspace(0, seconds, seconds * fs, False)
print(t)
# Generate wave form
a1 = 1
a2 = 0.3*np.greater(t,1)
a3 = 0.1*np.greater(t,2)
note = a1*np.sin(f1*t*2*np.pi) + a2*np.sin(f2*t*2*np.pi) + a3*np.sin(f3*t*2*np.pi)

# Ensure that highest value is in 16-bit range
audio = note * (2**15 - 1) / np.max(np.abs(note))
# Convert to 16-bit data
audio = audio.astype(np.int16)

# Start playback
play_obj = sa.play_buffer(audio, 1, 2, fs)

# Wait for playback to finish before exiting
play_obj.wait_done()
