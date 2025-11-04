import numpy as np
import sounddevice as sd

def waitRecord(samplingRate):
    waitSoundVolume = sd.rec(int(samplingRate*0.5), samplerate = samplingRate, channels = 1, dtype = 'float64')
    sd.wait()
    waitSoundVolume = waitSoundVolume.flatten()
    waitSoundVolume = np.linalg.norm(waitSoundVolume)
    return waitSoundVolume

def recordSound(duration, samplingRate):
    sound = sd.rec(int(duration*samplingRate), samplerate=samplingRate, channels=1, dtype= 'float64')
    sd.wait()
    sound = sound.flatten()
    return sound

def findFrequency(duration, samplingRate):
    sound = recordSound(duration, samplingRate)
    sound = sound * np.hanning(len(sound)) # Apply Hanning window to minimize spectral leakage
    sound = abs(np.fft.rfft(sound)) # Take magnitude
    hpsSpectrum = sound.copy()
    roughFrequency = np.argmax(sound) * samplingRate / len(sound)
    harmonic = 0
    if roughFrequency < 250:
        harmonic = 3
    elif roughFrequency < 500:
        harmonic = 4
    else:
        harmonic = 5
    # Apply Harmonic Product Spectrum (HPS) to remove harmonic frequencies to find fundamental frequency    
    for i in range(2, harmonic + 1):
        downsampled = sound[::i]
        hpsSpectrum[:len(downsampled)] *= downsampled
    index = np.argmax(hpsSpectrum)
    frequency = index * samplingRate / (samplingRate * duration) # frequency = index * samplingRate / number of samples
    return frequency


def noteToFreq(noteName):
    noteToFreqDict = {
                      'C': 16.35,
                      'C#': 17.32,
                      'D': 18.35,
                      'D#': 19.45,
                      'E': 20.60,
                      'F': 21.83,
                      'F#': 23.12,
                      'G': 24.50,
                      'G#': 25.96,
                      'A': 27.50,
                      'A#': 29.14,
                      'B': 30.87
                     }
    return noteToFreqDict[noteName]

def main():
    noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    samplingRate = 44100
    duration = 1
    threshold = 1
    
    while waitRecord(samplingRate) < threshold:
        pass
    frequency = findFrequency(duration, samplingRate)
    frequency = round(frequency, 2) # Rounds frequency to 2 decimal places
    
    print(f"{frequency} Hz:")
while True:
    main()