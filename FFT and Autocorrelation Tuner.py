import sounddevice as sd
import numpy as np

# waitRecord keeps the loop going until sound above ambient noise level occurs
def waitRecord(samplingRate):
    waitSoundVolume = sd.rec(int(samplingRate*0.5), samplerate=samplingRate, channels=1, dtype='float64')
    sd.wait()
    waitSoundVolume = waitSoundVolume.flatten()
    waitSoundVolume = np.linalg.norm(waitSoundVolume)
    return waitSoundVolume

def recordSound(duration, samplingRate):
    # Record actual sound
    sound = sd.rec(int(duration*samplingRate), samplerate=samplingRate, channels=1, dtype='float64')
    sd.wait()
    sound = sound.flatten()
    return sound

"""
By the Wiener-Khinchin theorem, The autocorrelation of a signal is
inverseFFT(abs(FFT(signal)^2))
"""
# fmin and fmax represent the min and max frequency respectively
def autocorrelation_fft(sound, duration, samplingRate, fmin=15, fmax=4000):
    # Centers average waveform value to zero
    sound = sound - np.mean(sound)
    
    # Check for silent signal
    if np.linalg.norm(sound) < 1e-6:
        return 0

    # Compute FFT of audio
    fft = np.fft.rfft(sound)
    # Take the fft^2
    powerSpectrum = np.abs(fft)**2

    # Autocorrelation = Inverse FFT(powerSpectrum)
    autocorrelation = np.fft.ifft(powerSpectrum).real

    # Keep only positive lags
    autocorrelation = autocorrelation[:int(duration*samplingRate)]  

    # Generally, short lags correspond to high frequencies 
    lagMin = int(samplingRate / fmax)
    lagMax = int(samplingRate / fmin)
    # Makes sure lagMax will not exceed the length of autocorrelation
    lagMax = min(lagMax, len(autocorrelation))

    # Sets all values less than lagMin to 0 
    autocorrelation[:lagMin] = 0
    # Sets all values greater than lagMax to 0
    autocorrelation[lagMax:] = 0

    lag = np.argmax(autocorrelation)

    # Checks for empty or extremely short signal
    if len(autocorrelation) < lagMin:
        return 0

    # Returns frequency if peak is found
    if lag != 0:
        frequency = samplingRate / lag
        return frequency
    else: 
        return 0

def frequencyToMidi(frequency):
    # 440 Hz is MIDI number 69, each step is 12*log2(frequency/440) away
    return 69.0 + 12.0 * np.log2(frequency / 440)

def midiToNote(midi):
    # Convert MIDI number to note name and octave
    midiRounded = int(round(midi))
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midiRounded // 12) - 1
    noteIndex = midiRounded % 12
    return notes[noteIndex], octave

def cents(frequency):
    # Calculate cents deviation from nearest note
    midi = frequencyToMidi(frequency)
    midiNearest = round(midi)
    return (midi - midiNearest) * 100

def main():
    samplingRate = 44100
    duration = 1 
    threshold = 0.1

    # Wait for sound above threshold
    while waitRecord(samplingRate) < threshold:
        pass  
    
    # Record sound and estimate frequency
    frequency = autocorrelation_fft(recordSound(duration, samplingRate), duration, samplingRate, 15, 4000)
    
    if frequency == 0:
        print("No sound detected or no clear pitch.")
        return

    # Get closest note and cents deviation
    note, octave = midiToNote(frequencyToMidi(frequency))
    centsDeviation = cents(frequency)

    print(f"Frequency: {frequency:.2f} Hz")
    print(f"Closest Note: {note}{octave}")
    print(f"Cents deviation: {centsDeviation:+.2f}")
