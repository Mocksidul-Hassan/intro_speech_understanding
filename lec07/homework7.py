import numpy as np

def major_chord(f, Fs):
    '''
    Generate a one-half-second major chord, based at frequency f, with sampling frequency Fs.

    @param:
    f (scalar): frequency of the root tone, in Hertz
    Fs (scalar): sampling frequency, in samples/second

    @return:
    x (array): a one-half-second waveform containing the chord
    
    A major chord is three notes, played at the same time:
    (1) The root tone (f)
    (2) A major third, i.e., four semitones above f
    (3) A major fifth, i.e., seven semitones above f
    '''
    N = int(0.5 * Fs)
    n = np.arange(N)
    
    f2 = f * (2 ** (4/12))   # major third
    f3 = f * (2 ** (7/12))   # major fifth
    
    x = (np.cos(2 * np.pi * f * n / Fs) +
         np.cos(2 * np.pi * f2 * n / Fs) +
         np.cos(2 * np.pi * f3 * n / Fs))
    return x

def dft_matrix(N):
    '''
    Create a DFT transform matrix, W, of size N.
    
    @param:
    N (scalar): number of columns in the transform matrix
    
    @result:
    W (NxN array): a matrix of dtype='complex' whose (k,n)^th element is:
           W[k,n] = cos(2*np.pi*k*n/N) - j*sin(2*np.pi*k*n/N)
    '''
    k = np.arange(N).reshape(-1, 1)  # column vector
    n = np.arange(N).reshape(1, -1)  # row vector
    W = np.cos(2 * np.pi * k * n / N) - 1j * np.sin(2 * np.pi * k * n / N)
    return W

def spectral_analysis(x, Fs):
    '''
    Find the three loudest frequencies in x.

    @param:
    x (array): the waveform
    Fs (scalar): sampling frequency (samples/second)

    @return:
    f1, f2, f3: The three loudest frequencies (in Hertz)
      These should be sorted so f1 < f2 < f3.
    '''
    X = np.fft.fft(x)
    mag = np.abs(X)
    

    N = len(x)
    half = N // 2
    mag = mag[:half]
    
    
    indices = np.argsort(mag)[-3:]   # top 3 indices
    freqs = indices * Fs / N         # convert bin to Hz
    
    f1, f2, f3 = sorted(freqs)
    return f1, f2, f3