from pylab import plot, show, title, xlabel, ylabel, subplot, savefig
from scipy import fft, arange, ifft
from numpy import sin, linspace, pi
from scipy.io.wavfile import read, write

__author__ = 'ptoth'

def plot_spectrum(data, sampling_rate=44100):
    n = len(data)
    k = arange(n)
    T = n/sampling_rate
    frq = k/T # two sides frequency range
    frq = frq[range(n/2)] # one side frequency range

    Y = fft(data)/n # fft computing and normalization
    Y = Y[range(n/2)]

    plot(frq, abs(Y), 'r') # plotting the spectrum
    xlabel('Freq (Hz)')
    ylabel('|Y(freq)|')

if __name__ == '__main__':
    filename = '../resources/bach_goldberg/wave/aria.wav'#bach_output.wav'
    filename_test = '../resources/bach_goldberg/wave/aria.wav'

    Fs = 44100  # sampling rate

    rate_test, data_test = read(filename_test)
    y_test = data_test
    t_test = linspace(0, len(y_test)/Fs, len(y_test))

    rate, data = read(filename)
    y = data
    t = linspace(0, len(y)/Fs, len(y))

    subplot(2, 2, 1)
    plot(t_test, y_test)
    xlabel('Time')
    ylabel('Amplitude')
    title("test")
    subplot(2, 2, 2)
    plot(t, y)
    xlabel('Time')
    ylabel('Amplitude')
    title("model")
    subplot(2, 2, 3)
    plot_spectrum(y_test, Fs)
    subplot(2, 2, 4)
    plot_spectrum(y, Fs)
    show()