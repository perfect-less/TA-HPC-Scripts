## Butter low pass filter
from scipy.signal import butter,filtfilt# Filter requirements.
fs = 1.0       # sample rate, Hz
cutoff = 0.06 #0.055 #0.07     # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 2 Hz
order = 8       
def butter_lowpass_filter(data, cutoff, fs, order):
    
    nyq = 0.5 * fs # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a,data)
    return y