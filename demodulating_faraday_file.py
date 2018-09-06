""" Importing packages """
import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy import signal

""" Defining functioions. Low pass from third year project"""
# Defining a function to apply our low pass filter. It takes as input the dataset to be filtered, the FilterFrequency and the data
# acquisition timestep. It outputs only the filtered dataset. 
def LPFilter(Dataset,FilterFrequency,TimeStep): 
	NyquistRate = 0.5*(TimeStep**-1)
	FractionNyquist =  FilterFrequency/(NyquistRate) 
	b, a = signal.butter(5, FractionNyquist)
	FilteredOut = 2*signal.filtfilt(b, a, Dataset, padlen=0)
	return FilteredOut

def find_index_from_time(time, sampling_rate):
	""" 
	Gives the index corresponding to an input time. This relies on
	the sampling rate 

	Arguments:
		time - time needed to convert to index
		sampling_rate - sampling rate of the list
	"""
	index = int(round(time*sampling_rate))
	return index

def find_index_from_freq(freq, frequency_step):
	""" 
	Gives the index corresponding to a specific frequency (eg to find a frequency
	from a fourier transform list). Requires the frequency step of the list

	Arguments:
		freq - frequency being looked for
		frequency_step - the step between consecutive indexes
	"""
	index = int(round(freq/frequency_step))
	return index

def make_periodogram(data, actual_capture_rate, filename, saveFig = False, plot_graph = False, title = 'Periodogram', start_freq = 0, end_freq = 'end', label = ''):
	""" 
	Returns a periodogram of data, with the option to plot it as well while selecting the 
	plot range (in frequency).

	Arguments:
		data - data to be made into a spectrogram
		actual_capture_rate - sampling rate of the data
		plot_graph - True or false argument for whether to plot the graph or not
		title - title of the graph
		start_freq - start frequency 
		end_freq - end frequency of plot
	"""
	f, Pxx_spec = signal.periodogram(data, actual_capture_rate, 'flattop', scaling='spectrum')
	if plot_graph == True:
		start_index = find_index_from_freq(start_freq, f[1])
		if end_freq == 'end':
			end_index = int(len(f) - 1)
		else:
			end_index = find_index_from_freq(end_freq, f[1])
			assert end_index<len(f), "Error in make_periodogram; end index is too long for array"
		plt.figure(5, figsize = (10,15))
		plt.plot(f[start_index:end_index], Pxx_spec[start_index:end_index], label = label)
		plt.title(filename)
		plt.xlabel('Frequency (Hz)')
		plt.ylabel('Amplitude')
		plt.grid()

		if saveFig is True:
			path = 'C:/Users/Boundsy/Desktop/Uni Work/PHS2360/Sim Results/' + str(filename) + '.png'
			print('Periodogram plot saved to Sim Results folder')
			plt.savefig(path)
	return f, Pxx_spec

def demod_from_h5(h5file_name, h5data_path, faraday_sampling_rate = 5e6, start_time = 0, end_time = 'max', reference_frequency = 736089.8, reference_phase_deg = 0, lowpas_freq = 10000, plot_demod = False, decimate_factor = 4 ):
	""" 
	Sweet lord above this function washes the dishes as well. In summary it opens a h5 
	file, in the same directory as the file, extracts data according to specified 
	directory and then demodulates. Demodulation requires phase and frequency to be 
	entered. After this is decimates the data, done to upload into limited memory of
	and arbitary waveform generator. Along the way there are options to plot graphs
	in order to check outputs. It returns the decimated waveform in a numpy array. 
	
	Arguments; 
		h5file_name - Name of the h5 file data is being loaded from
		h5data_path - Path in the h5 file to the data
		faraday_sampling_rate - Sampling rate the data was recorded at 
		start_time - Selecting the start time of the data in the h5 file after which data will
		be data
		end_time - Selecting the end time of the data before which data will be included. Enter 'max' for 
		the entire array
		reference_frequency - Frequency of the reference waveform used for demodulation
		reference_phase_deg - Phase of the reference waveform used for demodulation. 
		lowpas_freq - frequency of the lowpass filter. May be the 6db point, don't know. 
		plot_demod - Plots a figure of the demodulated data
		decimate_factor - Multiple for decimating the data. 
	"""

	""" Importing the Faraday data """
	f = h5py.File(h5file_name, 'r')
	Faraday = f[h5data_path].value 

	""" Selecting Range of Faraday data. Will affect phase """
	start_index = find_index_from_time(start_time, faraday_sampling_rate)
	if end_time == 'max':
		end_index = len(Faraday) - 1
	else:
		end_index = find_index_from_time(end_time, faraday_sampling_rate) 
	Faraday_clipped = Faraday[start_index:end_index]
	assert end_index<len(Faraday), "Runtime asked for in demod_from_h5() exceeds length of initial data. Please use a shorter end_time; rounded max of {:f}. Enter = 'max' for longest span".format(float(int(len(Faraday)) -1)/faraday_sampling_rate)

	""" Creating the reference """
	dt = 1/faraday_sampling_rate
	time_axis = np.arange(len(Faraday_clipped))*dt
	reference = np.sin(2*np.pi*reference_frequency*time_axis + reference_phase_deg*np.pi/180)

	""" Multiplying Faraday with reference and lowpassing """
	multiplied_waves = np.multiply(Faraday_clipped, reference)
	demodulated = LPFilter(multiplied_waves, lowpas_freq, dt)

	""" Figures to check demodulation """
	freq, amp = make_periodogram(demodulated, faraday_sampling_rate, plot_graph = plot_demod, title = 'Periodogram', start_freq = 0, end_freq = 10000)
	if plot_demod == True:
		plt.figure(figsize = (10,7.5))
		plt.plot(time_axis, demodulated)
		plt.xlabel('Time')
		plt.title('Demoulated Faraday')
		plt.show()

	""" decimating data """
	decimated_demod = signal.decimate(demodulated, decimate_factor, n=None, ftype='iir', axis=-1, zero_phase=True)
	
	return decimated_demod

def demod_from_array(mod_array, faraday_sampling_rate = 5e6, start_time = 0, end_time = 'max', reference_frequency = 736089.8, reference_phase_deg = 0, lowpas_freq = 10000, plot_demod = False, decimate_factor = 4, time_stamp = '', label = '', save = False):
	""" 
	Sweet lord above this function washes the dishes as well. In summary it opens a h5 
	file, in the same directory as the file, extracts data according to specified 
	directory and then demodulates. Demodulation requires phase and frequency to be 
	entered. After this is decimates the data, done to upload into limited memory of
	and arbitary waveform generator. Along the way there are options to plot graphs
	in order to check outputs. It returns the decimated waveform in a numpy array. 
	
	Arguments; 
		h5file_name - Name of the h5 file data is being loaded from
		h5data_path - Path in the h5 file to the data
		faraday_sampling_rate - Sampling rate the data was recorded at 
		start_time - Selecting the start time of the data in the h5 file after which data will
		be data
		end_time - Selecting the end time of the data before which data will be included. Enter 'max' for 
		the entire array
		reference_frequency - Frequency of the reference waveform used for demodulation
		reference_phase_deg - Phase of the reference waveform used for demodulation. 
		lowpas_freq - frequency of the lowpass filter. May be the 6db point, don't know. 
		plot_demod - Plots a figure of the demodulated data
		decimate_factor - Multiple for decimating the data. 
	"""

	
	""" Creating the reference """
	Faraday_clipped = mod_array
	dt = 1/faraday_sampling_rate
	time_axis = np.arange(len(Faraday_clipped))*dt
	reference = np.sin(2*np.pi*reference_frequency*time_axis + reference_phase_deg*np.pi/180)

	""" Multiplying Faraday with reference and lowpassing """
	multiplied_waves = np.multiply(Faraday_clipped, reference)
	demodulated = LPFilter(multiplied_waves, lowpas_freq, dt)

	""" Figures to check demodulation """
	fNameDemod = str(time_stamp) + '_Fx_demodulated'

	if plot_demod == True:
		plt.figure(4, figsize = (10,7.5))
		plt.plot(time_axis, demodulated, label = label)
		plt.xlabel('Time (s)')
		plt.ylabel('Demodulated <Fx>')
		plt.xlim(time_axis[0], time_axis[-1])
		plt.title(fNameDemod)
		plt.grid()

		if save is True:
			path = 'C:/Users/Boundsy/Desktop/Uni Work/PHS2360/Sim Results/' + str(fNameDemod) + '.png'
			print('Demodulated Fx plot saved to Sim Results folder')
			plt.savefig(path)
			
	fNamePeriodogram = str(time_stamp) + '_demod_periodogram'
	freq, amp = make_periodogram(demodulated, faraday_sampling_rate, fNamePeriodogram, saveFig = save, plot_graph = plot_demod, start_freq = 0, end_freq = 20000, label = label)

	""" decimating data """
	# decimated_demod = signal.decimate(demodulated, decimate_factor, n=None, ftype='iir', axis=-1, zero_phase=True)
	
	# return decimated_demod
	return demodulated