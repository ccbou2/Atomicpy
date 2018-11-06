#!python3
#from capy import *
from operators import *
from qChain import *
from utility import *
from demodulating_faraday_file import *
from shutil import copyfile
from simFunction import *
import numpy as np 
import scipy.signal as sp
import time
import matplotlib.pyplot as plt
import git
import yaml
import h5py
from lmfit import minimize, Parameters

detuning = np.arange(-500,500,5);
# detuning = np.arange(230,500,5);
# detuning = np.arange(-50,50,2);


# for i in range(len(detuning)):
# 	det = detuning[i];
# 	Iarray = smallSim(detFreq = det, output = True, save = False);

# 	# Write to hdf5 file
# 	h5name = 'C:/Users/Boundsy/Documents/GitHub/Atomicpy/SmallSignals/detScan3.h5'
# 	with h5py.File(h5name, 'a') as hf:
# 		hf.create_dataset('det' + str(det),  data=Iarray)

# 	print('Array saved to hdf5 in SmallSignals subfolder')

# print('sims complete')

# Iarray = smallSim(detFreq = det, output = True, save = False);

#######################################################################
# Plot the varying signals
#######################################################################

# Now import and plot to check;
# h5name = 'C:/Users/Boundsy/Documents/GitHub/Atomicpy/SmallSignals/detScan.h5'
# with h5py.File(h5name, 'r') as hf:
#     datam50 = hf['det-50'][:]
#     datam20 = hf['det-20'][:]
#     datam2 = hf['det-2'][:]
#     data10 = hf['det10'][:]
#     data28 = hf['det28'][:]
#     data49 = hf['det49'][:]

# plt.plot(xvals, datam50)
# plt.plot(xvals, datam20)
# plt.plot(xvals, datam2)
# plt.plot(xvals, data10)
# plt.plot(xvals, data28)
# plt.plot(xvals, data49)
# plt.grid()
# plt.legend(['-50Hz','-20Hz', '-2Hz', '10Hz', '28Hz', '49Hz'])
# plt.xlabel('time (s)')
# plt.ylabel('I(t) amplitude')
# plt.title('I(t) w/ Small Signal @ Different Detuning')
# plt.show()

# h5name = 'C:/Users/Boundsy/Documents/GitHub/Atomicpy/SmallSignals/detScan3.h5'
# with h5py.File(h5name, 'r') as hf:
#     datam50 = hf['det-50'][:]
#     datam30 = hf['det-30'][:]
#     datam10 = hf['det-10'][:]
#     data0 = hf['det0'][:]
#     data10 = hf['det10'][:]
#     data30 = hf['det30'][:]
#     data48 = hf['det48'][:]

# xvals = np.linspace(0,0.01,len(datam50))

# plt.plot(xvals, datam50)
# plt.plot(xvals, datam30)
# plt.plot(xvals, datam10)
# plt.plot(xvals, data0)
# plt.plot(xvals, data10)
# plt.plot(xvals, data30)
# plt.plot(xvals, data48)
# plt.grid()
# plt.legend(['-50Hz','-30Hz', '-10Hz', '0Hz', '10Hz', '30Hz', '48Hz'], fontsize = '16')
# plt.xlabel('time (s)',fontsize = '20')
# plt.ylabel('I(t) amplitude', fontsize = '20')
# plt.title('I(t) w/ Small Signal @ Different Detuning', fontsize = '24')
# plt.tick_params(labelsize = '16')
# plt.show()

# h5name = 'C:/Users/Boundsy/Documents/GitHub/Atomicpy/SmallSignals/detScan2.h5'
# with h5py.File(h5name, 'r') as hf:
#     data = hf['det0'][:]

# xvals = np.linspace(0,0.01,len(data))

# plt.plot(xvals, data)
# plt.grid()
# plt.xlabel('time (s)', fontsize = '20')
# plt.ylabel('I(t) amplitude', fontsize = '20')
# plt.title('I(t) w/ Small Signal @ Resonance', fontsize = '24')
# plt.tick_params(labelsize = '16')
# plt.show()

#######################################################################
# Plot the final point for each detuning
#######################################################################

Ifin = np.zeros(len(detuning))

h5name = 'C:/Users/Boundsy/Documents/GitHub/Atomicpy/SmallSignals/detScan2.h5'

for i in range(len(detuning)):
	dataName = 'det' + str(detuning[i]);
	with h5py.File(h5name, 'r') as hf:
		data = hf[dataName][:]
	Ifin[i] = data[-1]

plt.figure()
plt.plot(detuning, Ifin)
plt.grid()
plt.xlabel('Detuning (Hz)', fontsize = '20')
plt.ylabel('Final I(t) value @ 0.01s', fontsize = '20')
plt.title('Small Signal final value vs Detuning', fontsize = '24')
plt.tick_params(labelsize = '16')
plt.show()