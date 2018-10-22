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


for i in range(len(detuning)):
	det = detuning[i];
	Iarray = smallSim(detFreq = det, output = True, save = False);

	# Write to hdf5 file
	h5name = 'C:/Users/Boundsy/Documents/GitHub/Atomicpy/SmallSignals/detScan2.h5'
	with h5py.File(h5name, 'a') as hf:
		hf.create_dataset('det' + str(det),  data=Iarray)

	print('Array saved to hdf5 in SmallSignals subfolder')

print('sims complete')

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

# xvals = np.linspace(0,0.01,len(datam50))

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

plt.plot(detuning, Ifin)
plt.grid()
plt.xlabel('Detuning')
plt.ylabel('Final I(t) value @ 0.01s')
plt.title('Small Signal final value vs Detuning')
plt.show()