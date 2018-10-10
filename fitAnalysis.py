#!python3
import numpy as np
import matplotlib.pyplot as plt
import yaml

#######################################################################
# Import yaml fit parameters to analyse
#######################################################################

# Define filenames via timestamp
name10kHz = '20181004T124047';
name5kHz = '20181004T130715';
name1kHz = '20181004T130225';

# Convert timestamp name into file path w/ extension
fname10kHz = 'C:/Users/Boundsy/Documents/GitHub/Atomicpy/ParameterFiles/' \
	+ name10kHz + '_fitVals.yaml'
fname5kHz = 'C:/Users/Boundsy/Documents/GitHub/Atomicpy/ParameterFiles/' \
	+ name5kHz + '_fitVals.yaml'
fname1kHz = 'C:/Users/Boundsy/Documents/GitHub/Atomicpy/ParameterFiles/' \
	+ name1kHz + '_fitVals.yaml'

# Open and import fit parameters from each file
rabi10Stream = open(fname10kHz, 'r')
rabi10params = yaml.load(rabi10Stream)
rabi10Stream.close()

rabi5Stream = open(fname5kHz, 'r')
rabi5params = yaml.load(rabi5Stream)
rabi5Stream.close()

rabi1Stream = open(fname1kHz, 'r')
rabi1params = yaml.load(rabi1Stream)
rabi1Stream.close()

# Take relevant param strings from imports
freq10 = rabi10params['f']
Amp10 = rabi10params['A']
phase10 = rabi10params['phi']

freq5 = rabi5params['f']
Amp5 = rabi5params['A']
phase5 = rabi5params['phi']

freq1 = rabi1params['f']
Amp1 = rabi1params['A']
phase1 = rabi1params['phi']

# define rabi frequencies in Hz
rabi10 = 10000;
rabi5 = 5000;
rabi1 = 1000;

# define bias frequency
bias = 699900.0;

#######################################################################
# Convert into usable format of floats, including uncertainties
#######################################################################

# Break up string into value and uncertainty for Rabi freq of 10kHz;
f10 = freq10[:freq10.index('+')-1:]
f10 = np.float(f10)
u_f10 = freq10[freq10.index('/')+2:]
u_f10 = np.float(u_f10)

A10 = Amp10[:Amp10.index('+')-1:]
A10 = np.float(A10)
u_A10 = Amp10[Amp10.index('/')+2:]
u_A10 = np.float(u_A10)

phi10 = phase10[:phase10.index('+')-1:]
phi10 = np.float(phi10)
u_phi10 = phase10[phase10.index('/')+2:]
u_phi10 = np.float(u_phi10)

# Break up string into value and uncertainty for Rabi freq of 5kHz;
f5 = freq5[:freq5.index('+')-1:]
f5 = np.float(f5)
u_f5 = freq5[freq5.index('/')+2:]
u_f5 = np.float(u_f5)

A5 = Amp5[:Amp5.index('+')-1:]
A5 = np.float(A5)
u_A5 = Amp5[Amp5.index('/')+2:]
u_A5 = np.float(u_A5)

phi5 = phase5[:phase5.index('+')-1:]
phi5 = np.float(phi5)
u_phi5 = phase5[phase5.index('/')+2:]
u_phi5 = np.float(u_phi5)

# Break up string into value and uncertainty for Rabi freq of 1kHz;
f1 = freq1[:freq1.index('+')-1:]
f1 = np.float(f1)
u_f1 = freq10[freq1.index('/')+2:]
u_f1 = np.float(u_f1)

A1 = Amp1[:Amp1.index('+')-1:]
A1 = np.float(A1)
u_A1 = Amp1[Amp1.index('/')+2:]
u_A1 = np.float(u_A1)

phi1 = phase1[:phase1.index('+')-1:]
phi1 = np.float(phi1)
u_phi1 = phase1[phase1.index('/')+2:]
u_phi1 = np.float(u_phi1)

#######################################################################
# Assuming A ~ c, compute detuning from remnant ampltidue
#######################################################################

# Define function to compute detuning under small angle approximation
def detuning(rabi, epsilon):
	det = rabi*epsilon
	return det

# Compute detunings
det10 = detuning(rabi10, A10)
u_det10 = u_A10/A10 * det10

det5 = detuning(rabi5, A5)
u_det5 = u_A5/A5 * det5

det1 = detuning(rabi1, A1)
u_det1 = u_A1/A1 * det1

# print detunings
print('Detuning at 10kHz Rabi frquency: ' + str(det10) + ' +/- ' + str(u_det10) + ' Hz')
print('Detuning at 5kHz Rabi frquency: ' + str(det5) + ' +/- ' + str(u_det5) + ' Hz')
print('Detuning at 1kHz Rabi frquency: ' + str(det1) + ' +/- ' + str(u_det1) + ' Hz')

#######################################################################
# Compute theoretical Bloch-Seigert shift to compare
#######################################################################

# Define function to compute Bloch-Seigert shift
def blochSeigert(rabi, w_0):
	bss = 0.25 * rabi**2 / w_0
	return bss

# Compute Bloch-Seigert Shifts
bss10 = blochSeigert(rabi10, bias)
bss5 = blochSeigert(rabi5, bias)
bss1 = blochSeigert(rabi1, bias)

print('Bloch-Seigert shift at 10kHz Rabi frquency: ' + str(bss10) + ' Hz')
print('Bloch-Seigert shift at 5kHz Rabi frquency: ' + str(bss5) + ' Hz')
print('Bloch-Seigert shift at 1kHz Rabi frquency: ' + str(bss1) + ' Hz')