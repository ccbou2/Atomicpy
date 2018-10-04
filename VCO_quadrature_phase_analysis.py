#!python3
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt  
from lmfit import minimize, Parameters

#-------------------------------------------------
# Importing all data
#-------------------------------------------------
data = pd.read_csv('VCO_quadrature_phase_data.csv')

Freqs = np.asarray(data['F ext'].loc[3:])
Phases = np.asarray(data['Theta'].loc[3:])
Volts = np.asarray(data['Out 1'].loc[3:])

#-------------------------------------------------
# Setting up the fitting functions
#-------------------------------------------------
def residual(params, x, data):
	grad = params['gradient']
	offset = params['offset']

	model = grad*x + offset
	return data - model

def fit_line(x, data):
	params = Parameters()
	params.add('gradient', value = -0.2)
	params.add('offset', value = 0)
	
	out = minimize(residual, params, args = (x, data))
	return out

def model_function(x, grad, offset):
	y = grad*x + offset
	return y


#-------------------------------------------------
# Performing the voltage frequency fit
#-------------------------------------------------
linfit_out = fit_line(Volts, Freqs)
print(linfit_out.params['gradient'])
print(linfit_out.params['offset'])

plt.figure()
plt.plot(Volts, Freqs, 'o', label = 'data')
plt.plot(Volts, model_function(Volts, linfit_out.params['gradient'].value, linfit_out.params['offset'].value), label = 'fit')
plt.xlabel('DC modulation voltage')
plt.ylabel('Frequency')
plt.legend()

#-------------------------------------------------
# Plotting frequency vs phase difference
#-------------------------------------------------
plt.figure()
plt.plot(Freqs, Phases)
plt.title('Frequency vs channel offset phase')
plt.xlabel('Frequency')
plt.ylabel('Phase (degrees)')

plt.figure()
plt.plot(Freqs, Phases)
plt.title('Frequency vs channel offset phase')
plt.xlabel('Frequency')
plt.ylabel('Phase (degrees)')
plt.ylim(88,90)

plt.show()


