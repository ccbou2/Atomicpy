#!python3
#from capy import *
from operators import *
from qChain import *
from utility import *
from demodulating_faraday_file import *
from shutil import copyfile
import numpy as np 
import scipy.signal as sp
import time
import matplotlib.pyplot as plt
import git
import yaml
from lmfit import minimize, Parameters

def smallSim(detFreq=0, save=False, output=False):
		# sampling frequncy of unitary (Hz)
		fs = 1e8
		# time to evolve state for (seconds) 
		eperiod = 0.01

		# Define frequencies for the interacting EM field, detuning, Rabi frequency & phase, all in Hz
		f0 = gyro;   # lab value for bias field splitting
		det = detFreq;    # lab value for detuning
		rabi = 10000;    # lab value for Rabi frequency
		phi = 0;
		f = f0 + det;

		# Small signal params;
		rabi_2 = 10;
		f0_2 = 10000;
		det_2 = detFreq;
		f_2 = f0_2 + det_2;
		phi_2 = 0;

		# Legacy plot options
		fitIremnant = True;
		savePlots = False;

		# phase shift for reference determined from previous runs
		phiShift = 1.2605150131932397;
		fPass = 20000.0;

		print('det = ' + str(det_2))

		#######################################################################
		# B field definitions & simulation code
		#######################################################################

		# define B fields for lab frame
		# def Bx(t): return 2 * rabi * np.cos(2 * np.pi * f * t - phi * np.pi/180)
		def Bx(t): return 2 * rabi * np.cos(2 * np.pi * f * t - phi)	# For phi in rads
		def By(t): return 0 * (t/t)
		def Bz(t): return (det - f) + 2 * rabi_2 * np.cos(2 * np.pi * f_2 * t - phi_2)

		# # define B fields in first rotating frame
		# def Bx(t): return rabi * np.cos(phi * (t/t))
		# def By(t): return rabi * np.sin(phi * (t/t))
		# def Bz(t): return det * (t/t)

		# define generic Hamiltonian parameters with Zeeman splitting and rf dressing
		params = {"struct": ["custom",           # Fx is a sinusoidal dressing field field
												 "custom",           # Fy is a constant field
												 "custom"],          # Fz is a fade field 
							"freqb":  [0, 0, 0],           # frequency in Hz of each field vector
							"tau":    [None, None, None],  # time event of pulse along each axis
							"amp":    [0, 0, 0],           # amplitude in Gauss -> 1 Gauss ~= 700000 Hz precession
							"misc":   [Bx, By, Bz]}        # misc parameters

		# create specified magnetic fields and resulting Hamiltonian
		fields = field_gen(field_params=params)
		ham = Hamiltonian() 
		ham.generate_field_hamiltonian(fields)

		# initialise spin 1/2 system in zero (down) state
		atom = SpinSystem(init="zero")

		print('System prepared, running sim')

		# evolve state under system Hamiltonian
		start = time.time()
		tdomain, probs, pnts = atom.state_evolve(t=[1e-44, eperiod, 1/fs],          # time range and step size to evolve for
																					hamiltonian=ham.hamiltonian_cache, # system Hamiltonian
																					cache=True,                        # whether to cache calculations (faster)
																					project=meas1["+"],                # projection operator for measurement
																					bloch=[True, 1])                # Whether to save pnts for bloch state 
																																						 # and save interval
		end = time.time()


		# frame_anaylser(atom.state_cache, frmame)

		print("Atomic state evolution over {} seconds with Fs = {:.2f} MHz took {:.3f} seconds".format(eperiod, fs/1e6, end-start))
		print("Sim complete, beginning processing of signal...")
		
		# Save timestamp for current sim
		tStamp = time.strftime( "%Y%m%dT%H%M%S")

		# compute projection <Fx>
		fxProj = 2*probs - 1;

		#######################################################################
		# Demodulation of projection data via James' code
		#######################################################################

		# next, want to use James' code to try demodulate <Fx>
		# faraday_sampling_rate = fs

		# phiShift = 2.55700544e-02 * 180/np.pi;
		# phiShift = 1.84302201e-02 * 180/np.pi;
		phiShift = (2.55700544e-02 + 1.84302201e-02)/2 * (180/np.pi);

		demodFxProjQ = demod_from_array(fxProj, reference_frequency = f, faraday_sampling_rate = fs, \
			reference_phase_deg = phiShift, lowpas_freq = fPass, plot_demod = True, save = savePlots, time_stamp = str(tStamp), label = 'Q(t)')
		demodFxProjI = demod_from_array(fxProj, reference_frequency = f, faraday_sampling_rate = fs, \
			reference_phase_deg = (90 + phiShift), lowpas_freq = fPass, plot_demod = True, save = savePlots, time_stamp = str(tStamp), label = 'I(t)')
		tDemod = np.arange(len(demodFxProjQ)) * (100/fs)

		if fitIremnant is True:

			#######################################################################
			# Setting up lmfit for I(t) out of phase remnant
			#######################################################################

			# Set up fitting functions:
			def residual(params, t, data):
				f = params['frequency']
				phi = params['phaseShift']
				A = params['amplitude']
				c = params['offset']
				B = params['amp2']
				f_2 = params['frequency2']
				phi_2 = params['phase2']
				
				model = A * np.sin(2 * np.pi * f * t - phi) + B * np.cos(2 * np.pi * f_2 * t - phi_2) + c
				return data - model

			def fit_sine(t, data):
				params = Parameters()
				params.add('frequency', value = rabi)
				params.add('phaseShift', value = 0)
				params.add('amplitude', value = 0.01)
				params.add('offset', value = 0.005)
				params.add('frequency2', value = rabi_2)
				params.add('amp2', value = 0.1)
				params.add('phase2', value = 0)

				out = minimize(residual, params, args = (t, data))
				return out

			def model_function(t, f, phi, A, c, B, f_2, phi_2):
				y = A * np.sin(2 * np.pi * f * t - phi) + B * np.cos(2 * np.pi * f_2 * t - phi_2) + c
				return y

			#######################################################################
			# Perform fit of I(t) out of phase remnant
			#######################################################################
			
			# Cut initial and final parts of LP filter output to remove dodgy section for improved fit
			cutIndex = int(75 * 10000/rabi)
			tDemod = tDemod[cutIndex:-cutIndex];
			demodFxProjI = demodFxProjI[cutIndex:-cutIndex];

			# Perform fit
			sineFit_out = fit_sine(tDemod, demodFxProjI)
			fitData = model_function(tDemod, sineFit_out.params['frequency'], sineFit_out.params['phaseShift'], \
				sineFit_out.params['amplitude'], sineFit_out.params['offset'], sineFit_out.params['amp2'], \
				sineFit_out.params['frequency2'], sineFit_out.params['phase2'])

			# Convert parameters into printable format
			fitAmp = str(sineFit_out.params['amplitude']);
			fitAmp = fitAmp[fitAmp.index('=')+1:]
			fitAmp = fitAmp[:fitAmp.index(',')]
			print('Amplitude = ' + fitAmp)

			fitFreq = str(sineFit_out.params['frequency']);
			fitFreq = fitFreq[fitFreq.index('=')+1:]
			fitFreq = fitFreq[:fitFreq.index(',')]
			print('Frequency = ' + fitFreq)

			fitPhi = str(sineFit_out.params['phaseShift']);
			fitPhi = fitPhi[fitPhi.index('=')+1:]
			fitPhi = fitPhi[:fitPhi.index(',')]
			print('Phase = ' + fitPhi)

			fitOff = str(sineFit_out.params['offset']);
			fitOff = fitOff[fitOff.index('=')+1:]
			fitOff = fitOff[:fitOff.index(',')]
			print('Offset = ' + fitOff)

			fitFreq2 = str(sineFit_out.params['frequency2']);
			fitFreq2 = fitFreq2[fitFreq2.index('=')+1:]
			fitFreq2 = fitFreq2[:fitFreq2.index(',')]
			print('smallSig_Frequency = ' + fitFreq2)

			fitAmp2 = str(sineFit_out.params['amp2']);
			fitAmp2 = fitAmp2[fitAmp2.index('=')+1:]
			fitAmp2 = fitAmp2[:fitAmp2.index(',')]
			print('smallSig_Amplitude = ' + fitAmp2)

			fitPhi2 = str(sineFit_out.params['phase2']);
			fitPhi2 = fitPhi2[fitPhi2.index('=')+1:]
			fitPhi2 = fitPhi2[:fitPhi2.index(',')]
			print('Phase = ' + fitPhi2)

			# Plot the fit
			fNameRemnant = str(tStamp) + '_Iremnant'
			plt.figure(1)
			plt.plot(tDemod, demodFxProjI, label = 'I(t) data')
			plt.plot(tDemod, fitData, linestyle = '--', label = 'fit')
			plt.xlabel('time (s)')
			plt.ylabel('I(t) component')
			plt.title(fNameRemnant)
			# plt.annotate('fit params:', xy=(0.72,0.55), xycoords = 'axes fraction', fontsize=12)
			# plt.annotate('A = ' + fitAmp, xy=(0.72,0.51), xycoords = 'axes fraction', fontsize=12)
			# plt.annotate('f = ' + fitFreq, xy=(0.72,0.48), xycoords = 'axes fraction', fontsize=12)
			plt.legend()
			plt.grid()
			# plt.show()

			if output is True:
				return demodFxProjI

			if save is True:
				fNameFitParams = 'C:/Users/Boundsy/Documents/GitHub/Atomicpy/SmallSignals/' \
					 + str(det_2) + '_fitVals.yaml'
				stream = open(fNameFitParams, 'w+')
				yaml.dump({'A': fitAmp,
					'f': fitFreq,
					'phi': fitPhi,
					'c': fitOff,
					'A_2': fitAmp2,
					'f_2': fitFreq2,
					'phi_2': fitPhi2},
					stream,default_flow_style=False)
				print('Out of phase component fit parameters exported to .yaml file in ParameterFiles subfolder')