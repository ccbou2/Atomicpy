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


# use .yaml filetype for parameters, python package needed.

# Define whether we want to save plots
savePlots = False

#######################################################################
# Repository checks
#######################################################################

# Check git repository status
repo = git.Repo(search_parent_directories=True)
dirtyRepo = repo.is_dirty()

# If repository is dirty i.e. out of date, ask for input to continue and exit if desired
if dirtyRepo is True:
	status = input("Script has changed since last commit, continue? (y/n): ")
	if status is 'n':
		raise SystemExit

# Take current commit ID and print
sha = repo.head.object.hexsha
commitID = sha[:7]
print('Current commit version is ' + str(commitID))

# Run main script
if __name__ == "__main__":

		#######################################################################
		# Import simulation parameters
		#######################################################################

		# Open desired .yaml parameter file, set by nameParams, and read data to yamlParams
		nameParams = 'test_v3_shift'
		fNameParams = 'C:/Users/Boundsy/Documents/GitHub/Atomicpy/ParameterFiles/' \
			 + nameParams + '_params.yaml'
		paramStream = open(fNameParams, 'r')
		yamlParams = yaml.load(paramStream)
		paramStream.close()

		# Set simulation parameters from yamlParams
		# sampling frequncy of unitary (Hz)
		fs = yamlParams['stepFreq']
		# time to evolve state for (seconds) 
		eperiod = yamlParams['evolutionTime']
		# bias frequency
		f0 = yamlParams['biasFreq']
		# detuning frequency
		det = yamlParams['detuningFreq']
		# rabi frequency
		rabi = yamlParams['rabiFreq']
		# phase offset
		phi = yamlParams['phaseOffset']
		# phi = 0
		# total frequency
		f = f0 + det
		# low pass filter frequency
		if 'lowPassFreq' in yamlParams:
			fPass = yamlParams['lowPassFreq'] 
			runDemod = True
		else:
			runDemod = False
		# phase shift computed from previous phase error correcting run
		if 'refPhiShift' in yamlParams:
			phiShift = yamlParams['refPhiShift']
			fitIremnant = True
		else:
			phiShift = 0
			fitIremnant = False

		# Small signal params;
		rabi_2 = 1
		f0_2 = 10000
		det_2 = 0
		f_2 = f0_2 + det_2
		phi_2 = 0
		eperiod = 0.1
		fitIremnant = False

		#######################################################################
		# Legacy frequency parameter definitions, left for reference/use if yaml imports break
		#######################################################################

		# # sampling frequncy of unitary (Hz)
		# fs = 1e8
		# # time to evolve state for (seconds) 
		# eperiod = 1e-4

		# # Define frequencies for the interacting EM field, detuning, Rabi frequency & phase, all in Hz
		# f0 = gyro;   # lab value for bias field splitting
		# det = 0;    # lab value for detuning
		# rabi = 1e4;    # lab value for Rabi frequency
		# phi = 0;
		# f = f0 + det;

		#######################################################################
		# B field definitions & simulation code
		#######################################################################

		# define B fields for lab frame
		# def Bx(t): return 2 * rabi * np.cos(2 * np.pi * f * t - phi * np.pi/180)
		def Bx(t): return 2 * rabi * np.cos(2 * np.pi * f * t - phi)	# For phi in rads
		def By(t): return 0 * (t/t)
		def Bz(t): return (det - f) * (t/t) + 2 * rabi_2 * np.cos(2 * np.pi * f_2 * t - phi_2)
		# def Bz(t): return (det - f) * (t/t)

		# define B fields in first rotating frame
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

		# evolve state under system Hamiltonian
		start = time.time()
		tdomain, probs, pnts = atom.state_evolve(t=[1e-44, eperiod, 1/fs],          # time range and step size to evolve for
																					hamiltonian=ham.hamiltonian_cache, # system Hamiltonian
																					cache=True,                        # whether to cache calculations (faster)
																					project=meas1["+"],                # projection operator for measurement
																					bloch=[True, 100])                # Whether to save pnts for bloch state 
																																						 # and save interval
		end = time.time()


		# frame_anaylser(atom.state_cache, frmame)

		print("Atomic state evolution over {} seconds with Fs = {:.2f} MHz took {:.3f} seconds".format(eperiod, fs/1e6, end-start))
		
		# Save timestamp for current sim
		tStamp = time.strftime( "%Y%m%dT%H%M%S")

		#######################################################################
		# Initial plotting via imported functions
		#######################################################################

		# plot on Bloch sphere, saving timestamped filename if savePlots is true
		fNameBloch = str(tStamp) + '_blochplot'
		atom.bloch_plot(fNameBloch, pnts, savePlots)  

		# plot |<0|psi(t)>|^2 against time, saving timestamed filename if savePlots is true
		# with commit ID annotated to plot
		fNameProb = str(tStamp) + '_probplot'
		atom.prob_plot(tdomain, probs, fNameProb, commitID, savePlots)

		# plt.show()
		# raise SystemExit

		# compute projection <Fx>
		fxProj = 2*probs - 1;

		# plot <F_x> against time, saving timestamed filename if savePlots is true
		# with commit ID annotated to plot
		fNameProb = str(tStamp) + '_projectionplot'
		# atom.project_plot(tdomain[1:2000], fxProj[1:2000], fNameProb, commitID, savePlots)
		atom.project_plot(tdomain, fxProj, fNameProb, commitID, savePlots)

		# Overlay dressing field
		# dressField = np.cos(2 * np.pi * f * tdomain[1:100] - (phi * np.pi/180))
		# dressField = np.cos(2 * np.pi * f * tdomain)
		# plt.plot(tdomain[1:2000],dressField[1:2000])

		# # test function
		# testTime = np.arange(1e-44,eperiod,1/fs)
		# testSignal = np.sin(2*np.pi*f0*testTime - phi * np.pi/180)

		#######################################################################
		# Demodulation of projection data via James' code
		#######################################################################

		# next, want to use James' code to try demodulate <Fx>
		# faraday_sampling_rate = fs

		# phiShift = 2.55700544e-02 * 180/np.pi;
		# phiShift = 1.84302201e-02 * 180/np.pi;
		# phiShift = (2.55700544e-02 + 1.84302201e-02)/2 * (180/np.pi);

		if runDemod is True:
			demodFxProjQ = demod_from_array(fxProj, reference_frequency = f, faraday_sampling_rate = fs, \
				reference_phase_deg = phiShift, lowpas_freq = fPass, plot_demod = True, save = savePlots, time_stamp = str(tStamp), label = 'Q(t)')
			demodFxProjI = demod_from_array(fxProj, reference_frequency = f, faraday_sampling_rate = fs, \
				reference_phase_deg = (90 + phiShift), lowpas_freq = fPass, plot_demod = True, save = savePlots, time_stamp = str(tStamp), label = 'I(t)')
			plt.figure(4)
			plt.legend()
			plt.grid()
			plt.figure(5)
			plt.legend()
			plt.grid()

		#######################################################################
		# Phase error checking & plotting
		#######################################################################

		# Check arctangent of the two components Q and I of the demodulated <Fx> projection
		phasecheck = np.arctan2(demodFxProjQ,demodFxProjI)
		phaseMarker1 = np.pi / 2 * np.ones(len(phasecheck))
		phaseMarker2 = - np.pi / 2 * np.ones(len(phasecheck))

		tDemod = np.arange(len(demodFxProjQ)) * (100/fs)

		fNameArctan = str(tStamp) + '_arctanplot'

		plt.figure(7)
		# plt.plot(tdomain[1:1000], phasecheck[1:1000], label = 'arctan(Q/I)')
		# plt.plot(tdomain[1:1000], phaseMarker1[1:1000], label = 'phi = pi/2')
		# plt.plot(tdomain[1:1000], phaseMarker2[1:1000], label = 'phi = -pi/2')
		plt.plot(tDemod, phasecheck, label = 'arctan(Q/I)')
		plt.plot(tDemod, phaseMarker1, label = 'phi = pi/2')
		plt.plot(tDemod, phaseMarker2, label = 'phi = -pi/2')
		plt.xlabel('time (s)')
		plt.ylabel('arctan(Q(t)/I(t))')
		plt.title(fNameArctan)
		plt.grid()
		plt.legend()

		# Save arctan plot
		if savePlots is True:
			path = 'C:/Users/Boundsy/Desktop/Uni Work/PHS2360/Sim Results/' + str(fNameArctan) + '.png'
			print('Arctan of demodulated projection components plot saved to Sim Results folder')
			plt.savefig(path)

		# Now, find extrema positions of demod data using scipy.signal functions
		demodMaxInd = np.asarray(sp.argrelmax(demodFxProjQ))
		demodMinInd = np.asarray(sp.argrelmin(demodFxProjQ))

		# Using indices, find difference between arctan of Q and I components
		# and expected +- pi/2 for the extrema positions
		minimaPhase = np.zeros(np.shape(demodMinInd)[1])
		maximaPhase = np.zeros(np.shape(demodMaxInd)[1])
		Qmaxima = np.zeros(np.shape(demodMaxInd)[1])
		Qminima = np.zeros(np.shape(demodMinInd)[1])
		Imaxima = np.zeros(np.shape(demodMaxInd)[1])
		Iminima = np.zeros(np.shape(demodMinInd)[1])

		for i in range(0, np.shape(demodMinInd)[1]):
			index = demodMinInd[0][i]
			minimaPhase[i] = (-np.pi/2) - phasecheck[index]
			Qminima[i] = demodFxProjQ[index]
			Iminima[i] = demodFxProjI[index]

		for i in range(0, np.shape(demodMaxInd)[1]):
			index = demodMaxInd[0][i]
			maximaPhase[i] = (np.pi/2) - phasecheck[index]
			Qmaxima[i] = demodFxProjQ[index]
			Imaxima[i] = demodFxProjI[index]

		# Remove first and last maxima where LP filter is spinning up/winding down figuratively
		minimaPhase = minimaPhase[1:-1]
		maximaPhase = maximaPhase[:-1]
		# # Need to remove extra minima that pops up as LP filter spins down for f_Rabi = 1kHz
		# minimaPhase = minimaPhase[0:-1]

		# ERROR CHECKING: Find and print amplitude of Q and I components
		QmaxAv = np.mean(Qmaxima)
		QminAv = np.mean(Qminima)
		Qamp = (QmaxAv - QminAv) / 2

		ImaxAv = np.mean(Imaxima)
		IminAv = np.mean(Iminima)
		Iamp = (ImaxAv - IminAv) / 2

		print('Amplitude of Q(t) is ' + np.format_float_scientific(Qamp, precision = 6))
		print('Amplitude of I(t) is ' + np.format_float_scientific(Iamp, precision = 6))

		# Compute average of minima and maxima seperately and standard deviation
		phaseDiffMin = np.format_float_scientific(np.mean(minimaPhase), precision=8)
		u_phaseDiffMin = np.format_float_scientific(np.std(minimaPhase), precision=2)

		phaseDiffMax = np.format_float_scientific(np.mean(maximaPhase), precision=8)
		u_phaseDiffMax = np.format_float_scientific(np.std(maximaPhase), precision=2)

		# Print results of average w/ standard deviation for uncertainty
		print('phaseDiff for minima is ' + str(phaseDiffMin) + '(' + str(u_phaseDiffMin) + ')')
		print('phaseDiff for maxima is ' + str(phaseDiffMax) + '(' + str(u_phaseDiffMax) + ')')

		# Plot results of difference for extrema
		fNamePhase = str(tStamp) + '_phaseErrorPlot'
		maxText = 'maximaAv:' + str(phaseDiffMax) + '(' + str(u_phaseDiffMax) + ')'
		minText = 'minimaAv:' + str(phaseDiffMin) + '(' + str(u_phaseDiffMin) + ')'

		plt.figure(8)
		plt.plot(minimaPhase)
		plt.plot(maximaPhase)
		plt.ylabel('arctan(Q/I)')
		plt.xlabel('Extrema index')
		plt.title(fNamePhase)
		plt.annotate('pi/2 - abs(arctan) averages:', xy=(0.72,0.55), xycoords = 'axes fraction', fontsize=12)
		plt.annotate(maxText, xy=(0.72,0.51), xycoords = 'axes fraction', fontsize=12)
		plt.annotate(minText, xy=(0.72,0.48), xycoords = 'axes fraction', fontsize=12)
		plt.legend(['Minima','Maxima'])

		# Print average of the minima and maxima phases
		extremaAv = (np.mean(minimaPhase) + np.mean(maximaPhase))/2 * (180/np.pi);
		print('reference phi shift required : ' + str(extremaAv))

		# Save phaseError plot
		if savePlots is True:
			path = 'C:/Users/Boundsy/Desktop/Uni Work/PHS2360/Sim Results/' + str(fNamePhase) + '.png'
			print('Phase error of demodulated projection components plot saved to Sim Results folder')
			plt.savefig(path)

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
				
				model = A * np.sin(2 * np.pi * f * t - phi) + c
				return data - model

			def fit_sine(t, data):
				params = Parameters()
				params.add('frequency', value = rabi)
				params.add('phaseShift', value = 0)
				params.add('amplitude', value = 0.01)
				params.add('offset', value = 0.005)

				out = minimize(residual, params, args = (t, data))
				return out

			def model_function(t, f, phi, A, c):
				y = A * np.sin(2 * np.pi * f * t - phi) + c
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
				sineFit_out.params['amplitude'], sineFit_out.params['offset'])

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

			# Plot the fit
			fNameRemnant = str(tStamp) + '_Iremnant'
			plt.figure(9)
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

			# Save fit results if plots are being saved too
			if savePlots is True:
				path = 'C:/Users/Boundsy/Desktop/Uni Work/PHS2360/Sim Results/' + str(fNameRemnant) + '.png'
				print('plot of I(t) remnant fit saved to Sim Results folder')
				plt.savefig(path)

				fNameFitParams = 'C:/Users/Boundsy/Documents/GitHub/Atomicpy/ParameterFiles/' \
					 + str(tStamp) + '_fitVals.yaml'
				stream = open(fNameFitParams, 'w+')
				yaml.dump({'A': fitAmp,
					'f': fitFreq,
					'phi': fitPhi,
					'c': fitOff},
					stream,default_flow_style=False)
				print('Out of phase component fit parameters exported to .yaml file in ParameterFiles subfolder')


		#######################################################################
		# Show figs & export parameters
		#######################################################################

		# show all plotted figures
		plt.show()

		# Export parameters to timestamped .yaml file as record of parameters used for shot
		# can then load in to replicate 
		# Note: Have added such that only saved when plots are saved
		if savePlots is True:
			fNameYaml = 'C:/Users/Boundsy/Documents/GitHub/Atomicpy/ParameterFiles/' \
				 + str(tStamp) + '_params.yaml'
			copyfile(fNameParams, fNameYaml)
			print('Parameters exported to .yaml file in ParameterFiles subfolder')