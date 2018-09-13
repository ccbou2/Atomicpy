#!python3
#from capy import *
from operators import *
from qChain import *
from utility import *
from demodulating_faraday_file import *
from shutil import copyfile
import numpy as np 
import time
import matplotlib.pyplot as plt
import git
import yaml

# use .yaml filetype for parameters, python package needed.

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

		# Open desired .yaml parameter file, set by nameParams, and read data to yamlParams
		nameParams = 'test_v4'
		fNameParams = 'C:/Users/Boundsy/Documents/GitHub/Atomicpy/ParameterFiles/' \
			 + nameParams + '_params.yaml'
		paramStream = open(fNameParams, 'r')
		yamlParams = yaml.load(paramStream)
		paramStream.close()

		# Set frequencies from yamlParams
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

		# Legacy frequency parameter definitions, left for reference/use if yaml imports break

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

		# define B fields for lab frame
		def Bx(t): return 2 * rabi * np.cos(2 * np.pi * f * t - phi * np.pi/180)
		def By(t): return 0 * (t/t)
		def Bz(t): return (det - f) * (t/t)

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

		# Define whether we want to save plots
		savePlots = False

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

		# test function
		testTime = np.arange(1e-44,eperiod,1/fs)
		testSignal = np.sin(2*np.pi*f0*testTime - phi * np.pi/180)

		# compute & plot fft of signal to try figure out why we have this phase error
		# ftProj = np.fft.fft(fxProj); # Making a fourier transform
		# absftProj = np.abs(ftProj);
		# ftFreqs = np.fft.fftfreq(len(testTime), 1/fs); 
		# plt.figure(6)
		# plt.plot(ftFreqs, ftProj)

		# next, want to use James' code to try demodulate <Fx>
		# faraday_sampling_rate = fs
		if runDemod is True:
			demodFxProjQ = demod_from_array(fxProj, reference_frequency = f, faraday_sampling_rate = fs, \
				lowpas_freq = fPass, plot_demod = True, save = savePlots, time_stamp = str(tStamp), label = 'Q(t)')
			demodFxProjI = demod_from_array(fxProj, reference_frequency = f, faraday_sampling_rate = fs, \
				reference_phase_deg = 90, lowpas_freq = fPass, plot_demod = True, save = savePlots, time_stamp = str(tStamp), label = 'I(t)')
			plt.figure(4)
			plt.legend()
			plt.grid()
			plt.figure(5)
			plt.legend()
			plt.grid()

		# Check arctangent of the two components Q and I of the demodulated <Fx> projection
		phasecheck = np.arctan2(demodFxProjQ,demodFxProjI)
		phaseMarker1 = np.pi / 2 * np.ones(len(phasecheck))
		phaseMarker2 = - np.pi / 2 * np.ones(len(phasecheck))

		fNameArctan = str(tStamp) + '_arctanplot'

		# plt.figure(7)
		# plt.plot(tdomain[1:1000], phasecheck[1:1000], label = 'arctan(Q/I)')
		# plt.plot(tdomain[1:1000], phaseMarker1[1:1000], label = 'phi = pi/2')
		# plt.plot(tdomain[1:1000], phaseMarker2[1:1000], label = 'phi = -pi/2')
		# plt.xlabel('time (s)')
		# plt.ylabel('arctan(Q(t)/I(t))')
		# plt.title(fNameArctan)
		# plt.grid()
		# plt.legend()

		# if savePlots is True:
		# 	path = 'C:/Users/Boundsy/Desktop/Uni Work/PHS2360/Sim Results/' + str(fNameArctan) + '.png'
		# 	print('Arctan of demodulated projection components plot saved to Sim Results folder')
		# 	plt.savefig(path)

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