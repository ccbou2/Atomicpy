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
		nameParams = 'test_v2'
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
		def Bx(t): return 2 * rabi * np.cos(2 * np.pi * f * t - phi)
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
		savePlots = True

		# plot on Bloch sphere, saving timestamped filename if savePlots is true
		fNameBloch = str(tStamp) + '_blochplot'
		atom.bloch_plot(fNameBloch, pnts, savePlots)  

		# plot |<0|psi(t)>|^2 against time, saving timestamed filename if savePlots is true
		# with commit ID annotated to plot
		fNameProb = str(tStamp) + '_probplot'
		atom.prob_plot(tdomain, probs, fNameProb, commitID, savePlots)

		# compute projection <Fx>
		fxProj = 2*probs - 1;

		# plot <F_x> against time, saving timestamed filename if savePlots is true
		# with commit ID annotated to plot
		fNameProb = str(tStamp) + '_projectionplot'
		atom.project_plot(tdomain, fxProj, fNameProb, commitID, savePlots)

		# next, want to use James' code to try demodulate <Fx>
		if runDemod is True:
			demodFxProj = demod_from_array(fxProj, faraday_sampling_rate = fs, reference_frequency = f0, \
				lowpas_freq = fPass, plot_demod = True, save = savePlots, time_stamp = str(tStamp))

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