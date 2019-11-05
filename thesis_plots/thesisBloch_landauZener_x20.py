#!python3
#from capy import *
import sys
sys.path.insert(0, "C:/Users/ccbou2/GitHub/Atomicpy")
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

savePlots = True;
# Folder for saving output files
folder = 'C:/Users/ccbou2/GitHub/HonoursThesis/Figures/'

# Run main script
if __name__ == "__main__":

		#######################################################################
		# Legacy frequency parameter definitions, left for reference/use if yaml imports break
		#######################################################################

		# sampling frequncy of unitary (Hz)
		# fs = 1e8
		# fs = 1e8
		fs = 1e5
		# fs = 5e4
		# time to evolve state for (seconds) 
		# eperiod = 1e-1
		eperiod = 10/10
		# eperiod = 1e-1

		# Define frequencies for the interacting EM field, detuning, Rabi frequency & phase, all in Hz
		f0 = gyro;   # lab value for bias field splitting
		det = 0;    # lab value for detuning
		# rabi = 7.5e3;    # lab value for Rabi frequency
		rabi = 5e4;    # lab value for Rabi frequency
		phi = 0;
		f = f0 + det;

		# Small signal params;
		rabi_2 = 100;
		# f0_2 = 10000;
		f0_2 = rabi;
		# f0_2 = rabi/(2*np.pi);
		det_2 = 0;
		f_2 = f0_2 + det_2;
		phi_2 = 0;
		# eperiod = 0.1;
		fitIremnant = False;

		# Landau-Zener params;
		# sweepRate = 2e4;
		# eperiod = 100/rabi_2
		# t0 = -eperiod/2
		# sweepRate = 500;
		sweepRate = 500*10;
		sweep_init = -sweepRate*eperiod/(2*10)
		# eperiod = 0.02/2

		# initDet = -eperiod/2 * sweepRate
		# rabi_init = rabi - eperiod/2 * sweepRate

		#######################################################################
		# B field definitions & simulation code
		#######################################################################

		# define B fields for lab frame
		# def Bx(t): return 2 * rabi * np.cos(2 * np.pi * f * t - phi * np.pi/180)
		# def Bx(t): return 2 * rabi * np.cos(2 * np.pi * f * t - phi)	# For phi in rads
		# def Bx(t): return 0 * (t/t)	# For phi in rads
		# def By(t): return 0 * (t/t)
		# def Bz(t): return (det - f) * (t/t)
		# def Bz(t): return (det - f) * (t/t) + 2 * rabi_2 * np.sin(2 * np.pi * f_2 * t - phi_2)

		# define B fields in second, tipped rotating frame for LZ
		def Bx(t): return rabi_2 * np.cos(phi_2 * (t/t))
		def By(t): return rabi_2 * np.sin(phi_2 * (t/t))
		# def Bz(t): return det * (t/t)
		def Bz(t): return (sweep_init + sweepRate*t)

		# define B fields in second rotating frame for Landau-Zener
		# def Bx(t): return rabi * np.cos(phi * (t/t))
		# def Bx(t): return (rabi_init + sweepRate * t) * np.cos(phi * (t/t))
		# def Bx(t): return 0 * (t/t)
		# def By(t): return rabi * np.sin(phi * (t/t))
		# def By(t): return (rabi_init + sweepRate * t) * np.sin(phi * (t/t))
		# def By(t): return 0 * (t/t)
		# def Bz(t): return det * (t/t) + 2 * rabi_2 * np.sin(2 * np.pi * np.sqrt(f_2**2 + (rabi_init + sweepRate * t)) * t - phi_2)
		# def Bz(t): return det * (t/t) + 2 * rabi_2 * np.sin(2 * np.pi * f_2 * t - phi_2)
		# def Bz(t): return det * (t/t) + rabi_2

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
		# atom = SpinSystem(init="super")

		# evolve state under system Hamiltonian
		start = time.time()
		tdomain, probs, pnts = atom.state_evolve(t=[1e-44, eperiod, 1/fs],          # time range and step size to evolve for
																					hamiltonian=ham.hamiltonian_cache, # system Hamiltonian
																					cache=True,                        # whether to cache calculations (faster)
																					project=meas1["+"],                # projection operator for measurement
																					bloch=[True, 3])                # Whether to save pnts for bloch state 
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
		fNameBloch = 'bloch_rot2LZ_x10rate'
		vecs = np.array([[1,0,0],[0,0,1]])
		vec_col = ['royalblue', 'g']
		view = [70,20]
		# fig = plt.figure(1, figsize = (9,9))
		# ax = fig.add_subplot(1,1,1)
		# ax = atom.bloch_plot2(fNameBloch, fig = fig, ax = ax, points=pnts, view = view, save=savePlots, vecList = vecs, vecColour = vec_col, folder = folder)
		fig, ax = atom.bloch_plot2(fNameBloch, points=pnts, view = view, save=False, vecList = vecs, vecColour = vec_col, folder = folder)
		vec1label = r'$\va{\Omega}$'
		vec2label = r'$\va{B}_z$'
		# print(type(ax))
		ax.text3D(0.05, -0.45, 0.1, vec1label, color = vec_col[0], size = 'xx-large')
		ax.text3D(0.1, -0.05, 0.45, vec2label, color = vec_col[1], size = 'xx-large')
		# atom.bloch_animate(fNameAnim, pnts, save = savePlots)
		
		print('Bloch plot saved to ' + str(folder))
		path1 = folder + str(fNameBloch) + '.png'
		path2 = folder + str(fNameBloch) + '.pdf'
		fig.savefig(path1, dpi=300, transparent=True)
		fig.savefig(path2, dpi=300, transparent=True)

		# show all plotted figures
		plt.show()
