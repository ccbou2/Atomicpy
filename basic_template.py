#!python3
# from capy import *
from qChain import *
from utility import *
from operators import *

import numpy as np 
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # bias field strength
    bias_amp = gyro

    # create annoyingly complex Fx magnetic field function
    # wrf = lambda t: bias_amp/gyro

    # create Fx field
    # Fx = lambda t: wrf(t)

    # Define frequencies for the interacting EM field, detuning, Rabi frequency & phase, all in Hz
    w = 1000;   # Get lab value
    det = 0;    # Get lab value
    rabi = 50;  # Get lab value 
    phi = 0;

    # Create lab frame fields for Fx and Fz
    xf = lambda t: 2*rabi*np.cos(w*t - phi)
    Fx = lambda t: xf(t)

    zf = lambda t: det - w
    Fz = lambda t: zf(t)


    # define generic Hamiltonian parameters with Zeeman splitting and rf dressing
    params = {"struct": ["custom",                              # Fx is a sinusoidal dressing field field
                         "constant",                              # Fy is a constant field
                         "custom"],                                # Fz is a fade field 
                         "freqb": [0, 0, 0],                   # frequency in Hz of each field vector
                         "tau":   [None, None, 1e-4],               # time event of pulse along each axis
                         "amp":   [gyro/gyro, 0/gyro, 1/gyro],        # amplitude in Gauss -> 1 Gauss ~= 700000 Hz precession
                         "misc":  [Fx, None, Fz]}          # misc parameters

    # create specified magnetic fields and resulting Hamiltonian
    fields = field_gen(field_params=params)

    ham = Hamiltonian()
    ham.generate_field_hamiltonian(fields)

    #field_plot(fields, time=np.linspace(0,1e-2,1e5))
    #exit()

    # initialise spin system in up state
    atom = SpinSystem(init="zero")

    # evolve state under system Hamiltonian
    time, probs, pnts = atom.state_evolve(t=[0, 1e-4, 1e-7],             # time range and step size to evolve for
                                          hamiltonian=ham.hamiltonian,   # system Hamiltonian
                                          project=meas1["0"],            # projection operator for measurement
                                          bloch=[True, 1])               # Whether to save pnts for bloch state 
                                                                         # and save interval

    atom.bloch_plot(pnts)                                                                    
    atom.prob_plot(time, probs)                  

                                        


  


