import numpy as np
import matplotlib.pyplot as plt


# Plancks constant
pbar = 6.626070040e-34
# reduced
hbar = pbar/(2*np.pi)
# Bohr magneton in J/Gauss
mub = (9.274009994e-24)/1e4
# g factor
gm = 2.00231930436
# Gyromagnetic ratio
gyro = 699.9e3
# pi is pi
pi = np.pi

# identity matrix
_ID = np.asarray([[1, 0], [0, 1]])
# X gate
_X = np.asarray([[0, 1], [1, 0]])
# Z gate
_Z = np.asarray([[1, 0], [0, -1]])
# Hadamard gate
_H = (1/np.sqrt(2))*np.asarray([[1, 1], [1, -1]])
# Y Gate
_Y = np.asarray([[0, -1j], [1j, 0]])
# S gate
_S = np.asarray([[1, 0], [0, 1j]])
# Sdg gate
_Sdg = np.asarray([[1, 0], [0, -1j]])
# T gate
_T = np.asarray([[1, 0], [0, (1 + 1j)/np.sqrt(2)]])
# Tdg gate
_Tdg = np.asarray([[1, 0], [0, (1 - 1j)/np.sqrt(2)]])
# CNOT gate
_CX = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
# zero state
_pz = np.matrix([[1],[0]])
# one state
_po = np.matrix([[0],[1]])



# define operators for spin 1/2 
op1 = {'h':   _H,
        'id':  _ID,
        'x':   _X,
        'y':   _Y,
        'z':   _Z,
        't':   _T,
        'tdg': _Tdg,
        's':   _S,
        'sdg': _Sdg,
        'cx':  _CX,
        'pz': _pz,
        'po': _po}




# define operators for spin 1 
op2 = {
		
	  }


# measurement projections for spin 1/2
meas1 = {"0":np.asarray([[1,0]]),
		 "1":np.asarray([[0,1]]),
		 "+":np.asarray([[1,1]]/np.sqrt(2)),
		 "-":np.asarray([[1,-1]]/np.sqrt(2)),
		 "+i":np.asarray([[1,1j]]/np.sqrt(2)),
		 "-i":np.asarray([[1,-1j]]/np.sqrt(2)),
		}