import os 
import time
import copy
import h5py
import numpy as np 
# from capy import *
from utility import *
from operators import *
from qutip import Bloch
from time import gmtime
from IPython import embed
import scipy.linalg as sla
import matplotlib.pyplot as plt


# define location of archive file
archivepath = "C:\\Users\\Joshua\\Documents\\Projects\\Code\\Python\\Modules\\qChain\\archive.h5"


class SpinSystem(object):
    """
    class that defines a spin-[1/2, 1] system, keeping track of its state and evolution
    """

    def __init__(self, spin: str="half", init=None):
        # spin of system to be defined
        self.spin = spin
        # initial state of each particle [zero, one, super]
        self.init = init
        

        # initialise the system 
        self.initialise()

    def initialise(self):
        """
        initialises the spin system
        """
        if self.spin == "half":
            # dimension of system
            self.dim = 2
            # initial state
            self.state = op1["pz"]
        elif self.spin =="one":
            # dimension of system
            self.dim = 3
            # initial state
            self.state = op2["po"]

        if self.init is not None: 
            if type(self.init) is str:
                if self.init is "super":
                    self.state = op1["h"] @ self.state
                elif self.init is "isuper":
                    self.state = op1["s"] @ op1["h"] @ self.state
                elif self.init is "zero":
                    self.state = op1["pz"]
                elif self.init is "one":
                    self.state = op1["po"]
                else:
                    raise ValueError("Unrecognised initial state: {}".format(self.init))
            else:
                # custom start state
                self.state = np.matrix(self.init)
            
    def evolve(self, unitary, save=False):
        """
        evolve the spin system 
        """
        
        # evolve state 
        if save:
            self.state = unitary @ self.state
            return self.state
        else:
            return unitary @ self.state

    def measure(self, state=None, project=np.asarray([[1,0]])):
        """
        perform projective measurement in computational basis
        """
        if state is None:
            state = self.state

        return np.abs(np.dot(project, state))**2

    def state_evolve(self, hamiltonian, t=[0,1,1e-3], cache=True, bloch=[False, 10], **kwargs):
        """
        computes the probability of finding the system in the projection state over the given range
        """
        if len(t)<3: t.append(1e-3)
        # time array
        time = np.arange(t[0], t[1], t[2])   
        self.time = time
        # preallocate probability array
        probs = np.zeros((len(time)), dtype=np.float64)
        # create list for snap shot states to plot
        if bloch:
            bloch_points = []

        # compute unitary given hamiltonian for 0 -> t_delta
        if callable(hamiltonian):
            # hamiltonian is time dependent
            flag = True
        else:
            flag = False 
            unitary = sla.expm(-1j*hamiltonian*t[2]/hbar)
        
        if cache and flag:
            # create hamiltonian cache to reduce repeated call overhead
            h_cache = hamiltonian(time)
            u_cache = [myexpm(-1j*h_cache[:,:,i]*t[2]/hbar) for i in range(len(time)) if i!=0]
            state_cache = np.zeros((2,1,len(time)), dtype=np.complex128)

            # add initial state to caches
            state_cache[:,:,0] = self.state
            probs[0] = self.measure(state_cache[:,:,0], **kwargs)

            # iterate through cached hamiltonians
            for j,tstep in enumerate(time):
                if j!=0:
                    # compute measurement probablity of projector
                    state_cache[:,:,j] = self.evolve(u_cache[j-1], save=True)
                    probs[j] = self.measure(state=state_cache[:,:,j], **kwargs)

                # add state to bloch plot
                if bloch[0] and j % bloch[1] == 0:
                    bloch_points.append(self.get_bloch_vec(np.outer(self.state.H, self.state)))
            
            # save caches
            self.h_cache = h_cache
            self.u_cache = u_cache
            self.state_cache = state_cache

        else:
            # compute unitary in a piecewise fashion (this would be really easy to parellise but I probably shouldn't)
            for i,tstep in enumerate(time):
                print(tstep)
                if flag:
                    unitary = expm_eig(-1j*hamiltonian(tstep)/hbar, t[2])
                
                # compute measurement probablity of projector
                probs[i] = self.measure(state=self.evolve(unitary, save=True), **kwargs)
                # add state to bloch plot
                if bloch[0] and i % bloch[1] == 0:
                    bloch_points.append(self.get_bloch_vec(np.outer(self.state.H, self.state)))
                    
                
        # plot evolution on the Bloch sphere
        if bloch:
            # convert to qutips annoying format
            # TODO: preallocate
            x,y,z = [],[],[]
            for vec in bloch_points:
                x.append(vec[0])
                y.append(vec[1])
                z.append(vec[2])

            bloch_points = [x,y,z]
            return time, probs, bloch_points
        else:
            return time, probs

    def frame_transform(self, cstate=None, frame=["interaction", "lab"], project=meas1["0"], bloch=[False, 10]):
        """
        EXPERIMENTAL
        Transform a state or set of states to a specified reference frame from another. This method
        is still in the experimental phase. It works well for going from simpler reference frames to
        complicated oned but the reverse is prone to numerical instability. 
        """
        if cstate is None:
            cstate = np.copy(self.state_cache)

        # compute reference frame map
        if callable(frame):
            unitary_map = frame
        # determine transition case and define appropriate time dependent unitary operator
        elif frame[0] == "lab":
            if frame[1] == "interaction":
                def unitary_map(t, larmor=gyro): 
                    return np.asarray([[np.exp(1j*np.pi*larmor*t), 0], [0, np.exp(-1j*np.pi*larmor*t)]])
            elif frame == "dressed":
                # define dressed state transform
                def dressed(t, omega=1e4, detuning=0): return np.asarray([[np.cos(np.arctan(
                    omega/detuning)/2), -np.sin(np.arctan(omega/detuning)/2)], [np.sin(np.arctan(omega/detuning)/2), np.cos(np.arctan(omega/detuning)/2)]])

                def unitary_map(t, larmor=larmor): return dressed(t) @ np.asarray([[np.exp(1j*np.pi*larmor*t), 0], [0, np.exp(-1j*np.pi*larmor*t)]])
            else:
                raise ValueError("Unrecognised output reference frame")

        elif frame[0] == "interaction":
            if frame[1] == "lab":
                def unitary_map(t, larmor=gyro): 
                    return np.asarray([[np.exp(-1j*np.pi*larmor*t), 0], [0, np.exp(1j*np.pi*larmor*t)]])
            elif frame == "dressed":
                # define dressed state transform
                def unitary_map(t, omega=1e4, detuning=0): return np.asarray([[np.cos(np.arctan(
                    omega/detuning)/2), np.sin(np.arctan(omega/detuning)/2)], [-np.sin(np.arctan(omega/detuning)/2), np.cos(np.arctan(omega/detuning)/2)]])
            else:
                raise ValueError("Unrecognised output reference frame")

        elif frame[0] == "dressed":
            if frame[1] == "interaction":
                def unitary_map(t, larmor=gyro): 
                    return np.asarray([[np.cos(np.arctan(omega/detuning)/2), -np.sin(np.arctan(omega/detuning)/2)], [np.sin(np.arctan(omega/detuning)/2), np.cos(np.arctan(omega/detuning)/2)]])
            elif frame == "lab":
                # define dressed state transform
                def dressed(t, omega=1e4, detuning=0): return np.asarray([[np.cos(np.arctan(
                    omega/detuning)/2), -np.sin(np.arctan(omega/detuning)/2)], [np.sin(np.arctan(omega/detuning)/2), np.cos(np.arctan(omega/detuning)/2)]])

                def unitary_map(t, larmor=larmor): return dressed(t) @ np.asarray([[np.exp(1j*np.pi*larmor*t), 0], [0, np.exp(-1j*np.pi*larmor*t)]])
            else:
                raise ValueError("Unrecognised output reference frame")

        else:
            raise ValueError("Unrecognised input reference frame")

        # apply transform to states
        if len(np.shape(cstate))==3:
            new_states = [unitary_map(t) @ cstate[:,:,step] for step,t in enumerate(self.time)]
            nprobs = np.squeeze([np.abs(project @ nstate)**2 for i,nstate in enumerate(new_states)])
            # save new states and projection probabilities
            self.state_cache = new_states
            self.probs = nprobs
        else:
            new_states = unitary_map(self.time[-1]) @ cstate
            return new_states




    def bloch_plot(self, filename, points=None, save=False):
        """
        Plot the current state on the Bloch sphere using
        qutip. 
        """
        if points is None:
            # convert current state into density operator
            rho = np.outer(self.state.H, self.state)
            # get Bloch vector representation
            points = self.get_bloch_vec(rho)
            # Can only plot systems of dimension 2 at this time
            assert len(points) == 3, "System dimension must be spin 1/2 for Bloch sphere plot"
        
        # create instance of 3d plot
        bloch = Bloch(fig=1, figsize=[9,9], view=[190,10])
        # add state
        bloch.add_points(points)
        bloch.render()

        if save is True:
            print('Bloch plot saved to Sim Results folder')
            path = 'C:/Users/Boundsy/Desktop/Uni Work/PHS2360/Sim Results/' + str(filename) + '.png'
            bloch.fig.savefig(path, dpi=800, transparent=True)

    def bloch_plot2(self, filename, points=None, save=False):
        """
        Plot the current state on the Bloch sphere using
        qutip. 
        """
        if points is None:
            # convert current state into density operator
            rho = np.outer(self.state.H, self.state)
            # get Bloch vector representation
            points = self.get_bloch_vec(rho)
            # Can only plot systems of dimension 2 at this time
            assert len(points) == 3, "System dimension must be spin 1/2 for Bloch sphere plot"
        
        # create instance of 3d plot
        bloch = Bloch(fig=1, figsize=[9,9], view=[190,10])
        # add state
        bloch.add_points(points)
        # add field vector
        # bloch.add_vectors([1,0,0.15])
        bloch.add_vectors([1,0,0])
        bloch.render()

        if save is True:
            print('Bloch plot saved to Sim Results folder')
            path = 'C:/Users/Boundsy/Desktop/Uni Work/PHS2360/Sim Results/' + str(filename) + '.png'
            bloch.fig.savefig(path, dpi=800, transparent=True)


    def get_bloch_vec(self, rho):
        """
        compute the bloch vector for some 2 dimensional density operator rho
        """
        u = 2*np.real(rho[0,1])
        v = 2*np.imag(rho[1,0])
        w = np.real(rho[0,0] - rho[1,1])
        return [u,v,w]

    def prob_plot(self, time, probs, filename, commitVersion,save=False):
        """
        Formatted code for plot (why must plot code always be hideous?)
        """

        commitText = 'commit: ' + str(commitVersion)

        plt.figure(2)
        plt.plot(time, probs)
        plt.ylim([np.min(probs)*1.05, np.max(probs)*1.05])
        plt.xlim([time[0], time[-1]])
        plt.grid()
        plt.xlabel("Time (s)")
        plt.ylabel("Probability")
        plt.annotate(commitText, xy=(0.85,0.97), xycoords = 'axes fraction', fontsize=12)
        plt.title(filename)

        if save is True:
            path = 'C:/Users/Boundsy/Desktop/Uni Work/PHS2360/Sim Results/' + str(filename) + '.png'
            print('Probabiblity plot saved to Sim Results folder')
            plt.savefig(path)

        # plt.show()

    def project_plot(self, time, projections, filename, commitVersion, save=False):
        """
        Formatted code for plot (why must plot code always be hideous?)
        """

        commitText = 'commit: ' + str(commitVersion)
        # projections = 2*probs - 1;

        plt.figure(3)
        plt.plot(time, projections)
        plt.ylim([np.min(projections)*1.05, np.max(projections)*1.05])
        plt.xlim([time[0], time[-1]])
        plt.grid()
        plt.xlabel("Time (s)")
        plt.ylabel("F_x projection (<F_x>)")
        plt.annotate(commitText, xy=(0.85,0.97), xycoords = 'axes fraction', fontsize=12)
        plt.title(filename)

        if save is True:
            path = 'C:/Users/Boundsy/Desktop/Uni Work/PHS2360/Sim Results/' + str(filename) + '.png'
            print('Projection plot saved to Sim Results folder')
            plt.savefig(path)

        # plt.show()

    def bloch_animate(self, pnts, name="Bloch_animate"):
        """
        Animates the path of a state through the set of pure states - requires ffmpeg
        """
        from pylab import figure
        import matplotlib.animation as animation
        from mpl_toolkits.mplot3d import Axes3D

        # set up plot environment
        fig = figure()
        ax = Axes3D(fig, azim=-40, elev=30)
        sphere = Bloch(axes=ax)

        # define animation function (from qutip docs)
        def animate(i):
            sphere.clear()
            sphere.add_points([pnts[0][:i+1],pnts[1][:i+1],pnts[2][:i+1]])
            sphere.make_sphere()
            return ax

        def init():
            sphere.vector_color = ['r']
            return ax

        ani = animation.FuncAnimation(fig, animate, np.arange(len(pnts[0])),
                            init_func=init, repeat=False)
        
        ani.save(name + ".mp4", fps=20)


def field_gen(field_params):
    """
    Creates a 3 element list of functions that wholly defines a classical electromagnetic field 
    given a dictionary of parameters
    """

    # list of field functions
    field_vector = []
    for i,struct in enumerate(field_params["struct"]):

        # generate a simple sinusoid function with amplitude and frequency
        if struct is "sinusoid":
            field_vector.append(lambda t,j=i: field_params["amp"][j]*(np.cos(2*np.pi*field_params["freqb"][j]*t))) 
        # generate a constant DC bias field
        elif struct is "constant":
            field_vector.append(lambda t,j=i: field_params["amp"][j]*t/t)
        # generate a pulsed sinusoid with frequency omega beginning at time tau (seconds)    
        elif struct is "pulse":
            # generate field callable with num pulsed sinusoids
            field_vector.append(pulse_gen(field_params["freqb"][i], field_params["tau"][i], amp=field_params["amp"][i]))
        elif struct is "tchirp":
            # define chirp component
            chirp = lambda t,j=i: np.heaviside(t-field_params["tau"][j],1.0)*field_params["misc"][j]*np.tanh(0.01*(t-field_params["tau"][j])/field_params["tau"][j])
            # generate field with time varying amplitude
            constant = lambda t,j=i: field_params["amp"][j]
            # add to field vectors
            field_vector.append(lambda t: constant(t) + chirp(t))        
        elif struct is "custom":
            field_vector.append(field_params["misc"][i])
        else:
            raise ValueError("Unrecognised field type: {}".format(struct))


    return field_vector

def field_plot(field_vector, time=np.linspace(0,0.5,1e4)):
    """
    Plots the signal components of some field vector over the specified time doman
    """
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=False)
    for i, row in enumerate(ax):
        row.plot(time, field_vector[i](time))
        row.set_title("Field vector along {} axis".format(['x','y','z'][i]))
    plt.xlabel("Time (s)")
    #plt.ylabel("Ampltiude ($Hz/Gauss$)")
    plt.show()


class Hamiltonian(object):
    """
    Defines the Hamiltonian that acts on a SpinSystem class - dimension of Hilbert spaces of the two must match
    """
    def __init__(self, spin="half",  freq=500):
        # spin of system
        self.spin = spin
        # define oscillation frequency 
        self.freq = freq
        # define energy seperation
        self.energy = hbar*self.freq
        # define internal Hamiltonian for spin system
        if self.spin == "half":
            # two energy eigenstates (0,1)
            self.free_ham = 0.5*hbar*np.asarray([[-self.freq, 0],[0, self.freq]], dtype=np.complex128)
        elif self.spin == "one":
            # TODO: three energy eigenstates (0,1,2)
            pass
        else:
            print("Unrecognised spin system {}: aborting".format(self.spin))


    def generate_simple(self, potential=None,):
        """
        Creates a simple Hamiltonian based off the input Potential or defaults
        """
        # potential component of Hamiltonian
        self.potential = potential

        # check if potential is custom or one of defaults of style [coupling, phase, rwa]
        if self.potential is not None:
            if type(self.potential) == list:
                coupling = self.potential[0]
                tuning = self.potential[1]
            
            elif self.init is "zero":
                self.state = op1["pz"]

                # apply rotating wave approximation
                if potential[2]:
                    self.potential = 0.0
                    self.hamiltonian = 0.5*hbar*np.asarray([[tuning, coupling],
                                                 [coupling, -tuning]], dtype=np.complex128)
                # full hamiltonian in interaction picture (t dependency is annoying)
                else:
                    # define the off diagonal components as lambda functions for t dependency 
                    offdiag = lambda t: coupling*(1 + np.exp(-2j*(self.freq+tuning)*t))
                    self.hamiltonian = lambda t: 0.5*hbar*np.asarray([[tuning, offdiag(t)],
                                                           [np.conj(offdiag(t)), -tuning]],
                                                           dtype=np.complex128)
            else:
                pass
        else:
            self.potential = 0.0


    def generate_field_hamiltonian(self, fields):
        """
        generates an arbitrary magnetic field hamiltonian. Field must
        be a function with time as its only argument.
        """
        # store field 
        self.fields = fields
        # enforce function type
        for field in fields: assert callable(field), "Field {} must be callable".format(i)
        # redundant constants for clarity
        self.hamiltonian = lambda t: 0.5*hbar*2*np.pi*(fields[0](t)*op1["x"] + fields[1](t)*op1["y"] + fields[2](t)*op1["z"])

        # cache version of hamiltonian
        def hamiltonian2(time):
            return 0.5*hbar*2*np.pi*np.asarray([[fields[2](time), fields[0](time)+1j*fields[1](time)],
                                                    [fields[0](time)-1j*fields[1](time), -1*fields[2](time)]])
        self.hamiltonian_cache = hamiltonian2


    def bloch_field(self, time):
        """
        Plot magnetic field vector normalised to the bloch sphere
        """
        # seperate coordinates into axis set
        xb = self.fields[0](time)
        yb = self.fields[1](time)
        zb = self.fields[2](time)
        # add to list and normalise
        points = [list(xb), list(yb), list(zb)]
        sqsum = 0.0
        for i,point in enumerate(points[0]):
            # compute magnitude and store largest value
            sqsum_test = np.sqrt(points[0][i]**2 + points[1][i]**2 + points[2][i]**2)
            if sqsum_test > sqsum:
                sqsum = sqsum_test
        
        points = points/sqsum

        # create Bloch sphere instance
        bloch = Bloch()
        # add points and show
        bloch.add_points(points)
        bloch.show()

def data_retrieval(sim_params):
    """
    Retrieves a data set from the archive if it has already been simulated with identical parameters
    """

    #TODO: Fix this mess

    # define base name for retrieval
    root_group = "Atomic_Sense" 
    
    # open archive and check if atomic sensor data exists
    with h5py.File(archivepath, 'a') as archive:
        # create group if it doesn't exist
        if root_group not in archive:
            atomic_sense = archive.create_group(root_group)
            print("No data found, exiting")
            return None
        else:
            atomic_sense = archive[root_group]

        # iterate through parameter list
        for dataset in atomic_sense:
            flag = False
            for key,val in sim_params.items():

                # check for equivalency between floats
                if isinstance(val, float):
                    if np.isclose(atomic_sense[dataset].attrs[key], val):
                        flag = True
                    else:
                        flag = False
                        break
                else:
                    if atomic_sense[dataset].attrs[key] == val:
                        flag = True
                    else:
                        flag = False
                        break
            if flag:
                print("Data set found in archive")
                # get sensor data
                print(dataset)
                return np.asarray(atomic_sense[dataset])
        else:
            print("Data set not found in archive")

    return None

def data_store(sim_params, data, name=None, verbose=True):
    """
    Stores a simulation instance 
    """

    # define base name for retrieval
    root_group = "Atomic_Sense"
    # turn parameter set into name string
    date = gmtime()
    root = sim_params["struct"] + "_{}_{}_{}_{}_{}_{}".format(date.tm_sec, date.tm_min, date.tm_hour, date.tm_mday, date.tm_mon, date.tm_year)
    if name is not None:
        root += str(name)

    # open archive and check if atomic sensor data exists
    with h5py.File(archivepath, 'a') as archive:
        # create group if it doesn't exist
        if root_group not in archive:
            atomic_sense = archive.create_group(root_group)
        else:
            atomic_sense = archive[root_group]

        for dataset in atomic_sense:
            flag = False
            for key, val in sim_params.items():
                if atomic_sense[dataset].attrs[key] == val:
                    flag = True
                else:
                    flag = False
                    break
            if flag:
                print("Simulation event already exists, ignoring save request")
                return
    
        if verbose:
            print("Saving simulation results to archive file")
        dataset = atomic_sense.create_dataset(root, data=np.asarray(data))
        # save attributes
        for key,val in sim_params.items():
            dataset.attrs[key] = val


def pseudo_fourier(struct, sig_amp=1, sig_freq=360, f_range=[250,450, 1], sig=None, tau=[0.01], t=0.5, noise=0.0, plot=False, verbose=True):
    """
    Computes the projection |<1|psi>|^2 of a two level system for a given point in the signal parameter space with different frequency tunings
    """

    # define magnetic fields
    pulse = pulse_gen(freq=341, tau=[1e-2], amp=20)

    # set detune max value
    detune_max = 5e4
    # time at which detuning sweep occurs
    detune_tau = 4e-4
    # time for sweep to occur
    swt = 0.01
    # bias field strength
    bias_amp = gyro

    # create parameter dictionary
    sim_params = {"struct": struct, 
                  "sig_amp": sig_amp,
                  "sig_freq": sig_freq,
                  "f_range_low": f_range[0],
                  "f_range_high": f_range[1],
                  "f_range_step": f_range[2],
                  "num": len(tau)}

    result = None #data_retrieval(sim_params)
    if result is None:
        # projection array
        projs = []
        # frequencies to sample
        freqs = np.arange(f_range[0],f_range[1], f_range[2])
        # initiate Hamiltonian instance
        ham = Hamiltonian()
        for freq in freqs:
            if verbose:
                print("Computing evolution with tuning frequency: {:.2f} Hz\r".format(freq), end="",flush=True)

            def Bx(t, omega_amp=freq): return omega_amp*(t/t)
            def By(t): return 0
            def Bz(t): return  pulse(t)

    

            # define magnetic field vector parameters
            params = {"struct": ["custom", "constant", "custom"], 
                      "freqb": [sig_freq, 50, 0],          # frequency in Hz
                      "tau": [tau, None, None],          # time event of pulse
                      "amp":[sig_amp, 0, freq], # amplitude in Gauss -> 1 Gauss ~= 700000 Hz precession
                      "misc": [Bz, None, Bx]}          # misc parameters
            # generate magentic fields
            fields = field_gen(field_params=params)
            # compute Hamiltonian for updated magnetic field
            ham.generate_field_hamiltonian(fields)  
            # redefine atom spin system
            atom = SpinSystem(init="super")
            # evolve state using hamiltonian
            time, probs, pnts = atom.state_evolve(t=[1e-44, t, 1/2e5], hamiltonian=ham.hamiltonian_cache, project=meas1["1"], bloch=[False, 5])
            #atom.frame_transform(project=meas1["0"])
            projs.append(probs[-1])

        projs = np.asarray(projs)
        # format into single chunk for hdf5 compression                    
        data_matrix = np.zeros((2, len(projs)))
        signal_matrix = np.zeros((4, len(time)))
        data_matrix[0,:] = freqs
        data_matrix[1,:] = projs
        signal_matrix[0,:] = time
        signal_matrix[1,:] = fields[0](time)
        signal_matrix[2,:] = fields[1](time)
        signal_matrix[3,:] = fields[2](time)

        sim_params["name"] = "Fourier"
        #data_store(sim_params, data=data_matrix, name="_Fourier_measurements")
        sim_params["name"] = "Field"
        #data_store(sim_params, data=data_matrix, name="_Field_signal")

    else:
        freqs = result[0,:]
        projs = result[1,:]
        # define magnetic field vector parameters
        params = {"struct": [struct, "constant", "constant"], 
                  "freqb": [sig_freq, 0, 0],          # frequency in Hz
                  "tau": [tau, None, None],          # time event of oulse
                  "amp":[sig_amp, 0, 0], # amplitude in Gauss -> 1 Gauss ~= 700000 Hz precession
                  "misc": [sig, None,None]}          # misc parameters
        # generate magentic fields
        fields = field_gen(field_params=params)
       
    # set max of uniform distribution to be max of projector oscillations
    #noise_lvl = noise*np.max(projs-np.mean(projs))
    # add white noise to projector results
    #projs += np.random.normal(scale=noise_lvl,size=np.size(projs))
    if plot:
        plt.plot(freqs, projs, 'o--')
        plt.xlabel("Frequency of Bias field (Hz)")
        plt.ylabel("$|<1|\psi(t_1)>|^2$")
        plt.title("Probability vs tuning frequency for {} Hz {} beginning at {} seconds".format(sig_freq, params["struct"][0], params["tau"][0]))
        plt.grid()
        plt.show()

    return np.asarray(freqs), np.asarray(projs), fields[0]

