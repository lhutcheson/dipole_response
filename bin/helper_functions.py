# -*- coding: utf-8 -*-

"""
Functions to help with the analysis
@authors: Lynda Hutcheson, Maximillian Hartmann
"""

import numpy as np
import scipy.constants as cnt
from argparse import ArgumentParser as AP
from scipy.optimize import curve_fit

# -----------------------------------------------------------------------
#                   Constants
# -----------------------------------------------------------------------

pldp_au = 0.77              # path-length-density-product in atomic units
alpha = cnt.constants.fine_structure
lineshape_constant = pldp_au/np.log(10)*4*np.pi*alpha
literature_linewidth = 0.122

e_res = 55.37  # resonance energy (eV)
energy_shift = -5.5  # energy shift (eV) to match RMT result with experiment
gv_resolution_M4_M5 = 0.043  # experimental resolution


def au_to_fs(time_in_au):
    """
    Converts time from atomic units (au) to femtoseconds (fs).

    Parameters
    ----------
    time_in_au : float
        The time value in atomic units (au) to be converted to femtoseconds.

    Returns
    -------
    float
        The equivalent time in femtoseconds (fs).
    """
    return time_in_au * cnt.value("atomic unit of time") / cnt.femto


def fs_to_au(time_in_femto_seconds):
    """
    Converts time from femtoseconds (fs) to atomic units (au).

    Parameters
    ----------
    time_in_femto_seconds : float
        The time value in femtoseconds (fs) to be converted to atomic units (au).

    Returns
    -------
    float
        The equivalent time in atomic units (au).
    """
    return time_in_femto_seconds * cnt.femto / cnt.value("atomic unit of time")


def au_to_ev(energy_in_au):
    """
    Converts energy from atomic units (au) to electron volts (eV).

    Parameters
    ----------
    energy_in_au : float
        The energy value in atomic units (au) to be converted to electron volts (eV).

    Returns
    -------
    float
        The equivalent energy in electron volts (eV).
    """
    ev_to_joule = energy_in_au * cnt.value("atomic unit of energy")
    return ev_to_joule / cnt.value("electron volt-joule relationship")


def ev_to_au(energy_in_ev):
    """
    Converts energy from electron volts (eV) to atomic units (au).

    Parameters
    ----------
    energy_in_ev : float
        The energy value in electron volts (eV) to be converted to atomic units (au).

    Returns
    -------
    float
        The equivalent energy in atomic units (au).
    """
    ev_to_joule = energy_in_ev * cnt.value("electron volt-joule relationship")
    return ev_to_joule / cnt.value("atomic unit of energy")


def AugerDecayFactor(time_in_au, t_zero=904.1058):
    """
    Simulates the Auger decay in the dipole.

    The Auger decay is modeled as an exponential decay with the form:
    exp(-t / lifetime), where the lifetime is derived from the resonance linewidth. 
    This function calculates the decay factor at each time point relative to the time
    at which the XUV pulse ends (defined as `t_zero`), which is typically one full
    width at half maximum (FWHM) after the peak of the XUV pulse.

    Parameters
    ----------
    time_in_au : list-like
        An array of times at which the dipole has been evaluated, in atomic units (au).

    t_zero : float, optional
        The time at which to start the exponential decay, by default taken to be the
        end of the XUV pulse (1 FWHM after the peak of the XUV pulse): 904.1058.

    Returns
    -------
    numpy.ndarray
        An array for the decay factors, calculated for each time point in `time_in_au`.
    """
    time = time_in_au - t_zero
    lifetime = 1/(ev_to_au(literature_linewidth)/2)
    decay = np.exp(- time/lifetime)
    decay[time < 0] = 1
    return decay


def get_complex_dipole(dipole_data, e_res=60.85):
    """
    From the time-dependent dipole expectation value obtained from RMT,
    calculate the frequency-resolved dipole evaluated at the energy of the
    transition.
    The dipole is treated in the same as when the absorption profile is calculated:
    padded with zeros and windowed with an exponential decay to simulate Auger decay.

    Parameters
    ----------
    dipole_data : Pandas DataFrame
        DataFrame containing the time-dependent dipole expectation value for each
        time delay in the scan.
    e_res : float
        Default = 60.87
        the energy of the transition under investigation in the ATAS study.
    """
    file_length = len(dipole_data)
    desired_length = 2**22
    num_zeros_to_pad = desired_length - file_length

    decay = AugerDecayFactor(dipole_data['Time'])

    dt = dipole_data['Time'][1] - dipole_data['Time'][0]
    w = np.fft.fftfreq(desired_length, dt/(2*np.pi))
    w_ev = au_to_ev(w)

    res_index = np.argmin(np.abs(np.array(w_ev) - e_res))

    complex_number = []

    for col in dipole_data.columns[1:]:
        dipole = np.pad(dipole_data[col]*decay, (0, num_zeros_to_pad),
                        'constant', constant_values=(0, 0))
        fft_dipole = np.fft.fft(dipole)
        complex_number = np.append(complex_number, fft_dipole[res_index])
    return complex_number


def DCM_lineshape(energy_axis, z, phi, resonance_energy, gamma):
    """
    Dipole control model (DCM) line shape function for a single absorption line

    Parameters
    ----------
    energy_axis : np.array
        the array of values that defines the photon energy axis
    z :   float
        line strength
    phi : float
        dipole phase
    resonance_energy : float
        resonance energy of the absorption line
    gamma :   float
        line width

    Returns:
        np.array size of energy axis
        line shape function as a function of photon energy

    """

    lineshape = (gamma/2*np.cos(phi) - (energy_axis-resonance_energy)
                 * np.sin(phi)) / ((energy_axis-resonance_energy)**2 + gamma**2/4)

    return z * lineshape


def fit_lineshapes(energy_axis, z, phi, gamma, background):
    """
    Fit function to extract line shape parameters from several absorption lines
    from the measurement data. 

    Parameters
    ----------
    energy_axis : np.array
        the array of values that defines the photon energy axis
    z :   float
        line strength
    phi : float
        dipole phase
    resonance_energy : float
        resonance energy of the absorption line
    gamma :   float
        line width

    Returns:
    model : np.array size of energy axis
            Calculates an optical density as a function of photon energy. 
            Includes a constant offset to fit the non-resonant background.
    """
    model = DCM_lineshape(energy_axis, z*gamma, phi, e_res, gamma)
    model *= energy_axis * lineshape_constant
    model += background
    return model


def truncate_td(time_delay_axis, lower=-2.75, upper=2.75):
    """
    Returns the indexes required for slicing data over a given time delay range [lower_time, upper_time]

    Parameters
    ----------
    time_delay_axis : array
        list of time delays time delays
    lower: float
        the lower bound of the desired range of time delays
        default = -2.75
    upper: float
        the upper bound of the desired range of time delays
        default = 2.75
    """
    lower_index = 0
    upper_index = -1
    for time in time_delay_axis:
        if time < lower:
            lower_index = np.where(time_delay_axis == time)[0][0]
        elif time > upper:
            upper_index = np.where(time_delay_axis == time)[0][0]
            break
    return lower_index, upper_index


def gauss_envelope(intensity, time, FWHM=186):
    """
    Function to get the gaussian envelope of the NIR pulse. 

    Parameters
    ----------
    intensity : float
        intensity of the NIR pulse in 10^14 Wcm^-2

    time :    array
    time axis

    FWHM :    float
        Full-width half-maximum of the NIR pulse in atomic units
        Default is 4.5fs

    Returns:
        np.array size of time axis
        The corresponding gaussian envelope.
    """
    time_au = fs_to_au(time)
    E0 = 0.05336*np.sqrt(intensity)
    return E0 * np.exp(-((time_au)**2)/(FWHM**2))


def pulse(intensity, time=np.arange(-10, 10, 0.1), FWHM=186, IR_freq=0.06798):
    """
    Function to get the NIR pulse with a gaussian envelope. 

    Parameters
    ----------
    intensity : float
        intensity of the NIR pulse in 10^14 Wcm^-2

    time :    array
        time axis

    FWHM :    float
        Full-width half-maximum of the NIR pulse in atomic units
        Default is 4.5fs

    IR_freq : float
        Frequency of the NIR pump pulse in atomic units

    Returns:
        np.array size of time axis
        The corresponding NIR pulse.
    """
    time_au = fs_to_au(time)
    envelope = gauss_envelope(intensity, time, FWHM)
    pulse = envelope*np.cos(IR_freq*(time_au))
    return pulse


def fit_model_line(line, energy_axis):
    """
    Function to fit an absorption profile with the generalised lineshape,
    obtains the amplitude (z) and dipole phase (phi)
    and return the complex dipole response in the form:
            z*exp(i*phi)

    Parameters
    ----------
    line : array
                absorption line to be fitted

    energy_axis :    array
                energy axis

    Returns:
        the complex dipole response
    """
    # fit parameters:
    p_init = [1, 0, 0.122, 0]
    bounds = ([1e-6,  -2*np.pi, 0.2*0.122, -15],
              [np.inf, 2*np.pi, 15*0.122, 20])
    popt, pcov = curve_fit(fit_lineshapes, energy_axis, line, p_init, maxfev=1000000,
                           bounds=bounds)
    strength = popt[0]
    phase = popt[1]
    return strength*np.exp(1j*phase)


def model(IR_FWHM=186,
          IR_freq=0.06798,
          Max_ION=0.08,
          Max_E1=0.03,
          Max_E2=0.03,
          Phi_ION=0,
          Phi_E1=-2.38,
          Phi_E2=-1.31,
          excited_delay=0.0,
          time=np.arange(-10, 10, 0.1),
          smooth=False):
    """
    Minimal model for the complex dipole response induced by a strong-field.


    Parameters
    ----------
    IR_FWHM : float
                Default = 186 
                FWHM of NIR pulse in atomic units of time

    IR_freq : float
                Default = 0.06798, 
                Frequency of NIR pulse in atomic units

    Max_ION : float
                Default = 0.08, 
                Maximum amplitude of the ionisation channel

    Max_E1 : float
                Default = 0.03,
                Maximum amplitude of the 1st excitation channel

    Max_E2 : float
                Default = 0.03, 
                Maximum amplitude of the 2nd excitation channel

    Phi_ION : Float
                Default = 0, 
                Phase assigned to the ionisation channel (radians)

    Phi_E1 : Float
                Default = -2.2, 
                Phase assigned to the 1st excitation channel (radians)

    Phi_E2 : Float
                Default = -1.3, 
                Phase assigned to the 2nd excitation channel (radians)

    excited_delay :  Float
                Default = 0.3, 
                delay between the two excitation channels in femtoseconds

    time : numpy array
                Default = np.arange(-10, 10, 0.1),
                the time axis to be used

    smooth : Boolean
                Default = False 
                Option to model the overall response using the NIR envelope instead of the pulse

    Returns:
        The time-dependent complex dipole response
    """

    from scipy.integrate import cumulative_trapezoid

    e_res = 55  # energy of the resonance
    e_axis = np.arange(e_res - 4, e_res + 4 + 0.01, 0.01)
    L = 0.122  # Literature Linewidth

    time_au = fs_to_au(time)
    if smooth:
        IR = gauss_envelope(0.8, time, IR_FWHM)
        IR_delay = gauss_envelope(0.8, time-excited_delay, IR_FWHM)
    else:
        IR = pulse(0.8, time, IR_FWHM, IR_freq)
        IR_delay = pulse(0.8, time-excited_delay, IR_FWHM, IR_freq)

    ION_amplitude = cumulative_trapezoid(IR**2, time_au, initial=0)
    E1_amplitude = IR**2
    E2_amplitude = IR_delay**2

    ION_amplitude *= Max_ION/np.amax(ION_amplitude)
    E1_amplitude *= Max_E1/np.amax(E1_amplitude)
    E2_amplitude *= Max_E2/np.amax(E2_amplitude)

    dipole_response = []

    for ion, e1, e2 in zip(ION_amplitude, E1_amplitude, E2_amplitude):
        ion_line = fit_lineshapes(e_axis, ion, Phi_ION, L, 0)
        e1_line = fit_lineshapes(e_axis, e1, Phi_E1, L, 0)
        e2_line = fit_lineshapes(e_axis, e2, Phi_E2, L, 0)

        summed_line = ion_line + e1_line + e2_line

        dipole_response = np.append(
            dipole_response, fit_model_line(summed_line, e_axis))
    return dipole_response


def get_complex_data(dataframe, transition='T1', truncate=True):
    """
    Reads in a DataFrame containing the line strength (Z) and dipole phase (phi)
    fit parameters to the absorption spectrum for each time delay. Returns
    the complex-valued dipole response of the form:
            Z*exp(i*phi)
    for the desired transition (T1, T2 or T3).

    Parameters:
    -----------
    dataframe: Pandas DataFrame
        dataframe containing the fit parameters for each time delay
    transition: string
        which transition to use
        default = T1
    truncate: boolean
        option to truncate the time-delay scan to a subset of time delays

    Returns:
    real: array
        the real part of the complex dipole response for each time delay
    image: array
        the imaginary part of the complex dipole response for each time delay
    """
    time = dataframe['Time Delays'].to_numpy()
    complex_dat = (
        dataframe[f'Line Strength {transition}']*np.exp(1j*(dataframe[f'Phase {transition}'])))

    if truncate:
        lower_index, upper_index = truncate_td(
            time, lower=-2.75, upper=2.75)
        time = time[lower_index:upper_index]
        complex_dat = complex_dat[lower_index:upper_index]

    real = np.real(complex_dat)
    imag = np.imag(complex_dat)

    return real, imag


def read_command_line():
    parser = AP()
    parser.add_argument('-p', '--plot', help="show the plotted data",
                        action='store_true', default=False)
    parser.add_argument('-o', '--output', help="save data to file",
                        action='store_true', default=False)
    parser.add_argument('-i', '--IR_intensity',
                        type=float, help="IR intensity", default=0.8)
    parser.add_argument('-r', '--read_all',
                        help="read all data from file rather than recalculate",
                        action='store_true', default=False)

    args = vars(parser.parse_args())

    if not args['plot']:
        args['output'] = True
    roi_lo = e_res - energy_shift - 5
    roi_hi = e_res - energy_shift + 10
    args['roi'] = [roi_lo, roi_hi]
    args['energy_shift'] = energy_shift
    return args
