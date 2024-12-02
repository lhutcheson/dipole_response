import matplotlib.pyplot as plt
from helper_functions import (fit_lineshapes,
                              get_complex_data,
                              get_complex_dipole,
                              au_to_fs)
import numpy as np


def plot_fit_params(params, label, colour, truncate=True, fig=None, axs=None):
    """
    Given the time-delays and the fit parameters, plot the line strength,
    phase, and line width as a function of time delay on the provided figure
    and axes, or create new ones if not provided.

    Parameters
    ----------
    params : pd.DataFrame
        Fit parameters and fitting errors for each time delay
    label : str, optional
        The label for the plot, used in the legend.
    colour : str, optional
        The color to use for the plot. Can be a color string (e.g., 'C0', 'r',
        etc.).
    truncate : bool, optional
        Whether to truncate the data, removing time delay values outside the
        range [-2.75, 2.75]. Default is True.
    fig : matplotlib.figure.Figure, optional
        The figure on which to plot. If None, a new figure is created.
    axs : list, optional
        The list of axes to plot on. If None, new axes are created.

    Returns
    -------
    fig, axs : matplotlib.figure.Figure, list
        The figure and axes on which the plot was made.

    Notes
    -----
    - The function plots the line strength and phase as a function of time
      delay, with shaded regions representing the error margins (± error)
      around each data point.
    - The function assumes that the 'params' DataFrame contains the following
      columns:
        * 'Time Delays' — the time delay values (in fs).
        * 'Line Strength T1' — the line strength values.
        * 'Phase T1' — the phase values (in radians).
        * 'Line Strength T1 Error' — the error values for line strength.
        * 'Phase T1 Error' — the error values for phase.
    - The function provides titles for the subplots ('Strength $z$' for the
      first plot and 'Phase $\\varphi$ (rad.)' for the second).
    """
    if truncate:
        params.drop(params[params['Time Delays'] <= -2.75].index, inplace=True)
        params.drop(params[params['Time Delays'] >= 2.75].index, inplace=True)

    # If no figure or axes are provided, create them
    if fig is None or axs is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        axs = [ax1, ax2]
        ax1.set_title(r'Strength $z$')
        ax2.set_title(r'Phase $\varphi$ (rad.)')

    # Plot data on the axes
    for parameter, ax in zip(['Line Strength T1', 'Phase T1'], axs):
        ax.plot(params['Time Delays'], params[parameter],
                color=colour, label=label)
        ax.fill_between(
            params['Time Delays'],
            params[parameter] + params[f'{parameter} Error'],
            params[parameter] - params[f'{parameter} Error'],
            facecolor=colour, alpha=0.35)

    # Label axes
    axs[0].set_xlabel('Time delay (fs)')
    axs[1].set_xlabel('Time delay (fs)')
    plt.legend()
    return fig, axs


def plot_OD(OD_exper, OD_RMT):
    """
    Given the experimental and RMT optical density (OD) data as a function of
    energy and time delay, this function generates a surface plot comparing
    both datasets.

    Parameters
    ----------
    OD_exper : pd.DataFrame
        A DataFrame containing the experimental optical density values, where
        the index represents photon energy and the columns represent time
        delayss. Each cell corresponds to the OD at a specific energy and time
        delay.
    OD_RMT : pd.DataFrame
        A DataFrame containing the RMT optical density values, with the same
        structure as `OD_exper`, representing the calculateded OD at each time
        delay and energy.
    """
    # Define plotting parameters
    paramdict = {'cmap': 'plasma', 'shading': 'nearest'}

    # Create subplots for experimental and reconstructed OD
    fig, ax = plt.subplots(nrows=1, ncols=2, num=5)
    fig.subplots_adjust(right=0.9, left=0.1, top=0.9, bottom=0.15, wspace=0.2)

    # Plot the experimental optical density
    ax[0].pcolor(OD_exper.index.values,
                 [float(x) for x in OD_exper.columns],
                 OD_exper.transpose(),
                 **paramdict)

    # Plot the reconstructed optical density
    im2 = ax[1].pcolor(OD_RMT.index.values,
                       [float(x) for x in OD_RMT.columns],
                       OD_RMT.transpose(),
                       **paramdict)

    # Add a colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.75])
    fig.colorbar(im2, cax=cbar_ax)
    cbar_ax.set_title('OD')

    # Set axis labels and titles
    ax[0].set_ylabel(r'Time delay (fs)')
    ax[0].set_xlabel(r'Photon energy (eV)')
    ax[1].set_xlabel(r'Photon energy (eV)')
    ax[0].set_xlim([55, 56])
    ax[1].set_xlim([55, 56])
    ax[0].set_title('Experiment', fontsize=12, pad=1)
    ax[1].set_title('RMT', fontsize=12, pad=1)

    # Show the plot
    plt.show()


def plot_model_amplitudes(time,
                          ion_pop,
                          excited_pop,
                          excited_pop2,
                          smooth_excited):
    """
    Plot the time-dependent amplitudes for each channel used in the model. The
    function visualizes the ionization channel, two excitation channels, and
    the overall excitation (smoothed) as a function of time.

    Parameters
    ----------
    time : list-like of int or float
        The time axis (in femtoseconds, fs) for the data points. This list
        represents the time delays at which the amplitudes for the various
        channels are computed.

    ion_pop : list-like of int or float
        The time-dependent amplitudes for the ionization channel. This list
        corresponds to the population of ions at each time delay.

    excited_pop : list-like of int or float
        The time-dependent amplitudes for the first excitation channel. This
        list represents the population of the first excited state at each time
        delay.

    excited_pop2 : list-like of int or float
        The time-dependent amplitudes for the second excitation channel. This
        list corresponds to the population of the second excited state at each
        time delay.

    smooth_excited : list-like of int or float
        The smoothed time-dependent amplitudes for the overall excitation. This
        is the population of excited states modelled using the NIR envelope
        (rather than the full pulse).
    """
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.plot(time, ion_pop,
            label='Ionisation Channel',
            color='#7f3aacfd',
            linewidth=2.5)
    ax.plot(time, excited_pop,
            label='Excitation Channel 1',
            color='#71c837ff',
            linewidth=2.5)
    ax.plot(time, excited_pop2,
            label='Excitation Channel 2',
            color='#5f5fd3ff',
            linewidth=2.5)
    ax.plot(time, smooth_excited,
            label='Overall Excitation',
            linestyle='dotted',
            dashes=(1, 1),
            color='#ff5555ff',
            linewidth=4)

    ax.set_xlabel('Time Delay (fs)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_model_lineshapes(energy_axis,
                          phases,
                          strength=1,
                          linewidth=0.122,
                          background=0):
    """
    Plot the modeled lineshapes for the different effective channels--
    ionisation (ION), excitation 1 (E1), excitation 2 (E2), `bulk' excitation
    (OVERALL)) as a function of photon energy.

    Parameters
    ----------
    energy_axis : list-like of int or float
        The photon energy axis (in eV) over which the lineshapes are to be
        calculated and plotted.

    phases : list-like of float
        A list of phase values (in radians) corresponding to the different
        channels (ION, E1, E2, OVERALL). Each phase is used to generate a
        lineshape for the corresponding channel.

    strength : float, optional
        The strength factor for the lineshapes (default is 1). This controls
        the amplitude of the lineshape.

    linewidth : float, optional
        The linewidth parameter for the lineshape model (default is 0.122).
        This controls the width of the peak.

    background : float, optional
        The background value to be added to the lineshape (default is 0). This
        can be used to simulate a constant offset in the spectrum.
    """
    labels = ['ION', 'E1', 'E2', 'OVERALL']
    colours = ['#7f3aacfd', '#71c837ff', '#5f5fd3ff', '#ff5555ff']

    fig = plt.figure(2)
    ax = fig.add_subplot(111)

    for phase, lab, colour in zip(phases, labels, colours):
        line = fit_lineshapes(energy_axis, strength,
                              phase, linewidth, background)
        ax.plot(energy_axis, line,
                label=rf'$\varphi$ {lab} = {phase}',
                color=colour,
                linewidth=2.5)
    ax.legend()
    ax.set_xlabel('Energy (eV)')
    plt.xlim([54.5, 56.1])
    plt.show()


def plot_model_complex(complex_dipole_response,
                       colours=['#7f3aacfd', '#71c837ff',
                                '#5f5fd3ff', '#ff5555ff'],
                       labels=['1', '2', '3', '4']):
    """
    Plot the real vs. imaginary components of multiple complex dipole
    responses. This function generates a 2D plot for each complex-valued
    motion, where the real part is plotted on the x-axis and the imaginary part
    on the y-axis.

    Parameters
    ----------
    complex_dipole_response : list of complex ndarray or list-like
        A list of complex-valued motions (or trajectories) to be plotted. Each
        entry in the list should be a 1D array of complex dipole responses.

    colours : list of str, optional
        A list of colors to be used for each motion in the plot.
        Default is ['#7f3aacfd', '#71c837ff', '#5f5fd3ff', '#ff5555ff'].
                     Purple,      Green,       Blue,        Red

    labels : list of str, optional
        A list of labels to be used in the plot legend. Each label corresponds
        to a different response in the `complex_dipole_response` list.
        Default is ['1', '2', '3', '4'].
    """
    plt.figure(3)
    plt.axis('equal')

    for motion, lab, colour in zip(complex_dipole_response, labels, colours):
        plt.plot(np.real(motion), np.imag(motion),
                 label=lab,
                 color=colour,
                 linewidth=2)
    plt.legend()
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.show()


def plot_populations(pop_file):
    """
    Plot the time-dependent populations of different regions/states: ground
    state, outer region, and bound state as a function of time. The function
    uses a twin y-axis to plot the ground state population separately from the
    outer region and bound state populations, which are plotted on the primary
    y-axis.

    Parameters
    ----------
    pop_file : pd.DataFrame
        A DataFrame containing the population data. It should include the
        following columns:
        - 'Time': The time points at which the populations are measured
                    (in atomic units).
        - 'Ground': The population of the ground state at each time point.
        - 'Outer': The population of the outer region at each time point.
        - 'Bound': The total population of the bound states in the 1PO symmetry
                    at each time point.
    """
    # Convert time to femtoseconds and shift such that centre of NIR pulse is
    # at 0fs
    time_fs = au_to_fs(pop_file['Time']-900)

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()

    ax2.plot(time_fs, pop_file['Ground'],
             color='darkorange', linewidth=2, alpha=0.8)
    ax2.set_ylabel('GS Pop.')
    ax2.set_xlabel('Time (fs)')
    ax2.yaxis.label.set_color('darkorange')
    ax2.set_ybound(0.6, 1.01)
    ax2.tick_params(axis='y', colors='darkorange')

    ax.plot(time_fs, pop_file['Outer'],
            color='#7f3aacfd', linewidth=2, label='Outer Region Population')
    ax.plot(time_fs, pop_file['Bound'],
            color='#71c837ff', linewidth=2, label='Bound Population')

    ax.set_ylabel('Population')
    ax.set_xlabel('Time (fs)')
    ax.set_xbound(-8, 10)
    plt.tight_layout()
    ax.legend(loc='upper right')
    plt.show()


def plot_complex_single_intensity(RMT, exper, RMT_detuned, RMT_dipole):
    """
    Plot the complex dipole response obtained from fitting the RMT and
    experimental absorption lineshapes to get the dipole amplitude/phase. This
    is plotted along with the direct dipole response from RMT calculations in
    the complex plane.

    Parameters
    ----------
    RMT : pd.DataFrame
        A DataFrame containing the fitting results to the RMT absorption
        spectrum for the transition 'T1'. This should include the fitted line
        strength ['Line Strength T1'] and phase ['Phase T1'] for the
        transition.

    exper : pd.DataFrame
        A DataFrame containing the fitting results to the experimental spectrum
        for the transition 'T1'. Should be of the same structure as RMT.

    RMT_detuned : pd.DataFrame
        A DataFrame containing the fitting results to the RMT absorption
        spectrum for the transition 'T1' when the XUV energy is detuned off
        resonance. Should be of the same structure as RMT.

    RMT_dipole : pd.DataFrame
        A DataFrame containing the time-dependent expectation value of the
        dipole, obtained from RMT. To be plotted alongside the other data.
    """
    real_rmt, imag_rmt = get_complex_data(
        RMT, transition='T1', truncate=True)
    real_rmt2, imag_rmt2 = get_complex_data(
        RMT_detuned, transition='T1', truncate=True)
    real_exper, imag_exper = get_complex_data(
        exper, transition='T1', truncate=True)
    complex_dipole = get_complex_dipole(RMT_dipole)

    fig = plt.figure(1)
    ax1 = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax1.plot(real_rmt, imag_rmt,
             label='RMT',
             color='#0d0887ff',
             linewidth=2)
    ax1.plot(real_rmt2, imag_rmt2,
             label='RMT, detuned',
             color='#cc4778ff',
             linewidth=2)
    ax1.plot(real_exper, imag_exper,
             label='Exper.',
             color='#f89540ff',
             linewidth=2)

    ax1.set_xlabel(r'Real')
    ax1.set_ylabel(r'Imaginary')
    ax1.legend()
    ax1.axis('equal')

    ax2.plot(-1*np.real(complex_dipole), np.imag(complex_dipole),
             label='RMT direct dipole',
             color='#1fae1fff',
             linewidth=2)
    ax2.set_xlabel(r'Direct RMT dipole')
    ax2.set_ylabel(r'Direct RMT dipole')
    ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')

    ax2.xaxis.label.set_color('#1fae1fff')
    ax2.tick_params(axis='x', colors='#1fae1fff')
    ax2.yaxis.label.set_color('#1fae1fff')
    ax2.tick_params(axis='y', colors='#1fae1fff')
    ax2.xaxis.tick_top()
    ax2.yaxis.tick_right()
    ax2.axis('equal')
    plt.show()
