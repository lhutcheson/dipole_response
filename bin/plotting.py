import matplotlib.pyplot as plt
from helper_functions import fit_lineshapes
import numpy as np


def plot_model_amplitudes(time, ion_pop, excited_pop, excited_pop2, smooth_excited):
    """
    Plot the time-dependent amplitudes for each channel used in the model.

    Arguments:
        - time:     <list-like <int, float>>
                    The time-axis to be used
        - ion_pop:  <list-like <int, float>>
        - excited_pop:  <list-like <int, float>>
        - excited_pop2:  <list-like <int, float>>
        - smooth_excited:  <list-like <int, float>>
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


def plot_model_lineshapes(energy_axis, phases, strength=1, linewidth=0.122, background=0):
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


def plot_model_complex(complex_motions,
                       colours=['#7f3aacfd', '#71c837ff',
                                '#5f5fd3ff', '#ff5555ff'],
                       labels=['1', '2', '3', '4']):

    plt.figure(3)
    plt.axis('equal')

    for motion, lab, colour in zip(complex_motions, labels, colours):
        plt.plot(np.real(motion), np.imag(motion),
                 label=lab,
                 color=colour,
                 linewidth=2)
    plt.legend()
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.show()
