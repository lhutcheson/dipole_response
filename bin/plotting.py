import matplotlib.pyplot as plt
from helper_functions import fit_lineshapes, get_complex_data, get_complex_dipole
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


def plot_populations(pop_file):
    time_fs = (pop_file['Time']-900)/41.341

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
             label='RMT direct (scaled)',
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
