import pandas as pd
from plotting import plot_fit_params, plot_OD

exper_fits = pd.read_csv('data/experiment/fit_params0.94482114.csv')
RMT_fits = pd.read_csv('data/rmt/fit_params0.8.csv')
RMT_detune_fits = pd.read_csv('data/rmt_xuv_detuned/fit_params0.8.csv')

# Create the first set of plots
fig, axs = plot_fit_params(exper_fits,
                           label='Experiment',
                           colour='#f89540ff')

# Add the second set of plots to the same figure
fig, axs = plot_fit_params(RMT_fits,
                           colour='#0d0887ff',
                           label='RMT',
                           fig=fig,
                           axs=axs)

# Add another set if needed
fig, axs = plot_fit_params(RMT_detune_fits,
                           colour='#cc4778ff',
                           label='RMT, detuned',
                           fig=fig,
                           axs=axs)

exper_data = pd.read_csv(
    'data/experiment/OD0.94482114.csv').set_index('Energy')
RMT_data = pd.read_csv('data/RMT/OD0.8.csv').set_index('Energy')

plot_OD(exper_data, RMT_data)
