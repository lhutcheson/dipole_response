import pandas as pd
from plotting import plot_complex_single_intensity

data_rmt = pd.read_csv('data/rmt/fit_params0.8.csv')
data_dipole = pd.read_csv('data/rmt/dipole0.8.csv')
data_rmt2 = pd.read_csv('data/rmt_xuv_detuned/fit_params0.8.csv')
data_exper = pd.read_csv('data/experiment/fit_params0.94482114.csv')

plot_complex_single_intensity(data_rmt, data_exper, data_rmt2, data_dipole)
