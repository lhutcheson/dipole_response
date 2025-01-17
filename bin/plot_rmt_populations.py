import pandas as pd
from plotting import plot_populations
from helper_functions import pulse

pop_data = pd.read_csv('data/rmt/RMTpopulations.csv')
NIR_pulse = pulse(0.8, IR_freq=0.06798)
plot_populations(pop_data, pulse=NIR_pulse, intensity=0.8)
