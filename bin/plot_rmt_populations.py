import pandas as pd
from plotting import plot_populations

pop_data = pd.read_csv('data/rmt/RMTpopulations0.8.csv')
plot_populations(pop_data)
