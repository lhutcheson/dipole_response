# Dipole response induced by strong-field ionisation
Data and code for replicating results in the paper:

["Phase evolution of strong-field ionization"](https://doi.org/10.48550/arXiv.2502.19010)

#### Try the minimal model online now with Binder (may take 30-60s to start):
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lhutcheson/dipole_response/HEAD?labpath=.%2Fbin%2Fminimal_model.ipynb)


The RMT code is part of the UK-AMOR suite, and can be obtained for free **[here](https://gitlab.com/Uk-amor/RMT)**. 

## Requirements

A minimalist conda virtual environment with the required packages and versions
can be built from the environment.yml file provided.
To create the environment (named dipole_response), run

    conda env create -f environment.yml

and to activate the environment, run

    conda activate dipole_response

To launch the minimal model notebook, run

    jupyter notebook bin/minimal_model.ipynb

Alternatively, you can use the link above to try the minimal model with Binder.

