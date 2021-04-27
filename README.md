# MODEL REDUCTION FRAMEWORK

A framework for defining differential equation based models used in neuroscience
and reducing their dimensionality with mathematical model order reduction (MOR) methods.

## Installation

This project requires [Python 3.5](https://www.python.org/downloads/) or greater. 
Use of virtual environments is
strongly encouraged, but not required, to avoid clashes with numpy and matplotlib versions! See
[the venv documentation](https://docs.python.org/3/tutorial/venv.html) for more
information and installation instructions.

Once your brand new virtual environment is activated, navigate to the folder of
this repository and install it with 
``` 
pip install -e .  
```
or replace the `.` with the path to this project. All the dependencies will be downloaded automatically.

Finally, confirm that the installation was successfull by executing the provided tests
```
python tests/model_test.py
python tests/reduction_test.py
```
Numpy warnings about the `np.matrix` class can safely be ignored.

## Usage

Launch a Jupyter Notebook instance with
```
jupyter notebook
```
and find the notebook named `fitzhugh_nagumo_meanfield_reduction.ipynb`. Follow the notebook!

Citation for this project will be provided soon! For the time being, our previous work studying MOR
of a synaptic plasiticy model can be cited as 

- Lehtimäki, Mikko, Ippa Seppälä, Lassi Paunonen, and Marja-Leena Linne. 
  "Accelerated Simulation of a Neuronal Population via Mathematical Model Order Reduction." 
  In 2020 2nd IEEE International Conference on Artificial Intelligence Circuits and Systems (AICAS), pp. 118-122. IEEE, 2020.
- Lehtimäki, Mikko, Lassi Paunonen, and Marja-Leena Linne. 
  "Projection-based order reduction of a nonlinear biophysical neuronal network model." 
  In 2019 IEEE 58th Conference on Decision and Control (CDC), pp. 1-6. IEEE, 2019.
- Lehtimäki, M., Paunonen, L., Pohjolainen, S. and Linne, M.L., 2017. Order
  reduction for a signaling pathway model of neuronal synaptic plasticity.
  IFAC-PapersOnLine, 50(1), pp.7687-7692.
  
Our work can also be followed in my [ResearchGate profile](https://www.researchgate.net/profile/Mikko_Lehtimaeki)!
