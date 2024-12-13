# StoichiometricRay
SI Codes for "Theoretical basis for cell deaths"
https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.6.043217

## Brusselator
The codes for computing the stoichiometric ray of the reversible Brusselator (Fig.4).

`main_module.py` contains the main modules used to compute the Rays. 

`ComputeRay.ipynb` is the ipython notebook for the computation of the Ray. 

You do not need to see `main_module.py` unless you want to check what is happening behind.  

## Metabolic
The codes for computing the dynamics, Stoichiometric Rays, and the transition diagram of the metabolic model (Figs.5~7).

The folder has a similar structure to that of the Brusselator model. 

`main_module.py` and `plot_module.py` are the function modules for the dynamics computation, ray computation, and visualization. You do not need to check them unless you want to check what is going on behind.  

To generate the data files for the figures, first run `model.ipynb` which computes the model dynamics from several initial conditions and exports the information of the live and dead attractor.

Next, run `Ray.ipynb` which computes the stoichiometric rays of the active attractor, equivalently, the understimated dead set (Fig.5A). 

`plot.ipynb` visualizes the obtained non-returnable set using `plotly` package.

`Hasse.ipynb` computes the transition diagram of the sampled states. Note that the following; If we segment each axis into $n$ segments, then we need to compute the transitivity of $n^6$ times. The computation of even a single pair of points needs several optimizations because there are multiple paths of the reaction flips. Be careful about setting the $n$ value. 

`Linear_Flux.ipynb` is an independent notebook. This describes that the model with multiple substrates (e.g. E+2X→E+2Y) but shows a linear dependence of the reaction rate on the concentration of X.




