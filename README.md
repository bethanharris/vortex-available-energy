# vortex-available-energy

Analyse available energetics of a dry axisymmetric atmospheric vortex as described by Tailleux and Harris (2019):
> The generalised buoyancy/inertial forces and available energy of axisymmetric compressible stratified vortex motions

[Vortex.py](src/Vortex.py) contains the Vortex class, which is used to define a dry axisymmetric atmospheric vortex in thermal wind balance.

The vortex used as an example in Tailleux and Harris (2019), which itself is taken from [Smith (2005)](http://doi.org/10.1016/J.DYNATMOCE.2005.03.003), can be instantiated using the Vortex.smith() class method.

Vortices created using the Vortex class have methods to compute typical air parcel properties. For example, to compute the entropy of the Smith vortex at a radius of 5km and a height of 1km:
```python
vortex = Vortex.smith()
entropy = vortex.entropy(5000., 1000.)
```

[available_energy.py](src/available_energy.py) contains functions for computing the various energies defined by Tailleux and Harris (2019):
* Available acoustic energy <img src="https://render.githubusercontent.com/render/math?math=\Pi_1">
* Vortex available energy <img src="https://render.githubusercontent.com/render/math?math=A_e">
* Thermodynamic (<img src="https://render.githubusercontent.com/render/math?math=\Pi_e">) and mechanical (<img src="https://render.githubusercontent.com/render/math?math=\Pi_k">) components of <img src="https://render.githubusercontent.com/render/math?math=A_e">,

using a Vortex object as a reference state. Functions for computing vortex available energy for a range of perturbations are also provided, as well as functions for comparing <img src="https://render.githubusercontent.com/render/math?math=\Pi_k"> to the eddy kinetic energy.

[plot_vortex.py](src/plot_vortex.py) contains functions to plot various vortex and available energy features, including the figures featured in Tailleux and Harris (2019).

Running
```bash
python plot_vortex.py
```
from the src directory will generate all figures from Tailleux and Harris (2019) and save them to the [results](results) folder.

Tested using python 3.7.3 with numpy 1.16.4 and matplotlib 3.1.0.
