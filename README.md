# advectionGP
A model of advection and diffusion to determine pollution sources using a Fourier GP approximation

# Installation
## If you just want to use it...
```
pip install git+https://github.com/SheffieldML/advectionGP.git
```

## If you want to develop it...

Go to where you want to save your repo:
```
git clone https://github.com/SheffieldML/advectionGP.git
pip install -e advectionGP
```
So you can now import this in a python script anywhere you happen to be (no need to worry about paths etc):
```
from advectionGP.models import AdjointAdvectionDiffusionModel as adjModel
```

Enjoy!
