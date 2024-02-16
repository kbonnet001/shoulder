# shoulder

# Installation

## User

Simply use conda to create an environment from the `environment.yml` by running the following command:

```bash
conda create -f environment.yml
```

## Developer

If you plan to modify the `biorbd` [https://github.com/pyomeca/biorbd](https://github.com/pyomeca/biorbd) core, you will need to install the dependencies without installing `biorbd` from conda. 
The following command does most of it:

```bash
conda install pkgconfig cmake swig numpy scipy matplotlib rbdl eigen ipopt pyqt pyomeca vtk timyxml -cconda-forg
```

Please note that `bioviz` does not need to be installed. 
If you initialize the submodule, then the `PYTHONPATH` should points to `{$ROOT_SHOULDER}/external/bioviz`. 
