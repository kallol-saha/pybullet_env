### Create an environment:

```bash
conda create -n pybullet_env python=3.9
conda activate pybullet_env
```

### Install requirements:

```bash
pip install -r requirements.txt
```

### OMPL installation steps:

Go to https://ompl.kavrakilab.org/installation.html <br>
Or download directly from https://ompl.kavrakilab.org/install-ompl-ubuntu.sh

Navigate to the folder you downloaded OMPL in, and make the script executable.
Then, install OMPL with python bindings:

```bash
chmod u+x install-ompl-ubuntu.sh
./install-ompl-ubuntu.sh --python
```

Copy the `ompl-<version>/py-bindings/ompl` folder into `pybullet_env/ompl` (the home folder of this repository)
