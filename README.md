# Multi-Scale Object Search

## Installation

The following instructions have been tested with Python 3. First, clone the repo.

```
git clone --recursive --branch devel git@github.com:ethz-asl/multi_scale_search.git
```

Next, we recommend to create a new virtual environment.

```
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The package can either be built using catkin,

```
catkin build multi_scale_search
source path/to/catkin_ws/devel/setup.zsh
```

or alternatively

```
pip install -e .
```

Finally, compile [SARSOP](https://github.com/AdaCompNUS/sarsop).

```
cd third_party/sarsop/src
make
```

## Running the Experiments

```
python scripts/run_experiment.py [--gui]
```
