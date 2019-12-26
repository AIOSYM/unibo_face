# unibo_face

Work in progress....

### Environments

macOS 10.14.6

python 3.7.5

tensorflow 2.0.0

face_recognition 

### Setup

1. download and install anaconda https://docs.conda.io/en/latest/miniconda.html

2. create new environment

   ```
   conda create -n unibo
   ```

3. activate the environment

   ```python
   conda activate unibo
   ```

4.  install requirement libraries (using `conda`)

   ```
   conda install python=3.7
   conda install -c conda-forge opencv=4.1.1
   conda install -c conda-forge keras=2.3.1
   conda install -c conda-forge tensorflow=2.0.0
   ```

5. install requirement libraries (using `pip`)

   ```
   pip install -U numpy
   pip install matplotlib
   pip install face_recognition (take a while to install!)
   ```

6. download weight files from https://github.com/iwantooxxoox/Keras-OpenFace/tree/master/weights and put it in `weights/` folder.

### Run demo

With `python 3.7+`, run the main.py for demo

```
python main.py
```

### my module

unibo_face.py

utils.py

inception_blocks_v2.py



