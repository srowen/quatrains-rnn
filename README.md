# quatrains-rnn

Simple example applying Keras, TensorFlow to Nostradamus's prophecies to generate new ones

## Setup

To run this example, first download the GLoVe embedding from https://nlp.stanford.edu/projects/glove/ 
Warning: it's over 800MB.

```bash
curl -O https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

The following Python packages must be installed in the Python 3 environment you use:

- `tensorflow` (or `tensorflow-gpu`)
- `tensorboard`
- `keras`
- `numpy`

TensorFlow itself requires some additional setup, especially when using GPUs:
https://www.tensorflow.org/install/

### Building CPU-optimized TensorFlow

You can get a moderate speedup from CPU-based training by compiling TensorFlow from source. 
This is optional, and may not work in all cases.

To build and install the very latest:

```bash
pip3 uninstall tensorflow

# Or latest release of Bazel
curl -L -o bazel.sh https://github.com/bazelbuild/bazel/releases/download/0.5.2/bazel-0.5.2-without-jdk-installer-linux-x86_64.sh
bash bazel.sh --user
export PATH=~/bin:$PATH

git clone https://github.com/tensorflow/tensorflow
cd tensorflow
./configure
#Set Python path to /usr/local/bin/python3 and accept all other defaults

bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip3 install `ls /tmp/tensorflow_pkg/tensorflow*.whl`
```

## Running

Edit the configuration parameters at the top of the file as desired, and:

```bash
python3 quatrains.py
```

## Output

Output will look something like:

```
Epoch 655
Train on 24533 samples, validate on 2726 samples
Epoch 1/1
9s - loss: 5.3967 - acc: 0.2597 - val_loss: 6.9084 - val_acc: 0.1599

the realm of the great trouble their great rivers brave and the elder locked up
of the foreign city through land for last will be second mountains so
the a great seized another promise age of surname of the great
over the mountains in a aquarius when the sea
...
```

The fourth line shows the time taken to compute one epoch, and loss/accuracy on training data vs held-out 
validation data. It also shows some texts randomly generated from the model.
