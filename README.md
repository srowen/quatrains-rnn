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
- `keras`
- `numpy`

TensorFlow itself requires some additional setup, especially when using GPUs:
https://www.tensorflow.org/install/

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
