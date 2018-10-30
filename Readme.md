### Introduction

The code in this repository will let you train a Neural Network to play Pong solely based on the input frames of the game and the results of each round.

### Prerequisites

We have provided a requirements.txt for you to setup your **python3.6.6 environment.**

First off, create your virtual environment by entering the below command

### Mac OSX / Linux

If on OSX or Linux, enter the following into terminal
```

python3.6 -m venv pongenv

source pongenv/bin/activate

```

### Windows

If on windows, you may have to run the following in order to install the virtual environment tool

```
pip3.6 install virtualenv
```

Then, you have to run the following to make and activate the venv
```
virtualenv pongenv

pongenv\Scripts\activate.bat

```

### Setup

1. Install Openai Gym [here](https://gym.openai.com/docs). 
2. Run " pip install gym[atari] "

### Seeing the game

Set the variable "render" on line 14 to True if you wish to see the game. Note this will significantly slow down the training process.

### Performance

According to Karpathy's blog post, this algorithm should take around 3 days of training on a Macbook to start beating the computer.
Consider tweaking the hyperparameters or using CNNs to boost the performance further.

### Run
```
python pong.py
```
### Credit

This is based off of the work of Andrej Karpathy's great blog post and code [here](http://karpathy.github.io/2016/05/31/rl/)

