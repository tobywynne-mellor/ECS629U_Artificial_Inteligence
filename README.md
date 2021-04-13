# ECS629U Artificial Inteligence: Coursework  Report

[See Jupyter Notebook here](https://github.com/tobywynne-mellor/ECS629U_Artificial_Inteligence/blob/9039e2c16f85f1ef471ae5373fd39e9fd7185903/Toby_wynne-mellor_ECS629U_AI_CW.ipynb)

## Task 1 - **Training Dataset and Data loader**

I initiliased the dataset with `Nf` and `Npts` variables, denoting the number of functions to generate and the number of points to generate per function. During initialisation of the `Dataset`, the function generate_functions is called which uses the code provided to generate `Nf` functions and saves `Npts` points per function to the member variables `x_values` and `y_values`.

Then using a data split of 80% training_data:test_data ratio, I split the data set in two parts. The DataLoaders `train_iter` and `test_iter` are instantiated using their respective datasets, the `batch_size` and shuffle set to `True` to reduce non-representative batches.

## Task 2 - **Encoder and Decoder Models**

I created a Net class that is used for both the encoder and decoder. The Net class has 2 linear layers and uses a ReLU activation function between the linear layers.

The inputs to the encoder and decoder are both 2 and the hidden layers are both of dimention $h_{dim}$. The output of the encoder is $r_{dim}$ and the output of the decoder is 1.

## Task 3 - **Optimiser and Loss Function**

I am using a single Adam optimiser with learning rate set to 0.001 and weight decay set to 0.0005 for both the encoder and decoder models.

I am using MSELoss as it is well suited to regression problems.

## Task 4 - **Training: Loss and Hyperparameter**

In the training loop, I firstly set the models to train mode. I use the Accumulator class from `my_utils.py` to collect the loss metrics. For every batch, I sample the context points and pass them into the encoder and get $r_c$. To get the total context feature I average over all features in $r_c$ to produce $r_C$. $r_C$ and $x_t$ are combined and passed to the decoder to produce $\hat{y}$. Then I calculate the loss use MSE.

### Training Loss

I was able to achieve a training loss of around 0.073. With a validation loss of 0.077.

![ECS629U%20Artificial%20Inteligence%20Coursework%20Report%207d21f815ed18419b8bc505833284677e/training_plot16.svg](ECS629U%20Artificial%20Inteligence%20Coursework%20Report%207d21f815ed18419b8bc505833284677e/training_plot16.svg)

### Learning Rate

I experimented with several learning rates between 0.01 and 0.001 and 0.005 seemed to produce the lowest loss.

### $h_{dim}$ and $r_{dim}$

It seems that lower the bestter with $r_{dim}$, so I set it as 1 and $h_{dim}$ works well at ~256  to provide a low loss.

### Batch Size

Out of 5, 8, 16 and 100, batch size 16 provided the most consistently low loss.

### Optimiser

I tried both SGD and Adam but the loss seemed to get stuck at 0.27 with SGD. Adam allowed for the loss to be reduced further.

### Number of Hidden Layers

When I added an additional layer to the encoder and decoder the loss increased by ~0.02. So it seems that 2 layers is sufficient for the problem and adding more may be detrimental to the performance of the model both in terms of computation and accuracy.

## Task 5 - Evaluation

Using the test dataset I created in task 1, calculated the loss and plotted several functions to see how my model is performing in real terms.

![ECS629U%20Artificial%20Inteligence%20Coursework%20Report%207d21f815ed18419b8bc505833284677e/validation_plot17.svg](ECS629U%20Artificial%20Inteligence%20Coursework%20Report%207d21f815ed18419b8bc505833284677e/validation_plot17.svg)

The results from applying my model to the test sequences `test_data.pkl`:

![ECS629U%20Artificial%20Inteligence%20Coursework%20Report%207d21f815ed18419b8bc505833284677e/test_plot.svg](ECS629U%20Artificial%20Inteligence%20Coursework%20Report%207d21f815ed18419b8bc505833284677e/test_plot.svg)
