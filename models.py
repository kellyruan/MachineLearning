import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(self.get_weights(), x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        if nn.as_scalar(self.run(x)) < 0:
            return -1
        else:
            return 1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        accuracy = 0
        count = 0
        for data in dataset.iterate_once(1):
            count += 1
        while True:
            for data in dataset.iterate_once(1):
                if nn.as_scalar(data[1]) == self.get_prediction(data[0]):
                    accuracy += 100 / count
                else:
                    nn.Parameter.update(self.get_weights(), data[0], nn.as_scalar(data[1]))
            if accuracy < 100:
                accuracy = 0
            else:
                break


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        self.weight = nn.Parameter(1, 100)
        self.bias = nn.Parameter(1, 100)
        self.weight2 = nn.Parameter(100, 1)
        self.bias2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        return nn.AddBias(nn.Linear(nn.ReLU(nn.AddBias(nn.Linear(x, self.weight), self.bias)), self.weight2), self.bias2)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        while True:
            temp = None
            for data in dataset.iterate_once(1):
                temp = data
                gradients = nn.gradients(self.get_loss(data[0], data[1]), [self.weight, self.weight2, self.bias, self.bias2])
                self.weight.update(gradients[0], -0.001)
                self.bias.update(gradients[2], -0.001)
                self.weight2.update(gradients[1], -0.001)
                self.bias2.update(gradients[3], -0.001)
            if nn.as_scalar(self.get_loss(temp[0], temp[1])) < 0.02:
                break

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).
    """
    def __init__(self):
        self.weight = nn.Parameter(784, 100)
        self.bias = nn.Parameter(1, 100)
        self.weight2 = nn.Parameter(100, 10)
        self.bias2 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        return nn.AddBias(nn.Linear(nn.ReLU(nn.AddBias(nn.Linear(x, self.weight), self.bias)), self.weight2), self.bias2)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        while dataset.get_validation_accuracy() < 0.975:
            for data in dataset.iterate_once(1):
                temp = data
                gradients = nn.gradients(self.get_loss(data[0], data[1]), [self.weight, self.weight2, self.bias, self.bias2])
                self.weight.update(gradients[0], -0.005)
                self.bias.update(gradients[2], -0.005)
                self.weight2.update(gradients[1], -0.005)
                self.bias2.update(gradients[3], -0.005)
