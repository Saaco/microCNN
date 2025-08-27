import torchvision
import numpy as np
import microcnn_v2

# dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True)

# normalization
avg = 0.13066047626710706
std = 0.30810780394137605

# hyperparameters
batch_size = 1
learning_rate = 0.001
epochs = 1

# one-hot encoding of the labels
one_h_y = {}
for i in range(10):
    temp = [0] * 10
    temp[i] = 1
    one_h_y[i] = temp

# loss function (MSE), this function needs more work to make sure the it outputs Values so we can backprop on them
def batch_mse(predictions, labels):
    result = []
    for prediction, label in zip(predictions, labels):
        error = [i - j for i, j in zip(prediction, label)]
        error = [diff**2 for diff in error]
        error = sum(error)
        error = error/len(predictions)
        result.append(error)
    # average sample losses to calculate batch loss
    result = sum(result)/len(result)
    return result



# model definition
class SimpleCnn():
    def __init__(self):
        self.conv1 = microcnn_v2.CLayer(1, 2, k_size=3, padding=1) # 2*28*28
        self.pool = microcnn_v2.MaxPool(4,4) # 2*7*7
        self.fc1 = microcnn_v2.Layer(2*7*7, 10)
        self.layers = [self.conv1, self.fc1]

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.fc1(x.flatten()) # automatically applies relu (neuron definition)
        return x

    def parameters(self):
        return [value for layer in self.layers for value in layer.parameters()]

    def gradients(self):
        return [value.grad for layer in self.layers for value in layer.parameters()]

# instantiate model
model = SimpleCnn()

# training loop

# loop through epochs
for epoch in range(epochs):
    # loop through dataset in batches
    for first_batch_sample in range(0, len(train_dataset), batch_size):
        batch = [train_dataset[i] for i in range(first_batch_sample, first_batch_sample + batch_size)]
        batch_predictions = []
        batch_labels = []
        # loop through batch
        for x, y in batch:
            # normalize sample image
            x = [(a - avg)/std for a in x.getdata()]
            # reshape the sample image
            x = np.array(x).reshape((1,28,28))
            # one-hot encode label
            y = one_h_y[y]
            batch_labels.append(y)
            # forward pass
            prediction = model.forward(x)
            batch_predictions.append(prediction)

        # calculate batch loss
        batch_loss = batch_mse(batch_predictions, batch_labels)
        # get model parameters
        param = model.parameters()
        # zero-out gradients
        for w in param:
            w.grad = 0
        # backprop (backward pass)
        batch_loss.backward()
        # optimization step (gradient descent)
        for w in param:
            w.data = w.data - learning_rate * w.grad




# testing loop
correct = 0
total = len(test_dataset)

for x,y in test_dataset:
    # normalize sample image
    x = [(a - avg)/std for a in x.getdata()]
    # reshape the sample image
    x = np.array(x).reshape((1,28,28))
    # forward pass
    prediction = model.forward(x)
    # interpret output using argmax
    prediction = np.argmax(prediction)
    # evaluate prediction against label
    if prediction == y:
        correct += 1

# results
print(f"test accuracy = {correct/total}")
