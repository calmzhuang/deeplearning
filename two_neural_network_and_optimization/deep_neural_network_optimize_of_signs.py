import h5py
from two_neural_network_and_optimization.basic_function import *

def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def get_data():
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    # Flatten the training and test images
    X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
    X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
    # Normalize image vectors
    X_train = X_train_flatten / 255.
    X_test = X_test_flatten / 255.
    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig, 6)
    Y_test = convert_to_one_hot(Y_test_orig, 6)
    return X_train, Y_train, X_test, Y_test

def model(X_train, Y_train, X_test, Y_test, layers_dims, learning_rate = 0.0001, lambd = 0, num_epochs = 4000, minibatch_size = 32, print_cost = True):
    np.random.seed(1)
    m = X_train.shape[1]
    seed = 3
    t = 0
    costs = []
    parameters = initialize_parameters_deep(layers_dims)
    v, s = initialize_adam(parameters)
    accuracy_train = get_accuracy(X_train, Y_train, parameters)
    print(accuracy_train)
    for epoch in range(num_epochs):
        epoch_cost = 0
        num_minibatches = int(m / minibatch_size)
        seed = seed + 1
        minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
        for minibatch in minibatches:
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch
            AL, caches = L_model_forward(minibatch_X, parameters, keep_prob=[0.85, 1])
            minibatch_cost = compute_cost_with_regularization(AL + 1e-15, minibatch_Y, parameters, lambd)
            grads = L_model_backward(AL, minibatch_Y, caches, lambd)
            epoch_cost += minibatch_cost / num_minibatches
            t += 1
            parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate)
            # parameters = update_parameters(parameters, grads, learning_rate)
        # Print the cost every epoch
        if print_cost is True and epoch % 100 == 0:
            print("Cost after epoch %i: %f" % (epoch, epoch_cost))
        if print_cost is True and epoch % 5 == 0:
            costs.append(epoch_cost)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    accuracy_train = get_accuracy(X_train, Y_train, parameters)
    accuracy_test = get_accuracy(X_test, Y_test, parameters)
    print("accuracy_train: {}".format(accuracy_train))
    print("accuracy_test: {}".format(accuracy_test))

def get_accuracy(X, Y, parameters):
    AL, _ = L_model_forward(X, parameters, keep_prob=[1, 1])
    correct_prediction = np.equal(np.argmax(AL, axis=0), np.argmax(Y, axis=0))
    accuracy = np.mean(correct_prediction.astype(float))
    return accuracy

def run():
    X_train, Y_train, X_test, Y_test = get_data()
    layers_dims = [12288, 25, 12, 6]
    model(X_train, Y_train, X_test, Y_test, layers_dims)


if __name__ == '__main__':
    run()