import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500))) #prevent overflow

def sigmoid_derivative(x):
    return x * (1 - x)
    #sigmoid has already been called on x prior to sigmoid_derivative being called

class MLP:
    def __init__(self, nbr_layers, units_per_layer):
        assert nbr_layers == len(units_per_layer)
        self.activations = [None] * (nbr_layers+2)
        self.layers = []
        self.biases=[]
        scale = 2.0/(nbr_layers+2)
        for i in range(nbr_layers-1): # -1 because each weight matrix is used between two layers
            weight_matrix = np.random.normal(0,scale,(units_per_layer[i], units_per_layer[i+1]))    #input x output
            self.layers.append(weight_matrix)

            bias_vector = np.zeros(self.layers[i].shape[1])
            self.biases.append(bias_vector)

    
    def fit(self, x, y, lr, epochs, minibatch_size, print_updates):
        
        scale = 2.0/len(self.layers)
        
        input_weight_matrix = np.random.normal(0,scale,(len(x[0]), self.layers[0].shape[0]))
        self.layers.insert(0, input_weight_matrix)

        output_weight_matrix = np.random.normal(0,scale,(self.layers[-1].shape[1], len(y[0])))
        self.layers.append(output_weight_matrix)

        input_bias_vector = np.zeros(self.layers[1].shape[0])
        self.biases.insert(0,input_bias_vector)

        output_bias_vector = np.zeros(self.layers[-1].shape[1])
        self.biases.append(output_bias_vector)

        
        for epoch in range(epochs):
            cumulative_grad = [np.zeros(l.shape) for l in self.layers]
            minibatch_ids = np.random.choice(len(x), minibatch_size, False)
            minibatch_x = x[minibatch_ids]
            minibatch_y = y[minibatch_ids]
            cumulative_loss = 0

            dW = [np.zeros_like(layer) for layer in self.layers]
            dB = [np.zeros_like(bias) for bias in self.biases]

            for j in range(minibatch_size):
                # Forward pass
                y_hat = self.predict(minibatch_x[j])

                loss = -np.log(y_hat @ (minibatch_y[j].T)) / len(minibatch_y[j])
                #loss = -np.sum(minibatch_y[j] * np.log(y_hat + 1e-15))
                cumulative_loss += loss

                # Backward pass
                ## Gradients at the last layer
                error = y_hat - minibatch_y[j]

                #output layer gradient
                dW[-1] += np.outer(self.activations[-2], error)
                dB[-1] += error


                # Gradients at hidden layers
                curr_error = error
                for i in range(len(self.layers)-2, -1, -1): #starting from the second to last layer (last layer was computed above)

                  curr_error = (curr_error @ (self.layers[i+1].T)) * (sigmoid_derivative(self.activations[i+1])) #Gn​=(Gn+1​⋅Wn​)∘σ′(An​)

                  # Update gradients
                  dW[i] += np.outer(self.activations[i], curr_error)
                  dB[i] += curr_error

            if print_updates and not epoch % print_updates:
                print(f"[TRAINING] epoch = {epoch}, loss={cumulative_loss}")

            # Update the weights from the total of the gradients
            for i in range(len(self.layers)):
                self.layers[i] -= lr * dW[i] / minibatch_size
                self.biases[i] -= lr * dB[i] / minibatch_size

    def one_hot_encoded (self, y, num_classes):
         
        #converts labels to one-hot encoding form (N x K matrix)
        y = y.astype(int)
        Y_one_hot = np.zeros((y.shape[0], num_classes))
        Y_one_hot[np.arange(y.shape[0]), y] = 1 
        return Y_one_hot
    
    def softmax(self,z):
      z = z - np.max(z)
      exp = np.exp(z)
      return exp/np.sum(exp)

    def ReLU(self,z):
        return np.max(0, z)
    
    def predict(self, x):
        self.activations[0] = x #A1=X

        for i in range(len(self.layers)):

            activation = self.activations[i]
            z = activation @ self.layers[i] + self.biases[i]

            if i<len(self.layers) -1:
              self.activations[i+1] = sigmoid(z).flatten()
            else:
              self.activations[i+1] = self.softmax(z.flatten())


        return self.activations[-1]  # Final output layer activation


