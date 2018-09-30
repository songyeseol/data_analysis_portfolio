%% GentriHackers Neural Network Learning
%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 9;  
hidden_layer_size = 1;  
num_labels = 2;     
INIT_EPSILON=3;
lambda = 3;
%=========== Part 1: Loading Data =============

% Load Training Data
X=xlsread('train.xls');
y=X(:,end);
X=X(:,1:end-1);

%m = size(X, 1);

%% ================ Part 2: Loading Parameters ================

Theta1 = rand(hidden_layer_size,input_layer_size+1) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta2 = rand(num_labels,hidden_layer_size+1) * (2 * INIT_EPSILON) - INIT_EPSILON;
% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];

%% ================ Part 3: Compute Cost (Feedforward) ================

% Weight regularization parameter (we set this to 0 here).


J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

%% =================== Part 8: Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 10000);

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


%% ================= Part 10: Implement Predict =================
P=xlsread("test.xls");
m = size(P, 1);
%% 0
h1 = sigmoid([ones(m, 1) P] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');

prediction=h2(:,2);
csvwrite('result.csv',prediction);