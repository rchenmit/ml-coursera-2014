function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


%% Part 1 - feedforward the NN and return the cost in J(theta)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X = [ones(m, 1) X];
a1 = X;
z2 = a1*Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2];
z3 = a2*Theta2'; %Z for OUTPUT LAYER - initialize to matrix all 0's with the correct dimensions
a3 = sigmoid(z3);

%FIND the LARGEST OUTPUT for (h_theta(x))_k; index should hold the INDEX of
%the MOST PROBABLE CLASS LABEL (out of num_labels *labels*)
[maxP,index] = max(a3,[],2); 
p = index;
%cost (NOT regularized)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Ymat = zeros(m, num_labels);
Jmat = Ymat;
for i = 1:m
    Ymat(i, y(i)) = 1;
end
Jmat = -Ymat.*log(a3) - (1-Ymat).*log(1-a3);
J = sum(sum(Jmat))/m;

%regularized cost:%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_layers = 3;
%%note: number of input units = input_layer_size;
%%note: number of hidden units = hidden_layer_size;
%%note: number of output units = num_labels;

Theta1_squared = Theta1.^2;
Theta2_squared = Theta2.^2;
penalize_factor_cost = lambda/(2*m) * (sum(sum(Theta1_squared(:,2:end))) + sum(sum(Theta2_squared(:,2:end))));
J = J + penalize_factor_cost;
%%regularized grad:
%penalize_factor_grad = zeros(length(theta),1);
%penalize_factor_grad(2:end) = lambda/m * theta(2:end);
%grad = grad + penalize_factor_grad;


%% Part 2: Backpropagation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Concretely, you should implement a for-loop for t = 1:m and
%place steps 1-4 below inside the for-loop, with the tth iteration performing
%the calculation on the t'th training example (x(t); y(t)).Step 5 will divide the
%accumulated gradients by m to obtain the gradients for the neural network
%cost function.

for t = 1:m
%     %1. perform feedforward pass -- we did this already in Part 1!
    
    %%2. For each output unit k in layer 3 (the output layer), set 
    %%delta^(3)_k = (a^(3)_k - y_k)
    y_for_this_t = y(t);
    yVect_for_this_t = zeros(num_labels,1);
    yVect_for_this_t(y_for_this_t) = 1;
    
    %%3. For the hidden layer l = 2, setL delta2 = (Theta2')*delta3.*g'(z2)
    %d2(t,2:end) = d3(t,:)*Theta2(:,2:end).*sigmoidGradient(z2(t,:));
    delt3 = a3(t,:)' - yVect_for_this_t;
    delt2 = (Theta2' * delt3) .* [1; sigmoidGradient(z2(t,:))'];
    delt2 = delt2(2:end);
    
    %%4. Accumulate the gradient from this example using the following formula. 
    %%Note that you should skip or remove $delta^(2)_0$. In Octave, removing
    %%$delta^(2)_0 corresponds to delta2 = delta2(2:end).
    Theta1_grad = Theta1_grad + delt2 * a1(t,:);
    Theta2_grad = Theta2_grad + delt3 * a2(t,:);

end
%5. Obtain the (unregularized) gradient for the neural network cost 
%function by multiplying the accumulated gradients by 1/m:
Theta1_grad = 1/m * Theta1_grad;
Theta2_grad = 1/m * Theta2_grad;

%% Part 3: Regularization with cost function and gradients
Theta1_grad = Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];
% -------------------------------------------------------------

%% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
