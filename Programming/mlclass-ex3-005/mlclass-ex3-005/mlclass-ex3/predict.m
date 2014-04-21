function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
X = [ones(m, 1) X];
a1 = X;
z2 = 0*a1*Theta1'; %Z for HIDDEN LAYER - initialize to matrix all 0's with the correct dimensions
a2 = z2;
num_nodes_Theta1 = size(Theta1,1);
for k = 1:num_nodes_Theta1
    z2(:,k) = a1*Theta1(k,:)';
    a2(:,k) = sigmoid(z2(:,k));
end
a2 = [ones(m, 1) a2];
z3 = 0*a2*Theta2'; %Z for OUTPUT LAYER - initialize to matrix all 0's with the correct dimensions
for k = 1:num_labels
   z3(:,k) = a2*Theta2(k,:)'; 
   a3(:,k) = sigmoid(z3(:,k));
end

%FIND the LARGEST OUTPUT for (h_theta(x))_k; index should hold the INDEX of
%the MOST PROBABLE CLASS LABEL (out of num_labels *labels*)
[maxP,index] = max(a3,[],2); 
p = index;

% =========================================================================


end
