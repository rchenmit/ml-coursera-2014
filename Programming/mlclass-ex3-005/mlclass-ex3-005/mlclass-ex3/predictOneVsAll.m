function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

p_each_k = p;
for k = 1:num_labels
    %this_p = zeros(m, 1);
    X_theta_T = X*all_theta(k,:)';
    logistic_of_inputs = sigmoid(X_theta_T);
    %this_p(logistic_of_inputs>=0.5) = 1;
    %this_p(logistic_of_inputs<0.5) = 0;
    p_each_k = [p_each_k, logistic_of_inputs];
end

[maxP,index] = max(p_each_k,[],2); %get the max Probability and the index of that (index corresponding to the feature class)
p = index;
p = p-1; %because we appended a list of 1's to the matrix preceding the first "label" (For k labels)







% =========================================================================


end
