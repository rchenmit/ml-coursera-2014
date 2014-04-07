function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%non-regularized (copied from costFunction.m):
J = -1/m *sum(y' * log( sigmoid(X*theta)) + (1-y') *log(1-sigmoid(X*theta)));
grad = 1/m* X'*(sigmoid(X*theta)-y);
%regularized cost:
theta_squared = theta.^2;
penalize_factor_cost = lambda/(2*m) * sum(theta_squared(2:end));
J = J + penalize_factor_cost;
%regularized grad:
penalize_factor_grad = zeros(length(theta),1);
penalize_factor_grad(2:end) = lambda/m * theta(2:end);
grad = grad + penalize_factor_grad;
% =============================================================

end
