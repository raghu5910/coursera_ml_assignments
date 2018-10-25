function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
%X = [ones(m, 1) X];
m = length(y); 
prediction = X*theta;
Error = prediction-y;
sqrErrors = (Error).^2;
theta2=theta;
theta2(1,1)=0;
J = (sum(sqrErrors)/(2*m)) + (lambda*sum(theta2.^2))/(2*m);

grad(1) = sum(Error)/m ;
for i=2:size(theta),
	grad(i) = ((X(:,i)'*Error)/m) +(lambda*theta(i))/m;
end;
% =========================================================================

grad = grad(:);

end
