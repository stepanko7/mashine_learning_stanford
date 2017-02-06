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

%input_layer_size  = 400;  % 20x20 Input Images of Digits
%hidden_layer_size = 25;   % 25 hidden units
%num_labels = 10;          % 10 labels, from 1 to 10   



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


%size(Theta1) % 25 x 401
%size(Theta2) % 10 x 26



a1 = [ones(size(X, 1), 1)  X]; % 500x401

z2 = a1*(Theta1');  % 5000 x 25

a2 = sigmoid(z2); % 5000 x 25
a2 = [ones(size(a2, 1), 1)  a2]; % 5000 x 26

z3 = a2*(Theta2'); %5000 x 10
a3 = sigmoid(z3); %5000 x 10

yy = zeros(size(a3)); % %5000 x 10
v = m*(y'-1) + [1:m];
yy(v) = 1; 


J = sum(sum( -log(a3).*yy - log(1-a3).*(1-yy) ))/m;

t1 = Theta1(:,2:end); % 25 x 400
t2 = Theta2(:,2:end); % 10 x 25

J = J + ( sum(sum(t1.*t1)) + sum(sum(t2.*t2)) )*lambda/(2*m);

% Backpropagation

%t1 = Theta1;
%t2 = Theta2;

Delta_1 = zeros(size(Theta1));
Delta_2 = zeros(size(Theta2));
for i = 1:m
%  a1i = a1(i,2:end)'; % 400x1
%  a2i = a2(i,2:end)'; % 25x1
  a1i = a1(i,:)'; % 401x1
  a2i = a2(i,:)'; % 26x1
  a3i = a3(i,:)'; % 10x1

  z2i = z2(i,:)'; % 25x1
  z3i = z3(i,:)'; % 10x1

%  yyi = yy(i,:)';
  yyi = zeros(num_labels,1);
  yyi(y(i),1) = 1;

  delta3 = a3i - yyi; % 10x1
%  
  delta2 = (t2'*delta3).*sigmoidGradient(z2i);  %
%  delta2 = delta2(2:end);

%  size(delta2)
%  size(a1i)
  Delta_1 = Delta_1 + delta2*a1i';
  Delta_2 = Delta_2 + delta3*a2i';
end

%Theta1_grad = zeros(size(Theta1));
%Theta2_grad = zeros(size(Theta2));

%size(Theta1) % 25 x 401
%size(Theta2) % 10 x 26

%size([zeros(t1,1) t1])
Theta1_grad = Delta_1/m + lambda*[zeros(size(t1,1),1) t1]/m;
Theta2_grad = Delta_2/m + lambda*[zeros(size(t2,1),1) t2]/m;


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



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
