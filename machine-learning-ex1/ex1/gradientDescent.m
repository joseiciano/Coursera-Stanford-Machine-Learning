function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
% theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
% taking num_iters gradient steps with learning rate alpha

m = length(y);
J_history = zeros(num_iters, 1);
theta_len = length(theta);
theta_arr = [theta_len, 1];

for iter = 1:num_iters
  
  error = (X * theta) - y;
  for j = 1: theta_len
    theta_arr(j) = theta(j) - (alpha / m) * sum(error .* X(:, j));
  endfor
  
  for j = 1: theta_len
    theta(j) = theta_arr(j);
  endfor
  
  % Save the cost J in every iteration    
  J_history(iter) = computeCost(X, y, theta);
  
end

end