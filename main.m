
%% Initialization
clear ; close all; clc


%Loading data
data = csvread('ecommerce.csv')

x = data(:,1);
y=data(:,2);


%Plotting the figure
figure
plot(x,y,'pv', 'MarkerSize',4);
xlabel('Length of Membership')
ylabel('Yearly Amount Spent')


alpha = 0.1;
num_of_iterations = 1000;

theta = zeros(2,1);

%Length of dataset
m=length(x);

%Length of training set
x=[ones(m,1), x]


%Cost function
J=zeros(num_of_iterations,1);

%Gradient Decent
for i=1:num_of_iterations

  h_of_x=(x*theta).-y;

  theta(1)=theta(1)-(alpha/m)*h_of_x'*x(:,1);
  theta(2)=theta(2)-(alpha/m)*h_of_x'*x(:,2);

  J(i)=1/(2*m)*sum(h_of_x.^2);
end

%computeCost(X, y, theta)


fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('theta 0 = %f theta 1 = %f \n', theta(1), theta(2));

%Drawing regression line
hold on;
plot(x(:,2), x*theta, 'r-','linewidth',1);
legend('Training data', 'Linear regression');


fprintf('Program paused. Press enter to continue.\n');
pause;


%Normal equation, for confirmation
thetaNormal = (pinv(x'*x))*x'*y;
hold on;
plot(x(:,2), x*thetaNormal, 'k-','linewidth',1);


%Predicting values
predict=[2,5, 6];

hold on;

for i=predict
	plot(i, [1, i]*theta, 'm*', 'MarkerSize',10);
end

legend('Training data', 'Linear regression')

hold off;
