x = load('ml4Linx.dat');
y = load('ml4Liny.dat');
m = length(y);
plot(x,y,'o','MarkerFacecolor','r');
x = [ones(m,1),x,x.^2,x.^3,x.^4,x.^5];
theta = zeros(size(x(1,:)))';
lambda = 0;
L = lambda.*eye(6);
L(1) = 0;
theta = (x'*x+L)\x' * y;
norm_theta = norm(theta)
hold on;
x_vals = (-1:0.05:1)';
features = [ones(size(x_vals)), x_vals, x_vals.^2,x_vals.^3,x_vals.^4,x_vals.^5];
plot(x_vals, features*theta,'--','LineWidth',2);
legend('Training data', '5th order fit')
hold off

x = load('ml4Logx.dat');
y = load('ml4Logy.dat');
pos = find(y);
neg = find(y==0);
plot(x(pos,1),x(pos,2),'k+','LineWidth',2);
hold on
plot(x(neg,1),x(neg,2),'ko','MarkerFaceColor','y');

x = map_feature(x(:,1),x(:,2));
[m,n] = size(x);
g = inline('1.0/(1.0+exp(-z))');
MAX_ITERATION = 15;
J = zeros(MAX_ITERATION, 1);
lambda = 0;

for i = 1:MAX_ITERATION
z = x * theta;
h = g(z);
J(i) = (1/m)*sum(-y.*log(h)-(1-y).*log(1-h))+(lambda/(2*m))*norm(theta([2:end]))^2;
G = (lambda/m).*theta;
G(1) = 0;
L = (lambda/m).*eye(n);
L(1) = 0;
grad = ((1/m).*x'*(h-y)) + G;
H = ((1/m).*x' * diag(h) * diag(1-h) * x) + L;
theta = theta - H\grad;
end

u = linspace(-1, 1.5, 200);
v = linspace(-1, 1.5, 200);

z = zeros(length(u), length(v));

for i = 1:length(u)
    for j = 1:length(v)
        % Notice the order of j, i here!
        z(j,i) = map_feature(u(i), v(j))*theta;
    end
end

z = z';

contour(u,v,z, [0, 0], 'LineWidth', 2);
legend('y=1', 'y=0', 'Decision Boundary')
title(sprintf('\\lambda = %g', lambda), 'FontSize', 14);

norm_theta = norm(theta)
