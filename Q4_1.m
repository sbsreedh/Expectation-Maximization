clear; close all;

% Generate data X
n = 300;
mu1 = [ 2 5];
sigma1 = [3 1; 1 0.5];
mu2 = [0 1];
sigma2 = [1 0.5; 0.5 2];

for i = 1:n/2
    X1(i,:) = (chol(sigma1)*[normrnd(0,1) normrnd(0,1)]' + mu1')';
    X2(i,:) = (chol(sigma2)*[normrnd(0,1) normrnd(0,1)]' + mu2')';
end

X = [X1; X2];

% Initialization
k = randperm(n);
mu = X(k(1:2), :);

sigma1 = cov(X);
sigma2 = cov(X);

p = [0.2 0.8];

ud1 = bsxfun(@minus, X, mu(1,:));
ud2 = bsxfun(@minus, X, mu(2,:));

phi(:,1)=exp(-1/2*sum((ud1*inv(sigma1).*ud1),2))/sqrt((2*pi)^2*det(sigma1));
phi(:,2)=exp(-1/2*sum((ud2*inv(sigma2).*ud2),2))/sqrt((2*pi)^2*det(sigma2));
    
L_before = sum(log(p(1).*phi(:,1)+p(2).*phi(:,2)))/n;

for i = 1:2000
    %E-step
    phi_w = bsxfun(@times, phi, p);

    gamma = bsxfun(@rdivide, phi_w, sum(phi_w, 2));
    
    mu_temp = mu;
    
    % M-step
    mu(1,:) = gamma(:, 1)' * X ./ sum(gamma(:, 1), 1);
    mu(2,:) = gamma(:, 2)' * X ./ sum(gamma(:, 2), 1);

    XS1 = bsxfun(@minus, X, mu(1, :));
    XS2 = bsxfun(@minus, X, mu(2, :));

    for j=1:n
        sigma1 = sigma1 + (gamma(j, 1) .* (XS1(j, :)' * XS1(j, :)));
        sigma2 = sigma2 + (gamma(j, 2) .* (XS2(j, :)' * XS2(j, :)));
    end

    sigma1 = sigma1 ./ sum(gamma(:, 1));
    sigma2 = sigma2 ./ sum(gamma(:, 2));
     
    p = [mean(gamma(:,1)) mean(gamma(:,2))];
    
    ud1 = bsxfun(@minus, X, mu(1,:));
    ud2 = bsxfun(@minus, X, mu(2,:));

    phi(:,1)=exp(-1/2*sum((ud1*inv(sigma1).*ud1),2))/sqrt((2*pi)^2*det(sigma1));
    phi(:,2)=exp(-1/2*sum((ud2*inv(sigma2).*ud2),2))/sqrt((2*pi)^2*det(sigma2));
    
    L_after = sum(log(p(1).*phi(:,1)+p(2).*phi(:,2)))/n;
    
    % Check convergence
    if L_after == L_before
        break;
    else
        L_before = L_after;
    end
            
end

sigma = cat(3,sigma1,sigma2);
% build GMM
obj = gmdistribution(mu,sigma,p);

% 2D projection
figure;
ezcontourf(@(x,y) pdf(obj,[x y]));

mu
sigma1
sigma2
p