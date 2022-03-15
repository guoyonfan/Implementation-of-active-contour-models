function gauValue = Gaussian(x, sigma)
% Gaussian function
gauValue = sqrt(2*pi*sigma^2)^(-1) * exp( - x.^2 / (2*sigma^2) );
end