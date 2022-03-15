clc; clear all; close all
imgPath = '../images';
imgName = '1';
suffix = '.png';

rgbImg = imread(fullfile(imgPath, [imgName, suffix]));
grayImg = double(rgb2gray(rgbImg));

[row, col] = size(grayImg);

% parameters setting
c = 2; % the value for the initial level set function
deltaT = 1; % the time step
nu = 0.1; % the weight for the distance term
mu = 2; % the weight for the region term
lambda = 0.5 * mu; % the weight for the edge term
k = 0.2; % the smooth weight
a = 0.04;
w = k * lambda;
epsilon = 1;
alpha = 0.5;
beta = 0.5;
numIter = 200;

% initialize the level set function
s = 3;
initialLSF = ones(row, col) * c;
ymin = round( row / s );
ymax = round( (s-1) * row / s );
xmin = round( col / s);
xmax = round( (s-1) * col / s );
initialLSF(ymin:ymax, xmin:xmax) = -c;

% edge indicator function
sigma = 1.0; % scale parameter in Gaussian kernel
G = fspecial('gaussian', 3, sigma); % Gaussian kernel
smoothImg = conv2(grayImg, G, 'same');
[Gx, Gy] = gradient(smoothImg);
g = 1./ (1 + Gx.^2 + Gy.^2);

% image show
figure(1);
imshow(rgbImg); hold on;
set(gcf, 'position', [300, 100, 300, 300]);
set(gca, 'position', [0, 0, 1, 1]);
[cont, hn] = contour(initialLSF, [0, 0], 'g', 'LineWidth', 2); hold off;

phi = initialLSF;

for n = 1:numIter
    [phi, f] = RESLS(grayImg, phi, g, deltaT, nu, mu, lambda, w, a, epsilon, alpha, beta);
    
    if mod(n, 10) == 0
        pause(0.1);
        figure(1);
        imshow(rgbImg); hold on;
        set(gcf, 'position', [300, 100, 300, 300]);
        set(gca, 'position', [0, 0, 1, 1]);
        [cont, hn] = contour(phi, [0,0], 'g', 'LineWidth', 2);
        suptitle( num2str(n) );
        hold off;
    end
    
end