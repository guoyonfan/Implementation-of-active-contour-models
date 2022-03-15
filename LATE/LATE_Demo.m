clc;clear all;close all;

Img = imread('../images/1.png');
Img0=Img;
% Img = double(Img);
Img = double(Img(:,:,1));

[row, col] = size(Img);

% set the parameters
iterNum = 200;
mu =  0.02*255*255; % coefficient of the length term
timestep = 1;
nu = 0.1; % coefficient of the regularization term
epsilon = 1.0;
innerIter = 3;
sigma = 3.0; % scale parameter in Gaussian kernel
Ksigma = fspecial('gaussian', round(2*sigma)*2+1,sigma); % the Gaussian kernel
KOne = conv2(ones(size(Img)),Ksigma,'same');

% set the initial level set function
c = 2;
s = 3;
initialLSF = ones(row, col) * c;
ymin = round( row / s );
ymax = round( (s-1) * row / s );
xmin = round( col / s);
xmax = round( (s-1) * col / s );
initialLSF(ymin:ymax, xmin:xmax) = -c;

% set the inital b'
b0 = ones(size(Img));

phi = initialLSF;
b=b0;

% image show
figure(1);
imshow(Img0); hold on;
set(gcf, 'position', [300, 100, 300, 300]);
set(gca, 'position', [0, 0, 1, 1]);
[cont, hn] = contour(initialLSF, [0, 0], 'g', 'LineWidth', 2); hold off;

pause()
% start the level set evolution
for n=1:iterNum
    [phi,b]=LATE_Evolution(phi,b,KOne,Ksigma,Img,timestep,nu,mu,epsilon, innerIter);
    
    if mod(n,10)==0
        figure(1);
        imshow(Img0); hold on;
        set(gcf, 'position', [300, 100, 300, 300]);
        set(gca, 'position', [0, 0, 1, 1]);
        [cont, hn] = contour(phi, [0,0], 'g', 'LineWidth', 2);
        suptitle( num2str(n) );
        hold off;
    end
end

figure(2);
mesh(phi);
title('Final level set function');