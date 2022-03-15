clc; clear all; close all

ImgPath = '../images';
ImgName = '1';
rgbImg = imread( fullfile(ImgPath, [ImgName, '.png']) );

Img = double( rgb2gray(rgbImg) );
[h, w] = size(Img);

% the initial active contour by using the circular initialization
ih = h / 2;
jw = w / 2;
r = h / 3;
[X, Y] = meshgrid(1:w, 1:h);
phi0 = sqrt( (X-jw).^2 + (Y-ih).^2 ) - r;

% Parameters setting
epsilon = 1;
sigma = std( Img(:) ) * (h*w) ^(-1/5);
nuCur = 0.01*255*255; % the weight of the length term
nuP = 0.01; % the weight of the regularization term
timeStep = 5;
iterNum = 40;

figure(1);
imshow(rgbImg); hold on;
set(gcf, 'position', [300, 100, 300, 300]);
set(gca, 'position', [0, 0, 1, 1]);
[cont, hn] = contour(phi0, [0, 0], 'g', 'LineWidth', 2); hold off;

phi = phi0;
IND = ones(iterNum, 1);
savedPhi = zeros(2, h, w);
pause();

for k = 1:iterNum
    [phi, pOut, pIn, aOut, aIn] = INDAC(Img, phi, sigma, nuCur, nuP, epsilon, timeStep);
    
    % save the phi in two adjacent steps
    if mod(k, 2) == 0
        savedIdx = 2;
    else
        savedIdx = 1;
    end
    savedPhi(savedIdx, :, :) = phi;
    
    % save the Inclusion Degree Energy
    aImg = h * w;
    M = sum( min(pOut, pIn) );
    IND(k) = (aIn / aImg) * M / sum(pIn) + ...
                    (aOut / aImg) * M / sum(pOut);
    
    % determine whether to stop iteration
    if k >= 20
        if IND(k - 1) <= IND(k) || any( any( isnan(phi)) )
            optimalIdx = mod(savedIdx, 2) + 1;
            optimalPhi = squeeze(savedPhi(optimalIdx, :, :));
            break
        end
    end
    
    % image show
    if mod(k, 1) == 0
        figure(1);
        imshow(rgbImg); hold on;
        set(gca, 'position', [0, 0, 1, 1]);
        [cont, hn] = contour(phi, [0,0], 'g', 'LineWidth', 2);
        suptitle( num2str(k) );
        hold off;
        
        figure(2);
        plot(0:255,pOut,'r',0:255,pIn,'b','LineWidth',2);axis square;
        legend('pOut', 'pIn');
        suptitle( num2str(k) );
    end
    
end

% plot IND Energy
figure(3);
plot(IND(1:k-1));
suptitle('Inclusion Degree Energy');