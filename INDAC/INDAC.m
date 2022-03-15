function [phi, pOut, pIn, aOut, aIn] = INDAC(Img, phi0, sigma, nuCur, nuP, epsilon, timeStep)
% reinitilize the phi
[h, w] = size(Img);
% binaryPhi = double(phi0 <= 0);
% distIn = bwdist(1 - binaryPhi);
% distOut = bwdist(binaryPhi);
% phi = distOut - distIn;
phi = NeumannBoundCond(phi0);

aIn = sum( sum( phi < 0 ) );
aOut = sum( sum( phi >= 0 ) );
aImg = h * w;
z = linspace(0, 255, 256); % the intensity variable
i = reshape(Img, [], 1);

deltaPhi = deltaF(phi, epsilon);

pIn = KDE( Img(phi < 0), z, sigma );
pOut = KDE( Img(phi >= 0), z, sigma );

BInOut = double( pIn > pOut ); % the two-valued function
BOutIn = double( pOut > pIn);

Uin = sum(pIn .* BInOut);
Uout = sum(pOut .* BOutIn);

allTmp = bsxfun(@minus, i, z);
gauTmp = Gaussian(allTmp, sigma);

W1 = (aIn / aImg) * BInOut * ...
            ( (Uin*aIn + Uout*aOut) / (Uin^2 * aIn * aOut + 1e-8) );
W2 = -(aOut / aImg) * BOutIn * ...
            ( (Uin*aIn + Uout*aOut) / (Uout^2 * aIn * aOut + 1e-8) );

W = W1 + W2;
IND = bsxfun(@times, gauTmp, W);
IND = sum(IND, 2);
INDI = reshape(IND, h, w);

G1 = -(aIn / aImg) * ...
            (Uout * (aIn + aOut) / (Uin * aIn * aOut + 1e-8) );
G2 = (aOut / aImg) * ...
            ( Uin * (aIn + aOut) / (Uout * aIn * aOut + 1e-8) );
G = G1 + G2;
        
V =  -(INDI + G) .* deltaPhi; % Equation (12)
maxV = max( max(V(:)) );
scaleRate = 100 / maxV;
V = V * scaleRate; % Normalize the max V to be 100

% the length term
if nuCur ~= 0
    Cur = Curvature(phi);
    P = nuP * ( 4 * del2(phi) - Cur);
    C = nuCur * Cur .* deltaPhi;
else
    C = 0;
    P = 0;
end

phi = phi + timeStep * (C + V + P);

end

function deltaPhi = deltaF(phi, epsilon)
% the smoother version of the deltafunction
% deltaPhi = (1 / (2*epsilon) ) * ...
%     (1 + cos( (pi * phi) / epsilon) );
% deltaPhi( abs(phi) > epsilon ) = 0;
deltaPhi = (epsilon / pi) ./ (epsilon^2 + phi.^2);
end

function density = KDE(Img, z, sigma)
% Kernel density esatimation
i = reshape(Img, [], 1); % the intensity value in image
ni = size(i, 1); % the number of pixel
tmp = bsxfun(@minus, z, i);
allDensity = Gaussian(tmp, sigma);
density = sum(allDensity, 1) / ni;
end

function gauValue = Gaussian(x, sigma)
% Gaussian function
gauValue = sqrt(2*pi*sigma^2)^(-1) * exp( - x.^2 / (2*sigma^2) );
end

function g = NeumannBoundCond(f)
% Neumann boundary condition
[nrow, ncol] = size(f);
g = f;
g([1 nrow], [1 ncol]) = g([3 nrow-2], [3 ncol-2]);
g([1 nrow], 2:end-1) = g([3 nrow-2], 2:end-1);
g(2:end-1, [1 ncol]) = g(2:end-1, [3 ncol-2]);
end

function K = Curvature(f)
% compute curvature
[f_fx, f_fy] = forward_gradient(f);
[f_bx, f_by] = backward_gradient(f);

mag1 = sqrt(f_fx.^2 + f_fy.^2 + 1e-10);
n1x = f_fx ./ mag1;
n1y = f_fy ./ mag1;

mag2 = sqrt(f_bx.^2 + f_fy.^2 + 1e-10);
n2x = f_bx ./ mag2;
n2y = f_fy ./ mag2;

mag3 = sqrt(f_fx.^2 + f_by.^2 + 1e-10);
n3x = f_fx ./ mag3;
n3y = f_by ./ mag3;

mag4 = sqrt(f_bx.^2 + f_by.^2 + 1e-10);
n4x = f_bx ./ mag4;
n4y = f_by ./ mag4;

nx = n1x + n2x + n3x + n4x;
ny = n1y + n2y + n3y + n4y;

magn = sqrt(nx.^2 + ny.^2);
nx = nx ./ (magn + 1e-10);
ny = ny ./ (magn + 1e-10);

[nxx, nxy] = gradient(nx);
[nyx, nyy] = gradient(ny);

K = nxx + nyy;
end

function [fdy, fdx] = forward_gradient(f)
% function [fdx,fdy]=forward_gradient(f);
%   created on 04/26/2004
%   author: Chunming Li
%   email: li_chunming@hotmail.com
%   Copyright (c) 2004-2006 by Chunming Li
[nr, nc] = size(f);
fdx = zeros(nr, nc);
fdy = zeros(nr, nc);

a = f(2:nr, :) - f(1:nr - 1, :);
fdx(1:nr - 1, :) = a;
b = f(:, 2:nc) - f(:, 1:nc - 1);
fdy(:, 1:nc - 1) = b;
end

function [bdy, bdx] = backward_gradient(f)
% function [bdx,bdy]=backward_gradient(f);
%   created on 04/26/2004
%   author: Chunming Li
%   email: li_chunming@hotmail.com
%   Copyright (c) 2004-2006 by Chunming Li
[nr, nc] = size(f);
bdx = zeros(nr, nc);
bdy = zeros(nr, nc);

bdx(2:nr ,:)=f(2:nr, :) - f(1:nr - 1, :);
bdy(:, 2:nc)=f(:, 2:nc) - f(:, 1:nc - 1);
end