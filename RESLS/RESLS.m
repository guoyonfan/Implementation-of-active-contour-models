function [phi, f] = RESLS(I, phi0, g, deltaT, nu, mu, lambda, w, a, epsilon, alpha, beta)
phi = NeumannBoundCond(phi0);

[Gx, Gy] = gradient(g); 
[Nx, Ny] = normD(phi);

deltaPhi = deltaF(phi, epsilon);
H = heavisideF(phi, epsilon);
f = normIndicator(I, H, alpha, beta, a);

term1 = nu * distRegP2(phi); % the first term in Eq. (50)
term2 = lambda * (Gx .* Nx + Gy .* Ny) .* deltaPhi; % the second term
term3 = (lambda * g + w) .* div(Nx, Ny) .* deltaPhi; % the third term
term4 = mu * f .* deltaPhi;

phi = phi + deltaT * (term1 + term2 + term3 + term4);

end

% compute the normalized indicator
function f = normIndicator(I, H, alpha, beta, a)
[f1, f2] = fittingTerm(H, I);
P = f1 - f2;
signP = signF(P);
zx = approixmatedNorm(alpha * f1 + beta * f2 - I, a);
f = signP .* zx;
end

% the the sign function
function signP = signF(P)
if P < 0
    signP = -1;
elseif P == 0
    signP = 0;
else
    signP = 1;
end
end

% compute the fitting terms f1, f2
function [f1, f2] = fittingTerm(H, I)
a1 = (1 - H) .* I;
numer1 = sum( a1(:) );
c = 1 - H;
denom1 = sum( c(:) );
f1 = numer1 / denom1;

a2 = H .* I;
numer2 = sum( a2(:) );
denom2 = sum( H(:) );
f2 = numer2 / denom2;
end

% compute the distance regularization term with the double-well potential p2 in eqaution (16)
function f = distRegP2(phi)
[phi_x, phi_y]=gradient(phi);
s=sqrt(phi_x.^2 + phi_y.^2);
a=(s>=0) & (s<=1);
b=(s>1);
ps=a.*sin(2*pi*s) / (2*pi) + b.*(s-1);  % compute first order derivative of the double-well potential p2 in eqaution (16)
dps=((ps~=0) .* ps + (ps==0)) ./ ((s~=0) .* s + (s==0));  % compute d_p(s)=p'(s)/s in equation (10). As s-->0, we have d_p(s)-->1 according to equation (18)
f = div(dps .* phi_x - phi_x, dps .* phi_y - phi_y) + 4*del2(phi);
end

% Make a function satisfy Neumann boundary condition
function g = NeumannBoundCond(f)
[nrow, ncol] = size(f);
g = f;
g([1 nrow], [1 ncol]) = g([3 nrow-2], [3 ncol-2]);
g([1 nrow], 2:end-1) = g([3 nrow-2], 2:end-1);
g(2:end-1, [1 ncol]) = g(2:end-1, [3 ncol-2]);
end

% the approximated normalized term
function ZX = approixmatedNorm(X, a)
ZX = 2 ./ (1 + exp(-a * X)) - 1;
end

% compute the normalized gradient
function [Nx, Ny] = normD(u)
[ux, uy] = gradient(u);
normDu = sqrt(ux.^2 + uy.^2 + 1e-10);

Nx = ux ./ normDu;
Ny = uy ./ normDu;
end

% compute the div
function f = div(nx, ny)
[nxx, ~]=gradient(nx);
[~, nyy]=gradient(ny);
f=nxx+nyy;
end

% the smoothed delta function
function deltaPhi = deltaF(phi, epsilon)
deltaPhi = (epsilon / pi) ./ (epsilon^2 + phi.^2);
end

% the smoothed Heaviside function
function H = heavisideF(phi, epsilon)
H = 0.5 * (1 + (2/pi) *atan(phi ./ epsilon));
end