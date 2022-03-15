function [phi,b] = LATE_Evolution(phi0,b0,KOne,Ksigma,Img,timestep,nu,mu,epsilon, innerIter)
% phi0 : the initial level set function
% b0 : the initial variation degree of intensity inhomogeneity b'
% Ksigma : the Gaussian kernel with standard deviation sigma
% nu : the parameter that controls the regularization term
% mu : the parameter that controls the length term
phi=phi0;
b=b0;

maxB=max(max(abs(b)));
scaleRate = 1 / maxB;
b = b * scaleRate; % normalize the max abs(b) to be 1

Hphi = Heaviside(phi,epsilon);

[LIM1,LIM2] = LocalIntensityMean(Img,Hphi,Ksigma);
[C1,C2] = AdjustingTermC(Ksigma,Img,LIM1,LIM2,Hphi,b);
b = AdjustingTermB(Img,LIM1,LIM2,C1,C2,Hphi,Ksigma);
A1 = KOne.*(Img.^2+LIM1.^2-2*Img.*LIM1);
A2 = (C1*b).^2-2*C1*Img.*b+2*C1*LIM1.*b;
A2 = conv2(A2,Ksigma,'same');
A = A1+A2; % the first term in equation (29)

B1 = KOne.*(Img.^2+LIM2.^2-2*Img.*LIM2);
B2 = (C2*b).^2-2*C2*Img.*b+2*C2*LIM2.*b;
B2 = conv2(B2,Ksigma,'same');
B = B1+B2; % the second term in equation (29)

for i = 1:innerIter
    phi = NeumannBoundCond(phi);
    K = Curvature(phi);
    DiracPhi = Delta(phi,epsilon);
    RegTerm = (4*del2(phi)-K);
    phi = phi+timestep*(-DiracPhi.*(A-B)+mu*DiracPhi.*K+nu*RegTerm);
end
end


function b = AdjustingTermB(Img,LIM1,LIM2,C1,C2,Hphi,Ksigma)
% calculate the adjusting term b'
bNumer=C1*(Img-LIM1).*Hphi+C2*(Img-LIM2).*(1-Hphi);
bdenom=C1^2*Hphi+C2^2*(1-Hphi);

numer=conv2(bNumer,Ksigma,'same');
denom=conv2(bdenom,Ksigma,'same');

b=numer./denom;
end

function [C1, C2] = AdjustingTermC(Ksigma,Img,LIM1,LIM2,Hphi,b)
% calculate the adjusting term C1 and C2
a1=(Img-LIM1).*(Hphi);
numer1=conv2(a1,Ksigma,'same');
denom1=b.*conv2(Hphi,Ksigma,'same');
C1=sum(numer1(:))/sum(denom1(:));

a2=(Img-LIM2).*(1-Hphi);
numer2=conv2(a2,Ksigma,'same');
denom2=b.*conv2((1-Hphi),Ksigma,'same');
C2=sum(numer2(:))/sum(denom2(:));
end

function [LIM1,LIM2] = LocalIntensityMean(Img,Hphi,Ksigma)
% calculate the local intensity mean (LIM)
lim1=conv2(Img.*Hphi,Ksigma,'same');
area1=conv2(Hphi,Ksigma,'same');
LIM1=lim1./area1;

lim2=conv2(Img.*(1-Hphi),Ksigma,'same');
area2=conv2(1-Hphi,Ksigma,'same');
LIM2=lim2./area2;
end

function g=NeumannBoundCond(f)
% Neumann boundary condition
[nrow,ncol]=size(f);
g=f;
g([1 nrow],[1 ncol]) = g([3 nrow-2],[3 ncol-2]);
g([1 nrow],2:end-1) = g([3 nrow-2],2:end-1);
g(2:end-1,[1 ncol]) = g(2:end-1,[3 ncol-2]);
end

function k = Curvature(u)
% compute curvature
[ux,uy] = gradient(u);
normDu = sqrt(ux.^2+uy.^2+1e-10);

Nx = ux./normDu;
Ny = uy./normDu;
[nxx,~] = gradient(Nx);
[~,nyy] = gradient(Ny);
k = nxx+nyy;  % compute diveragence
end

function Hphi = Heaviside(phi,epsilon)
Hphi = 0.5*(1+(2/pi)*atan(phi./epsilon));
end

function Delta_h = Delta(phi,epsilon)
Delta_h = (epsilon/pi)./(epsilon^2+phi.^2);
end