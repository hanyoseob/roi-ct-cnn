
function [IMG_FOV, FBP_FOV, BPF_FOV, R] = getFOV(param)
%%

%%
DSD                 = param.DSD;
DSO                 = param.DSO;

nDCT                = param.nNumDct;
dDCT                = param.dStepDct;

nY                  = param.nY;
nX                  = param.nX;

dY                  = param.dY;
dX                  = param.dX;

nVIEW               = param.nNumView;
dVIEW               = param.dStepView;

dMaxGamma           = atan2(0.5*dDCT*nDCT, DSD);
dOffset             = 0;

%%
DCT                 = dDCT*nDCT/2;
IMG_X               = dX*nX/2;
IMG_Y               = dY*nY/2;

TRI_R               = sqrt(DCT.^2 + DSD.^2);

DCT_R               = DCT*DSO./TRI_R;
IMG_R               = sqrt(IMG_X.^2 + IMG_Y.^2);

R                   = min(DCT_R, IMG_R);

[mx, my]            = meshgrid(linspace(-IMG_X, IMG_X, nX), linspace(-IMG_Y, IMG_Y, nY));
IMG_FOV             = sqrt(mx.^2 + my.^2) < R;

FBP_FOV             = ones(nDCT, nVIEW, 'single');
BPF_FOV             = ones(nDCT, nVIEW, 'single');

if ~param.bshort
    return;
end

%% BPF SHORT SCAN WEIGHITNG
idx                 = linspace(0, 2*pi, nVIEW + 1);
idx(end)            = [];

idx                 = idx > atan2(R, DSO) & idx < pi - atan2(R, DSO);
BPF_FOV(:, idx)     = 0;

%% FBP SHORT SCAN WEIGHITNG
for iVIEW = 1:nVIEW
    dBeta   = dVIEW*(iVIEW - 1);
    
    for idct    = 1:nDCT
        dPosDct = dDCT*(-(nDCT - 1)/2.0 + (idct - 1)) + dOffset;
        dGAMMA	= atan2(dPosDct, DSD);
        
        if (dBeta >= 0 && dBeta <= 2*(dMaxGamma - dGAMMA))
            PRJ_WGT_                = sin(pi/4*dBeta/(dMaxGamma - dGAMMA))^2;
            FBP_FOV(idct, iVIEW)	= PRJ_WGT_;
        elseif (dBeta >= (pi - 2*dGAMMA) && dBeta <= (pi + 2*dMaxGamma))
            PRJ_WGT_                = sin(pi/4*(pi + 2*dMaxGamma - dBeta)/(dMaxGamma + dGAMMA))^2;
            FBP_FOV(idct, iVIEW)    = PRJ_WGT_;
        elseif (dBeta >= 2*(dMaxGamma - dGAMMA) && dBeta <= (pi - 2*dGAMMA))
            FBP_FOV(idct, iVIEW)    = 1;
        else
            FBP_FOV(idct, iVIEW)    = 0;
        end
    end
end

idx                 = find(BPF_FOV(1, :) == 0);
FBP_FOV             = circshift(flipud(2*FBP_FOV), idx(end), 2);

end