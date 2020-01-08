function hu = mu2hu(mu)
% HU (CORRECT)
muw = 0.0192;
hu  = (mu - muw) / muw * 1000;

% HU
% muw = 0.192;
% hu  = (mu - muw) / muw * 1000;