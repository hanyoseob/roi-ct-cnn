clear ;
gpus        = 1;

%%
reset(gpuDevice(gpus));

restoredefaultpath();

run(fullfile(fileparts(mfilename('fullpath')),...
    '..', '..', 'matlab', 'vl_setupnn.m')) ;

%% 
dataDir         = './data/';
netDir          = './network/';

sz              = [512, 512];

%% Hyper parameter to train convolutional neural network
network_img     = 'cnn_image_i2i_unet_init';

networkType     = 'dagnn';

method          = 'image';

solver_handle	= [];

imageRange      = [0, 100];
imageSize       = [sz(1), sz(2), 1];
inputSize       = [sz(1), sz(2), 1];

wgt             = 3e3;

numEpochs       = 300;

batchSize       = 4;
subbatchSize    = 4;
numSubBatches   = ceil(batchSize/subbatchSize);
batchSample     = 1;

lrnrate         = logspace(-4, -5, numEpochs);
wgtdecay        = 1e-4;

meanNorm        = false;
varNorm         = false;

train           = struct('gpus', gpus);

smp             = 2;

%% IMAGE
type                    = 'fbp';
expDir_type1_a1         = [netDir network_img '_' type '_type1_a1'];

type                    = 'fbp';
expDir_type1_a2         = [netDir network_img '_' type '_type1_a2'];

type                    = 'dbp';
expDir_type2_b1         = [netDir network_img '_' type '_type2_b1'];

type                    = 'dbp';
expDir_type2_b2         = [netDir network_img '_' type '_type2_b2'];

%% Load networks and stats
modelPath                           = @(epdir, ep) fullfile(epdir, sprintf('net-epoch-%d.mat', ep));

epoch_type1_a1                   	= findLastCheckpoint(expDir_type1_a1);
[net_type1_a1, ~, stats_type1_a1]  	= loadState(modelPath(expDir_type1_a1, epoch_type1_a1));

epoch_type1_a2                     	= findLastCheckpoint(expDir_type1_a2);
[net_type1_a2, ~, stats_type1_a2] 	= loadState(modelPath(expDir_type1_a2, epoch_type1_a2));

epoch_type2_b1                  	= findLastCheckpoint(expDir_type2_b1);
[net_type2_b1, ~, stats_type2_b1]  	= loadState(modelPath(expDir_type2_b1, epoch_type2_b1));

epoch_type2_b2                    	= findLastCheckpoint(expDir_type2_b2);
[net_type2_b2, ~, stats_type2_b2] 	= loadState(modelPath(expDir_type2_b2, epoch_type2_b2));

%% Delete Loss parts
net_type1_a1.removeLayer('mse_img');
net_type1_a1.removeLayer('loss_img');

net_type1_a2.removeLayer('mse_img');
net_type1_a2.removeLayer('loss_img');

net_type2_b1.removeLayer('mse_img');
net_type2_b1.removeLayer('loss_img');

net_type2_b2.removeLayer('mse_img');
net_type2_b2.removeLayer('loss_img');

%% Conserve the result memory
vid_type1_a1   	= net_type1_a1.getVarIndex('regr_img') ;
net_type1_a1.vars(vid_type1_a1).precious	= true ;

vid_type1_a2 	= net_type1_a2.getVarIndex('regr_img') ;
net_type1_a2.vars(vid_type1_a2).precious  	= true ;

vid_type2_b1  	= net_type2_b1.getVarIndex('regr_img') ;
net_type2_b1.vars(vid_type2_b1).precious	= true ;

vid_type2_b2 	= net_type2_b2.getVarIndex('regr_img') ;
net_type2_b2.vars(vid_type2_b2).precious  	= true ;

%% Load cost valuse along the epoch
epoch_num	= min([epoch_type1_a1, epoch_type1_a2, epoch_type2_b1, epoch_type2_b2]);

for i = 1:epoch_num
    obj_train_type1_a1(i)  	= stats_type1_a1.train(i).objective;
    obj_train_type1_a2(i) 	= stats_type1_a2.train(i).objective;
    obj_train_type2_b1(i)  	= stats_type2_b1.train(i).objective;
    obj_train_type2_b2(i)  	= stats_type2_b2.train(i).objective;
    
    obj_val_type1_a1(i)     = stats_type1_a1.val(i).objective;
    obj_val_type1_a2(i)   	= stats_type1_a2.val(i).objective;
    obj_val_type2_b1(i)     = stats_type2_b1.val(i).objective;
    obj_val_type2_b2(i)     = stats_type2_b2.val(i).objective;
end

%% Reproduce the figure 5: Convergence plots for the objective function
figure('name', '[Fig5] Convergence plots for the objective function','NumberTitle', 'off');
set(gcf,'PaperPositionMode','auto');

plot(obj_train_type1_a1, 'color', [255, 192, 0]./255,'linestyle', '--', 'linewidth', 4);   hold on;
plot(obj_train_type1_a2, 'color', [0, 176, 80]./255, 'linestyle', '--', 'linewidth', 4);
plot(obj_train_type2_b1, 'color', [0, 0, 255]./255,  'linestyle', '--', 'linewidth', 4);
plot(obj_train_type2_b2, 'color', [255, 0, 0]./255,	'linestyle', '--', 'linewidth', 4);

plot(obj_val_type1_a1,   'color', [255, 192, 0]./255,'linestyle', '-', 'linewidth', 4);
plot(obj_val_type1_a2,   'color', [0, 176, 80]./255, 'linestyle', '-', 'linewidth', 4);
plot(obj_val_type2_b1,	 'color', [0, 0, 255]./255,  'linestyle', '-', 'linewidth', 4);
plot(obj_val_type2_b2,   'color', [255, 0, 0]./255,	'linestyle', '-', 'linewidth', 4);   hold off;

legend('[Train] Type I  : Fig.3 (a-1)', ...
    '[Train] Type I  : Fig.3 (a-2)', ...
    '[Train] Type II : Fig.3 (b-1)', ...
    '[Train] Type II : Fig.3 (b-2)', ...
    '[Valid] Type I  : Fig.3 (a-1)', ...
    '[Valid] Type I  : Fig.3 (a-2)', ...
    '[Valid] Type II : Fig.3 (b-1)', ...
    '[Valid] Type II : Fig.3 (b-2)', ...
    'location', 'NorthEast');

ylabel('Objective', 'FontSize', 20, 'FontWeight', 'bold');
xlabel('The number of epochs', 'FontSize', 20, 'FontWeight', 'bold');
ylim([0, 20]);
xlim([1, epoch_num]);
grid on;
grid minor;

ax  = gca;
ax.FontSize     = 20;
ax.FontWeight 	= 'bold';
ax.FontName     = 'Adobe';

% return ;

%% NETWORK MODE : TEST
mode                = 'test';         % 'test' / 'normal'

net_type1_a1.mode   = mode;
net_type1_a2.mode 	= mode;
net_type2_b1.mode   = mode;
net_type2_b2.mode   = mode;

%% Set up the system parameters
load([dataDir 'param_TRUNC.mat']);
load([dataDir 'param_FULL.mat']);

param_FULL.bshort               = false;
param_TRUNC.bshort              = false;

param_FULL_forward              = param_FULL;
param_FULL_backward             = param_FULL;

param_TRUNC_forward             = param_TRUNC;
param_TRUNC_backward            = param_TRUNC;

param_FULL_backward.dX          = 1.0;
param_FULL_backward.dY          = 1.0;
param_FULL_backward.nX          = 512;
param_FULL_backward.nY          = 512;

param_TRUNC_backward.dX         = 1.0;
param_TRUNC_backward.dY         = 1.0;
param_TRUNC_backward.nX         = 512;
param_TRUNC_backward.nY         = 512;

param_TRUNC_forward.nNumView    = 1200;
param_TRUNC_forward.dStepView  	= 2*pi/param_TRUNC_forward.nNumView;

param_TRUNC_backward.nNumView   = 1200;
param_TRUNC_backward.dStepView	= 2*pi/param_TRUNC_backward.nNumView;

%% Several detector size
dct_set = [240, 300, 600, 800, 1440];

for id = 1:length(dct_set)
    idct = dct_set(id);
    
    param_TRUNC_forward.dStepDct	= 1;
    param_TRUNC_forward.nNumDct     = idct;
    
    param_TRUNC_backward.dStepDct	= 1;
    param_TRUNC_backward.nNumDct	= idct;
    
    %%
    load(num2str(idct, [dataDir 'data_dct%d.mat']))
    
    %% Make FOV
    FOV             = getFOV(param_TRUNC_backward);
    
    %%
    bnd          	= 1:param_TRUNC_backward.nY;
    nbnd          	= length(bnd);
    
    %% Set up the inferece parametor of the networks
    opts.gpus       = gpus;
    
    opts.wgt        = wgt;
    opts.offset     = 0;
    
    opts.set        = 1;
    
    opts.meanNorm	= meanNorm;
    opts.varNorm	= varNorm;
    opts.batchSize  = 1;
    
    opts.imageSize  = [nbnd, nbnd, 1];
    opts.inputSize  = [nbnd, nbnd, 1];
    opts.kernalSize	= [0, 0, 1];
    
    opts.input      = 'input_img';
    
    opts.size       = [opts.imageSize, length(opts.set)];
    
    %% Inference the networks
    opts.vid        = vid_type1_a1;
    rec_type1_a1  	= recon_cnn4img_i2i(net_type1_a1, bsxfun(@times, data_fbp, FOV), opts);
    
    opts.vid        = vid_type1_a2;
    rec_type1_a2    = recon_cnn4img_i2i(net_type1_a2, bsxfun(@times, data_fbp, FOV), opts);
    
    opts.vid        = vid_type2_b1;
    rec_type2_b1    = recon_cnn4img_i2i(net_type2_b1, bsxfun(@times, data_dbp, FOV), opts);
    
    opts.vid        = vid_type2_b2;
    rec_type2_b2    = recon_cnn4img_i2i(net_type2_b2, bsxfun(@times, data_dbp, FOV), opts);
    
    %%
    idy            	= find(sum(FOV, 2));
    bnd             = idy(1):idy(end);
    nbnd            = length(bnd);
    
    %% Crop the FOV region
    labels_fov      	= bsxfun(@times, labels, FOV);
    data_fbp_fov        = bsxfun(@times, data_fbp, FOV);
    
    rec_type1_a1_fov    = bsxfun(@times, rec_type1_a1, FOV);
    rec_type1_a2_fov    = bsxfun(@times, rec_type1_a2, FOV);
    
    rec_type2_b1_fov    = bsxfun(@times, rec_type2_b1, FOV);
    rec_type2_b2_fov    = bsxfun(@times, rec_type2_b2, FOV);
    
    labels_fov          = labels_fov(bnd, bnd);
    data_fbp_fov        = data_fbp_fov(bnd, bnd);
    
    rec_type1_a1_fov    = rec_type1_a1_fov(bnd, bnd);
    rec_type1_a2_fov    = rec_type1_a2_fov(bnd, bnd);
    
    rec_type2_b1_fov    = rec_type2_b1_fov(bnd, bnd);
    rec_type2_b2_fov    = rec_type2_b2_fov(bnd, bnd);

    %% Calculate the quantitative values such as nmse, psnr, and ssim
    nmse_fbp       	= [];
    nmse_type1_a1   = [];
    nmse_type1_a2   = [];
    nmse_type2_b1   = [];
    nmse_type2_b2   = [];
    
    wnd_img         = [0, 5e-1];
    wnd_hu          = [-150, 300];
    
    %%
    label_      	= labels_fov;
    norval          = max(label_(:));
    
    label_          = max(labels_fov, 0)./norval;
    data_fbp_       = max(data_fbp_fov, 0)./norval;
    
    rec_type1_a1_   = max(rec_type1_a1_fov, 0)./norval;
    rec_type1_a2_   = max(rec_type1_a2_fov, 0)./norval;
    rec_type2_b1_   = max(rec_type2_b1_fov, 0)./norval;
    rec_type2_b2_   = max(rec_type2_b2_fov, 0)./norval;
    
    %%
    nmse_fbp        = nmse(data_fbp_, label_);
    nmse_type1_a1   = nmse(rec_type1_a1_, label_);
    nmse_type1_a2   = nmse(rec_type1_a2_, label_);
    nmse_type2_b1   = nmse(rec_type2_b1_, label_);
    nmse_type2_b2   = nmse(rec_type2_b2_, label_);
    
    %%
    label_          = max(labels_fov, 0);
    data_fbp_       = max(data_fbp_fov, 0);
    
    rec_type1_a1_   = max(rec_type1_a1_fov, 0);
    rec_type1_a2_   = max(rec_type1_a2_fov, 0);
    rec_type2_b1_   = max(rec_type2_b1_fov, 0);
    rec_type2_b2_   = max(rec_type2_b2_fov, 0);
    
    %%
    label_hu_           = mu2hu(label_);
    data_fbp_hu_        = mu2hu(data_fbp_);
    
    rec_type1_a1_hu_	= mu2hu(rec_type1_a1_);
    rec_type1_a2_hu_    = mu2hu(rec_type1_a2_);
    rec_type2_b1_hu_    = mu2hu(rec_type2_b1_);
    rec_type2_b2_hu_    = mu2hu(rec_type2_b2_);
    
    %%
    figure('name', num2str(idct, '[Fig7] Number of detector: %d'),'NumberTitle', 'off');
    colormap gray;
    
    %%
    switch idct
        case {240, 600, 1440}
            switch idct
                case 240
                    iy = 68;
                case 600
                    iy = 130;
                case 1440
                    iy = 245;
            end
            
            label_hu_(iy, :)        = wnd_hu(end);
            data_fbp_hu_(iy, :)     = 4*wnd_hu(end);
            
            rec_type1_a1_hu_(iy, :) = wnd_hu(end);
            rec_type1_a2_hu_(iy, :)	= wnd_hu(end);
            rec_type2_b1_hu_(iy, :) = wnd_hu(end);
            rec_type2_b2_hu_(iy, :) = wnd_hu(end);
            
            label__                 = label_(iy, :);
            data_fbp__              = data_fbp_(iy, :);
            rec_fbp_img_full__      = rec_type1_a2_(iy, :);
            rec_dbp_img_full__      = rec_type2_b2_(iy, :);
            
        case {300, 800}
            if idct == 300
                ix  = 100;
            else
                ix  = 219;
            end
                
            label_hu_(:, ix)      	= wnd_hu(end);
            data_fbp_hu_(:, ix)   	= 4*wnd_hu(end);
            
            rec_type1_a1_hu_(:, ix) = wnd_hu(end);
            rec_type1_a2_hu_(:, ix) = wnd_hu(end);
            rec_type2_b1_hu_(:, ix) = wnd_hu(end);
            rec_type2_b2_hu_(:, ix) = wnd_hu(end);
            
            label__                 = label_(:, ix);
            data_fbp__              = data_fbp_(:, ix);
            rec_fbp_img_full__      = rec_type1_a2_(:, ix);
            rec_dbp_img_full__      = rec_type2_b2_(:, ix);
    end
    
    subplot(2, 3, [3, 6]);
    set(gcf,'PaperPositionMode','auto');
    plot(label__,               'color', [0, 0, 0]./255,        'linestyle', '--', 'linewidth', 2);   hold on;
    plot(data_fbp__,            'color', [192, 192, 192]./255,  'linestyle', '-', 'linewidth', 2);	
    plot(rec_fbp_img_full__,    'color', [0, 176, 80]./255,     'linestyle', '-', 'linewidth', 2);
    plot(rec_dbp_img_full__,	'color', [255, 0, 0]./255,      'linestyle', '-', 'linewidth', 2);  hold off;
    
    legend('(i) Ground Truth', '(ii) FBP', '(iii) Type I', '(iv) Type II');
    
    grid minor;
    grid on;
    ylim([0, 0.05]);
    xlim([1, nbnd]);
    
    ax              = gca;
    ax.FontWeight 	= 'bold';
    ax.FontName     = 'Adobe';
    
    %%
    if (idct == 240 || idct == 300) sc = 4;
    else sc = 1;
    end
    
    subplot(2, 3, 1); imagesc(label_hu_, wnd_hu); axis off image; title({'(i) Ground Truth'});
    subplot(2, 3, 2); imagesc(data_fbp_hu_, sc*wnd_hu); axis off image; title({'(ii) FBP', num2str(nmse_fbp, 'NMSE: %.4e')});
    subplot(2, 3, 4); imagesc(rec_type1_a2_hu_, wnd_hu); axis off image; title({'(iii) Type I', num2str(nmse_type1_a2, 'NMSE: %.4e')});
    subplot(2, 3, 5); imagesc(rec_type2_b2_hu_, wnd_hu); axis off image; title({'(iv) Type II', num2str(nmse_type2_b2, 'NMSE: %.4e')});
    
    drawnow();
    
    %%
    disp(['[# of detectors] ' num2str(idct)]);
    disp(['    FBP: ' num2str(nmse_fbp, '%.4e')]);
    disp([' Type I: ' num2str(nmse_type1_a2, '%.4e')]);
    disp(['Type II: ' num2str(nmse_type2_b2, '%.4e')]);
    
    disp(' ');
    
end