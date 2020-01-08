function net = cnn_image_i2i_unet_init(varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NETWORK	: CNP
% PAPER     : Convolutional Neural Pyramid for Image Processing
%               (https://arxiv.org/pdf/1704.02071.pdf)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opts.meanNorm           = true ;
opts.varNorm            = true ;
opts.networkType        = 'dagnn' ;

opts.method             = 'image';

opts.param              = [];

[opts, ~]               = vl_argparse(opts, varargin) ;

opts.imageRange         = [0, 1];
opts.imageSize          = [256, 256, 1];

opts.inputSize          = [40, 40, 1];
opts.windowSize         = [80, 80, 1];
opts.wrapSize           = [0, 0, 1];

opts.wgt                = 1;

opts.numEpochs          = 1e2;
opts.batchSize          = 16;
opts.numSubBatches      = 1;
opts.batchSample        = 1;

opts.lrnrate            = [-3, -5];
opts.wgtdecay           = 1e-4;

opts.solver             = [];
[opts, ~]               = vl_argparse(opts, varargin) ;

%%
ch      = opts.inputSize(3);

flt0    = [3, 3, ch, 64];
flt1	= [3, 3, 64, 128];
flt2	= [3, 3, 128, 256];
flt3	= [3, 3, 256, 512];
flte    = [1, 1, 64, ch];


%%
opts.bBias      = true;
opts.bBnorm     = true;
opts.bReLU      = true;
opts.nConnect   = 2;        % +1 : sum, +2 : concat

net             = dagnn.DagNN();

%%
opts.input      = 'input_img';
opts.numStage   = 4;
opts.scope      = [];
opts.poolSize   = [2, 2];

%% [STAGE 0] CONTRACT & EXTRACT PATH
nstg                = 0;
[net, lastLayer]	= add_block_multi_img(net, nstg, flt0, opts);

%% [STAGE 1] CONTRACT & EXTRACT PATH
nstg                = 1;
net                 = add_block_multi_img(net, nstg, flt1, opts);

%% [STAGE 2] CONTRACT & EXTRACT PATH
nstg                = 2;
net                 = add_block_multi_img(net, nstg, flt2, opts);

%% [STAGE 3] CONTRACT & EXTRACT PATH
nstg                = 3;
net                 = add_block_multi_img(net, nstg, flt3, opts);

%% FULLY CONNECTED PATH
cfy         = flte(1);
cfx         = flte(2);
cfz         = flte(3);
cfd         = flte(4);

hcfy        = floor((cfy - 1)/2);
hcfx        = floor((cfx - 1)/2);

hcf         = [hcfy, hcfy, hcfx, hcfx]; % [TOP BOTTOM LEFT RIGHT]

l_fc_conv 	= dagnn.Conv('size', flte, 'pad', hcf, 'stride', 1, 'hasBias', true);
net.addLayer('l_fc_conv', 	l_fc_conv, {lastLayer}, {'regr_img'}, {'fc_cf', 'fc_cb'});

% l_fc_conv 	= dagnn.Conv('size', flte, 'pad', hcf, 'stride', 1, 'hasBias', true);
% net.addLayer('l_fc_conv', 	l_fc_conv, {lastLayer}, {'reg_img'}, {'fc_cf', 'fc_cb'});
% 
% l_sum   	= dagnn.Sum();
% net.addLayer('l_sum', l_sum, {'input_img', 'reg_img'}, {'regr_img'});

%%
l_loss_img          = dagnn.EuclideanLoss('p', 2);
l_mse_img         	= dagnn.Error('loss', 'mse'); 

net.addLayer('loss_img',    l_loss_img,	{'regr_img', 'label_img'},	{'objective'});
net.addLayer('mse_img',     l_mse_img,  {'regr_img', 'label_img'},  {'mse_img'});

%% YOU HAVE TO RUN THE FUNCTION for INITIALIZATION OF PARAMETERS
% load order.mat
% for i = 1:length(order)
%     net.vars(order(i) + 1).precious = true; 
%     disp([net.vars(order(i) + 1).name ',        ' num2str(size(net.vars(order(i) + 1).value))]);
% end;

net.initParams();

%% Meta parameters
net.meta.inputSize                  = opts.inputSize ;

net.meta.trainOpts.method           = opts.method;

if length(opts.lrnrate) == 2
    net.meta.trainOpts.learningRate     = logspace(opts.lrnrate(1), opts.lrnrate(2), opts.numEpochs) ;
else
    net.meta.trainOpts.learningRate     = opts.lrnrate;
end

net.meta.trainOpts.errorFunction	= 'euclidean';

net.meta.trainOpts.numEpochs        = opts.numEpochs ;
net.meta.trainOpts.batchSize        = opts.batchSize ;
net.meta.trainOpts.numSubBatches    = opts.numSubBatches ;
net.meta.trainOpts.batchSample      = opts.batchSample ;

net.meta.trainOpts.weightDecay      = opts.wgtdecay;
net.meta.trainOpts.momentum         = 9e-1;

net.meta.trainOpts.imageRange    	= opts.imageRange;

net.meta.trainOpts.solver           = opts.solver ;
net.meta.trainOpts.param            = opts.param ;

%%
for l = 1:numel(net.layers)
  if isa(net.layers(l).block, 'dagnn.WaveDec') || isa(net.layers(l).block, 'dagnn.WaveRec')
    k = net.getParamIndex(net.layers(l).params{1}) ;
    net.params(k).learningRate = 0 ;
  end
end

