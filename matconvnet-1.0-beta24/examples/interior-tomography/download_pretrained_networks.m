%% Download the trained network
% Type I: Fig.3 (a-1)
network_path    = './network/cnn_image_i2i_unet_init_fbp_type1_a1/';
network_name	= [network_path 'net-epoch-300.mat'];
network_url     = 'https://www.dropbox.com/s/4v0bfak27dhyh5j/net-epoch-300.mat?dl=1';

mkdir(network_path);
fprintf('downloading the pretrained network of Type I: Fig.3 (a-1) from %s\n', network_url) ;
websave(network_name, network_url);

% Type I: Fig.3 (a-2)
network_path    = './network/cnn_image_i2i_unet_init_fbp_type1_a2/';
network_name	= [network_path 'net-epoch-150.mat'];
network_url     = 'https://www.dropbox.com/s/3ru8qb6bk5hm03j/net-epoch-150.mat?dl=1';

mkdir(network_path);
fprintf('downloading the pretrained network of Type I: Fig.3 (a-2) from %s\n', network_url) ;
websave(network_name, network_url);

% Type II: Fig.3 (b-1)
network_path    = './network/cnn_image_i2i_unet_init_dbp_type2_b1/';
network_name	= [network_path 'net-epoch-300.mat'];
network_url     = 'https://www.dropbox.com/s/cn06wzy7h0xtnff/net-epoch-300.mat?dl=1';

mkdir(network_path);
fprintf('downloading the pretrained network of Type II: Fig.3 (b-1) from %s\n', network_url) ;
websave(network_name, network_url);

% Type II: Fig.3 (b-2)
network_path    = './network/cnn_image_i2i_unet_init_dbp_type2_b2/';
network_name	= [network_path 'net-epoch-150.mat'];
network_url     = 'https://www.dropbox.com/s/zbep1xum5mb8kc5/net-epoch-150.mat?dl=1';

mkdir(network_path);
fprintf('downloading the pretrained network of Type II: Fig.3 (b-2) from %s\n', network_url) ;
websave(network_name, network_url);
