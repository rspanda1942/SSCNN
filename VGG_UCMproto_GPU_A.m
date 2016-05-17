

ucmnet.layers = {
    struct('type', 'input', 'height', 109, 'width', 109, 'outputmaps', 3) %input layer
     
%     struct('type', 'padding', 'padsize', 1) %activition layer   
    struct('type', 'conv', 'outputmaps', 128, 'kernelsize', 3, ...
    'weight', 'gaussian', 'Std', 0.01, 'weight_multiplter', 1.0, ...
    'bias', 'constant', 'value', 0, 'bias_multiplter', 2.0) %convolution layer       
    struct('type', 'pool', 'poolsize', 5, 'stride', 3) %poolling layer           
    struct('type', 'relu') %activition layer      
    struct('type', 'lrn', 'local_size', 5, 'lrn_alpha', 0.0001, 'lrn_beta', 0.75) %activition layer        
 
%     struct('type', 'padding', 'padsize', 1) %activition layer   
    struct('type', 'conv', 'outputmaps', 192, 'kernelsize', 3, ...
    'weight', 'gaussian', 'Std', 0.01, 'weight_multiplter', 1.0, ...
    'bias', 'constant', 'value', 0, 'bias_multiplter', 2.0) %convolution layer  
    struct('type', 'pool', 'poolsize', 3, 'stride', 2) %poolling layer    
    struct('type', 'relu') %activition layer       
    struct('type', 'lrn', 'local_size', 5, 'lrn_alpha', 0.0001, 'lrn_beta', 0.75) %activition layer       
    
%     struct('type', 'padding', 'padsize', 1) %activition layer    
    struct('type', 'conv', 'outputmaps', 192, 'kernelsize', 3, ...
    'weight', 'gaussian', 'Std', 0.01, 'weight_multiplter', 1.0, ...
    'bias', 'constant', 'value', 0, 'bias_multiplter', 2.0) %convolution layer
    struct('type', 'pool', 'poolsize', 2, 'stride', 2) %poolling layer 
    struct('type', 'relu') %activition layer           
    struct('type', 'lrn', 'local_size', 5, 'lrn_alpha', 0.0001, 'lrn_beta', 0.75) %activition layer      
   
    struct('type', 'full', 'outputmaps', 420, ...
    'weight', 'gaussian', 'Std', 0.01, 'weight_multiplter', 1.0, ...
    'bias', 'constant', 'value', 0, 'bias_multiplter', 2.0) %fullconnect layer3
    struct('type', 'relu') %activition layer 
    struct('type', 'dropout', 'fraction', 0.5) %activition layer   
    
    struct('type', 'full', 'outputmaps', 21, ...
    'weight', 'gaussian', 'Std', 0.01, 'weight_multiplter', 1.0, ...
    'bias', 'constant', 'value', 0, 'bias_multiplter', 2.0) %fullconnect layer    
%     struct('type', 'sigmoid') %activition layer 
    struct('type', 'loss','classnum', 21) %softmax_loss layer        
};

ucmnet = cnnsetup(ucmnet);
ucmnet = cnn2GPU(ucmnet);
%% 
% load .\data\UCM_alldata64.mat;
dataPatch = '/home/panda/Ureserch/SSCNN/image_128';

data = zeros(128,128,3,2100);
labels = zeros(2100,1);
for ii= 1:2100
    
    patchT = [dataPatch, '/', num2str(ii), '.jpg'];
    data(:,:,:,ii) = imread(patchT);
    labels(ii) = ceil(ii/100);
end

opts.meanimage = mean(data,4);

[trainS  , textS ]= randsampleMY(labels , 0.8);

         
opts.dataAug = 'crop';
opts.cropsize = 109;

opts.jetting = 'on';   % Size manipulation
opts.jetting_para = 0.20;

opts.colormul = 'on';   % Color manipulation
opts.meansubt = 'on';   % Mean removal

opts.fixweight = 'on';   % fix largest weight
opts.weight_const = 0.1;

opts.base_lr = 0.01;
opts.momentum = 0.9;
opts.weight_decay = 0.0005;
opts.bias_decay = 0;

opts.iter = 1;

opts.display = 100;           % every 10 iterations display the kernel
opts.test_interval = 300;    %Carry out testing every 300 training iterations.

opts.lr_decay = 1.0;  

opts.Maxiter = 10000;

% The learning rate policy
opts.lr_policy = 'inv' ;
opts.gamma = 0.0001;
opts.power = 0.75;

opts.batchsize = 64;   % GPU size   training size
opts.testbatchsize = 2;

%% 

[ucmnet , opts]= cnntrain_GPU_Fin(ucmnet, data(:,:,:,trainS), labels(trainS), data(:,:,:,textS), labels(textS), opts); 

