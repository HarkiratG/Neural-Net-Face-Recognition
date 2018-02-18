%% reset command/variables
clc
clearvars
close all

%% variables
trainingNumFiles = 9;
rng(1)

%% importing images for database & doing fft
faceDatasetPath = fullfile('orl_faces');
faceData = imageDatastore(faceDatasetPath,...
	'IncludeSubfolders',true, 'LabelSource', 'foldernames');

%% showing sample pictures
figure;
% perm = randperm(400,20);
% for i = 1:20
%     subplot(4,5,i);
%     imshow(faceData.Files{perm(i)});
% end

% read one image to get pixel size
img = readimage(faceData,1);

% splitting the testing and training data
[trainFaceData,testFaceData] = splitEachLabel(faceData, ...
	trainingNumFiles,'randomize');

%% defining CNN parameters
% defining layers
layers = [imageInputLayer([size(img,1) size(img,2) 1])
	%middle layers
	convolution2dLayer(5,3,'Padding', 2, 'Stride',3)
	reluLayer
	maxPooling2dLayer(3,'Stride',3)
	%final layers
	fullyConnectedLayer(40)
	softmaxLayer
	classificationLayer()];

% options to train the network
options = trainingOptions('sgdm', ...
	'MiniBatchSize', 40, ...
	'InitialLearnRate', 1e-4, ...
	'MaxEpochs', 25, ...
	'LearnRateSchedule', 'piecewise', ...
	'LearnRateDropFactor', 0.875, ...
	'LearnRateDropPeriod', 12, ...
	'VerboseFrequency', 5);

% training the network
% convnet = load('convnet.mat');
% convnet = convnet.convnet;
convnet = trainNetwork(trainFaceData,layers,options);

%% classifying
YTest = classify(convnet,testFaceData);
TTest = testFaceData.Labels;

%% Calculate the accuracy.
accuracy = sum(YTest == TTest)/numel(TTest)