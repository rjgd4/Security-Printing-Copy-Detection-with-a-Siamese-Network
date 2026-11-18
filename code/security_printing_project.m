function security_printing_project
% SECURITY_PRINTING_PROJECT
% Detection de copies de codes anti-copie avec un reseau siamois.
% Auteurs : Robin Jegard, Gaspard Milek
%
% Ce que fait le script :
%   1. genere les images (p, q = originaux, r = copie) pour train et test
%   2. entraine le reseau siamois sur des paires d'images
%   3. choisit un seuil sur le train puis evalue sur le test (accuracy, ROC)
%
% Pour lancer : taper  security_printing_project  dans la Command Window.
% Toolboxes : Deep Learning, Image Processing, Statistics and Machine Learning.

%% Parametres
close all
rng(1,"twister");               % graine fixe pour retrouver les memes resultats
ratio = 2;                      % 1 point du code = 2x2 pixels
codeSize = [56 46];             % taille du code (en points)
imageSize = ratio * codeSize;   % taille de l'image : 112 x 92
numTrainCodes = 200;
numTestCodes = 100;
numIterations = 200;
miniBatchSize = 10;             % 10 paires par iteration
learningRate = 6e-5;

dotG = 1.0;                     % bavure de l'encre (0.6 = copie trop dure a detecter)
exponent = 1.0;

% true = on regenere les images a chaque lancement
regenerateData = true;

%% 1. Creation des donnees (p, q, r)
if regenerateData || ~isfolder(fullfile(pwd,"training","p"))
    fprintf("Generation des donnees...\n");
    berTrain = makeDataset("training",numTrainCodes,ratio,codeSize,dotG,exponent);
    berTest  = makeDataset("test",numTestCodes,ratio,codeSize,dotG,exponent);
    fprintf("BER de l'estimation du code : %.2f %% (train), %.2f %% (test)\n", ...
        100*berTrain,100*berTest);
end
showExample("training",1);

%% 2. Datastores d'images
% l'ordre compte : les 200 premiers sont les p, puis les q, puis les r
trainFiles = orderedFiles("training",numTrainCodes);
testFiles  = orderedFiles("test",numTestCodes);
imdsTrain = imageDatastore(trainFiles);
imdsTest  = imageDatastore(testFiles);

%% 3. Reseau jumeau (les deux branches partagent les memes poids)
layers = [
    imageInputLayer([imageSize 1],"Name","input","Normalization","none")
    convolution2dLayer(10,64,"Name","conv1", ...
        "WeightsInitializer","narrow-normal","BiasInitializer","narrow-normal")
    reluLayer("Name","relu1")
    maxPooling2dLayer(2,"Stride",2,"Name","pool1")
    convolution2dLayer(7,128,"Name","conv2", ...
        "WeightsInitializer","narrow-normal","BiasInitializer","narrow-normal")
    reluLayer("Name","relu2")
    maxPooling2dLayer(2,"Stride",2,"Name","pool2")
    convolution2dLayer(4,128,"Name","conv3", ...
        "WeightsInitializer","narrow-normal","BiasInitializer","narrow-normal")
    reluLayer("Name","relu3")
    maxPooling2dLayer(2,"Stride",2,"Name","pool3")
    convolution2dLayer(5,256,"Name","conv4", ...
        "WeightsInitializer","narrow-normal","BiasInitializer","narrow-normal")
    reluLayer("Name","relu4")
    fullyConnectedLayer(4096,"Name","embedding", ...
        "WeightsInitializer","narrow-normal","BiasInitializer","narrow-normal")];
dlnet = dlnetwork(layerGraph(layers));

% derniere couche (4096 -> 1 note) : poids et biais a part
fcParams.FcWeights = dlarray(0.01 * randn(1,4096,"single"));
fcParams.FcBias    = dlarray(single(2));   % biais de depart a +2, sinon les paires
                                           % similaires restent bloquees vers 0.5

%% 4. Entrainement
% variables pour l'optimiseur Adam
trailingAvgSubnet = [];
trailingAvgSqSubnet = [];
trailingAvgParams = [];
trailingAvgSqParams = [];
gradDecay = 0.9;
gradDecaySq = 0.999;
useGPU = canUseGPU;
lossHistory = zeros(numIterations,1);

fprintf("Entrainement (%d iterations)...\n",numIterations);
tic
for iteration = 1:numIterations
    % on tire 10 paires au hasard et on normalise entre 0 et 1
    [X1,X2,pairLabels] = getSiameseBatchBis(imdsTrain,miniBatchSize);
    dlX1 = dlarray(single(X1)/255,"SSCB");
    dlX2 = dlarray(single(X2)/255,"SSCB");
    if useGPU
        dlX1 = gpuArray(dlX1);
        dlX2 = gpuArray(dlX2);
    end

    % calcul de la perte et des gradients
    [gradientsSubnet,gradientsParams,loss] = dlfeval( ...
        @modelGradients,dlnet,fcParams,dlX1,dlX2,pairLabels);

    % mise a jour des poids (Adam)
    [dlnet,trailingAvgSubnet,trailingAvgSqSubnet] = adamupdate( ...
        dlnet,gradientsSubnet,trailingAvgSubnet,trailingAvgSqSubnet, ...
        iteration,learningRate,gradDecay,gradDecaySq);
    [fcParams,trailingAvgParams,trailingAvgSqParams] = adamupdate( ...
        fcParams,gradientsParams,trailingAvgParams,trailingAvgSqParams, ...
        iteration,learningRate,gradDecay,gradDecaySq);

    lossHistory(iteration) = double(gather(extractdata(loss)));
    if mod(iteration,50) == 0 || iteration == 1
        fprintf("Iteration %3d / %3d : perte = %.4f\n", ...
            iteration,numIterations,lossHistory(iteration));
    end
end
fprintf("Entrainement termine en %.1f s.\n",toc);

save("security_printing_network.mat","dlnet","fcParams","lossHistory", ...
    "ratio","dotG","exponent");

figure("Name","Perte d'entrainement");
plot(lossHistory,"LineWidth",1.2); grid on
xlabel("Iteration"); ylabel("Entropie croisee binaire");
title("Evolution de la perte pendant l'entrainement");

%% 5. Seuil (calcule sur le train) puis test
% le seuil est choisi sur des paires d'entrainement pour ne pas "tricher"
% avec les donnees de test
calibrationBatchSize = 1000;
[XCal1,XCal2,labelsCalibration] = getSiameseBatchBis(imdsTrain,calibrationBatchSize);
dlXCal1 = dlarray(single(XCal1)/255,"SSCB");
dlXCal2 = dlarray(single(XCal2)/255,"SSCB");
if useGPU
    dlXCal1 = gpuArray(dlXCal1);
    dlXCal2 = gpuArray(dlXCal2);
end
calibrationScores = gather(extractdata(predictSiamese(dlnet,fcParams,dlXCal1,dlXCal2)));

% on garde le seuil qui maximise (taux vrais positifs - taux faux positifs)
[fprCal,tprCal,thresholdsCal] = perfcurve(labelsCalibration,calibrationScores,1);
[~,bestIndex] = max(tprCal-fprCal);
decisionThreshold = thresholdsCal(bestIndex);

% test : 1000 paires tirees au hasard parmi les 100 codes de test
testBatchSize = 1000;
[XTest1,XTest2,labelsTest] = getSiameseBatchBis(imdsTest,testBatchSize);
dlXTest1 = dlarray(single(XTest1)/255,"SSCB");
dlXTest2 = dlarray(single(XTest2)/255,"SSCB");
if useGPU
    dlXTest1 = gpuArray(dlXTest1);
    dlXTest2 = gpuArray(dlXTest2);
end
scores = gather(extractdata(predictSiamese(dlnet,fcParams,dlXTest1,dlXTest2)));
predictions = scores >= decisionThreshold;    % note >= seuil -> "similaire"
accuracy = mean(predictions == labelsTest);
fprintf("Seuil calibre : %.4f | Accuracy de test : %.2f %%\n", ...
    decisionThreshold,100*accuracy);

% histogramme des notes
figure("Name","Scores de similarite");
histogram(scores(labelsTest == 1),"Normalization","probability"); hold on
histogram(scores(labelsTest == 0),"Normalization","probability"); grid on
xlabel("Score predit : probabilite de similarite"); ylabel("Proportion")
legend("p_i / q_i : similaires","p_i ou q_i / r_i : copie","Location","best");
title(sprintf("Test : accuracy = %.2f %%",100*accuracy));

% courbe ROC et AUC
[fpr,tpr,~,auc] = perfcurve(labelsTest,scores,1);
figure("Name","ROC");
plot(fpr,tpr,"LineWidth",1.5); hold on; plot([0 1],[0 1],"k--"); grid on
xlabel("Taux de faux positifs"); ylabel("Taux de vrais positifs")
title(sprintf("Courbe ROC - AUC = %.3f",auc));

save("security_printing_results.mat","scores","labelsTest", ...
    "predictions","accuracy","fpr","tpr","auc","lossHistory", ...
    "decisionThreshold","calibrationScores","labelsCalibration");
fprintf("Fichiers sauvegardes : security_printing_network.mat et security_printing_results.mat\n");
end

%% ========================================================================
%  Fonctions
%  ========================================================================

function berMean = makeDataset(setName,N,ratio,codeSize,dotG,exponent)
% Cree N codes et sauvegarde 3 images par code :
%   p, q : deux impressions/scans du code original x
%   r    : la copie (le faussaire reimprime son estimation de x a partir de p)
for folder = ["p","q","r"]
    target = fullfile(setName,folder);
    if ~isfolder(target), mkdir(target); end
end
ber = zeros(N,1);
for i = 1:N
    x = rand(codeSize) > 0.5;      % code aleatoire, 1 = encre

    % les deux originaux
    p = printScan(255*double(x),ratio,dotG,exponent);
    q = printScan(255*double(x),ratio,dotG,exponent);

    % le faussaire retrouve le code a partir de p :
    % moyenne par bloc 2x2 (enleve du bruit) puis seuil d'Otsu
    blockMean = mean_extract(p,ratio);
    blockNorm = mat2gray(blockMean);
    threshold = graythresh(blockNorm);
    % l'encre est sombre sur la capture, donc encre = "<" seuil
    % (avec ">" la copie etait le negatif de l'original, trop facile a detecter)
    xhat = blockNorm < threshold;
    ber(i) = mean(xhat(:) ~= x(:));     % proportion de points mal retrouves

    % la copie : on reimprime xhat puis on cale luminosite/contraste sur p
    rawCopy = printScan(255*double(xhat),ratio,dotG,exponent);
    copy = matchContrast(rawCopy,p);

    imwrite(uint8(p),   fullfile(setName,"p",sprintf("p_%03d.bmp",i)));
    imwrite(uint8(q),   fullfile(setName,"q",sprintf("q_%03d.bmp",i)));
    imwrite(uint8(copy),fullfile(setName,"r",sprintf("r_%03d.bmp",i)));
end
berMean = mean(ber);
end

function output = matchContrast(input,reference)
% Donne a "input" la meme moyenne et le meme ecart-type que "reference"
inputStd = std(input(:));
output = (input - mean(input(:))) / max(inputStd,eps) * std(reference(:)) ...
    + mean(reference(:));
output = min(max(output,0),255);
end

function files = orderedFiles(setName,N)
% Liste des chemins : p_001..p_N, puis q_001..q_N, puis r_001..r_N
files = strings(3*N,1);
folder = ["p","q","r"];
for group = 1:3
    for i = 1:N
        files((group-1)*N+i) = fullfile(pwd,setName,folder(group), ...
            sprintf("%s_%03d.bmp",folder(group),i));
    end
end
files = cellstr(files);
end

function showExample(setName,index)
% Affiche p, q et r cote a cote pour verifier que tout est coherent
p = imread(fullfile(setName,"p",sprintf("p_%03d.bmp",index)));
q = imread(fullfile(setName,"q",sprintf("q_%03d.bmp",index)));
r = imread(fullfile(setName,"r",sprintf("r_%03d.bmp",index)));
figure("Name","Exemple p, q et r");
imshow([p q r],[]);
title("p : primo-impression | q : seconde primo-impression | r : copie");
end

function [X1,X2,pairLabels] = getSiameseBatchBis(imds,miniBatchSize)
% Tire miniBatchSize paires au hasard.
%   paire impaire : (p_i, q_i)         -> etiquette 1 (similaires)
%   paire paire   : (r_i, p_i ou q_i)  -> etiquette 0 (original vs copie)
N = numel(imds.Files)/3;               % nombre de codes
oneImage = readimage(imds,1);
X1 = zeros([size(oneImage) 1 miniBatchSize],"like",oneImage);
X2 = zeros([size(oneImage) 1 miniBatchSize],"like",oneImage);
pairLabels = zeros(1,miniBatchSize);
for i = 1:miniBatchSize
    imageNumber = randi(N);
    if mod(i,2) == 1
        group1 = 1; group2 = 2; pairLabels(i) = 1;       % p_i / q_i
    else
        group1 = 3; group2 = randi(2); pairLabels(i) = 0; % r_i / p_i ou q_i
    end
    % on echange parfois l'ordre pour que le reseau ne l'apprenne pas
    if rand < 0.5, tmp = group1; group1 = group2; group2 = tmp; end
    X1(:,:,1,i) = readimage(imds,(group1-1)*N + imageNumber);
    X2(:,:,1,i) = readimage(imds,(group2-1)*N + imageNumber);
end
end

function Y = forwardSiamese(dlnet,fcParams,dlX1,dlX2)
% Passage avant (entrainement) : meme reseau sur les deux images,
% difference absolue des empreintes, puis couche finale + sigmoide
F1 = sigmoid(forward(dlnet,dlX1));
F2 = sigmoid(forward(dlnet,dlX2));
Y = fullyconnect(abs(F1-F2),fcParams.FcWeights,fcParams.FcBias);
Y = sigmoid(Y);
end

function [gradientsSubnet,gradientsParams,loss] = modelGradients(dlnet,fcParams,dlX1,dlX2,pairLabels)
% Perte (entropie croisee binaire) et gradients
Y = forwardSiamese(dlnet,fcParams,dlX1,dlX2);
precision = underlyingType(Y);
Y = min(max(Y,eps(precision)),1-eps(precision));   % evite log(0)
loss = -pairLabels.*log(Y) - (1-pairLabels).*log(1-Y);
loss = sum(loss,"all") / numel(pairLabels);
[gradientsSubnet,gradientsParams] = dlgradient(loss,dlnet.Learnables,fcParams);
end

function Y = predictSiamese(dlnet,fcParams,dlX1,dlX2)
% Pareil que forwardSiamese mais avec predict (pour le test)
F1 = sigmoid(predict(dlnet,dlX1));
F2 = sigmoid(predict(dlnet,dlX2));
Y = fullyconnect(abs(F1-F2),fcParams.FcWeights,fcParams.FcBias);
Y = sigmoid(Y);
end

function B = mean_extract(A,eltSize)
% Moyenne de chaque bloc eltSize x eltSize (filtre moyenneur)
nr = floor(size(A,1)/eltSize);
nc = floor(size(A,2)/eltSize);
B = zeros(nr,nc);
for i = 1:nr
    for j = 1:nc
        block = A((i-1)*eltSize+1:i*eltSize,(j-1)*eltSize+1:j*eltSize);
        B(i,j) = mean(block(:));
    end
end
end

function [y,xbis] = printScan(x,eltSize,dotG,p)
% Simule impression + scan :
%   - on agrandit chaque point (eltSize x eltSize pixels)
%   - flou gaussien = l'encre qui bave (dotG regle la bavure)
%   - bruit ajoute avec ggrnd, comme un vrai scanner
xbis = kron(x,ones(eltSize));
kernelSize = 2*ceil(2*dotG)+1;
blurred = imfilter(double(xbis).^p, ...
    fspecial("gaussian",[kernelSize kernelSize],dotG),"same");
y = 255 - blurred.^(1/p);
y = ggrnd(130*(1+tanh(0.01*(y-130))),25*exp(-((y-126)/42).^2),2);
end

function r = ggrnd(m,a,b)
% Bruit gaussien generalise (moyenne m, amplitude a, forme b).
% Un tirage gamma par pixel (sinon le bruit serait le meme partout).
v = 2*rand(size(m))-1;
y = gamrnd((1+1./b)*ones(size(m)),1);
r = m + a.*v.*y.^(1./b);
r(a == 0 | b == 0) = NaN;
end
