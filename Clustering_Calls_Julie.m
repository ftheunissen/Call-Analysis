%% Post processing description of calls

% Load the data.
% load('/Users/frederictheunissen/Documents/Data/Julie/FullVocalizationBank/vocCutsAnal.mat');
% This one includes the PSD, the temporal enveloppe and has the correct
% calibration for sound level
load('/Users/frederictheunissen/Documents/Data/Julie/FullVocalizationBank/vocCutsAnalwPSD.mat');

% the file with _test has the fix for the saliency but does not include power and rms - it is called _test
% just in case it had problem.

% Clean up the data
ind = find(strcmp({callAnalData.type},'C-'));   % This corresponds to unknown-11
callAnalData(ind) = [];     % Delete this bird because calls are going to be all mixed
ind = find(strcmp({callAnalData.type},'WC'));   % These are copulation whines...
callAnalData(ind) = [];
ind = find(strcmp({callAnalData.type},'-A'));
for i=1:length(ind)
   callAnalData(ind(i)).type = 'Ag';
end

ind = find(strcmp({callAnalData.bird}, 'HpiHpi4748'));
for i=1:length(ind)
   callAnalData(ind(i)).bird = 'HPiHPi4748';
end

% Read the Bird info file
fid = fopen('/Users/frederictheunissen/Documents/Data/Julie/FullVocalizationBank/Birds_List_Acoustic.txt', 'r');
birdInfo = textscan(fid, '%s %s %s %s %s %d');
nInfo = length(birdInfo{1});
fclose(fid);

% Check to see if we have info for all the birds
birdNames = unique({callAnalData.bird});
nBirds = length(birdNames);

birdInfoInd = zeros(1, nBirds);
for ibird=1:nBirds
    for iinfo=1:nInfo
        if (strcmp(birdInfo{1}(iinfo), birdNames{ibird}) )
            birdInfoInd(ibird) = iinfo;
            break;
        end
    end
    
    ind = find(strcmp({callAnalData.bird}, birdNames{ibird}));
    for i=1:length(ind)
        if birdInfoInd(ibird) ~= 0
            callAnalData(ind(i)).birdSex = birdInfo{2}{birdInfoInd(ibird)};
            callAnalData(ind(i)).birdAge = birdInfo{3}{birdInfoInd(ibird)};
        else
            callAnalData(ind(i)).birdSex = 'U';
            callAnalData(ind(i)).birdAge = 'U';           
        end
            
    end
end

notFoundInd = find(birdInfoInd == 0 );
for i=1:length(notFoundInd)
    fprintf(1, 'Warning no information for bird %s\n', birdNames{notFoundInd(i)});
end

nameGrp = unique({callAnalData.type},'stable');   % Names in the order found in original data set
ngroups = length(nameGrp);
indSong = find(strcmp(nameGrp, 'So'));

indSex = find(strcmp({callAnalData.birdSex}, 'M') | strcmp({callAnalData.birdSex}, 'F')); 
indAge = find(strcmp({callAnalData.birdAge}, 'A') | strcmp({callAnalData.birdAge}, 'C'));
indSexNoSo = find((strcmp({callAnalData.birdSex}, 'M') | strcmp({callAnalData.birdSex}, 'F')) & ~(strcmp({callAnalData.type}, 'So')));

name_grp_plot = {'Be', 'LT', 'Tu', 'Th', 'Di', 'Ag', 'Wh', 'Ne', 'Te', 'DC', 'So'};
colorVals = [ [0 230 255]; [0 95 255]; [255 200 65]; [255 150 40]; [255 105 15];...
    [255 0 0]; [255 180 255]; [255 100 255]; [140 100 185]; [100 50 200]; [100 100 100] ];


if (length(name_grp_plot) ~= ngroups)
    fprintf(1, 'Error: missmatch between the length of name_grp_plot and the number of groups\n');
end

colorplot = zeros(ngroups, 3);

for ig1=1:ngroups
    for ig2=1:ngroups
        if strcmp(nameGrp(ig1), name_grp_plot{ig2})
            colorplot(ig1, :) = colorVals(ig2, :)./255;
            break;
        end       
    end
end

nameGrp2 = cell(1,ngroups*2-1);
colorplot2 = zeros(ngroups*2-1, 3);

j = 1;
for i=1:ngroups
    if strcmp(nameGrp{i}, 'So')
        nameGrp2{j} = 'So,M';
        for ig2=1:ngroups
            if strcmp(nameGrp(i), name_grp_plot{ig2})
                colorplot2(j, :) = colorVals(ig2, :)./255;
                break;
            end
        end
        j = j+1;
        
    else
        for ig2=1:ngroups
            if strcmp(nameGrp(i), name_grp_plot{ig2})
                colorplot2(j, :) = colorVals(ig2, :)./255;
                colorplot2(j+1, :) = colorVals(ig2, :)./255;
                break;
            end
        end
        nameGrp2{j} = sprintf('%s,M', nameGrp{i});
        j = j+1;
        nameGrp2{j} = sprintf('%s,F', nameGrp{i});
        j = j+1;
    end
end



%% Reformat Data Base

% Extract the grouping variables from data array
birdNameCuts = {callAnalData.bird};
birdSexCuts = {callAnalData.birdSex};
birdNames = unique(birdNameCuts);
nBirds = length(birdNames);

vocTypeCuts = {callAnalData.type};
vocTypes = unique(vocTypeCuts);   % This returns alphabetical 
name_grp = unique(vocTypeCuts, 'stable');  % This is the order returned by grpstats, manova, etcc
ngroups = length(vocTypes);


nAcoust = 22;
% Make a matrix of the Acoustical Parameters
Acoust = zeros(length(vocTypeCuts), nAcoust);
Acoust(:,1) = [callAnalData.fund];
Acoust(:,2) = [callAnalData.sal];
Acoust(:,3) = [callAnalData.fund2];
Acoust(:,4) = [callAnalData.voice2percent];
Acoust(:,5) = [callAnalData.maxfund];
Acoust(:,6) = [callAnalData.minfund];
Acoust(:,7) = [callAnalData.cvfund];
Acoust(:,8) = [callAnalData.meanspect];
Acoust(:,9) = [callAnalData.stdspect];
Acoust(:,10) = [callAnalData.skewspect];
Acoust(:,11) = [callAnalData.kurtosisspect];
Acoust(:,12) = [callAnalData.entropyspect];
Acoust(:,13) = [callAnalData.q1];
Acoust(:,14) = [callAnalData.q2];
Acoust(:,15) = [callAnalData.q3];
Acoust(:,16) = [callAnalData.meantime];
Acoust(:,17) = [callAnalData.stdtime];
Acoust(:,18) = [callAnalData.skewtime];
Acoust(:,19) = [callAnalData.kurtosistime];
Acoust(:,20) = [callAnalData.entropytime];
Acoust(:,21) = [callAnalData.rms];
Acoust(:,22) = [callAnalData.maxAmp];

% Tags
xtag{1} = 'fund';
xtag{2} = 'sal';
xtag{3} = 'fund2';
xtag{4} = 'voice2percent';
xtag{5} = 'maxfund';
xtag{6} = 'minfund';
xtag{7} = 'cvfund';
xtag{8} = 'meanspect';
xtag{9} = 'stdspect';
xtag{10} = 'skewspect';
xtag{11} = 'kurtosisspect';
xtag{12} = 'entropyspect';
xtag{13} = 'q1';
xtag{14} = 'q2';
xtag{15} = 'q3';
xtag{16} = 'meantime';
xtag{17} = 'stdtime';
xtag{18} = 'skewtime';
xtag{19} = 'kurtosistime';
xtag{20} = 'entropytime';
xtag{21} = 'rms';
xtag{22} = 'maxamp';

% xtag for plotting
xtagPlot{1} = 'F0';
xtagPlot{2} = 'Sal';
xtagPlot{3} = 'Pk2';
xtagPlot{4} = '2nd V';
xtagPlot{5} = 'Max F0';
xtagPlot{6} = 'Min F0';
xtagPlot{7} = 'CV F0';
xtagPlot{8} = 'Mean S';
xtagPlot{9} = 'Std S';
xtagPlot{10} = 'Skew S';
xtagPlot{11} = 'Kurt S';
xtagPlot{12} = 'Ent S';
xtagPlot{13} = 'Q1';
xtagPlot{14} = 'Q2';
xtagPlot{15} = 'Q3';
xtagPlot{16} = 'Mean T';
xtagPlot{17} = 'Std T';
xtagPlot{18} = 'Skew T';
xtagPlot{19} = 'Kurt T';
xtagPlot{20} = 'Ent T';
xtagPlot{21} = 'RMS';
xtagPlot{22} = 'Max A';

%% Try an unservised clustering with Saliency and Mean Spectrum

options = statset('Display','final', 'MaxIter', 1000);  % Display final results

X = [Acoust(:,2) Acoust(:,8)];  % Use saliency and mean spectrum for clustering

% Min Max for displaying
minX1 = min(Acoust(:,2));
maxX1 = max(Acoust(:,2));
minX2 = min(Acoust(:,8));
maxX2 = max(Acoust(:,8));


% Plot the data on a color coded scatter plot
figure(1);
for ig=1:ngroups
    indGrp =  find( strcmp(vocTypeCuts, nameGrp(ig)) );
    if ig > 1
        hold on;
    end
    scatter(X(indGrp,1),X(indGrp,2),10, 'MarkerFaceColor', colorplot(ig, :), 'MarkerEdgeColor', colorplot(ig, :));
    hold off;
end
legend(nameGrp);
xlabel('Saliency');
ylabel('Spectral Mean (Hz)');


% Unsupervised clusters
figure(2);
cmap = colormap();
for icluster=5:10
    
    gm = fitgmdist(X,icluster,'Options',options);  % Gaussian mixture model
    idx = cluster(gm,X);   % Returns the clusters
    
    % plot each cluster
    subplot(1, 6, icluster-4);
    for ic = 1:icluster
        if ic > 1
            hold on;
        end
        clusteridx = idx == ic;
        scatter(X(clusteridx,1),X(clusteridx,2),10, 'MarkerFaceColor', cmap(fix(64*ic/10),:), 'MarkerEdgeColor', cmap(fix(64*ic/10),:));
        
        hold off;
    end
    
    % Add contour lines
    hold on
    ezcontour(@(x,y)pdf(gm,[x y]),[minX1 maxX1 minX2 maxX2],20);
    hold off
    if icluster == 5
        xlabel('Saliency');
        ylabel('Spectral Mean (Hz)');
    end
    title(sprintf('%d AIC = %f', ic, gm.AIC));
    
end

%% Try an unservised clustering with Saliency and Mean Spectrum - this time without song

indSong = find( strcmp(vocTypeCuts, 'So') );
XnoSo = X;
XnoSo(indSong, :) = [];


% Plot the data on a color coded scatter plot
figure(3);
for ig=1:ngroups
    if strcmp(nameGrp(ig), 'So')
        igSo = ig;
        continue;
    end
    indGrp =  find( strcmp(vocTypeCuts, nameGrp(ig)) );
    if ig > 1
        hold on;
    end
    scatter(X(indGrp,1),X(indGrp,2),10, 'MarkerFaceColor', colorplot(ig, :), 'MarkerEdgeColor', colorplot(ig, :));
    hold off;
end
nameGrpnoSo = nameGrp;
nameGrpnoSo(igSo) = [];
legend(nameGrpnoSo);
xlabel('Saliency');
ylabel('Spectral Mean (Hz)');
    
gm = fitgmdist(XnoSo,5,'Options',options);  % Gaussian mixture model
hold on
ezcontour(@(x,y)pdf(gm,[x y]),[minX1 maxX1 minX2 maxX2],20);
hold off


% Unsupervised clusters
figure(4);
cmap = colormap();
for icluster=5:10
    
    gm = fitgmdist(XnoSo,icluster,'Options',options);  % Gaussian mixture model
    idx = cluster(gm,XnoSo);   % Returns the clusters
    
    % plot each cluster
    subplot(1, 6, icluster-4);
    for ic = 1:icluster
        if ic > 1
            hold on;
        end
        clusteridx = idx == ic;
        scatter(XnoSo(clusteridx,1),XnoSo(clusteridx,2),10, 'MarkerFaceColor', cmap(fix(64*ic/10),:), 'MarkerEdgeColor', cmap(fix(64*ic/10),:));
        
        hold off;
    end
    
    % Add contour lines
    hold on
    ezcontour(@(x,y)pdf(gm,[x y]),[minX1 maxX1 minX2 maxX2],20);
    hold off
    if icluster == 5
        xlabel('Saliency');
        ylabel('Spectral Mean (Hz)');
    end
    title(sprintf('%d AIC = %f', ic, gm.AIC));
    
end

%% Try a clustering on the bird averaged data
% Find number of unique combinations
vocTypesBirds = unique(strcat(vocTypeCuts', birdNameCuts'), 'stable');
nvocTypesBirds = length(vocTypesBirds);
vocTypeCutsMeans = cell(1, nvocTypesBirds);
birdNameCutsMeans = cell(1, nvocTypesBirds);
birdSexCutsMeans = cell(1, nvocTypesBirds);
XMeans = zeros(nvocTypesBirds, 2);


for ic = 1: nvocTypesBirds
    indTypesBirds = find( strcmp(vocTypesBirds{ic}(1:2), vocTypeCuts') & strcmp(vocTypesBirds{ic}(3:end), birdNameCuts'));
    vocTypeCutsMeans{ic} = vocTypesBirds{ic}(1:2);
    birdNameCutsMeans{ic} = vocTypesBirds{ic}(3:end);
    birdSexCutsMeans{ic} = birdSexCuts{indTypesBirds(1)};
    if length(indTypesBirds) == 1
        XMeans(ic, :) = X(indTypesBirds, :);
    else
        XMeans(ic, :) = mean(X(indTypesBirds, :));
    end
end

figure(5);
for ig=1:ngroups
    indGrp =  find( strcmp(vocTypeCutsMeans, nameGrp(ig)) );
    if ig > 1
        hold on;
    end
    scatter(XMeans(indGrp,1),XMeans(indGrp,2),32, 'MarkerFaceColor', colorplot(ig, :), 'MarkerEdgeColor', colorplot(ig, :));
    hold off;
end

gm = fitgmdist(XMeans,6,'Options',options);  % Gaussian mixture model
hold on
ezcontour(@(x,y)pdf(gm,[x y]),[minX1 maxX1 minX2 maxX2],50);
hold off
legend(nameGrp);
xlabel('Saliency');
ylabel('Spectral Mean (Hz)');

% Unsupervised clusters
figure(6);
cmap = colormap();
for icluster=1:5
    
    gm = fitgmdist(XMeans,icluster,'Options',options);  % Gaussian mixture model
    idx = cluster(gm,XMeans);   % Returns the clusters
    
    % plot each cluster
    subplot(1, 5, icluster);
    for ic = 1:icluster
        if ic > 1
            hold on;
        end
        clusteridx = idx == ic;
        scatter(XMeans(clusteridx,1),XMeans(clusteridx,2),10, 'MarkerFaceColor', cmap(fix(64*ic/10),:), 'MarkerEdgeColor', cmap(fix(64*ic/10),:));
        
        hold off;
    end
    
    % Add contour lines
    hold on
    ezcontour(@(x,y)pdf(gm,[x y]),[minX1 maxX1 minX2 maxX2],20);
    hold off
    if icluster == 5
        xlabel('Saliency');
        ylabel('Spectral Mean (Hz)');
    end
    title(sprintf('%d AIC = %f', ic, gm.AIC));
    
end


