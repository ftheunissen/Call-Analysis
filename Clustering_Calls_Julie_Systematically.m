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


nAcoust = 21;
% Make a matrix of the Acoustical Parameters - we are going to remove the
% fund2 because it has too many missing values
Acoust = zeros(length(callAnalData), nAcoust);
Acoust(:,1) = [callAnalData.fund];
Acoust(:,2) = [callAnalData.sal];
% Acoust(:,3) = [callAnalData.fund2];
Acoust(:,3) = [callAnalData.voice2percent];
Acoust(:,4) = [callAnalData.maxfund];
Acoust(:,5) = [callAnalData.minfund];
Acoust(:,6) = [callAnalData.cvfund];
Acoust(:,7) = [callAnalData.meanspect];
Acoust(:,8) = [callAnalData.stdspect];
Acoust(:,9) = [callAnalData.skewspect];
Acoust(:,10) = [callAnalData.kurtosisspect];
Acoust(:,11) = [callAnalData.entropyspect];
Acoust(:,12) = [callAnalData.q1];
Acoust(:,13) = [callAnalData.q2];
Acoust(:,14) = [callAnalData.q3];
Acoust(:,15) = [callAnalData.meantime];
Acoust(:,16) = [callAnalData.stdtime];
Acoust(:,17) = [callAnalData.skewtime];
Acoust(:,18) = [callAnalData.kurtosistime];
Acoust(:,19) = [callAnalData.entropytime];
Acoust(:,20) = [callAnalData.rms];
Acoust(:,21) = [callAnalData.maxAmp];

% Tags
xtag{1} = 'fund';
xtag{2} = 'sal';
% xtag{3} = 'fund2';
xtag{3} = 'voice2percent';
xtag{4} = 'maxfund';
xtag{5} = 'minfund';
xtag{6} = 'cvfund';
xtag{7} = 'meanspect';
xtag{8} = 'stdspect';
xtag{9} = 'skewspect';
xtag{10} = 'kurtosisspect';
xtag{11} = 'entropyspect';
xtag{12} = 'q1';
xtag{13} = 'q2';
xtag{14} = 'q3';
xtag{15} = 'meantime';
xtag{16} = 'stdtime';
xtag{17} = 'skewtime';
xtag{18} = 'kurtosistime';
xtag{19} = 'entropytime';
xtag{20} = 'rms';
xtag{21} = 'maxamp';

% xtag for plotting
xtagPlot{1} = 'F0';
xtagPlot{2} = 'Sal';
% xtagPlot{3} = 'Pk2';
xtagPlot{3} = '2nd V';
xtagPlot{4} = 'Max F0';
xtagPlot{5} = 'Min F0';
xtagPlot{6} = 'CV F0';
xtagPlot{7} = 'Mean S';
xtagPlot{8} = 'Std S';
xtagPlot{9} = 'Skew S';
xtagPlot{10} = 'Kurt S';
xtagPlot{11} = 'Ent S';
xtagPlot{12} = 'Q1';
xtagPlot{13} = 'Q2';
xtagPlot{14} = 'Q3';
xtagPlot{15} = 'Mean T';
xtagPlot{16} = 'Std T';
xtagPlot{17} = 'Skew T';
xtagPlot{18} = 'Kurt T';
xtagPlot{19} = 'Ent T';
xtagPlot{20} = 'RMS';
xtagPlot{21} = 'Max A';

% remove missing values
[indr, indc] = find(isnan(Acoust));
Acoust(indr, :) = [];

% Extract the grouping variables from data array
birdNameCuts = {callAnalData.bird};
birdNameCuts(indr) = [];
birdSexCuts = {callAnalData.birdSex};
birdSexCuts(indr) = [];
birdNames = unique(birdNameCuts);
nBirds = length(birdNames);

vocTypeCuts = {callAnalData.type};
vocTypeCuts(indr) = [];
vocTypes = unique(vocTypeCuts);   % This returns alphabetical 
name_grp = unique(vocTypeCuts, 'stable');  % This is the order returned by grpstats, manova, etcc
ngroups = length(vocTypes);

%% Generate PCA

zAcoust = zscore(Acoust);
[pcAcoust, scoreAcoust, eigenval] = princomp(zAcoust);  % Use pca so that we can used weighted observations as well

figure(1);
plot(100*cumsum(eigenval)./sum(eigenval));
xlabel('Number of PCs');
ylabel('Variance Explained %');
title('PCs on Acoustical Features');

% 10 PCs explain 89.8430 of the variance


% %% Try an unservised clustering with increasing number of PCs and from 1 to 10 clusters
% 
% nClusters = 10;
% nPCs = length(pcAcoust);
% 
% gmList = cell(nPCs, nClusters);
% aicList = zeros(nPCs, nClusters);
% bicList = zeros(nPCs, nClusters);
% 
% options = statset('Display','final', 'MaxIter', 1000);  % Display final results
% 
% for ipc = 1:nPCs
%     for ic = 1:nClusters
%         
%         gmList{ipc, ic} = fitgmdist(scoreAcoust(:, 1:ipc), ic, 'Options', options);  % Gaussian mixture model
%         aicList(ipc, ic) = gmList{ipc, ic}.AIC;
%         bicList(ipc, ic) = gmList{ipc, ic}.BIC;
%         
%     end
% end
% 
% figure(1);
% surf(1:nClusters, 1:nPCs, aicList - min(min(aicList)));
% ylabel('# PCs');
% xlabel('# Clusters');
% title('AIC');
% 
% figure(2);
% surf(1:nClusters, 1:nPCs, bicList - min(min(bicList)));
% ylabel('# PCs');
% xlabel('# Clusters');
% title('BIC');
% 
% figure(3);
% logL = zeros(nPCs, nClusters);
% for ipc = 1:nPCs
%     for ic = 1:nClusters
%        logL(ipc, ic) = gmList{ipc, ic}.NegativeLogLikelihood;      
%     end
% end
% surf(1:nClusters, 1:nPCs, logL - min(min(logL)));
% ylabel('# PCs');
% xlabel('# Clusters');
% title('-Log Likelihood');

%% Find the optimal number of cluster for 10 pcs

nClusters = 40;
nPCs = 10;

gmList = cell(1, nClusters);
aicList = zeros(1, nClusters);
bicList = zeros(1, nClusters);

options = statset('Display','final', 'MaxIter', 1000);  % Display final results

parfor ic = 1:nClusters    
    gmList{ic} = fitgmdist(scoreAcoust(:, 1:nPCs), ic, 'Options', options, 'Replicates', 5);  % Gaussian mixture model
    aicList(ic) = gmList{ic}.AIC;
    bicList(ic) = gmList{ic}.BIC;
end

figure(2);
subplot(1,2,1);
plot(1:nClusters, aicList - min(aicList), 'k');
xlabel('Number of Gaussians');
ylabel('AIC');

subplot(1,2,2);
plot(1:nClusters, bicList, 'r');
xlabel('Number of Gaussians');
ylabel('BIC');

%% Look at distribution from 5 to 10 Groups

figure(3);  
nPCs = 10;
    
for icgroup=1:10
    subplot(2,5,icgroup);
    icluster = icgroup+4;
    
    idx = cluster(gmList{icluster} , scoreAcoust(:,1:nPCs));
    
    countTypePer = zeros(icluster, ngroups);
    countTypeTot = zeros(1, ngroups);
    for ig=1:ngroups
        indGrp =  find( strcmp(vocTypeCuts, nameGrp(ig)) );
        countTypeTot(ig) = length(indGrp);
    end
    
    for ic = 1:icluster
        clusteridx = find(idx == ic);
        fprintf(1,'Cluster %d:\n', ic);
        for ig=1:ngroups
            indGrp =  find( strcmp(vocTypeCuts(clusteridx), nameGrp(ig)) );
            countTypePer(ic,ig) = 100.0*length(indGrp)./countTypeTot(ig);
            fprintf(1,'\t%s %d (%.2f%%)\n', nameGrp{ig}, length(indGrp), countTypePer(ic,ig));
        end
    end
    
    bh = bar(countTypePer);
    for ibh=1:ngroups
        set(bh(ibh), 'EdgeColor', colorplot(ibh, :), 'FaceColor', colorplot(ibh, :));
    end
    if icgroup == 10
        legend(nameGrp);
    end
    title(sprintf('%d Groups', icluster));
end

% 10 is nice
figure(4);

icluster = 10;

idx = cluster(gmList{icluster} , scoreAcoust(:,1:nPCs));

countTypePer = zeros(icluster, ngroups);
countTypeTot = zeros(1, ngroups);
for ig=1:ngroups
    indGrp =  find( strcmp(vocTypeCuts, nameGrp(ig)) );
    countTypeTot(ig) = length(indGrp);
end

for ic = 1:icluster
    clusteridx = find(idx == ic);
    fprintf(1,'Cluster %d:\n', ic);
    for ig=1:ngroups
        indGrp =  find( strcmp(vocTypeCuts(clusteridx), nameGrp(ig)) );
        countTypePer(ic,ig) = 100.0*length(indGrp)./countTypeTot(ig);
        fprintf(1,'\t%s %d (%.2f%%)\n', nameGrp{ig}, length(indGrp), countTypePer(ic,ig));
    end
end

bh = bar(countTypePer);
for ibh=1:ngroups
    set(bh(ibh), 'EdgeColor', colorplot(ibh, :), 'FaceColor', colorplot(ibh, :));
end
legend(nameGrp);
title(sprintf('%d Groups', icluster));

% Plot the data using 1rst and 2nd PCs
figure(5);
subplot(1,2,1)
for ig=1:ngroups
    indGrp =  find( strcmp(vocTypeCuts, nameGrp(ig)) );
    if ig > 1
        hold on;
    end
    scatter(scoreAcoust(indGrp,1),scoreAcoust(indGrp,2), 2, 'MarkerFaceColor', colorplot(ig, :), 'MarkerEdgeColor', colorplot(ig, :));
    hold off;
end

% Add contour lines
% hold on
% for ic = 1:icluster
%     ezcontour(@(x,y)mvnpdf([x y], [gmList{icluster}.mu(ic,1) gmList{icluster}.mu(ic,2)], gmList{icluster}.Sigma(1:2,1:2,ic)), ...
%         [gmList{icluster}.mu(ic,1)-2 gmList{icluster}.mu(ic,1)+2 gmList{icluster}.mu(ic,2)-2 gmList{icluster}.mu(ic,2)+2], 60);
% end
% hold off

hold on
ang = 0:pi/20:2*pi;
circ = [cos(ang); sin(ang)];   % x,y coordinates of unit circle    
for ic = 1:icluster
    mu1 = gmList{icluster}.mu(ic,1);
    mu2 = gmList{icluster}.mu(ic,2);
    sig1 = gmList{icluster}.Sigma(1,1,ic);
    sig2 = gmList{icluster}.Sigma(2,2,ic);
    sig12 = gmList{icluster}.Sigma(1,2,ic);
    covMat = [sig1, sig12; sig12, sig2];
    [V, D] = eig(covMat);
    Dsd = sqrt(D);
    
    X = V*Dsd*circ;
    
    scatter(mu1, mu2, 30, 'k');
    plot( mu1 + X(1,:), mu2 + X(2,:) ,'-', 'LineWidth', 2, 'Color', [0.5 0.5 0.5]);
    
end
hold off
xlabel('PC1');
ylabel('PC2');
axis([-5 5 -5 5]);

subplot(1,2,2)
for ig=1:ngroups
    indGrp =  find( strcmp(vocTypeCuts, nameGrp(ig)) );
    if ig > 1
        hold on;
    end
    scatter(scoreAcoust(indGrp,1),scoreAcoust(indGrp,3),2, 'MarkerFaceColor', colorplot(ig, :), 'MarkerEdgeColor', colorplot(ig, :));
    hold off;
end
% Add contour lines
% hold on
% for ic = 1:icluster
%     ezcontour(@(x,y)mvnpdf([x y], [gmList{icluster}.mu(ic,1) gmList{icluster}.mu(ic,3)],...
%         [gmList{icluster}.Sigma(1,1,ic), gmList{icluster}.Sigma(1,3,ic); gmList{icluster}.Sigma(3,1,ic), gmList{icluster}.Sigma(3,3,ic)] ),...
%         [gmList{icluster}.mu(ic,1)-2 gmList{icluster}.mu(ic,1)+2 gmList{icluster}.mu(ic,3)-2 gmList{icluster}.mu(ic,3)+2], 60);
% end
% hold off
hold on
ang = 0:pi/20:2*pi;
circ = [cos(ang); sin(ang)];   % x,y coordinates of unit circle    
for ic = 1:icluster
    mu1 = gmList{icluster}.mu(ic,1);
    mu2 = gmList{icluster}.mu(ic,3);
    sig1 = gmList{icluster}.Sigma(1,1,ic);
    sig2 = gmList{icluster}.Sigma(3,3,ic);
    sig12 = gmList{icluster}.Sigma(1,3,ic);
    covMat = [sig1, sig12; sig12, sig2];
    [V, D] = eig(covMat);
    Dsd = sqrt(D);
    
    X = V*Dsd*circ;
    
    scatter(mu1, mu2, 30, 'k');
    plot( mu1 + X(1,:), mu2 + X(2,:) ,'-', 'LineWidth', 2, 'Color', [0.5 0.5 0.5]);
    
end
hold off

xlabel('PC1');
ylabel('PC3');
axis([-5 5 -5 5]);

%% Repeat without the song 
% Find the optimal number of cluster for 10 pcs

indSong = find( strcmp(vocTypeCuts, 'So') );
AcoustnoSo = Acoust;
AcoustnoSo(indSong, :) = [];
vocTypeCutsnoSo = vocTypeCuts;
vocTypeCutsnoSo(indSong) = [];

zAcoustnoSo = zscore(AcoustnoSo);
[pcAcoustnoSo, scoreAcoustnoSo, eigenvalnoSo] = princomp(zAcoustnoSo);  % Use pca so that we can used weighted observations as well

figure(6);
plot(100*cumsum(eigenvalnoSo)./sum(eigenvalnoSo));
xlabel('Number of PCs');
ylabel('Variance Explained %');
title('PCs on Acoustical Features');

nClusters = 40;
nPCs = 10;

gmListnoSo = cell(1, nClusters);
aicListnoSo = zeros(1, nClusters);
bicListnoSo = zeros(1, nClusters);

options = statset('Display','final', 'MaxIter', 1000);  % Display final results

parfor ic = 1:nClusters    
    gmListnoSo{ic} = fitgmdist(scoreAcoustnoSo(:, 1:nPCs), ic, 'Options', options, 'Replicates', 5);  % Gaussian mixture model
    aicListnoSo(ic) = gmListnoSo{ic}.AIC;
    bicListnoSo(ic) = gmListnoSo{ic}.BIC;
end

figure(7);
subplot(1,2,1);
plot(1:nClusters, aicListnoSo - min(aicListnoSo), 'k');
xlabel('Number of Gaussians');
ylabel('AIC');

subplot(1,2,2);
plot(1:nClusters, bicListnoSo, 'r');
xlabel('Number of Gaussians');
ylabel('BIC');

%% Look at distribution from 5 to 10 Groups

figure(8);  
nPCs = 10;
    
for icgroup=1:10
    subplot(2,5,icgroup);
    icluster = icgroup+4;
    
    idx = cluster(gmListnoSo{icluster} , scoreAcoustnoSo(:,1:nPCs));
    
    countTypePer = zeros(icluster, ngroups);
    countTypeTot = zeros(1, ngroups);
    for ig=1:ngroups
        indGrp =  find( strcmp(vocTypeCutsnoSo, nameGrp(ig)) );
        countTypeTot(ig) = length(indGrp);
    end
    
    for ic = 1:icluster
        clusteridx = find(idx == ic);
        fprintf(1,'Cluster %d:\n', ic);
        for ig=1:ngroups
            indGrp =  find( strcmp(vocTypeCutsnoSo(clusteridx), nameGrp(ig)) );
            countTypePer(ic,ig) = 100.0*length(indGrp)./countTypeTot(ig);
            fprintf(1,'\t%s %d (%.2f%%)\n', nameGrp{ig}, length(indGrp), countTypePer(ic,ig));
        end
    end
    
    bh = bar(countTypePer);
    for ibh=1:ngroups
        set(bh(ibh), 'EdgeColor', colorplot(ibh, :), 'FaceColor', colorplot(ibh, :));
    end
    if icgroup == 10
        legend(nameGrp);
    end
    title(sprintf('%d Groups', icluster));
end

% 7 is nice
figure(9);

icluster = 7;

idx = cluster(gmListnoSo{icluster} , scoreAcoustnoSo(:,1:nPCs));

countTypePer = zeros(icluster, ngroups);
countTypeTot = zeros(1, ngroups);
for ig=1:ngroups
    indGrp =  find( strcmp(vocTypeCutsnoSo, nameGrp(ig)) );
    countTypeTot(ig) = length(indGrp);
end

for ic = 1:icluster
    clusteridx = find(idx == ic);
    fprintf(1,'Cluster %d:\n', ic);
    for ig=1:ngroups
        indGrp =  find( strcmp(vocTypeCutsnoSo(clusteridx), nameGrp(ig)) );
        countTypePer(ic,ig) = 100.0*length(indGrp)./countTypeTot(ig);
        fprintf(1,'\t%s %d (%.2f%%)\n', nameGrp{ig}, length(indGrp), countTypePer(ic,ig));
    end
end

bh = bar(countTypePer);
for ibh=1:ngroups
    set(bh(ibh), 'EdgeColor', colorplot(ibh, :), 'FaceColor', colorplot(ibh, :));
end
legend(nameGrp);
title(sprintf('%d Groups', icluster));

% Plot the data using 1rst and 2nd PCs
figure(10);
subplot(1,2,1)
for ig=1:ngroups
    indGrp =  find( strcmp(vocTypeCutsnoSo, nameGrp(ig)) );
    if ig > 1
        hold on;
    end
    scatter(scoreAcoustnoSo(indGrp,1),scoreAcoustnoSo(indGrp,2), 2, 'MarkerFaceColor', colorplot(ig, :), 'MarkerEdgeColor', colorplot(ig, :));
    hold off;
end

% Add contour lines
% hold on
% for ic = 1:icluster
%     ezcontour(@(x,y)mvnpdf([x y], [gmList{icluster}.mu(ic,1) gmList{icluster}.mu(ic,2)], gmList{icluster}.Sigma(1:2,1:2,ic)), ...
%         [gmList{icluster}.mu(ic,1)-2 gmList{icluster}.mu(ic,1)+2 gmList{icluster}.mu(ic,2)-2 gmList{icluster}.mu(ic,2)+2], 60);
% end
% hold off

hold on
ang = 0:pi/20:2*pi;
circ = [cos(ang); sin(ang)];   % x,y coordinates of unit circle  
maxAmp = max(gmListnoSo{icluster}.ComponentProportion);
for ic = 1:icluster
    compAmp = gmListnoSo{icluster}.ComponentProportion(ic);
    mu1 = gmListnoSo{icluster}.mu(ic,1);
    mu2 = gmListnoSo{icluster}.mu(ic,2);
    sig1 = gmListnoSo{icluster}.Sigma(1,1,ic);
    sig2 = gmListnoSo{icluster}.Sigma(2,2,ic);
    sig12 = gmListnoSo{icluster}.Sigma(1,2,ic);
    covMat = [sig1, sig12; sig12, sig2];

    [V, D] = eig(covMat);
    Dsd = sqrt(D);
    
    X = V*Dsd*circ;
    
    scatter(mu1, mu2, fix(60*compAmp/maxAmp), 'k', 'filled');
    plot( mu1 + X(1,:), mu2 + X(2,:) ,'-', 'LineWidth', 2, 'Color', [0.5 0.5 0.5]);
    
end
hold off
xlabel('PC1');
ylabel('PC2');
axis([-5 5 -5 5]);

subplot(1,2,2)
for ig=1:ngroups
    indGrp =  find( strcmp(vocTypeCutsnoSo, nameGrp(ig)) );
    if ig > 1
        hold on;
    end
    scatter(scoreAcoustnoSo(indGrp,1),scoreAcoustnoSo(indGrp,3),2, 'MarkerFaceColor', colorplot(ig, :), 'MarkerEdgeColor', colorplot(ig, :));
    hold off;
end
% Add contour lines
% hold on
% for ic = 1:icluster
%     ezcontour(@(x,y)mvnpdf([x y], [gmListnoSo{icluster}.mu(ic,1) gmListnoSo{icluster}.mu(ic,3)],...
%         [gmListnoSo{icluster}.Sigma(1,1,ic), gmListnoSo{icluster}.Sigma(1,3,ic); gmListnoSo{icluster}.Sigma(3,1,ic), gmListnoSo{icluster}.Sigma(3,3,ic)] ),...
%         [gmListnoSo{icluster}.mu(ic,1)-2 gmListnoSo{icluster}.mu(ic,1)+2 gmListnoSo{icluster}.mu(ic,3)-2 gmListnoSo{icluster}.mu(ic,3)+2], 60);
% end
% hold off
hold on
ang = 0:pi/20:2*pi;
circ = [cos(ang); sin(ang)];   % x,y coordinates of unit circle  
maxAmp = max(gmListnoSo{icluster}.ComponentProportion);

for ic = 1:icluster
    compAmp = gmListnoSo{icluster}.ComponentProportion(ic);
    mu1 = gmListnoSo{icluster}.mu(ic,1);
    mu2 = gmListnoSo{icluster}.mu(ic,3);
    sig1 = gmListnoSo{icluster}.Sigma(1,1,ic);
    sig2 = gmListnoSo{icluster}.Sigma(3,3,ic);
    sig12 = gmListnoSo{icluster}.Sigma(1,3,ic);
    covMat = [sig1, sig12; sig12, sig2];
    [V, D] = eig(covMat);
    Dsd = sqrt(D);
    
    X = V*Dsd*circ;
    
    scatter(mu1, mu2, fix(60*compAmp/maxAmp), 'k', 'filled');
    plot( mu1 + X(1,:), mu2 + X(2,:) ,'-', 'LineWidth', 2, 'Color', [0.5 0.5 0.5]);
    
end
hold off

xlabel('PC1');
ylabel('PC3');
axis([-5 5 -5 5]);

%% Clustering on bird averaged data and with prior of equal probability.

vocTypesBirds = unique(strcat(vocTypeCuts', birdNameCuts'), 'stable');
nvocTypesBirds = length(vocTypesBirds);
vocTypeCutsMeans = cell(1, nvocTypesBirds);
birdNameCutsMeans = cell(1, nvocTypesBirds);
birdSexCutsMeans = cell(1, nvocTypesBirds);
AcoustMeans = zeros(nvocTypesBirds, size(Acoust,2));


for ic = 1: nvocTypesBirds
    indTypesBirds = find( strcmp(vocTypesBirds{ic}(1:2), vocTypeCuts') & strcmp(vocTypesBirds{ic}(3:end), birdNameCuts'));
    vocTypeCutsMeans{ic} = vocTypesBirds{ic}(1:2);
    birdNameCutsMeans{ic} = vocTypesBirds{ic}(3:end);
    birdSexCutsMeans{ic} = birdSexCuts{indTypesBirds(1)};
    if length(indTypesBirds) == 1
        AcoustMeans(ic, :) = Acoust(indTypesBirds, :);
    else
        AcoustMeans(ic, :) = mean(Acoust(indTypesBirds, :));
    end
end

% Trasnform into zscores 
zAcoustMeans = zscore(AcoustMeans);
[pcAcoustMeans, scoreAcoustMeans, eigenvalMeans] = princomp(zAcoustMeans);  % Use pca so that we can used weighted observations as well

figure(11);
plot(100*cumsum(eigenvalMeans)./sum(eigenvalMeans));
xlabel('Number of PCs');
ylabel('Variance Explained %');
title('PCs on Acoustical Features');



%% Perform the unsupervised learning with equal number of birds and equal distribution

nClusters = 4;  % Good upper limit
% nPCs = length(pcAcoustMeans);
nPCs = 10;    % Does not work well beyond
nPerm = 10;

indGrpLen = zeros(1,ngroups);     % Number of data points for each type...
indGrp = cell(1,ngroups);
for ig=1:ngroups
    indGrp{ig} =  find( strcmp(vocTypeCutsMeans, nameGrp(ig)) );
    indGrpLen(ig) = length(indGrp{ig});
    fprintf(1,'Number of birds for %s: %d\n', nameGrp{ig}, indGrpLen(ig));
end
minNumBirds = min(indGrpLen);

aicListM = zeros(nPerm, nClusters);
bicListM = zeros(nPerm, nClusters);
logLM = zeros(nPerm, nClusters);

options = statset('Display','final', 'MaxIter', 1000);  % Display final results

for ip = 1:nPerm
    indPerm = zeros(1,minNumBirds*ngroups);
    for ig=1:ngroups
        p = randperm(indGrpLen(ig), minNumBirds);
        indPerm((ig-1)*minNumBirds+1:ig*minNumBirds) = indGrp{ig}(p);
    end
    for ic = 1:nClusters  
        fprintf(1,'n Perm %d n Cluster %d\n', ip, ic);
        gm = fitgmdist(scoreAcoustMeans(indPerm, 1:nPCs), ic, 'Options', options, 'Replicates', 10);  % Gaussian mixture model
        aicListM(ip, ic) = gm.AIC;
        bicListM(ip, ic) = gm.BIC;
        logLM(ip, ic) = gm.NegativeLogLikelihood;         
    end
end

figure(12);
subplot(1,2,1);
plot(1:nClusters, mean(aicListM));
xlabel('# Clusters');
ylabel('AIC');

subplot(1,2,2);
plot(1:nClusters, mean(bicListM));
xlabel('# Clusters');
ylabel('BIC');

%% Looks like two clusters...
nPCs = 10;
icluster = 2;

indGrpLen = zeros(1,ngroups);     % Number of data points for each type...
indGrp = cell(1,ngroups);
for ig=1:ngroups
    indGrp{ig} =  find( strcmp(vocTypeCutsMeans, nameGrp(ig)) );
    indGrpLen(ig) = length(indGrp{ig});
    fprintf(1,'Number of birds for %s: %d\n', nameGrp{ig}, indGrpLen(ig));
end
minNumBirds = min(indGrpLen);


indPerm = zeros(1,minNumBirds*ngroups);
for ig=1:ngroups
    p = randperm(indGrpLen(ig), minNumBirds);
    indPerm((ig-1)*minNumBirds+1:ig*minNumBirds) = indGrp{ig}(p);
end

gm = fitgmdist(scoreAcoustMeans(indPerm, 1:nPCs), icluster, 'Options', options, 'Replicates', 30);  % Gaussian mixture model

figure(13);
idx = cluster(gm , scoreAcoustMeans(indPerm, 1:nPCs));

countTypePer = zeros(icluster, ngroups);
countTypeTot = zeros(1, ngroups);
vocTypeCutsPerm = vocTypeCutsMeans(indPerm);
for ig=1:ngroups
    indGrp =  find( strcmp(vocTypeCutsPerm, nameGrp(ig)) );
    countTypeTot(ig) = length(indGrp);
end

for ic = 1:icluster
    clusteridx = find(idx == ic);
    fprintf(1,'Cluster %d:\n', ic);
    for ig=1:ngroups
        indGrp =  find( strcmp(vocTypeCutsPerm(clusteridx), nameGrp(ig)) );
        countTypePer(ic,ig) = 100.0*length(indGrp)./countTypeTot(ig);
        fprintf(1,'\t%s %d (%.2f%%)\n', nameGrp{ig}, length(indGrp), countTypePer(ic,ig));
    end
end

bh = bar(countTypePer);
for ibh=1:ngroups
    set(bh(ibh), 'EdgeColor', colorplot(ibh, :), 'FaceColor', colorplot(ibh, :));
end
legend(nameGrp);
title(sprintf('%d Groups', icluster));

% Plot the data using 1rst and 2nd PCs
figure(14);
subplot(1,2,1)
for ig=1:ngroups
    indGrp =  find( strcmp(vocTypeCutsMeans, nameGrp(ig)) );
    if ig > 1
        hold on;
    end
    scatter(scoreAcoustMeans(indGrp,1),scoreAcoustMeans(indGrp,2), 10, 'MarkerFaceColor', colorplot(ig, :), 'MarkerEdgeColor', colorplot(ig, :));
    hold off;
end

% Add contour lines
% hold on
% for ic = 1:icluster
%     ezcontour(@(x,y)mvnpdf([x y], [gmList{icluster}.mu(ic,1) gmList{icluster}.mu(ic,2)], gmList{icluster}.Sigma(1:2,1:2,ic)), ...
%         [gmList{icluster}.mu(ic,1)-2 gmList{icluster}.mu(ic,1)+2 gmList{icluster}.mu(ic,2)-2 gmList{icluster}.mu(ic,2)+2], 60);
% end
% hold off

hold on
ang = 0:pi/20:2*pi;
circ = [cos(ang); sin(ang)];   % x,y coordinates of unit circle  
maxAmp = max(gm.ComponentProportion);
for ic = 1:icluster
    compAmp = gm.ComponentProportion(ic);
    mu1 = gm.mu(ic,1);
    mu2 = gm.mu(ic,2);
    sig1 = gm.Sigma(1,1,ic);
    sig2 = gm.Sigma(2,2,ic);
    sig12 = gm.Sigma(1,2,ic);
    covMat = [sig1, sig12; sig12, sig2];

    [V, D] = eig(covMat);
    Dsd = sqrt(D);
    
    X = V*Dsd*circ;
    
    scatter(mu1, mu2, fix(60*compAmp/maxAmp), 'k', 'filled');
    plot( mu1 + X(1,:), mu2 + X(2,:) ,'-', 'LineWidth', 2, 'Color', [0.5 0.5 0.5]);
    
end
hold off
xlabel('PC1');
ylabel('PC2');
axis([-5 5 -5 5]);

subplot(1,2,2)
for ig=1:ngroups
    indGrp =  find( strcmp(vocTypeCutsMeans, nameGrp(ig)) );
    if ig > 1
        hold on;
    end
    scatter(scoreAcoustMeans(indGrp,1),scoreAcoustMeans(indGrp,3), 10, 'MarkerFaceColor', colorplot(ig, :), 'MarkerEdgeColor', colorplot(ig, :));
    hold off;
end
% Add contour lines
% hold on
% for ic = 1:icluster
%     ezcontour(@(x,y)mvnpdf([x y], [gmListnoSo{icluster}.mu(ic,1) gmListnoSo{icluster}.mu(ic,3)],...
%         [gmListnoSo{icluster}.Sigma(1,1,ic), gmListnoSo{icluster}.Sigma(1,3,ic); gmListnoSo{icluster}.Sigma(3,1,ic), gmListnoSo{icluster}.Sigma(3,3,ic)] ),...
%         [gmListnoSo{icluster}.mu(ic,1)-2 gmListnoSo{icluster}.mu(ic,1)+2 gmListnoSo{icluster}.mu(ic,3)-2 gmListnoSo{icluster}.mu(ic,3)+2], 60);
% end
% hold off
hold on
ang = 0:pi/20:2*pi;
circ = [cos(ang); sin(ang)];   % x,y coordinates of unit circle  
maxAmp = max(gm.ComponentProportion);

for ic = 1:icluster
    compAmp = gm.ComponentProportion(ic);
    mu1 = gm.mu(ic,1);
    mu2 = gm.mu(ic,3);
    sig1 = gm.Sigma(1,1,ic);
    sig2 = gm.Sigma(3,3,ic);
    sig12 = gm.Sigma(1,3,ic);
    covMat = [sig1, sig12; sig12, sig2];
    [V, D] = eig(covMat);
    Dsd = sqrt(D);
    
    X = V*Dsd*circ;
    
    scatter(mu1, mu2, fix(60*compAmp/maxAmp), 'k', 'filled');
    plot( mu1 + X(1,:), mu2 + X(2,:) ,'-', 'LineWidth', 2, 'Color', [0.5 0.5 0.5]);
    
end
hold off

xlabel('PC1');
ylabel('PC3');
axis([-5 5 -5 5]);   

%% Repeat clustering with only Thuck and Tucks calls to see if there are two vs one group

indTT = find( strcmp(vocTypeCuts, 'Th') | strcmp(vocTypeCuts, 'Tu' ) );
AcoustTT = Acoust(indTT,:);
vocTypeCutsTT = vocTypeCuts(indTT);


zAcoustTT = zscore(AcoustTT);
[pcAcoustTT, scoreAcoustTT, eigenvalTT] = princomp(zAcoustTT);  % Use pca so that we can used weighted observations as well

figure(15);
plot(100*cumsum(eigenvalTT)./sum(eigenvalTT));
xlabel('Number of PCs');
ylabel('Variance Explained %');
title('PCs on Acoustical Features');

nClusters = 5;
nPCs = 10;

gmListTT = cell(1, nClusters);
aicListTT = zeros(1, nClusters);
bicListTT = zeros(1, nClusters);

options = statset('Display','final', 'MaxIter', 1000);  % Display final results

parfor ic = 1:nClusters    
    gmListTT{ic} = fitgmdist(scoreAcoustTT(:, 1:nPCs), ic, 'Options', options, 'Replicates', 5);  % Gaussian mixture model
    aicListTT(ic) = gmListTT{ic}.AIC;
    bicListTT(ic) = gmListTT{ic}.BIC;
end

figure(16);
subplot(1,2,1);
plot(1:nClusters, aicListTT - min(aicListTT), 'k');
xlabel('Number of Gaussians');
ylabel('AIC');

subplot(1,2,2);
plot(1:nClusters, bicListTT, 'r');
xlabel('Number of Gaussians');
ylabel('BIC');

%
figure(17);

icluster = 2;


idx = cluster(gmListTT{icluster} , scoreAcoustTT(:,1:nPCs));

countTypePer = zeros(icluster, ngroups);
countType = zeros(icluster, ngroups);
countTypeTot = zeros(1, ngroups);
for ig=1:ngroups
    indGrp =  find( strcmp(vocTypeCutsTT, nameGrp(ig)) );
    countTypeTot(ig) = length(indGrp);
end

for ic = 1:icluster
    clusteridx = find(idx == ic);
    fprintf(1,'Cluster %d:\n', ic);
    for ig=1:ngroups
        if countTypeTot(ig) == 0
            continue;
        end
        indGrp =  find( strcmp(vocTypeCutsTT(clusteridx), nameGrp(ig)) );
        countType(ic,ig) = length(indGrp);
        countTypePer(ic,ig) = 100.0*countType(ic,ig)./countTypeTot(ig);
        fprintf(1,'\t%s %d (%.2f%%)\n', nameGrp{ig}, countType(ic,ig), countTypePer(ic,ig));
    end
end

bh = bar(countTypePer(:,8:9));   % 8 and 9 are Thucks and Tucks
for ibh=1:2
    set(bh(ibh), 'EdgeColor', colorplot(ibh+7, :), 'FaceColor', colorplot(ibh+7, :));
    text(0.85+(ibh-1)*0.3, countTypePer(1,ibh+7) + 3, sprintf('%d',countType(1, ibh+7)));
    text(1.85+(ibh-1)*0.3, countTypePer(2,ibh+7) + 3, sprintf('%d',countType(2, ibh+7)));
end
legend(nameGrp(8:9));
axisVal = axis();
axisVal(4) = 100;
axis(axisVal);

% Test of two proportions
n1 = countType(1, 8) + countType(1, 9);
n2 = countType(2, 8) + countType(2, 9);
p1est = countType(1, 8)/n1;
p2est = countType(2, 8)/n2;
pest = (countType(1, 8) + countType(2, 8))/(n1 + n2);
zval = (p1est-p2est)./sqrt(pest*(1-pest)*(1/n1+1/n2));
pd = makedist('Normal');
pval = 2*(1-cdf(pd, abs(zval)));

title(sprintf('Proportion Test zval=%.2f pval=%.4f', zval, pval));

% Plot the data using 1rst and 2nd PCs
figure(18);
subplot(1,2,1)
for ig=8:9
    indGrp =  find( strcmp(vocTypeCutsTT, nameGrp(ig)) );
    if ig > 8
        hold on;
    end
    scatter(scoreAcoustTT(indGrp,1),scoreAcoustTT(indGrp,2), 8, 'MarkerFaceColor', colorplot(ig, :), 'MarkerEdgeColor', colorplot(ig, :));
    hold off;
end

% Add contour lines
% hold on
% for ic = 1:icluster
%     ezcontour(@(x,y)mvnpdf([x y], [gmList{icluster}.mu(ic,1) gmList{icluster}.mu(ic,2)], gmList{icluster}.Sigma(1:2,1:2,ic)), ...
%         [gmList{icluster}.mu(ic,1)-2 gmList{icluster}.mu(ic,1)+2 gmList{icluster}.mu(ic,2)-2 gmList{icluster}.mu(ic,2)+2], 60);
% end
% hold off

hold on
ang = 0:pi/20:2*pi;
circ = [cos(ang); sin(ang)];   % x,y coordinates of unit circle  
maxAmp = max(gmListTT{icluster}.ComponentProportion);
for ic = 1:icluster
    compAmp = gmListTT{icluster}.ComponentProportion(ic);
    mu1 = gmListTT{icluster}.mu(ic,1);
    mu2 = gmListTT{icluster}.mu(ic,2);
    sig1 = gmListTT{icluster}.Sigma(1,1,ic);
    sig2 = gmListTT{icluster}.Sigma(2,2,ic);
    sig12 = gmListTT{icluster}.Sigma(1,2,ic);
    covMat = [sig1, sig12; sig12, sig2];

    [V, D] = eig(covMat);
    Dsd = sqrt(D);
    
    X = V*Dsd*circ;
    
    scatter(mu1, mu2, fix(60*compAmp/maxAmp), 'k', 'filled');
    plot( mu1 + X(1,:), mu2 + X(2,:) ,'-', 'LineWidth', 2, 'Color', [0.5 0.5 0.5]);
    
end
hold off
xlabel('PC1');
ylabel('PC2');
axis([-5 5 -5 5]);

subplot(1,2,2)
for ig=8:9
    indGrp =  find( strcmp(vocTypeCutsTT, nameGrp(ig)) );
    if ig > 8
        hold on;
    end
    scatter(scoreAcoustTT(indGrp,1),scoreAcoustTT(indGrp,3),8, 'MarkerFaceColor', colorplot(ig, :), 'MarkerEdgeColor', colorplot(ig, :));
    hold off;
end
% Add contour lines
% hold on
% for ic = 1:icluster
%     ezcontour(@(x,y)mvnpdf([x y], [gmListTT{icluster}.mu(ic,1) gmListTT{icluster}.mu(ic,3)],...
%         [gmListTT{icluster}.Sigma(1,1,ic), gmListTT{icluster}.Sigma(1,3,ic); gmListTT{icluster}.Sigma(3,1,ic), gmListTT{icluster}.Sigma(3,3,ic)] ),...
%         [gmListTT{icluster}.mu(ic,1)-2 gmListTT{icluster}.mu(ic,1)+2 gmListTT{icluster}.mu(ic,3)-2 gmListTT{icluster}.mu(ic,3)+2], 60);
% end
% hold off
hold on
ang = 0:pi/20:2*pi;
circ = [cos(ang); sin(ang)];   % x,y coordinates of unit circle  
maxAmp = max(gmListTT{icluster}.ComponentProportion);

for ic = 1:icluster
    compAmp = gmListTT{icluster}.ComponentProportion(ic);
    mu1 = gmListTT{icluster}.mu(ic,1);
    mu2 = gmListTT{icluster}.mu(ic,3);
    sig1 = gmListTT{icluster}.Sigma(1,1,ic);
    sig2 = gmListTT{icluster}.Sigma(3,3,ic);
    sig12 = gmListTT{icluster}.Sigma(1,3,ic);
    covMat = [sig1, sig12; sig12, sig2];
    [V, D] = eig(covMat);
    Dsd = sqrt(D);
    
    X = V*Dsd*circ;
    
    scatter(mu1, mu2, fix(60*compAmp/maxAmp), 'k', 'filled');
    plot( mu1 + X(1,:), mu2 + X(2,:) ,'-', 'LineWidth', 2, 'Color', [0.5 0.5 0.5]);
    
end
hold off

xlabel('PC1');
ylabel('PC3');
axis([-5 5 -5 5]);
title(sprintf('P1 = %.2f P2 = %.2f', gmListTT{icluster}.ComponentProportion(1), gmListTT{icluster}.ComponentProportion(2)));

%% Now just with Nest Calls
indNe = find( strcmp(vocTypeCuts, 'Ne')  );
AcoustNe = Acoust(indNe,:);
vocTypeCutsNe = vocTypeCuts(indNe);


zAcoustNe = zscore(AcoustNe);
[pcAcoustNe, scoreAcoustNe, eigenvalNe] = princomp(zAcoustNe);  % Use pca so that we can used weighted observations as well

figure(19);
plot(100*cumsum(eigenvalNe)./sum(eigenvalNe));
xlabel('Number of PCs');
ylabel('Variance Explained %');
title('PCs on Acoustical Features');

nClusters = 10;
nPCs = 10;

gmListNe = cell(1, nClusters);
aicListNe = zeros(1, nClusters);
bicListNe = zeros(1, nClusters);

options = statset('Display','final', 'MaxIter', 1000);  % Display final results

parfor ic = 1:nClusters    
    gmListNe{ic} = fitgmdist(scoreAcoustNe(:, 1:nPCs), ic, 'Options', options, 'Replicates', 5);  % Gaussian mixture model
    aicListNe(ic) = gmListNe{ic}.AIC;
    bicListNe(ic) = gmListNe{ic}.BIC;
end

figure(20);
subplot(1,2,1);
plot(1:nClusters, aicListNe - min(aicListNe), 'k');
xlabel('Number of Gaussians');
ylabel('AIC');

subplot(1,2,2);
plot(1:nClusters, bicListNe, 'r');
xlabel('Number of Gaussians');
ylabel('BIC');


% Plot the data using 1rst and 2nd PCs
icluster = 2;
figure(21);
subplot(1,2,1)
ig=3;           % Corresponds to Nest calls
scatter(scoreAcoustNe(:,1),scoreAcoustNe(:,2),8, 'MarkerFaceColor', colorplot(ig, :), 'MarkerEdgeColor', colorplot(ig, :));


% Add contour lines
% hold on
% for ic = 1:icluster
%     ezcontour(@(x,y)mvnpdf([x y], [gmList{icluster}.mu(ic,1) gmList{icluster}.mu(ic,2)], gmList{icluster}.Sigma(1:2,1:2,ic)), ...
%         [gmList{icluster}.mu(ic,1)-2 gmList{icluster}.mu(ic,1)+2 gmList{icluster}.mu(ic,2)-2 gmList{icluster}.mu(ic,2)+2], 60);
% end
% hold off

hold on
ang = 0:pi/20:2*pi;
circ = [cos(ang); sin(ang)];   % x,y coordinates of unit circle  
maxAmp = max(gmListNe{icluster}.ComponentProportion);
for ic = 1:icluster
    compAmp = gmListNe{icluster}.ComponentProportion(ic);
    mu1 = gmListNe{icluster}.mu(ic,1);
    mu2 = gmListNe{icluster}.mu(ic,2);
    sig1 = gmListNe{icluster}.Sigma(1,1,ic);
    sig2 = gmListNe{icluster}.Sigma(2,2,ic);
    sig12 = gmListNe{icluster}.Sigma(1,2,ic);
    covMat = [sig1, sig12; sig12, sig2];

    [V, D] = eig(covMat);
    Dsd = sqrt(D);
    
    X = V*Dsd*circ;
    
    scatter(mu1, mu2, fix(60*compAmp/maxAmp), 'k', 'filled');
    plot( mu1 + X(1,:), mu2 + X(2,:) ,'-', 'LineWidth', 2, 'Color', [0.5 0.5 0.5]);
    
end
hold off
xlabel('PC1');
ylabel('PC2');
axis([-5 5 -5 5]);

subplot(1,2,2)
ig=3;           % Corresponds to Nest calls
scatter(scoreAcoustNe(:,1),scoreAcoustNe(:,3),8, 'MarkerFaceColor', colorplot(ig, :), 'MarkerEdgeColor', colorplot(ig, :));

% Add contour lines
% hold on
% for ic = 1:icluster
%     ezcontour(@(x,y)mvnpdf([x y], [gmListNe{icluster}.mu(ic,1) gmListNe{icluster}.mu(ic,3)],...
%         [gmListNe{icluster}.Sigma(1,1,ic), gmListNe{icluster}.Sigma(1,3,ic); gmListNe{icluster}.Sigma(3,1,ic), gmListNe{icluster}.Sigma(3,3,ic)] ),...
%         [gmListNe{icluster}.mu(ic,1)-2 gmListNe{icluster}.mu(ic,1)+2 gmListNe{icluster}.mu(ic,3)-2 gmListNe{icluster}.mu(ic,3)+2], 60);
% end
% hold off
hold on
ang = 0:pi/20:2*pi;
circ = [cos(ang); sin(ang)];   % x,y coordinates of unit circle  
maxAmp = max(gmListNe{icluster}.ComponentProportion);

for ic = 1:icluster
    compAmp = gmListNe{icluster}.ComponentProportion(ic);
    mu1 = gmListNe{icluster}.mu(ic,1);
    mu2 = gmListNe{icluster}.mu(ic,3);
    sig1 = gmListNe{icluster}.Sigma(1,1,ic);
    sig2 = gmListNe{icluster}.Sigma(3,3,ic);
    sig12 = gmListNe{icluster}.Sigma(1,3,ic);
    covMat = [sig1, sig12; sig12, sig2];
    [V, D] = eig(covMat);
    Dsd = sqrt(D);
    
    X = V*Dsd*circ;
    
    scatter(mu1, mu2, fix(60*compAmp/maxAmp), 'k', 'filled');
    plot( mu1 + X(1,:), mu2 + X(2,:) ,'-', 'LineWidth', 2, 'Color', [0.5 0.5 0.5]);
    
end
hold off

xlabel('PC1');
ylabel('PC3');
axis([-5 5 -5 5]);
title(sprintf('P1 = %.2f P2 = %.2f', gmListNe{icluster}.ComponentProportion(1), gmListNe{icluster}.ComponentProportion(2)));


%% Now just with Tets Calls
indTe = find( strcmp(vocTypeCuts, 'Te')  );
AcoustTe = Acoust(indTe,:);
vocTypeCutsTe = vocTypeCuts(indTe);
birdSexCutsTe = birdSexCuts(indTe);


zAcoustTe = zscore(AcoustTe);
[pcAcoustTe, scoreAcoustTe, eigenvalTe] = princomp(zAcoustTe);  % Use pca so that we can used weighted observations as well

figure(22);
plot(100*cumsum(eigenvalTe)./sum(eigenvalTe));
xlabel('Number of PCs');
ylabel('Variance Explained %');
title('PCs on Acoustical Features');

nClusters = 10;
nPCs = 10;

gmListTe = cell(1, nClusters);
aicListTe = zeros(1, nClusters);
bicListTe = zeros(1, nClusters);

options = statset('Display','final', 'MaxIter', 1000);  % Display final results

parfor ic = 1:nClusters    
    gmListTe{ic} = fitgmdist(scoreAcoustTe(:, 1:nPCs), ic, 'Options', options, 'Replicates', 5);  % Gaussian mixture model
    aicListTe(ic) = gmListTe{ic}.AIC;
    bicListTe(ic) = gmListTe{ic}.BIC;
end

figure(23);
subplot(1,2,1);
plot(1:nClusters, aicListTe - min(aicListTe), 'k');
xlabel('Number of Gaussians');
ylabel('AIC');

subplot(1,2,2);
plot(1:nClusters, bicListTe, 'r');
xlabel('Number of Gaussians');
ylabel('BIC');


% Plot the data using 1rst and 2nd PCs
icluster = 2;
figure(24);
subplot(1,2,1)
ig=4;           % Corresponds to Tets calls
scatter(scoreAcoustTe(:,1),scoreAcoustTe(:,2),8, 'MarkerFaceColor', colorplot(ig, :), 'MarkerEdgeColor', colorplot(ig, :));


% Add contour lines
% hold on
% for ic = 1:icluster
%     ezcontour(@(x,y)mvnpdf([x y], [gmList{icluster}.mu(ic,1) gmList{icluster}.mu(ic,2)], gmList{icluster}.Sigma(1:2,1:2,ic)), ...
%         [gmList{icluster}.mu(ic,1)-2 gmList{icluster}.mu(ic,1)+2 gmList{icluster}.mu(ic,2)-2 gmList{icluster}.mu(ic,2)+2], 60);
% end
% hold off

hold on
ang = 0:pi/20:2*pi;
circ = [cos(ang); sin(ang)];   % x,y coordinates of unit circle  
maxAmp = max(gmListTe{icluster}.ComponentProportion);
for ic = 1:icluster
    compAmp = gmListTe{icluster}.ComponentProportion(ic);
    mu1 = gmListTe{icluster}.mu(ic,1);
    mu2 = gmListTe{icluster}.mu(ic,2);
    sig1 = gmListTe{icluster}.Sigma(1,1,ic);
    sig2 = gmListTe{icluster}.Sigma(2,2,ic);
    sig12 = gmListTe{icluster}.Sigma(1,2,ic);
    covMat = [sig1, sig12; sig12, sig2];

    [V, D] = eig(covMat);
    Dsd = sqrt(D);
    
    X = V*Dsd*circ;
    
    scatter(mu1, mu2, fix(60*compAmp/maxAmp), 'k', 'filled');
    plot( mu1 + X(1,:), mu2 + X(2,:) ,'-', 'LineWidth', 2, 'Color', [0.5 0.5 0.5]);
    
end
hold off
xlabel('PC1');
ylabel('PC2');
axis([-5 5 -5 5]);

subplot(1,2,2)
ig=4;           % Corresponds to Tet calls
scatter(scoreAcoustTe(:,1),scoreAcoustTe(:,3),8, 'MarkerFaceColor', colorplot(ig, :), 'MarkerEdgeColor', colorplot(ig, :));

% Add contour lines
% hold on
% for ic = 1:icluster
%     ezcontour(@(x,y)mvnpdf([x y], [gmListTe{icluster}.mu(ic,1) gmListTe{icluster}.mu(ic,3)],...
%         [gmListTe{icluster}.Sigma(1,1,ic), gmListTe{icluster}.Sigma(1,3,ic); gmListTe{icluster}.Sigma(3,1,ic), gmListTe{icluster}.Sigma(3,3,ic)] ),...
%         [gmListTe{icluster}.mu(ic,1)-2 gmListTe{icluster}.mu(ic,1)+2 gmListTe{icluster}.mu(ic,3)-2 gmListTe{icluster}.mu(ic,3)+2], 60);
% end
% hold off
hold on
ang = 0:pi/20:2*pi;
circ = [cos(ang); sin(ang)];   % x,y coordinates of unit circle  
maxAmp = max(gmListTe{icluster}.ComponentProportion);

for ic = 1:icluster
    compAmp = gmListTe{icluster}.ComponentProportion(ic);
    mu1 = gmListTe{icluster}.mu(ic,1);
    mu2 = gmListTe{icluster}.mu(ic,3);
    sig1 = gmListTe{icluster}.Sigma(1,1,ic);
    sig2 = gmListTe{icluster}.Sigma(3,3,ic);
    sig12 = gmListTe{icluster}.Sigma(1,3,ic);
    covMat = [sig1, sig12; sig12, sig2];
    [V, D] = eig(covMat);
    Dsd = sqrt(D);
    
    X = V*Dsd*circ;
    
    scatter(mu1, mu2, fix(60*compAmp/maxAmp), 'k', 'filled');
    plot( mu1 + X(1,:), mu2 + X(2,:) ,'-', 'LineWidth', 2, 'Color', [0.5 0.5 0.5]);
    
end
hold off

xlabel('PC1');
ylabel('PC3');
axis([-5 5 -5 5]);
title(sprintf('P1 = %.2f P2 = %.2f', gmListTe{icluster}.ComponentProportion(1), gmListTe{icluster}.ComponentProportion(2)));

figure(25);
idx = cluster(gmListTe{icluster} , scoreAcoustTe(:,1:nPCs));

countTypePer = zeros(icluster, 2);
countType = zeros(icluster, 2);
countTypeTot = zeros(1, 2);
sexName = {'F', 'M'};
for ig=1:2
    indGrp =  find( strcmp(birdSexCutsTe, sexName(ig)) );
    countTypeTot(ig) = length(indGrp);
end

for ic = 1:icluster
    clusteridx = find(idx == ic);
    fprintf(1,'Cluster %d:\n', ic);
    for ig=1:2
        if countTypeTot(ig) == 0
            continue;
        end
        indGrp =  find( strcmp(birdSexCutsTe(clusteridx), sexName(ig)) );
        countType(ic,ig) = length(indGrp);
        countTypePer(ic,ig) = 100.0*countType(ic,ig)./countTypeTot(ig);
        fprintf(1,'\t%s %d (%.2f%%)\n', sexName{ig}, countType(ic,ig), countTypePer(ic,ig));
    end
end

bh = bar(countTypePer(:,1:2));   % 8 and 9 are Thucks and Tucks
for ibh=1:2
    text(0.85+(ibh-1)*0.3, countTypePer(1,ibh) + 3, sprintf('%d',countType(1, ibh)));
    text(1.85+(ibh-1)*0.3, countTypePer(2,ibh) + 3, sprintf('%d',countType(2, ibh)));
end
legend(sexName);
axisVal = axis();
axisVal(4) = 100;
axis(axisVal);

% Test of two proportions
n1 = countType(1, 1) + countType(1, 2);
n2 = countType(2, 1) + countType(2, 2);
p1est = countType(1, 1)/n1;
p2est = countType(2, 1)/n2;
pest = (countType(1, 1) + countType(2, 1))/(n1 + n2);
zval = (p1est-p2est)./sqrt(pest*(1-pest)*(1/n1+1/n2));
pd = makedist('Normal');
pval = 2*(1-cdf(pd, abs(zval)));

title(sprintf('Proportion Test zval=%.2f pval=%.4f', zval, pval));

%% Now just with Tets Calls from Males and Females separated.
indTeM = find( strcmp(vocTypeCuts, 'Te') & strcmp(birdSexCuts,'M') );
AcoustTeM = Acoust(indTeM,:);
vocTypeCutsTeM = vocTypeCuts(indTeM);
birdNameCutsTeM = birdNameCuts(indTeM);
birdNamesTeM = unique(birdNameCutsTeM);
nbirdsTeM = length(birdNamesTeM);

zAcoustTeM = zscore(AcoustTeM);
[pcAcoustTeM, scoreAcoustTeM, eigenvalTeM] = princomp(zAcoustTeM);  % Use pca so that we can used weighted observations as well

figure(26);
plot(100*cumsum(eigenvalTeM)./sum(eigenvalTeM));
xlabel('Number of PCs');
ylabel('Variance Explained %');
title('PCs on Acoustical Features');

nClusters = 10;
nPCs = 10;

gmListTeM = cell(1, nClusters);
aicListTeM = zeros(1, nClusters);
bicListTeM = zeros(1, nClusters);

options = statset('Display','final', 'MaxIter', 1000);  % Display final results

for ic = 1:nClusters    
    gmListTeM{ic} = fitgmdist(scoreAcoustTeM(:, 1:nPCs), ic, 'Options', options, 'Replicates', 5);  % Gaussian mixture model
    aicListTeM(ic) = gmListTeM{ic}.AIC;
    bicListTeM(ic) = gmListTeM{ic}.BIC;
end

figure(27);
subplot(1,2,1);
plot(1:nClusters, aicListTeM - min(aicListTeM), 'k');
xlabel('Number of Gaussians');
ylabel('AIC');

subplot(1,2,2);
plot(1:nClusters, bicListTeM, 'r');
xlabel('Number of Gaussians');
ylabel('BIC');
title('Male Tets only');


% Plot the data using 1rst and 2nd PCs
icluster = 2;
figure(28);
subplot(1,2,1)
ig=4;           % Corresponds to Tets calls
% scatter(scoreAcoustTeM(:,1),scoreAcoustTeM(:,2),8, 'MarkerFaceColor', colorplot(ig, :), 'MarkerEdgeColor', colorplot(ig, :));

% make a loop to color by bird
cmap = colormap;
for ibird = 1:nbirdsTeM
    icolor = floor(1 + 64*(ibird-1)/nbirdsTeM);
    indBird = find(strcmp(birdNameCutsTeM, birdNamesTeM{ibird}));
    
    scatter(scoreAcoustTeM(indBird,1),scoreAcoustTeM(indBird,2),8, 'MarkerFaceColor', cmap(icolor, :), 'MarkerEdgeColor', cmap(icolor, :));
    hold on;
end
hold off;

% Add contour lines
% hold on
% for ic = 1:icluster
%     ezcontour(@(x,y)mvnpdf([x y], [gmList{icluster}.mu(ic,1) gmList{icluster}.mu(ic,2)], gmList{icluster}.Sigma(1:2,1:2,ic)), ...
%         [gmList{icluster}.mu(ic,1)-2 gmList{icluster}.mu(ic,1)+2 gmList{icluster}.mu(ic,2)-2 gmList{icluster}.mu(ic,2)+2], 60);
% end
% hold off

hold on
ang = 0:pi/20:2*pi;
circ = [cos(ang); sin(ang)];   % x,y coordinates of unit circle  
maxAmp = max(gmListTeM{icluster}.ComponentProportion);
for ic = 1:icluster
    compAmp = gmListTeM{icluster}.ComponentProportion(ic);
    mu1 = gmListTeM{icluster}.mu(ic,1);
    mu2 = gmListTeM{icluster}.mu(ic,2);
    sig1 = gmListTeM{icluster}.Sigma(1,1,ic);
    sig2 = gmListTeM{icluster}.Sigma(2,2,ic);
    sig12 = gmListTeM{icluster}.Sigma(1,2,ic);
    covMat = [sig1, sig12; sig12, sig2];

    [V, D] = eig(covMat);
    Dsd = sqrt(D);
    
    X = V*Dsd*circ;
    
    scatter(mu1, mu2, fix(60*compAmp/maxAmp), 'k', 'filled');
    plot( mu1 + X(1,:), mu2 + X(2,:) ,'-', 'LineWidth', 2, 'Color', [0.5 0.5 0.5]);
    
end
hold off
xlabel('PC1');
ylabel('PC2');
axis([-5 5 -5 5]);

subplot(1,2,2)
ig=4;           % Corresponds to Tet calls
% scatter(scoreAcoustTeM(:,1),scoreAcoustTeM(:,3),8, 'MarkerFaceColor', colorplot(ig, :), 'MarkerEdgeColor', colorplot(ig, :));

% make a loop to color by bird
cmap = colormap;
for ibird = 1:nbirdsTeM
    icolor = floor(1 + 64*(ibird-1)/nbirdsTeM);
    indBird = find(strcmp(birdNameCutsTeM, birdNamesTeM{ibird}));
    
    scatter(scoreAcoustTeM(indBird,1),scoreAcoustTeM(indBird,3),8, 'MarkerFaceColor', cmap(icolor, :), 'MarkerEdgeColor', cmap(icolor, :));
    hold on;
end
hold off;

hold on
ang = 0:pi/20:2*pi;
circ = [cos(ang); sin(ang)];   % x,y coordinates of unit circle  
maxAmp = max(gmListTeM{icluster}.ComponentProportion);

for ic = 1:icluster
    compAmp = gmListTeM{icluster}.ComponentProportion(ic);
    mu1 = gmListTeM{icluster}.mu(ic,1);
    mu2 = gmListTeM{icluster}.mu(ic,3);
    sig1 = gmListTeM{icluster}.Sigma(1,1,ic);
    sig2 = gmListTeM{icluster}.Sigma(3,3,ic);
    sig12 = gmListTeM{icluster}.Sigma(1,3,ic);
    covMat = [sig1, sig12; sig12, sig2];
    [V, D] = eig(covMat);
    Dsd = sqrt(D);
    
    X = V*Dsd*circ;
    
    scatter(mu1, mu2, fix(60*compAmp/maxAmp), 'k', 'filled');
    plot( mu1 + X(1,:), mu2 + X(2,:) ,'-', 'LineWidth', 2, 'Color', [0.5 0.5 0.5]);
    
end
hold off

xlabel('PC1');
ylabel('PC3');
axis([-5 5 -5 5]);
title(sprintf('P1 = %.2f P2 = %.2f', gmListTeM{icluster}.ComponentProportion(1), gmListTeM{icluster}.ComponentProportion(2)));


% Look at some acoustical differences inclusters
figure(29);
idx = cluster(gmListTeM{icluster} , scoreAcoustTeM(:,1:nPCs));

% Print out the proportion of clusters for each bird
for ibird = 1:nbirdsTeM
    indBird = find(strcmp(birdNameCutsTeM, birdNamesTeM{ibird}));
    
    ncallsBird = length(indBird);
    fprintf(1, 'Bird %s Ncalls = %d\t', birdNamesTeM{ibird}, ncallsBird);
    for ic=1:icluster
        indBirdCluster = find(strcmp(birdNameCutsTeM(idx == ic), birdNamesTeM{ibird}));
        fprintf(1, 'P(%d) = %.1f%%\t', ic, 100.0*length(indBirdCluster)/ncallsBird);
    end
    fprintf(1,'\n');
end
        

% Let's calculate means, sem and t-test for sdtime, cvf0, fund, spectral mean and rms 

testingFeatures = [6,7,16,20];
plottingFactorFeatures = [1 0.001 1 1000];
nFeatures = length(testingFeatures);
meanValFeatures = zeros(icluster, nFeatures);
semValFeatures = zeros(icluster, nFeatures);

for ifeat=1:nFeatures
    id = testingFeatures(ifeat);
    fprintf(1, 'Differences for %s\n', xtagPlot{id});
    [meanValFeatures(:,ifeat), semValFeatures(:, ifeat)] = grpstats(AcoustTeM(:, id), idx, {'mean', 'sem'});
    [h,p,ci,stats] = ttest2(AcoustTeM(idx==1, id), AcoustTeM(idx==2, id));
    fprintf(1, 'Mean 1: %f +- %f Mean 2: %f +- %f t(%d)=%.2f p = %.4f\n', meanValFeatures(1, ifeat), semValFeatures(1, ifeat),...
        meanValFeatures(2, ifeat), semValFeatures(2, ifeat), stats.df, stats.tstat, p );
    
    subplot(2, nFeatures/2, ifeat);
    bar(meanValFeatures(:,ifeat).*plottingFactorFeatures(ifeat));
    hold on;
    errorbar(1:icluster, meanValFeatures(:,ifeat).*plottingFactorFeatures(ifeat),...
        semValFeatures(:, ifeat).*plottingFactorFeatures(ifeat), 'r+', 'LineWidth', 2);
    title(xtagPlot{id});
    if (p < 0.001)
        text((1+icluster)/2, 1.05*max(meanValFeatures(:, ifeat))*plottingFactorFeatures(ifeat) , '***');
    elseif (p < 0.01)
        text((1+icluster)/2, 1.05*max(meanValFeatures(:, ifeat))*plottingFactorFeatures(ifeat) , '**');
    elseif (p <0.05)
        text((1+icluster)/2, 1.05*max(meanValFeatures(:, ifeat))*plottingFactorFeatures(ifeat) , '*');
    end
    hold off;
    axis([0 3 0.7*min(meanValFeatures(:, ifeat))*plottingFactorFeatures(ifeat) 1.2*max(meanValFeatures(:, ifeat))*plottingFactorFeatures(ifeat)]);
        
end

%% Now just with Tets Calls from  Females.
indTeF = find( strcmp(vocTypeCuts, 'Te') & strcmp(birdSexCuts,'F') );
AcoustTeF = Acoust(indTeF,:);
vocTypeCutsTeF = vocTypeCuts(indTeF);
birdNameCutsTeF = birdNameCuts(indTeF);
birdNamesTeF = unique(birdNameCutsTeF);
nbirdsTeF = length(birdNamesTeF);

zAcoustTeF = zscore(AcoustTeF);
[pcAcoustTeF, scoreAcoustTeF, eigenvalTeF] = princomp(zAcoustTeF);  % Use pca so that we can used weighted observations as well

figure(30);
plot(100*cumsum(eigenvalTeF)./sum(eigenvalTeF));
xlabel('Number of PCs');
ylabel('Variance Explained %');
title('PCs on Acoustical Features');

nClusters = 10;
nPCs = 10;

gmListTeF = cell(1, nClusters);
aicListTeF = zeros(1, nClusters);
bicListTeF = zeros(1, nClusters);

options = statset('Display','final', 'MaxIter', 1000);  % Display final results

parfor ic = 1:nClusters    
    gmListTeF{ic} = fitgmdist(scoreAcoustTeF(:, 1:nPCs), ic, 'Options', options, 'Replicates', 5);  % Gaussian mixture model
    aicListTeF(ic) = gmListTeF{ic}.AIC;
    bicListTeF(ic) = gmListTeF{ic}.BIC;
end

figure(31);
subplot(1,2,1);
plot(1:nClusters, aicListTeF - min(aicListTeF), 'k');
xlabel('Number of Gaussians');
ylabel('AIC');

subplot(1,2,2);
plot(1:nClusters, bicListTeF, 'r');
xlabel('Number of Gaussians');
ylabel('BIC');
title('Female Tets only');


% Plot the data using 1rst and 2nd PCs
icluster = 2;
figure(32);
subplot(1,2,1)
ig=4;           % Corresponds to Tets calls
%scatter(scoreAcoustTeF(:,1),scoreAcoustTeF(:,2),8, 'MarkerFaceColor', colorplot(ig, :), 'MarkerEdgeColor', colorplot(ig, :));

% make a loop to color by bird
cmap = colormap;
for ibird = 1:nbirdsTeF
    icolor = floor(1 + 64*(ibird-1)/nbirdsTeF);
    indBird = find(strcmp(birdNameCutsTeF, birdNamesTeF{ibird}));
    
    scatter(scoreAcoustTeF(indBird,1),scoreAcoustTeF(indBird,2),8, 'MarkerFaceColor', cmap(icolor, :), 'MarkerEdgeColor', cmap(icolor, :));
    hold on;
end
hold off;

hold on
ang = 0:pi/20:2*pi;
circ = [cos(ang); sin(ang)];   % x,y coordinates of unit circle  
maxAmp = max(gmListTeF{icluster}.ComponentProportion);
for ic = 1:icluster
    compAmp = gmListTeF{icluster}.ComponentProportion(ic);
    mu1 = gmListTeF{icluster}.mu(ic,1);
    mu2 = gmListTeF{icluster}.mu(ic,2);
    sig1 = gmListTeF{icluster}.Sigma(1,1,ic);
    sig2 = gmListTeF{icluster}.Sigma(2,2,ic);
    sig12 = gmListTeF{icluster}.Sigma(1,2,ic);
    covMat = [sig1, sig12; sig12, sig2];

    [V, D] = eig(covMat);
    Dsd = sqrt(D);
    
    X = V*Dsd*circ;
    
    scatter(mu1, mu2, fix(60*compAmp/maxAmp), 'k', 'filled');
    plot( mu1 + X(1,:), mu2 + X(2,:) ,'-', 'LineWidth', 2, 'Color', [0.5 0.5 0.5]);
    
end
hold off
xlabel('PC1');
ylabel('PC2');
axis([-5 5 -5 5]);

subplot(1,2,2)
ig=4;           % Corresponds to Tet calls
scatter(scoreAcoustTeF(:,1),scoreAcoustTeF(:,3),8, 'MarkerFaceColor', colorplot(ig, :), 'MarkerEdgeColor', colorplot(ig, :));

% make a loop to color by bird
cmap = colormap;
for ibird = 1:nbirdsTeF
    icolor = floor(1 + 64*(ibird-1)/nbirdsTeF);
    indBird = find(strcmp(birdNameCutsTeF, birdNamesTeF{ibird}));
    
    scatter(scoreAcoustTeF(indBird,1),scoreAcoustTeF(indBird,3),8, 'MarkerFaceColor', cmap(icolor, :), 'MarkerEdgeColor', cmap(icolor, :));
    hold on;
end
hold off;

hold on
ang = 0:pi/20:2*pi;
circ = [cos(ang); sin(ang)];   % x,y coordinates of unit circle  
maxAmp = max(gmListTeF{icluster}.ComponentProportion);

for ic = 1:icluster
    compAmp = gmListTeF{icluster}.ComponentProportion(ic);
    mu1 = gmListTeF{icluster}.mu(ic,1);
    mu2 = gmListTeF{icluster}.mu(ic,3);
    sig1 = gmListTeF{icluster}.Sigma(1,1,ic);
    sig2 = gmListTeF{icluster}.Sigma(3,3,ic);
    sig12 = gmListTeF{icluster}.Sigma(1,3,ic);
    covMat = [sig1, sig12; sig12, sig2];
    [V, D] = eig(covMat);
    Dsd = sqrt(D);
    
    X = V*Dsd*circ;
    
    scatter(mu1, mu2, fix(60*compAmp/maxAmp), 'k', 'filled');
    plot( mu1 + X(1,:), mu2 + X(2,:) ,'-', 'LineWidth', 2, 'Color', [0.5 0.5 0.5]);
    
end
hold off

xlabel('PC1');
ylabel('PC3');
axis([-5 5 -5 5]);
title(sprintf('P1 = %.2f P2 = %.2f', gmListTeF{icluster}.ComponentProportion(1), gmListTeF{icluster}.ComponentProportion(2)));

figure(33);
idx = cluster(gmListTeF{icluster} , scoreAcoustTeF(:,1:nPCs));

% Print out the proportion of clusters for each bird
for ibird = 1:nbirdsTeF
    indBird = find(strcmp(birdNameCutsTeF, birdNamesTeF{ibird}));
    
    ncallsBird = length(indBird);
    fprintf(1, 'Bird %s Ncalls = %d\t', birdNamesTeF{ibird}, ncallsBird);
    for ic=1:icluster
        indBirdCluster = find(strcmp(birdNameCutsTeF(idx == ic), birdNamesTeF{ibird}));
        fprintf(1, 'P(%d) = %.1f%%\t', ic, 100.0*length(indBirdCluster)/ncallsBird);
    end
    fprintf(1,'\n');
end

% Let's calculate means, sem and t-test for sdtime, cvf0, fund and rms 
testingFeatures = [6,7,16,20];
plottingFactorFeatures = [1 0.001 1 1000];
nFeatures = length(testingFeatures);
meanValFeatures = zeros(icluster, nFeatures);
semValFeatures = zeros(icluster, nFeatures);

for ifeat=1:nFeatures
    id = testingFeatures(ifeat);
    fprintf(1, 'Differences for %s\n', xtagPlot{id});
    [meanValFeatures(:,ifeat), semValFeatures(:, ifeat)] = grpstats(AcoustTeF(:, id), idx, {'mean', 'sem'});
    [h,p,ci,stats] = ttest2(AcoustTeF(idx==1, id), AcoustTeF(idx==2, id));
    fprintf(1, 'Mean 1: %f +- %f Mean 2: %f +- %f t(%d)=%.2f p = %.4f\n', meanValFeatures(1, ifeat), semValFeatures(1, ifeat),...
        meanValFeatures(2, ifeat), semValFeatures(2, ifeat), stats.df, stats.tstat, p );
    
    subplot(2, nFeatures/2, ifeat);
    bar(meanValFeatures(:,ifeat).*plottingFactorFeatures(ifeat));
    hold on;
    errorbar(1:icluster, meanValFeatures(:,ifeat).*plottingFactorFeatures(ifeat),...
        semValFeatures(:, ifeat).*plottingFactorFeatures(ifeat), 'r+', 'LineWidth', 2);
    title(xtagPlot{id});
    if (p < 0.001)
        text((1+icluster)/2, 1.05*max(meanValFeatures(:, ifeat))*plottingFactorFeatures(ifeat) , '***');
    elseif (p < 0.01)
        text((1+icluster)/2, 1.05*max(meanValFeatures(:, ifeat))*plottingFactorFeatures(ifeat) , '**');
    elseif (p <0.05)
        text((1+icluster)/2, 1.05*max(meanValFeatures(:, ifeat))*plottingFactorFeatures(ifeat) , '*');
    end
    hold off;
    axis([0 3 0.7*min(meanValFeatures(:, ifeat))*plottingFactorFeatures(ifeat) 1.2*max(meanValFeatures(:, ifeat))*plottingFactorFeatures(ifeat)]);
        
end

%% Now mixing Tets and DC - first for Males

indTeDCM = find( (strcmp(vocTypeCuts, 'Te') | strcmp(vocTypeCuts, 'DC' )) &  strcmp(birdSexCuts,'M'));
AcoustTeDCM = Acoust(indTeDCM,:);
vocTypeCutsTeDCM = vocTypeCuts(indTeDCM);
birdNameCutsTeDCM = birdNameCuts(indTeDCM);
birdNamesTeDCM = unique(birdNameCutsTeDCM);
nbirdsTeDCM = length(birdNamesTeDCM);


zAcoustTeDCM = zscore(AcoustTeDCM);
[pcAcoustTeDCM, scoreAcoustTeDCM, eigenvalTeDCM] = princomp(zAcoustTeDCM);  % Use pca so that we can used weighted observations as well

figure(34);
plot(100*cumsum(eigenvalTeDCM)./sum(eigenvalTeDCM));
xlabel('Number of PCs');
ylabel('Variance Explained %');
title('PCs on Acoustical Features');

nClusters = 5;
nPCs = 10;

gmListTeDCM = cell(1, nClusters);
aicListTeDCM = zeros(1, nClusters);
bicListTeDCM = zeros(1, nClusters);

options = statset('Display','final', 'MaxIter', 1000);  % Display final results

for ic = 1:nClusters    
    gmListTeDCM{ic} = fitgmdist(scoreAcoustTeDCM(:, 1:nPCs), ic, 'Options', options, 'Replicates', 5);  % Gaussian mixture model
    aicListTeDCM(ic) = gmListTeDCM{ic}.AIC;
    bicListTeDCM(ic) = gmListTeDCM{ic}.BIC;
end

figure(35);
subplot(1,2,1);
plot(1:nClusters, aicListTeDCM - min(aicListTeDCM), 'k');
xlabel('Number of Gaussians');
ylabel('AIC');

subplot(1,2,2);
plot(1:nClusters, bicListTeDCM, 'r');
xlabel('Number of Gaussians');
ylabel('BIC');

%
figure(36);

icluster = 3;


idx = cluster(gmListTeDCM{icluster} , scoreAcoustTeDCM(:,1:nPCs));

countTypePer = zeros(icluster, ngroups);
countType = zeros(icluster, ngroups);
countTypeTot = zeros(1, ngroups);
for ig=1:ngroups
    indGrp =  find( strcmp(vocTypeCutsTeDCM, nameGrp(ig)) );
    countTypeTot(ig) = length(indGrp);
end

for ic = 1:icluster
    clusteridx = find(idx == ic);
    fprintf(1,'Cluster %d:\n', ic);
    for ig=1:ngroups
        if countTypeTot(ig) == 0
            continue;
        end
        indGrp =  find( strcmp(vocTypeCutsTeDCM(clusteridx), nameGrp(ig)) );
        countType(ic,ig) = length(indGrp);
        countTypePer(ic,ig) = 100.0*countType(ic,ig)./countTypeTot(ig);
        fprintf(1,'\t%s %d (%.2f%%)\n', nameGrp{ig}, countType(ic,ig), countTypePer(ic,ig));
    end
end

grpid = [2 4];
bh = bar(countTypePer(:,grpid));   % 8 and 9 are Thucks and Tucks
for ibh=1:2
    set(bh(ibh), 'EdgeColor', colorplot(grpid(ibh), :), 'FaceColor', colorplot(grpid(ibh), :));
    for ic=1:icluster
    text(ic - 0.15+(ibh-1)*0.3, countTypePer(ic,grpid(ibh)) + 3, sprintf('%d',countType(ic, grpid(ibh))));
    end
end
legend(nameGrp(grpid));
axisVal = axis();
axisVal(4) = 100;
axis(axisVal);


% Plot the data using 1rst and 2nd PCs
figure(37);
subplot(1,2,1)
for ig=1:2
    indGrp =  find( strcmp(vocTypeCutsTeDCM, nameGrp(grpid(ig))) );
    if ig > 1
        hold on;
    end
    scatter(scoreAcoustTeDCM(indGrp,1),scoreAcoustTeDCM(indGrp,2), 8, 'MarkerFaceColor', colorplot(grpid(ig), :), 'MarkerEdgeColor', colorplot(grpid(ig), :));
    hold off;
end


hold on
ang = 0:pi/20:2*pi;
circ = [cos(ang); sin(ang)];   % x,y coordinates of unit circle  
maxAmp = max(gmListTeDCM{icluster}.ComponentProportion);
for ic = 1:icluster
    compAmp = gmListTeDCM{icluster}.ComponentProportion(ic);
    mu1 = gmListTeDCM{icluster}.mu(ic,1);
    mu2 = gmListTeDCM{icluster}.mu(ic,2);
    sig1 = gmListTeDCM{icluster}.Sigma(1,1,ic);
    sig2 = gmListTeDCM{icluster}.Sigma(2,2,ic);
    sig12 = gmListTeDCM{icluster}.Sigma(1,2,ic);
    covMat = [sig1, sig12; sig12, sig2];

    [V, D] = eig(covMat);
    Dsd = sqrt(D);
    
    X = V*Dsd*circ;
    
    scatter(mu1, mu2, fix(60*compAmp/maxAmp), 'k', 'filled');
    plot( mu1 + X(1,:), mu2 + X(2,:) ,'-', 'LineWidth', 2, 'Color', [0.5 0.5 0.5]);
    
end
hold off
xlabel('PC1');
ylabel('PC2');
axis([-5 5 -5 5]);

subplot(1,2,2)
for ig=1:2
    indGrp =  find( strcmp(vocTypeCutsTeDCM, nameGrp(grpid(ig))) );
    if ig > 1
        hold on;
    end
    scatter(scoreAcoustTeDCM(indGrp,1),scoreAcoustTeDCM(indGrp,3),8, 'MarkerFaceColor', colorplot(grpid(ig), :), 'MarkerEdgeColor', colorplot(grpid(ig), :));
    hold off;
end


hold on
ang = 0:pi/20:2*pi;
circ = [cos(ang); sin(ang)];   % x,y coordinates of unit circle  
maxAmp = max(gmListTeDCM{icluster}.ComponentProportion);

for ic = 1:icluster
    compAmp = gmListTeDCM{icluster}.ComponentProportion(ic);
    mu1 = gmListTeDCM{icluster}.mu(ic,1);
    mu2 = gmListTeDCM{icluster}.mu(ic,3);
    sig1 = gmListTeDCM{icluster}.Sigma(1,1,ic);
    sig2 = gmListTeDCM{icluster}.Sigma(3,3,ic);
    sig12 = gmListTeDCM{icluster}.Sigma(1,3,ic);
    covMat = [sig1, sig12; sig12, sig2];
    [V, D] = eig(covMat);
    Dsd = sqrt(D);
    
    X = V*Dsd*circ;
    
    scatter(mu1, mu2, fix(60*compAmp/maxAmp), 'k', 'filled');
    plot( mu1 + X(1,:), mu2 + X(2,:) ,'-', 'LineWidth', 2, 'Color', [0.5 0.5 0.5]);
    
end
hold off

xlabel('PC1');
ylabel('PC3');
axis([-5 5 -5 5]);
title(sprintf('P1 = %.2f P2 = %.2f P3 = %.2f', gmListTeDCM{icluster}.ComponentProportion(1),...
    gmListTeDCM{icluster}.ComponentProportion(2), gmListTeDCM{icluster}.ComponentProportion(3)));

figure(38);
idx = cluster(gmListTeDCM{icluster} , scoreAcoustTeDCM(:,1:nPCs));

% Print out the proportion of clusters for each bird
for ibird = 1:nbirdsTeDCM
    indBird = find(strcmp(birdNameCutsTeDCM, birdNamesTeDCM{ibird}));
    
    ncallsBird = length(indBird);
    ncallsBirdTe = length(find(strcmp(vocTypeCutsTeDCM(indBird), 'Te')));
    ncallsBirdDC = length(find(strcmp(vocTypeCutsTeDCM(indBird), 'DC')));
    fprintf(1, 'Bird %s Ncalls = %d Te=%d DC=%d\t', birdNamesTeDCM{ibird}, ncallsBird, ncallsBirdTe, ncallsBirdDC);
    for ic=1:icluster
        indBirdCluster = find(strcmp(birdNameCutsTeDCM(idx == ic), birdNamesTeDCM{ibird}));
        indBirdClusterTe = find(strcmp(birdNameCutsTeDCM(idx == ic), birdNamesTeDCM{ibird}) & strcmp(vocTypeCutsTeDCM(idx == ic), 'Te'));
        indBirdClusterDC = find(strcmp(birdNameCutsTeDCM(idx == ic), birdNamesTeDCM{ibird}) & strcmp(vocTypeCutsTeDCM(idx ==ic), 'DC'));
        fprintf(1, 'P(%d) = %.0f%% Te=%.0f%% DC=%.0f%%\t', ic, ...
            100.0*length(indBirdCluster)/ncallsBird, 100.0*length(indBirdClusterTe)/ncallsBirdTe, 100.0*length(indBirdClusterDC)/ncallsBirdDC );
    end
    fprintf(1,'\n');
end

% Let's calculate means, sem and t-test for sdtime, cvf0, fund and rms 
testingFeatures = [1,6,7,16,20];
nFeatures = length(testingFeatures);
meanValFeatures = zeros(icluster, nFeatures);
semValFeatures = zeros(icluster, nFeatures);

for ifeat=1:nFeatures
    id = testingFeatures(ifeat);
    fprintf(1, 'Differences for %s\n', xtagPlot{id});
    [meanValFeatures(:,ifeat), semValFeatures(:, ifeat)] = grpstats(AcoustTeDCM(:, id), idx, {'mean', 'sem'});
    [p,table, stats] = anova1(AcoustTeDCM(:, id), idx, 'off');
    for ic=1:icluster
        fprintf(1, '\tMean 1: %f +- %f\n', meanValFeatures(ic, ifeat), semValFeatures(ic, ifeat));
    end
    fprintf(1,'\t\t F(%d,%d) = %.2f, p=%.4f\n',  table{2,3}, table{3,3}, table{2,5}, p );
    
    subplot(1, nFeatures, ifeat);
    bar(meanValFeatures(:,ifeat));
    hold on;
    errorbar(1:icluster, meanValFeatures(:,ifeat), semValFeatures(:, ifeat), 'r+', 'LineWidth', 2);
    title(xtagPlot{id});
    if (p < 0.001)
        text((1+icluster)/2, 1.05*max(meanValFeatures(:, ifeat)) , '***');
    elseif (p < 0.01)
        text((1+icluster)/2, 1.05*max(meanValFeatures(:, ifeat)) , '**');
    elseif (p <0.05)
        text((1+icluster)/2, 1.05*max(meanValFeatures(:, ifeat)) , '*');
    end
    hold off;
    axis([0 icluster+1 0 1.1*max(meanValFeatures(:, ifeat))])
        
end


%% Now mixing Tets and DC - Now for Females

indTeDCF = find( (strcmp(vocTypeCuts, 'Te') | strcmp(vocTypeCuts, 'DC' )) &  strcmp(birdSexCuts,'F'));
AcoustTeDCF = Acoust(indTeDCF,:);
vocTypeCutsTeDCF = vocTypeCuts(indTeDCF);
birdNameCutsTeDCF = birdNameCuts(indTeDCF);
birdNamesTeDCF = unique(birdNameCutsTeDCF);
nbirdsTeDCF = length(birdNamesTeDCF);


zAcoustTeDCF = zscore(AcoustTeDCF);
[pcAcoustTeDCF, scoreAcoustTeDCF, eigenvalTeDCF] = princomp(zAcoustTeDCF);  % Use pca so that we can used weighted observations as well

figure(39);
plot(100*cumsum(eigenvalTeDCF)./sum(eigenvalTeDCF));
xlabel('Number of PCs');
ylabel('Variance Explained %');
title('PCs on Acoustical Features');

nClusters = 5;
nPCs = 10;

gmListTeDCF = cell(1, nClusters);
aicListTeDCF = zeros(1, nClusters);
bicListTeDCF = zeros(1, nClusters);

options = statset('Display','final', 'MaxIter', 1000);  % Display final results

for ic = 1:nClusters    
    gmListTeDCF{ic} = fitgmdist(scoreAcoustTeDCF(:, 1:nPCs), ic, 'Options', options, 'Replicates', 5);  % Gaussian mixture model
    aicListTeDCF(ic) = gmListTeDCF{ic}.AIC;
    bicListTeDCF(ic) = gmListTeDCF{ic}.BIC;
end

figure(40);
subplot(1,2,1);
plot(1:nClusters, aicListTeDCF - min(aicListTeDCF), 'k');
xlabel('Number of Gaussians');
ylabel('AIC');

subplot(1,2,2);
plot(1:nClusters, bicListTeDCF, 'r');
xlabel('Number of Gaussians');
ylabel('BIC');

%
figure(41);

icluster = 3;


idx = cluster(gmListTeDCF{icluster} , scoreAcoustTeDCF(:,1:nPCs));

countTypePer = zeros(icluster, ngroups);
countType = zeros(icluster, ngroups);
countTypeTot = zeros(1, ngroups);
for ig=1:ngroups
    indGrp =  find( strcmp(vocTypeCutsTeDCF, nameGrp(ig)) );
    countTypeTot(ig) = length(indGrp);
end

for ic = 1:icluster
    clusteridx = find(idx == ic);
    fprintf(1,'Cluster %d:\n', ic);
    for ig=1:ngroups
        if countTypeTot(ig) == 0
            continue;
        end
        indGrp =  find( strcmp(vocTypeCutsTeDCF(clusteridx), nameGrp(ig)) );
        countType(ic,ig) = length(indGrp);
        countTypePer(ic,ig) = 100.0*countType(ic,ig)./countTypeTot(ig);
        fprintf(1,'\t%s %d (%.2f%%)\n', nameGrp{ig}, countType(ic,ig), countTypePer(ic,ig));
    end
end

grpid = [2 4];
bh = bar(countTypePer(:,grpid));   % 8 and 9 are Thucks and Tucks
for ibh=1:2
    set(bh(ibh), 'EdgeColor', colorplot(grpid(ibh), :), 'FaceColor', colorplot(grpid(ibh), :));
    for ic=1:icluster
    text(ic - 0.15+(ibh-1)*0.3, countTypePer(ic,grpid(ibh)) + 3, sprintf('%d',countType(ic, grpid(ibh))));
    end
end
legend(nameGrp(grpid));
axisVal = axis();
axisVal(4) = 100;
axis(axisVal);


% Plot the data using 1rst and 2nd PCs
figure(42);
subplot(1,2,1)
for ig=1:2
    indGrp =  find( strcmp(vocTypeCutsTeDCF, nameGrp(grpid(ig))) );
    if ig > 1
        hold on;
    end
    scatter(scoreAcoustTeDCF(indGrp,1),scoreAcoustTeDCF(indGrp,2), 8, 'MarkerFaceColor', colorplot(grpid(ig), :), 'MarkerEdgeColor', colorplot(grpid(ig), :));
    hold off;
end


hold on
ang = 0:pi/20:2*pi;
circ = [cos(ang); sin(ang)];   % x,y coordinates of unit circle  
maxAmp = max(gmListTeDCF{icluster}.ComponentProportion);
for ic = 1:icluster
    compAmp = gmListTeDCF{icluster}.ComponentProportion(ic);
    mu1 = gmListTeDCF{icluster}.mu(ic,1);
    mu2 = gmListTeDCF{icluster}.mu(ic,2);
    sig1 = gmListTeDCF{icluster}.Sigma(1,1,ic);
    sig2 = gmListTeDCF{icluster}.Sigma(2,2,ic);
    sig12 = gmListTeDCF{icluster}.Sigma(1,2,ic);
    covMat = [sig1, sig12; sig12, sig2];

    [V, D] = eig(covMat);
    Dsd = sqrt(D);
    
    X = V*Dsd*circ;
    
    scatter(mu1, mu2, fix(60*compAmp/maxAmp), 'k', 'filled');
    plot( mu1 + X(1,:), mu2 + X(2,:) ,'-', 'LineWidth', 2, 'Color', [0.5 0.5 0.5]);
    
end
hold off
xlabel('PC1');
ylabel('PC2');
axis([-5 5 -5 5]);

subplot(1,2,2)
for ig=1:2
    indGrp =  find( strcmp(vocTypeCutsTeDCF, nameGrp(grpid(ig))) );
    if ig > 1
        hold on;
    end
    scatter(scoreAcoustTeDCF(indGrp,1),scoreAcoustTeDCF(indGrp,3),8, 'MarkerFaceColor', colorplot(grpid(ig), :), 'MarkerEdgeColor', colorplot(grpid(ig), :));
    hold off;
end


hold on
ang = 0:pi/20:2*pi;
circ = [cos(ang); sin(ang)];   % x,y coordinates of unit circle  
maxAmp = max(gmListTeDCF{icluster}.ComponentProportion);

for ic = 1:icluster
    compAmp = gmListTeDCF{icluster}.ComponentProportion(ic);
    mu1 = gmListTeDCF{icluster}.mu(ic,1);
    mu2 = gmListTeDCF{icluster}.mu(ic,3);
    sig1 = gmListTeDCF{icluster}.Sigma(1,1,ic);
    sig2 = gmListTeDCF{icluster}.Sigma(3,3,ic);
    sig12 = gmListTeDCF{icluster}.Sigma(1,3,ic);
    covMat = [sig1, sig12; sig12, sig2];
    [V, D] = eig(covMat);
    Dsd = sqrt(D);
    
    X = V*Dsd*circ;
    
    scatter(mu1, mu2, fix(60*compAmp/maxAmp), 'k', 'filled');
    plot( mu1 + X(1,:), mu2 + X(2,:) ,'-', 'LineWidth', 2, 'Color', [0.5 0.5 0.5]);
    
end
hold off

xlabel('PC1');
ylabel('PC3');
axis([-5 5 -5 5]);
title(sprintf('P1 = %.2f P2 = %.2f P3 = %.2f', gmListTeDCF{icluster}.ComponentProportion(1),...
    gmListTeDCF{icluster}.ComponentProportion(2), gmListTeDCF{icluster}.ComponentProportion(3)));

figure(43);
idx = cluster(gmListTeDCF{icluster} , scoreAcoustTeDCF(:,1:nPCs));

% Print out the proportion of clusters for each bird
for ibird = 1:nbirdsTeDCF
    indBird = find(strcmp(birdNameCutsTeDCF, birdNamesTeDCF{ibird}));
    
    ncallsBird = length(indBird);
    ncallsBirdTe = length(find(strcmp(vocTypeCutsTeDCF(indBird), 'Te')));
    ncallsBirdDC = length(find(strcmp(vocTypeCutsTeDCF(indBird), 'DC')));
    fprintf(1, 'Bird %s Ncalls = %d Te=%d DC=%d\t', birdNamesTeDCF{ibird}, ncallsBird, ncallsBirdTe, ncallsBirdDC);
    for ic=1:icluster
        indBirdCluster = find(strcmp(birdNameCutsTeDCF(idx == ic), birdNamesTeDCF{ibird}));
        indBirdClusterTe = find(strcmp(birdNameCutsTeDCF(idx == ic), birdNamesTeDCF{ibird}) & strcmp(vocTypeCutsTeDCF(idx == ic), 'Te'));
        indBirdClusterDC = find(strcmp(birdNameCutsTeDCF(idx == ic), birdNamesTeDCF{ibird}) & strcmp(vocTypeCutsTeDCF(idx ==ic), 'DC'));
        fprintf(1, 'P(%d) = %.0f%% Te=%.0f%% DC=%.0f%%\t', ic, ...
            100.0*length(indBirdCluster)/ncallsBird, 100.0*length(indBirdClusterTe)/ncallsBirdTe, 100.0*length(indBirdClusterDC)/ncallsBirdDC );
    end
    fprintf(1,'\n');
end

% Let's calculate means, sem and t-test for sdtime, cvf0, fund and rms 
testingFeatures = [1,6,7,16,20];
nFeatures = length(testingFeatures);
meanValFeatures = zeros(icluster, nFeatures);
semValFeatures = zeros(icluster, nFeatures);

for ifeat=1:nFeatures
    id = testingFeatures(ifeat);
    fprintf(1, 'Differences for %s\n', xtagPlot{id});
    [meanValFeatures(:,ifeat), semValFeatures(:, ifeat)] = grpstats(AcoustTeDCF(:, id), idx, {'mean', 'sem'});
    [p,table, stats] = anova1(AcoustTeDCF(:, id), idx, 'off');
    for ic=1:icluster
        fprintf(1, '\tMean 1: %f +- %f\n', meanValFeatures(ic, ifeat), semValFeatures(ic, ifeat));
    end
    fprintf(1,'\t\t F(%d,%d) = %.2f, p=%.4f\n',  table{2,3}, table{3,3}, table{2,5}, p );
    
    subplot(1, nFeatures, ifeat);
    bar(meanValFeatures(:,ifeat));
    hold on;
    errorbar(1:icluster, meanValFeatures(:,ifeat), semValFeatures(:, ifeat), 'r+', 'LineWidth', 2);
    title(xtagPlot{id});
    if (p < 0.001)
        text((1+icluster)/2, 1.05*max(meanValFeatures(:, ifeat)) , '***');
    elseif (p < 0.01)
        text((1+icluster)/2, 1.05*max(meanValFeatures(:, ifeat)) , '**');
    elseif (p <0.05)
        text((1+icluster)/2, 1.05*max(meanValFeatures(:, ifeat)) , '*');
    end
    hold off;
    axis([0 icluster+1 0 1.1*max(meanValFeatures(:, ifeat))])
        
end

%% Repeat clustering with only Thuck and Tucks calls to see if there are two vs one group

indDA = find( strcmp(vocTypeCuts, 'Di') | strcmp(vocTypeCuts, 'Ag' ) );
AcoustDA = Acoust(indDA,:);
vocTypeCutsDA = vocTypeCuts(indDA);


zAcoustDA = zscore(AcoustDA);
[pcAcoustDA, scoreAcoustDA, eigenvalDA] = princomp(zAcoustDA);  % Use pca so that we can used weighted observations as well

figure(44);
plot(100*cumsum(eigenvalDA)./sum(eigenvalDA));
xlabel('Number of PCs');
ylabel('Variance Explained %');
title('PCs on Acoustical Features');

nClusters = 5;
nPCs = 10;

gmListDA = cell(1, nClusters);
aicListDA = zeros(1, nClusters);
bicListDA = zeros(1, nClusters);

options = statset('Display','final', 'MaxIter', 1000);  % Display final results

parfor ic = 1:nClusters    
    gmListDA{ic} = fitgmdist(scoreAcoustDA(:, 1:nPCs), ic, 'Options', options, 'Replicates', 5);  % Gaussian mixture model
    aicListDA(ic) = gmListDA{ic}.AIC;
    bicListDA(ic) = gmListDA{ic}.BIC;
end

figure(45);
subplot(1,2,1);
plot(1:nClusters, aicListDA - min(aicListDA), 'k');
xlabel('Number of Gaussians');
ylabel('AIC');

subplot(1,2,2);
plot(1:nClusters, bicListDA, 'r');
xlabel('Number of Gaussians');
ylabel('BIC');

%
figure(46);

icluster = 2;


idx = cluster(gmListDA{icluster} , scoreAcoustDA(:,1:nPCs));

countTypePer = zeros(icluster, ngroups);
countType = zeros(icluster, ngroups);
countTypeTot = zeros(1, ngroups);
for ig=1:ngroups
    indGrp =  find( strcmp(vocTypeCutsDA, nameGrp(ig)) );
    countTypeTot(ig) = length(indGrp);
end

for ic = 1:icluster
    clusteridx = find(idx == ic);
    fprintf(1,'Cluster %d:\n', ic);
    for ig=1:ngroups
        if countTypeTot(ig) == 0
            continue;
        end
        indGrp =  find( strcmp(vocTypeCutsDA(clusteridx), nameGrp(ig)) );
        countType(ic,ig) = length(indGrp);
        countTypePer(ic,ig) = 100.0*countType(ic,ig)./countTypeTot(ig);
        fprintf(1,'\t%s %d (%.2f%%)\n', nameGrp{ig}, countType(ic,ig), countTypePer(ic,ig));
    end
end

stimId = [1 6];  % Agressive and distress
bh = bar(countTypePer(:,stimId));   
for ibh=1:2
    set(bh(ibh), 'EdgeColor', colorplot(stimId(ibh), :), 'FaceColor', colorplot(stimId(ibh), :));
    text(0.85+(ibh-1)*0.3, countTypePer(1,stimId(ibh)) + 3, sprintf('%d',countType(1, stimId(ibh))));
    text(1.85+(ibh-1)*0.3, countTypePer(2,stimId(ibh)) + 3, sprintf('%d',countType(2, stimId(ibh))));
end
legend(nameGrp(stimId));
axisVal = axis();
axisVal(4) = 100;
axis(axisVal);

% Test of two proportions
n1 = countType(1, stimId(1)) + countType(1, stimId(2));
n2 = countType(2, stimId(1)) + countType(2, stimId(2));
p1est = countType(1, stimId(1))/n1;
p2est = countType(2, stimId(1))/n2;
pest = (countType(1, stimId(1)) + countType(2, stimId(1)))/(n1 + n2);
zval = (p1est-p2est)./sqrt(pest*(1-pest)*(1/n1+1/n2));
pd = makedist('Normal');
pval = 2*(1-cdf(pd, abs(zval)));

title(sprintf('Proportion Test zval=%.2f pval=%.4f', zval, pval));

% Plot the data using 1rst and 2nd PCs
figure(47);
subplot(1,2,1)
for ig=1:2
    indGrp =  find( strcmp(vocTypeCutsDA, nameGrp(stimId(ig))) );
    if ig > 1
        hold on;
    end
    scatter(scoreAcoustDA(indGrp,1),scoreAcoustDA(indGrp,2), 8, 'MarkerFaceColor', colorplot(stimId(ig), :), 'MarkerEdgeColor', colorplot(stimId(ig), :));
    hold off;
end


hold on
ang = 0:pi/20:2*pi;
circ = [cos(ang); sin(ang)];   % x,y coordinates of unit circle  
maxAmp = max(gmListDA{icluster}.ComponentProportion);
for ic = 1:icluster
    compAmp = gmListDA{icluster}.ComponentProportion(ic);
    mu1 = gmListDA{icluster}.mu(ic,1);
    mu2 = gmListDA{icluster}.mu(ic,2);
    sig1 = gmListDA{icluster}.Sigma(1,1,ic);
    sig2 = gmListDA{icluster}.Sigma(2,2,ic);
    sig12 = gmListDA{icluster}.Sigma(1,2,ic);
    covMat = [sig1, sig12; sig12, sig2];

    [V, D] = eig(covMat);
    Dsd = sqrt(D);
    
    X = V*Dsd*circ;
    
    scatter(mu1, mu2, fix(60*compAmp/maxAmp), 'k', 'filled');
    plot( mu1 + X(1,:), mu2 + X(2,:) ,'-', 'LineWidth', 2, 'Color', [0.5 0.5 0.5]);
    
end
hold off
xlabel('PC1');
ylabel('PC2');
axis([-5 5 -5 5]);

subplot(1,2,2)
for ig=1:2
    indGrp =  find( strcmp(vocTypeCutsDA, nameGrp(stimId(ig))) );
    if ig > 1
        hold on;
    end
    scatter(scoreAcoustDA(indGrp,1),scoreAcoustDA(indGrp,3),8, 'MarkerFaceColor', colorplot(stimId(ig), :), 'MarkerEdgeColor', colorplot(stimId(ig), :));
    hold off;
end
% Add contour lines
% hold on
% for ic = 1:icluster
%     ezcontour(@(x,y)mvnpdf([x y], [gmListDA{icluster}.mu(ic,1) gmListDA{icluster}.mu(ic,3)],...
%         [gmListDA{icluster}.Sigma(1,1,ic), gmListDA{icluster}.Sigma(1,3,ic); gmListDA{icluster}.Sigma(3,1,ic), gmListDA{icluster}.Sigma(3,3,ic)] ),...
%         [gmListDA{icluster}.mu(ic,1)-2 gmListDA{icluster}.mu(ic,1)+2 gmListDA{icluster}.mu(ic,3)-2 gmListDA{icluster}.mu(ic,3)+2], 60);
% end
% hold off
hold on
ang = 0:pi/20:2*pi;
circ = [cos(ang); sin(ang)];   % x,y coordinates of unit circle  
maxAmp = max(gmListDA{icluster}.ComponentProportion);

for ic = 1:icluster
    compAmp = gmListDA{icluster}.ComponentProportion(ic);
    mu1 = gmListDA{icluster}.mu(ic,1);
    mu2 = gmListDA{icluster}.mu(ic,3);
    sig1 = gmListDA{icluster}.Sigma(1,1,ic);
    sig2 = gmListDA{icluster}.Sigma(3,3,ic);
    sig12 = gmListDA{icluster}.Sigma(1,3,ic);
    covMat = [sig1, sig12; sig12, sig2];
    [V, D] = eig(covMat);
    Dsd = sqrt(D);
    
    X = V*Dsd*circ;
    
    scatter(mu1, mu2, fix(60*compAmp/maxAmp), 'k', 'filled');
    plot( mu1 + X(1,:), mu2 + X(2,:) ,'-', 'LineWidth', 2, 'Color', [0.5 0.5 0.5]);
    
end
hold off

xlabel('PC1');
ylabel('PC3');
axis([-5 5 -5 5]);
title(sprintf('P1 = %.2f P2 = %.2f', gmListDA{icluster}.ComponentProportion(1), gmListDA{icluster}.ComponentProportion(2)));

