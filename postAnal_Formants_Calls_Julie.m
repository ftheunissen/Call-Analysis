%% Post processing description of calls

% Load the data.
% load('/Users/frederictheunissen/Documents/Data/Julie/FullVocalizationBank/vocCutsAnal.mat');
% This one includes the PSD, the temporal enveloppe and has the correct
% calibration for sound level
load('/Users/frederictheunissen/Documents/Data/Julie/FullVocalizationBank/vocCutsAnalwFormants.mat');

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


nAcoust = 25;
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
Acoust(:,23) = [callAnalData.f1];
Acoust(:,24) = [callAnalData.f2];
Acoust(:,25) = [callAnalData.f3];

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
xtag{23} = 'F1';
xtag{24} = 'F2';
xtag{25} = 'F3';

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
xtagPlot{23} = 'F1';
xtagPlot{24} = 'F2';
xtagPlot{25} = 'F3';



%% Make a Histogram of F1, F2, F3.
figure(1);
histogram(Acoust(:, 23));
hold on;
histogram(Acoust(:, 24));
histogram(Acoust(:, 25));
hold off;

% F1, F2, F3 together
figure(2);
histogram([Acoust(:, 23) Acoust(:, 24) Acoust(:, 25)]);

% Fix values to put boundaries between formants
n = size(Acoust,1);

Acoust(Acoust(:, 23) > 8000, 23) = nan;
Acoust(Acoust(:, 24) > 8000, 24) = nan;
Acoust(Acoust(:, 25) > 8000, 25) = nan;

for i = 1:n
    if Acoust(i, 24) > 5000
        Acoust(i, 25) = Acoust(i, 24);
        Acoust(i, 24) = nan;
    end
    if Acoust(i, 23) > 3000
        if Acoust(i, 23) > 5000
            Acoust(i, 25) = Acoust(i, 23);
        else
            Acoust(i, 24) = Acoust(i, 23);
        end
        Acoust(i,23) = nan;
    end       
end

figure(3);
histogram(Acoust(:, 23));
hold on;
histogram(Acoust(:, 24));
histogram(Acoust(:, 25));
hold off;


%% Look at scatter plot of F1, F2 and F1, F3 for subset of data

indSel = find( strcmp(vocTypeCuts, 'DC') | strcmp(vocTypeCuts, 'Wh') | strcmp(vocTypeCuts, 'Te') | strcmp(vocTypeCuts, 'Ne') );

AcoustSel = Acoust(indSel,:);
vocTypeCutsSel = vocTypeCuts(indSel);


figure(4);
subplot(1,3,1);
gscatter(AcoustSel(:, 23), AcoustSel(:,24), vocTypeCutsSel');
xlabel(xtagPlot{23});
ylabel(xtagPlot{24});

subplot(1,3,2);
gscatter(AcoustSel(:, 23), AcoustSel(:,25), vocTypeCutsSel');
xlabel(xtagPlot{23});
ylabel(xtagPlot{25});

subplot(1,3,3);
gscatter(AcoustSel(:, 24), AcoustSel(:,25), vocTypeCutsSel');
xlabel(xtagPlot{24});
ylabel(xtagPlot{25});
