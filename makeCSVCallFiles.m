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

%% Make a csv file
fid = fopen('/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/zebraFinchCalls.csv', 'w');
indSel = find( strcmp(vocTypeCuts, 'DC') | strcmp(vocTypeCuts, 'LT') | strcmp(vocTypeCuts, 'Be') );
AcoustSel = Acoust(indSel,:);
vocTypeCutsSel = vocTypeCuts(indSel);

ncols = length(xtag);
nrows = length(indSel);

% print the header
fprintf(fid, '%s,', 'CallType');
for ic=1:ncols
    fprintf(fid,'%s', xtag{ic});
    if ic<ncols
        fprintf(fid,',');
    end
end
fprintf(fid, '\n');

% Now the data
for ir=1:nrows
    fprintf(fid, '%s,', vocTypeCutsSel{ir});
    for ic=1:ncols
        fprintf(fid,'%f', AcoustSel(ir, ic));
        if ic<ncols
            fprintf(fid,',');
        end
    end
    fprintf(fid, '\n');
end



