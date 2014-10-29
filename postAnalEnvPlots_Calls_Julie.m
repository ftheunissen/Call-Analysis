%% Post processing description of calls

% Load the data.
load('/Users/frederictheunissen/Documents/Data/Julie/FullVocalizationBank/vocCutsAnalwPSD.mat');

% the file with _test has the fix for the saliency - it is called _test
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

ncalls = length(callAnalData);
nf = length(callAnalData(1).psd.Data);
nt = length(callAnalData(1).tAmp);

callsPSD = zeros(ncalls, nf);
callsPSDNorm = zeros(ncalls, nf);
callstAmp = zeros(ncalls, nt);
callstAmpNorm = zeros(ncalls, nt);

for ic = 1: ncalls
    callsPSD(ic, :) = (callAnalData(ic).psd.Data)';
    callsPSDNorm(ic, :) = callsPSD(ic,:)./sum(callsPSD(ic,:));      % Normalizing by total power
    callstAmp(ic, :) = callAnalData(ic).tAmp;
    callstAmpNorm(ic, :) = callstAmp(ic, :)./max(callstAmp(ic,:));   % Normalizing by max amplituded
    
end

% Make averages for each bird
% Find number of unique combinations
vocTypesBirds = unique(strcat(vocTypeCuts', birdNameCuts'), 'stable');
nvocTypesBirds = length(vocTypesBirds);
vocTypeCutsMeans = cell(1, nvocTypesBirds);
birdNameCutsMeans = cell(1, nvocTypesBirds);
birdSexCutsMeans = cell(1, nvocTypesBirds);
callsPSDMeans = zeros(nvocTypesBirds, nf);
callsPSDNormMeans = zeros(nvocTypesBirds, nf);
callstAmpMeans = zeros(nvocTypesBirds, nt);
callstAmpNormMeans = zeros(nvocTypesBirds, nt);

for ic = 1: nvocTypesBirds
    indTypesBirds = find( strcmp(vocTypesBirds{ic}(1:2), vocTypeCuts') & strcmp(vocTypesBirds{ic}(3:end), birdNameCuts'));
    vocTypeCutsMeans{ic} = vocTypesBirds{ic}(1:2);
    birdNameCutsMeans{ic} = vocTypesBirds{ic}(3:end);
    birdSexCutsMeans{ic} = birdSexCuts{indTypesBirds(1)};
    if (length(indTypesBirds) == 1 )
        callsPSDMeans(ic, :) = callsPSD(indTypesBirds, :);
        callsPSDNormMeans(ic, :) = callsPSDNorm(indTypesBirds, :);
        callstAmpMeans(ic, :) = callstAmp(indTypesBirds, :);
        callstAmpNormMeans(ic, :) = callstAmpNorm(indTypesBirds, :);
    else
        callsPSDMeans(ic, :) = mean(callsPSD(indTypesBirds, :));
        callsPSDNormMeans(ic, :) = mean(callsPSDNorm(indTypesBirds, :));
        callstAmpMeans(ic, :) = mean(callstAmp(indTypesBirds, :));
        callstAmpNormMeans(ic, :) = mean(callstAmpNorm(indTypesBirds, :));
    end
end

%% Some stuff needed for graphs

nameGrp = unique(vocTypeCutsMeans,'stable');   % Names in the order found in original data set as used by grpmeans
ngroups = length(nameGrp);

name_grp_plot = {'Be', 'LT', 'Tu', 'Th', 'Di', 'Ag', 'Wh', 'Ne', 'Te', 'DC', 'So'};
colorVals = [ [0 230 255]; [0 95 255]; [255 200 65]; [255 150 40]; [255 105 15];...
    [255 0 0]; [255 180 255]; [255 100 255]; [140 100 185]; [100 50 200]; [100 100 100] ];


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


%% Plot the mean values of the PSD

meanPSD = grpstats(callsPSDMeans, vocTypeCutsMeans');             % This is the means of means of birds
meanPSDNorm = grpstats(callsPSDNormMeans, vocTypeCutsMeans');     % This is the means of means of birds

maxPow = max(max(meanPSD));
maxPowNorm = max(max(meanPSDNorm));

% This is a bit circular - this is the excel file that has the formnants
% that I chosse from running this code - you can plot one or the other...
formantFile = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/VocFormants.xlsx';
[numF, txtF] = xlsread(formantFile, 'Formants');
maxnF = size(numF,2);    % Maximum number of Formants


fvals = callAnalData(1).psd.Frequencies;
for ig = 1:ngroups
    powvals = 20*log10(meanPSD(ig, :)./maxPow)+100;
    smPow = sgolayfilt(powvals,3,51);
    powvalsNorm = 20*log10(meanPSDNorm(ig, :)./maxPowNorm)+100;
    smPowNorm = sgolayfilt(powvalsNorm,3,51);
    
    figure(1);
    plot(fvals, smPow, '-', 'Color', colorplot(ig, :), 'LineWidth', 2);
    hold on;
    figure(2);
    % plot(fvals, smPowNorm./max(smPowNorm), '-', 'Color', colorplot(ig, :), 'LineWidth', 2);
    plot(fvals, smPowNorm, '-', 'Color', colorplot(ig, :), 'LineWidth', 2);
    hold on;
    

    plotInd = find(strcmp(name_grp_plot, nameGrp(ig)));
    indF = find(strcmp(txtF(:,1), nameGrp{ig}));
    
    figure(10);
    subplot(1, ngroups, plotInd);
    semilogx(fvals./1000, smPowNorm, '-', 'Color', colorplot(ig, :), 'LineWidth', 2);
    [pks, locs] = findpeaks(smPowNorm, 'MINPEAKHEIGHT', 15, 'MINPEAKDISTANCE', 20);
    title(nameGrp(ig));
    if plotInd == 1
    xlabel('Frequency (kHz)');
    ylabel('Power dB');
%     else
%         axis off;
    end
    box off;
    axis([0.2 10 65 95]);
    hold on;
    npks = length(pks);
    fprintf(1,'Call Type %s:', nameGrp{ig});
    for ip=1:npks
%         semilogx([fvals(locs(ip))/1000 fvals(locs(ip))/1000], [40 100], 'k--');   %
        %  Uncomment this line if you want to plot the original guesses
        if (fvals(locs(ip)) < 10000)
            fprintf(1, '\t%4.2f', fvals(locs(ip))./1000);
        end       
    end
    fprintf(1,'\n');
    for ifm=1:maxnF
        if (~isnan(numF(indF-1, ifm)) )
            semilogx([numF(indF-1, ifm) numF(indF-1, ifm)], [0 100], 'k--', 'LineWidth', 1);
        end
    end
    
    hold off;
    
end

figure (1);
legend(nameGrp, 'Location', 'EastOutside');
xlabel('Frequency (Hz)');
ylabel('Power dB');
axis([250 10000 20 100]);
hold off;

figure (2);
legend(nameGrp, 'Location', 'EastOutside');
xlabel('Frequency (Hz)');
ylabel('Normalized Power dB');
axis([250 10000 20 100]);
% axis([250 10000 0.1 1.1]);
hold off;


%% Plot the mean values of the enveloppe

meanAmp = grpstats(callstAmpMeans, vocTypeCutsMeans');
meanAmpNorm = grpstats(callstAmpNormMeans, vocTypeCutsMeans');

maxAmp = max(max(meanAmp));
t = 0:nt-1;

for ig = 1:ngroups
    smAmp = sgolayfilt(meanAmp(ig,:),3,21);
    smAmpNorm = sgolayfilt(meanAmpNorm(ig,:),3,21);
    figure(3);
    plot(t, smAmp, '-', 'Color', colorplot(ig, :), 'LineWidth', 2);
    hold on;
    figure(4);
    plot(t, smAmpNorm, '-', 'Color', colorplot(ig, :), 'LineWidth', 2);
    hold on;
end
figure (3);
legend(nameGrp, 'Location', 'EastOutside');
xlabel('Time (ms)');
ylabel('Amplitude');

hold off;

figure (4);
legend(nameGrp, 'Location', 'EastOutside');
xlabel('Time (ms)');
ylabel('Normalized Amp');
hold off;


%% Plot the mean values of the PSD for distance call and tets and for male and female

indTets = find((strcmp(birdSexCutsMeans, 'M') | strcmp(birdSexCutsMeans, 'F') ) & (strcmp(vocTypeCutsMeans, 'Te')));
meanPSDTets = grpstats(callsPSDMeans(indTets,:), birdSexCutsMeans(indTets)');
meanPSDTetsNorm = grpstats(callsPSDNormMeans(indTets,:), birdSexCutsMeans(indTets)');

sex_name_Te = unique(birdSexCutsMeans(indTets), 'stable');

indDCs = find((strcmp(birdSexCutsMeans, 'M') | strcmp(birdSexCutsMeans, 'F') ) & (strcmp(vocTypeCutsMeans, 'DC')));
meanPSDDCs = grpstats(callsPSDMeans(indDCs,:), birdSexCutsMeans(indDCs)');
meanPSDDCsNorm = grpstats(callsPSDNormMeans(indDCs,:), birdSexCutsMeans(indDCs)');

sex_name_DC = unique(birdSexCutsMeans(indTets), 'stable');

maxPow = max(max(max(meanPSDDCs)),max(max(meanPSDTets)));
maxPowNorm = max(max(meanPSDDCsNorm), max(max(meanPSDTetsNorm)));


fvals = callAnalData(1).psd.Frequencies;
for ig = 1:2
    powvals = 20*log10(meanPSDTets(ig, :)./maxPow)+100;
    smPow = sgolayfilt(powvals,3,51);
    powvalsNorm = 20*log10(meanPSDTetsNorm(ig, :)./maxPowNorm)+100;
    smPowNorm = sgolayfilt(powvalsNorm,3,51);
    
    figure(5);
    if ig == 1
        plot(fvals, smPow, '-', 'Color', colorVals(9, :)./255, 'LineWidth', 1);
    else
        plot(fvals, smPow, ':', 'Color', colorVals(9, :)./255, 'LineWidth', 1);
    end
    hold on;
    figure(6);
    if ig == 1
        plot(fvals, smPowNorm, '-', 'Color', colorVals(9, :)./255, 'LineWidth', 1);
    else
        plot(fvals, smPowNorm, ':', 'Color', colorVals(9, :)./255, 'LineWidth', 1);
    end
    hold on;
end


for ig = 1:2
    powvals = 20*log10(meanPSDDCs(ig, :)./maxPow)+100;
    smPow = sgolayfilt(powvals,3,51);
    powvalsNorm = 20*log10(meanPSDDCsNorm(ig, :)./maxPowNorm)+100;
    smPowNorm = sgolayfilt(powvalsNorm,3,51);
    
    figure(5);
    if ig == 1
        plot(fvals, smPow, '-', 'Color', colorVals(10, :)./255, 'LineWidth', 1);
    else
        plot(fvals, smPow, ':', 'Color', colorVals(10, :)./255, 'LineWidth', 1);
    end
    hold on;
    figure(6);
    if ig == 1
        plot(fvals, smPowNorm, '-', 'Color', colorVals(10, :)./255, 'LineWidth', 1);
    else
        plot(fvals, smPowNorm, ':', 'Color', colorVals(10, :)./255, 'LineWidth', 1);
    end
    hold on;
end



figure (5);
legend([sex_name_Te, sex_name_DC]);
xlabel('Frequency (Hz)');
ylabel('Power dB');
axis([250 10000 40 100]);
hold off;

figure (6);
legend([sex_name_Te, sex_name_DC]);
xlabel('Frequency (Hz)');
ylabel('Power Normalized to Peak');
axis([250 10000 40 100]);
hold off;



%% Plot the mean values of the enveloppe for distance call and tets and for male and female

indTets = find((strcmp(birdSexCutsMeans, 'M') | strcmp(birdSexCutsMeans, 'F') ) & (strcmp(vocTypeCutsMeans, 'Te')));
meanAmpTets = grpstats(callstAmpMeans(indTets,:), birdSexCutsMeans(indTets)');
meanAmpTetsNorm = grpstats(callstAmpNormMeans(indTets,:), birdSexCutsMeans(indTets)');

sex_name_Te = unique(birdSexCutsMeans(indTets), 'stable');

t = 0:nt-1;
for ig = 1:2

    smAmp = sgolayfilt(meanAmpTets(ig,:),3,21);
    smAmpNorm = sgolayfilt(meanAmpTetsNorm(ig,:),3,21);
    figure(7);
    if ig == 1
        plot(t, smAmp, '-', 'Color', colorVals(9, :)./255, 'LineWidth', 1);
    else
        plot(t, smAmp, ':', 'Color', colorVals(9, :)./255, 'LineWidth', 1);
    end
    hold on;
    figure(8);
    if ig == 1
        plot(t, smAmpNorm, '-', 'Color', colorVals(9, :)./255, 'LineWidth', 1);
    else
        plot(t, smAmpNorm, ':', 'Color', colorVals(9, :)./255, 'LineWidth', 1);
    end
    hold on;
end

indDCs = find((strcmp(birdSexCutsMeans, 'M') | strcmp(birdSexCutsMeans, 'F') ) & (strcmp(vocTypeCutsMeans, 'DC')));
meanAmpDCs = grpstats(callstAmpMeans(indDCs,:), birdSexCutsMeans(indDCs)');
meanAmpDCsNorm = grpstats(callstAmpNormMeans(indDCs,:), birdSexCutsMeans(indDCs)');

sex_name_DC = unique(birdSexCutsMeans(indTets), 'stable');

for ig = 1:2
    smAmp = sgolayfilt(meanAmpDCs(ig,:),3,21);
    smAmpNorm = sgolayfilt(meanAmpDCsNorm(ig,:),3,21);
    figure(7);
    if ig == 1
        plot(t, smAmp, '-', 'Color', colorVals(10, :)./255, 'LineWidth', 1);
    else
        plot(t, smAmp, ':', 'Color', colorVals(10, :)./255, 'LineWidth', 1);
    end
    hold on;
    figure(8);
    if ig == 1
        plot(t, smAmpNorm, '-', 'Color', colorVals(10, :)./255, 'LineWidth', 1);
    else
        plot(t, smAmpNorm, ':', 'Color', colorVals(10, :)./255, 'LineWidth', 1);
    end
    hold on;
end



figure (7);
legend([sex_name_Te, sex_name_DC]);
xlabel('Time (ms)');
ylabel('Amplitude');

hold off;

figure (8);
legend([sex_name_Te, sex_name_DC]);
xlabel('Time (ms)');
ylabel('Normalized Amp');

hold off;

