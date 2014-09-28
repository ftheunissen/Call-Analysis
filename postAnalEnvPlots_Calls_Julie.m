%% Post processing description of calls

% Load the data.
load('/Users/frederictheunissen/Documents/Data/Julie/FullVocalizationBank/vocCutsAnalwPSD_test.mat');

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

nameGrp = unique({callAnalData.type},'stable');   % Names in the order found in original data set
ngroups = length(nameGrp);
indSong = find(strcmp(nameGrp, 'So'));

indSex = find(strcmp({callAnalData.birdSex}, 'M') | strcmp({callAnalData.birdSex}, 'F')); 
indAge = find(strcmp({callAnalData.birdAge}, 'A') | strcmp({callAnalData.birdAge}, 'C'));
indSexNoSo = find((strcmp({callAnalData.birdSex}, 'M') | strcmp({callAnalData.birdSex}, 'F')) & ~(strcmp({callAnalData.type}, 'So')));

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
callstAmp = zeros(ncalls, nt);

for ic = 1: ncalls
    callsPSD(ic, :) = (callAnalData(ic).psd.Data)';
    callstAmp(ic, :) = callAnalData(ic).tAmp;
    
end


%% Plot the mean values of the PSD

meanPSD = grpstats(callsPSD, vocTypeCuts');

maxPow = max(max(meanPSD));


fvals = callAnalData(1).psd.Frequencies;
for ig = 1:ngroups
    powvals = 20*log10(meanPSD(ig, :)./maxPow)+100;
    smPow = sgolayfilt(powvals,3,51);
    figure(1);
    plot(fvals, smPow, '-', 'Color', colorplot(ig, :), 'LineWidth', 2);
    hold on;
    figure(2);
    plot(fvals, smPow./max(smPow), '-', 'Color', colorplot(ig, :), 'LineWidth', 2);
    hold on;
    figure(10+ig);
    plot(fvals, smPow./max(smPow), '-', 'Color', colorplot(ig, :), 'LineWidth', 2);
    [pks, locs] = findpeaks(smPow./max(smPow), 'MINPEAKHEIGHT', 0.5, 'MINPEAKDISTANCE', 30);
    title(nameGrp(ig));
    xlabel('Frequency (Hz)');
    ylabel('Power dB');
    axis([250 10000 0.1 1.1]);
    hold on;
    npks = length(pks);
    fprintf(1,'Call Type %s:', nameGrp{ig});
    for ip=1:npks
        plot([fvals(locs(ip)) fvals(locs(ip))], [0.1 1.1], 'k--');
        if (fvals(locs(ip)) < 10000)
            fprintf(1, '\t%4.2f', fvals(locs(ip))./1000);
        end       
    end
    fprintf(1,'\n');
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
ylabel('Power Normalized to Peak');
axis([250 10000 0.1 1.1]);
hold off;


%% Plot the mean values of the enveloppe

meanAmp = grpstats(callstAmp, vocTypeCuts');

maxAmp = max(max(meanAmp));
t = 0:nt-1;

for ig = 1:ngroups
    smAmp = sgolayfilt(meanAmp(ig,:),3,21);
    figure(3);
    plot(t, smAmp, '-', 'Color', colorplot(ig, :), 'LineWidth', 2);
    hold on;
    figure(4);
    plot(t, smAmp./max(smAmp), '-', 'Color', colorplot(ig, :), 'LineWidth', 2);
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

indTets = find((strcmp({callAnalData.birdSex}, 'M') | strcmp({callAnalData.birdSex}, 'F') ) & (strcmp(vocTypeCuts, 'Te')));


meanPSDTets = grpstats(callsPSD(indTets,:), {callAnalData(indTets).birdSex}');
sex_name_Te = unique({callAnalData(indTets).birdSex}, 'stable');

maxPow = max(max(meanPSDTets));


fvals = callAnalData(1).psd.Frequencies;
for ig = 1:2
    powvals = 20*log10(meanPSDTets(ig, :)./maxPow)+100;
    smPow = sgolayfilt(powvals,3,51);
    figure(5);
    if ig == 1
        plot(fvals, smPow, '-', 'Color', colorVals(9, :)./255, 'LineWidth', 1);
    else
        plot(fvals, smPow, ':', 'Color', colorVals(9, :)./255, 'LineWidth', 1);
    end
    hold on;
    figure(6);
    if ig == 1
        plot(fvals, smPow./max(smPow), '-', 'Color', colorVals(9, :)./255, 'LineWidth', 1);
    else
        plot(fvals, smPow./max(smPow), ':', 'Color', colorVals(9, :)./255, 'LineWidth', 1);
    end
    hold on;
end

indDCs = find((strcmp({callAnalData.birdSex}, 'M') | strcmp({callAnalData.birdSex}, 'F') ) & (strcmp(vocTypeCuts, 'DC')));


meanPSDDCs = grpstats(callsPSD(indDCs,:), {callAnalData(indDCs).birdSex}');
sex_name_DC = unique({callAnalData(indDCs).birdSex}, 'stable');

maxPow = max(max(meanPSDDCs));

for ig = 1:2
    powvals = 20*log10(meanPSDDCs(ig, :)./maxPow)+100;
    smPow = sgolayfilt(powvals,3,51);
    figure(5);
    if ig == 1
        plot(fvals, smPow, '-', 'Color', colorVals(10, :)./255, 'LineWidth', 1);
    else
        plot(fvals, smPow, ':', 'Color', colorVals(10, :)./255, 'LineWidth', 1);
    end
    hold on;
    figure(6);
    if ig == 1
        plot(fvals, smPow./max(smPow), '-', 'Color', colorVals(10, :)./255, 'LineWidth', 1);
    else
        plot(fvals, smPow./max(smPow), ':', 'Color', colorVals(10, :)./255, 'LineWidth', 1);
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
axis([250 10000 0.1 1.1]);
hold off;



%% Plot the mean values of the enveloppe for distance call and tets and for male and female

indTets = find((strcmp({callAnalData.birdSex}, 'M') | strcmp({callAnalData.birdSex}, 'F') ) & (strcmp(vocTypeCuts, 'Te')));


meanAmpTets = grpstats(callstAmp(indTets,:), {callAnalData(indTets).birdSex}');
sex_name_Te = unique({callAnalData(indTets).birdSex}, 'stable');

t = 0:nt-1;
for ig = 1:2

    smAmp = sgolayfilt(meanAmpTets(ig,:),3,21);
    figure(7);
    if ig == 1
        plot(t, smAmp, '-', 'Color', colorVals(9, :)./255, 'LineWidth', 1);
    else
        plot(t, smAmp, ':', 'Color', colorVals(9, :)./255, 'LineWidth', 1);
    end
    hold on;
    figure(8);
    if ig == 1
        plot(t, smAmp./max(smAmp), '-', 'Color', colorVals(9, :)./255, 'LineWidth', 1);
    else
        plot(t, smAmp./max(smAmp), ':', 'Color', colorVals(9, :)./255, 'LineWidth', 1);
    end
    hold on;
end

indDCs = find((strcmp({callAnalData.birdSex}, 'M') | strcmp({callAnalData.birdSex}, 'F') ) & (strcmp(vocTypeCuts, 'DC')));


meanAmpDCs = grpstats(callstAmp(indDCs,:), {callAnalData(indDCs).birdSex}');
sex_name_DC = unique({callAnalData(indDCs).birdSex}, 'stable');

maxPow = max(max(meanAmpDCs));

for ig = 1:2
    smAmp = sgolayfilt(meanAmpDCs(ig,:),3,21);
    figure(7);
    if ig == 1
        plot(t, smAmp, '-', 'Color', colorVals(10, :)./255, 'LineWidth', 1);
    else
        plot(t, smAmp, ':', 'Color', colorVals(10, :)./255, 'LineWidth', 1);
    end
    hold on;
    figure(8);
    if ig == 1
        plot(t, smAmp./max(smAmp), '-', 'Color', colorVals(10, :)./255, 'LineWidth', 1);
    else
        plot(t, smAmp./max(smAmp), ':', 'Color', colorVals(10, :)./255, 'LineWidth', 1);
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

