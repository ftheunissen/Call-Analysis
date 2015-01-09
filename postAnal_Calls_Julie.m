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

%% Model for each Acoustical Parameter and Plotting

% Make space for statistical results
R2AType = zeros(nAcoust, 1);
pType = zeros(nAcoust, 1);        % p value for lme model with just Type (includes song)
R2ATypeSex = zeros(nAcoust, 1);
R2ATypeS = zeros(nAcoust, 1);
pTypeS = zeros(nAcoust, 1);       % p value in Wald test for Type in complete model (excludes song)
pSex = zeros(nAcoust, 1);         % p value in Wald test for Sex in complete model (excludes song)
pSexType = zeros(nAcoust, 1);     % p value in Wald test for InterAction in complete model (excludes song)
pCompare = zeros(nAcoust, 1);

for ia=1:nAcoust
    figure(ia);
    
    [acoustMean, acoustMeanCI, nameType] = grpstats(Acoust(:,ia),vocTypeCuts',{'mean','meanci', 'gname'});
    [acoustMeanSorted, indSorted] = sort(acoustMean);

    % Make a table of the data needed and average per bird.
    T = table(Acoust(:,ia), vocTypeCuts', birdNameCuts', 'VariableNames', { xtag{ia}, 'Type', 'Bird'});
    TBird = grpstats(T, {'Type', 'Bird'}, 'mean', 'DataVars', xtag{ia});
    
    modelV2 = fitlme(TBird, sprintf('mean_%s ~  Type + (1|Bird)', xtag{ia}), 'DummyVarCoding', 'effects');
    stats = anova(modelV2);
    
    % boxplot([callAnalData.stdspect],{callAnalData.type}, 'grouporder', nameGrp(indSorted), 'colorgroup', nameGrp(indSorted), 'colors', colorplot, 'boxstyle', 'filled', 'outliersize', 1, 'whisker', 100);
    %
    % xlabel('Call Type');
    % ylabel('Bandwidth (Hz)');
    R2AType(ia) = modelV2.Rsquared.Adjusted;
    pType(ia) = stats.pValue(2);
    
    modelV2Full = fitlme(TBird, sprintf('mean_%s ~  -1 + Type + (1|Bird)', xtag{ia}), 'DummyVarCoding', 'full');
    CI = coefCI(modelV2Full);
        % Calculate p-values for different from mean
    nTypes = length(acoustMean);
    pValMean = zeros(nTypes, 1);
    for i=1:nTypes
        H = zeros(1, nTypes);
        H(i) = 1;
        pValMean(i) = coefTest(modelV2Full,H, mean(acoustMean));
    end
       
    fprintf('\nLME for %s (no Sex) p=%.2g R2A=%.2f Overall Mean %.3f\n', xtag{ia}, stats.pValue(2), modelV2.Rsquared.Adjusted, mean(acoustMean));
    for i=1:nTypes
        fprintf(1, '\t%s %.3f (%.3f-%.3f) %.4f\n', nameType{indSorted(i)}, acoustMeanSorted(i), CI(indSorted(i),1), CI(indSorted(i),2), pValMean(indSorted(i)));
    end
    
    indSorted2 = zeros(1, ngroups*2-1);
    
    koff = 1;
    for i=1:ngroups
        if indSorted(i) == indSong
            indSorted2(2*i-koff) = 2*indSorted(i)-1;
            koff = 2;
        elseif indSorted(i) > indSong
            indSorted2(2*i-koff) = 2*indSorted(i)-2;
            indSorted2(2*i-koff+1) = 2*indSorted(i)-1;
        else
            indSorted2(2*i-koff) = 2*indSorted(i)-1;
            indSorted2(2*i-koff+1) = 2*indSorted(i);
        end
    end
    
    T = table(Acoust(indSexNoSo, ia), vocTypeCuts(indSexNoSo)', birdNameCuts(indSexNoSo)', birdSexCuts(indSexNoSo)', 'VariableNames', { xtag{ia}, 'Type', 'Bird', 'Sex'});
    TBird = grpstats(T, {'Type', 'Bird', 'Sex'}, 'mean', 'DataVars', xtag{ia});
    
    modelV2Sex = fitlme(TBird, sprintf('mean_%s ~ Type + Sex + Type:Sex + (1|Bird)', xtag{ia}), 'DummyVarCoding', 'effects');
    modelV2NoSex = fitlme(TBird, sprintf('mean_%s ~ Type + (1|Bird)', xtag{ia}), 'DummyVarCoding', 'effects');
    stats = anova(modelV2Sex);
    
    boxplot(Acoust(indSex,ia),{vocTypeCuts(indSex) birdSexCuts(indSex)},  'grouporder', nameGrp2(indSorted2),...
        'colorgroup', nameGrp2(indSorted2), 'colors', colorplot2, 'boxstyle', 'filled', 'outliersize', 1, 'whisker', 100, 'factorgap', [3 0.5]);
    xlabel('Call Type');
    ylabel(xtag{ia});
    title(sprintf('%s*Sex p(SM)=%.2g p(Sex)=%.2g p(Int)=%.2g R2A=%.2f',...
        xtag{ia}, stats.pValue(2), stats.pValue(3), stats.pValue(4), modelV2Sex.Rsquared.Adjusted));
    R2ATypeSex(ia) = modelV2Sex.Rsquared.Adjusted;
    R2ATypeS(ia) = modelV2NoSex.Rsquared.Adjusted;
    
    % P values
    pTypeS(ia) = stats.pValue(2);
    pSex(ia) = stats.pValue(3);
    pSexType(ia) = stats.pValue(4);
    
    % Likelihood ratio test
    Fval = ((modelV2NoSex.SSE - modelV2Sex.SSE)/(modelV2Sex.NumCoefficients-modelV2NoSex.NumCoefficients))/(modelV2Sex.SSE/modelV2Sex.DFE);
    pCompare(ia) = fcdf(Fval,modelV2Sex.NumCoefficients-modelV2NoSex.NumCoefficients,modelV2Sex.DFE,'upper');
    
    acoustMale = T{strcmp(T.Sex, 'M'),1};
    acoustFemale = T{strcmp(T.Sex, 'F'),1};
    
    fprintf(1,'LME %s*Sex p(SM)=%.2g p(Sex)=%.2g p(Int)=%.2g R2A=%.2f\n',...
        xtag{ia}, stats.pValue(2), stats.pValue(3), stats.pValue(4), modelV2Sex.Rsquared.Adjusted);
    fprintf(1,'\t%s Females = %f Males = %f \n\n', xtag{ia}, nanmean(acoustFemale), nanmean(acoustMale));
    
    for i=1:ngroups
        if strcmp(nameGrp{i}, 'So')   % skip the songs for the paired ttest.
            continue;
        end
        
        indTest = find((strcmp(birdSexCuts, 'M') | strcmp(birdSexCuts, 'F')) & (strcmp(vocTypeCuts, nameGrp{i})));
        T = table(Acoust(indTest,ia), birdNameCuts(indTest)', birdSexCuts(indTest)', 'VariableNames', {xtag{ia}, 'Bird', 'Sex'});
        TBird = grpstats(T, {'Bird', 'Sex'}, 'mean', 'DataVars', xtag{ia});
        modelV2est = fitlm(TBird, sprintf('mean_%s ~ Sex', xtag{ia}));
        stats = anova(modelV2est);
        
        birdID = unique(T.Bird);
        acoustID = grpstats(T{:,1}, T.Bird);
        nbirdsID = length(birdID);
        birdSexID = cell(nbirdsID,1);
        for j=1:nbirdsID
            iTable = find(strcmp(birdID(j), T.Bird), 1, 'first');
            birdSexID{j} = T.Sex{iTable};
        end
        [h, p] = ttest2(acoustID(strcmp(birdSexID, 'M')), acoustID(strcmp(birdSexID, 'F')));
        acoustMale = T{strcmp(T.Sex, 'M'),1};
        acoustFemale = T{strcmp(T.Sex, 'F'),1};
               
        fprintf(1, 'Sex differences for %s plm = %.2g ptest = %.2g Female = %f Male = %f\n', nameGrp{i}, stats.pValue(1), p, nanmean(acoustFemale), nanmean(acoustMale));
    end
    
end

%  Make a figure of r2 results
figure(nAcoust+1);
[R2ATypeSorted, indSorted] = sort(R2AType);
bh = bar(R2ATypeSorted, 0.5, 'EdgeColor','k', 'FaceColor', [0.3 0.3 0.3], 'LineWidth', 1 );
axis([0.5 nAcoust+0.5 0 1]);
ph = get(bh, 'Parent');
set(ph, 'XTick', 1:nAcoust);
set(ph, 'XTickLabel', xtagPlot(indSorted));
xlabel('Acoustical Feature');
ylabel('Adjusted R2');
hold on;
for ia=1:nAcoust
    if pType(indSorted(ia)) < 0.05
        text(ia-0.1, 0.9, '*');
    end
end
hold off;

figure(nAcoust+2);
R2ADiff = R2ATypeSex - R2ATypeS;
[R2ADiffSorted, indDiffSorted] = sort(R2ADiff);
bh = bar(R2ADiffSorted, 0.5, 'EdgeColor','k', 'FaceColor', [0.3 0.3 0.3], 'LineWidth', 1 );
axis([0.5 nAcoust+0.5 0 0.2]);
ph = get(bh, 'Parent');
set(ph, 'XTick', 1:nAcoust);
set(ph, 'XTickLabel', xtagPlot(indDiffSorted));
xlabel('Acoustical Feature');
ylabel('Adjusted R2 Difference Sex');

hold on;
for ia=1:nAcoust
    if pCompare(indDiffSorted(ia)) < 0.05
        text(ia-0.1, 0.18, '*');
    end
end
hold off;

% Calculate correlation coefficient
corAcoust = zeros(nAcoust, nAcoust);
ZAcoust = zeros(size(Acoust));
for ia=1:nAcoust
    meanval = nanmean(Acoust(:,ia));
    stdval = nanstd(Acoust(:,ia));
    ZAcoust(:, ia) = (Acoust(:,ia) - meanval)./stdval;
end
fprintf(1,' ');
for ia=1:nAcoust
    fprintf(1,'\t%s', xtag{ia});
end
fprintf(1,'\n');

for ia=1:nAcoust
    fprintf(1,'%s', xtag{ia});
    for ja=1:nAcoust
        corAcoust(ia,ja) = nanmean(ZAcoust(:,ia).*ZAcoust(:,ja));
        fprintf(1,'\t%.2f', corAcoust(ia,ja));
    end
    fprintf(1,'\n');
end



%% Classifier Performance

% Perform the cross-validation
nPerm = 200;
PCC_Acoust = struct( 'nvalid', 0, ...
    'Total_DFA', 0,  'group_DFA', zeros(1, ngroups) , ...
    'Total_DFA_CI', zeros(1,2),  'group_DFA_CI', zeros(ngroups, 2) , ...
    'Conf_DFA', zeros(ngroups, ngroups), ...
    'Total_RFP', 0,  'group_RFP', zeros(1, ngroups) , ...
    'Total_RFP_CI', zeros(1,2),  'group_RFP_CI', zeros(ngroups, 2), ...
    'Conf_RFP', zeros(ngroups, ngroups) );

% Allocate space for distance vector and confusion matrix
Dist = zeros(1, ngroups);
n_validTot = 0;

for iperm=1:nPerm

    % Choose a random bird from each group for validation
    ind_valid = [];
    for ig = 1:ngroups
        indGrp = find(strcmp(vocTypeCuts,vocTypes{ig}));
        nGrp = length(indGrp);
        birdGrp = unique(birdNameCuts(indGrp));
        nBirdGrp = length(birdGrp);
        birdValid = randi(nBirdGrp, 1);
        indGrpValid = find(strcmp(vocTypeCuts,vocTypes{ig}) & strcmp(birdNameCuts, birdGrp{birdValid}));
        ind_valid = [ind_valid indGrpValid];
    end
    
    % ind_valid = find(strcmp(birdNameCuts, birdNames{ibird}));    % index of the validation calls
    n_valid = length(ind_valid);
        fprintf(1, 'Starting Permutation %d with %d in validation\n', iperm, n_valid);
    
    % Separate data into fitting and validation
    X_valid = Acoust(ind_valid, :);
    X_fit = Acoust;
    X_fit(ind_valid, :) = [];
    
    % Similarly for the group labels.
    Group_valid = vocTypeCuts(ind_valid);
    Group_fit = vocTypeCuts;
    Group_fit(ind_valid) = [];
    
    % Perform the linear DFA using manova1 for the training set
    fprintf(1, '\t Starting DFA\n');
    [nDF, p, stats] = manova1(X_fit, Group_fit);
    [mean_bgrp, sem_bgrp, meanbCI_grp, range_bgrp, name_bgrp] = grpstats(stats.canon(:,1:nDF),Group_fit', {'mean', 'sem', 'meanci', 'range', 'gname'});
    nbgroups = size(mean_bgrp,1);
    
    % Project the validation data set into the DFA.
    mean_X_fit = nanmean(X_fit);
    Xc = X_valid - repmat(mean_X_fit, size(X_valid,1), 1);
    Canon = Xc*stats.eigenvec(:, 1:nDF);
    
    % Use Euclidian Distances
    for i = 1:n_valid
        for j = 1:nbgroups
            Dist(j) = sqrt((Canon(i,:) - mean_bgrp(j,:))*(Canon(i,:) - mean_bgrp(j,:))');
            if strcmp(name_bgrp(j),Group_valid(i))
                k_actual = j;
            end
        end
        k_guess = find(Dist == nanmin(Dist), 1, 'first');
        
        % Just in case a group is missing find the index that corresponds
        % to the groups when all the data is taken into account.
        for j=1:ngroups
            if strcmp(name_grp(j), name_bgrp(k_actual))
                k_actual_all = j;
                break;
            end
        end
        for j=1:ngroups
            if strcmp(name_grp(j), name_bgrp(k_guess))
                k_guess_all = j;
                break;
            end
        end
        
        PCC_Acoust.Conf_DFA(k_actual_all, k_guess_all) = PCC_Acoust.Conf_DFA(k_actual_all, k_guess_all) + 1;
    end
    % Repeat using a random forest classifier with and without equal
    % prior
        fprintf(1, '\t Starting RF\n');
    BPrior = TreeBagger(200, X_fit, Group_fit, 'FBoot', 1.0, 'OOBPred', 'on', 'MinLeaf', 5, 'NPrint', 500, 'Prior', ones(1,ngroups).*(1/ngroups));
    Group_predict_prior = predict(BPrior, X_valid);
    
    for i = 1:n_valid
        k_actual = find(strcmp(name_grp,Group_valid(i)));
        k_guess = find(strcmp(name_grp,Group_predict_prior(i)));
        PCC_Acoust.Conf_RFP(k_actual, k_guess) = PCC_Acoust.Conf_RFP(k_actual, k_guess) + 1;
    end
    
     n_validTot = n_validTot + n_valid;
    
end

[PCC_Acoust.Total_DFA, PCC_Acoust.Total_DFA_CI]= binofit(sum(diag(PCC_Acoust.Conf_DFA)), n_validTot);
[PCC_Acoust.Total_RFP, PCC_Acoust.Total_RFP_CI]= binofit(sum(diag(PCC_Acoust.Conf_RFP)), n_validTot);

for i = 1:ngroups
    [PCC_Acoust.group_DFA(i), PCC_Acoust.group_DFA_CI(i,:)] = binofit(PCC_Acoust.Conf_DFA(i,i), sum(PCC_Acoust.Conf_DFA(i, :), 2));
    [PCC_Acoust.group_RFP(i), PCC_Acoust.group_RFP_CI(i,:)] = binofit(PCC_Acoust.Conf_RFP(i,i), sum(PCC_Acoust.Conf_RFP(i, :), 2));
end
PCC_Acoust.nvalid = n_validTot;

save('/Users/frederictheunissen/Documents/Data/Julie/FullVocalizationBank/vocTypeAcoust.mat', 'PCC_Acoust');

%% Print out the confusion Matrix
load '/Users/frederictheunissen/Documents/Data/Julie/FullVocalizationBank/vocTypeAcoust.mat';
figure();
% first re-organize the confusion matrix so the call types are in the right
% order
subplot(1,2,1);
tosortMatrix = PCC_Acoust.Conf_RFP;

sortedMatrix = zeros(size(tosortMatrix));
for rr = 1:ngroups
    tosortMatrix(rr,:) = 100.*(tosortMatrix(rr,:)./sum(tosortMatrix(rr,:)));
end
for rr = 1:ngroups
    rInd = find(strcmp(name_grp_plot(rr), name_grp));
    for cc = 1:ngroups
        cInd = find(strcmp(name_grp_plot(cc), name_grp));
        sortedMatrix(rr,cc) = tosortMatrix(rInd, cInd);
    end
end

imagesc(sortedMatrix, [0 100]);
xlabel('Guess');
ylabel('Actual');
axis square;
colormap(gray);
colorbar;
%title(sprintf('Spectro Confusion Matrix DFA\nPCA = %d PC Total= %.1f%% PCC Mean = %.1f%%', ...
%    PCC_info_bird.nb(inb), 100*PCC_info_bird.PCC_Total_DFA(inb), 100*mean(PCC_info_bird.PCC_group_DFA(inb,:))));

title(sprintf('RF (Equal Prior) Total = %.1f%% Type = %.1f%%', 100*PCC_Acoust.Total_RFP, 100*mean(PCC_Acoust.group_RFP)));
set(gca(), 'Ytick', 1:ngroups);
set(gca(), 'YTickLabel', name_grp_plot);
set(gca(), 'Xtick', 1:ngroups);
set(gca(), 'XTickLabel', name_grp_plot);


% first re-organize the confusion matrix so the call types are in the right
% order
subplot(1,2,2);
tosortMatrix = PCC_Acoust.Conf_DFA;

sortedMatrix = zeros(size(tosortMatrix));
for rr = 1:ngroups
    tosortMatrix(rr,:) = 100.*(tosortMatrix(rr,:)./sum(tosortMatrix(rr,:)));
end
for rr = 1:ngroups
    rInd = find(strcmp(name_grp_plot(rr), name_grp));
    for cc = 1:ngroups
        cInd = find(strcmp(name_grp_plot(cc), name_grp));
        sortedMatrix(rr,cc) = tosortMatrix(rInd, cInd);
    end
end

imagesc(sortedMatrix, [0 100]);
xlabel('Guess');
ylabel('Actual');
axis square;
colormap(gray);
colorbar;
%title(sprintf('Spectro Confusion Matrix DFA\nPCA = %d PC Total= %.1f%% PCC Mean = %.1f%%', ...
%    PCC_info_bird.nb(inb), 100*PCC_info_bird.PCC_Total_DFA(inb), 100*mean(PCC_info_bird.PCC_group_DFA(inb,:))));

title(sprintf('DFA Total = %.1f%% Type = %.1f%%', 100*PCC_Acoust.Total_DFA, 100*mean(PCC_Acoust.group_DFA)));
set(gca(), 'Ytick', 1:ngroups);
set(gca(), 'YTickLabel', name_grp_plot);
set(gca(), 'Xtick', 1:ngroups);
set(gca(), 'XTickLabel', name_grp_plot);

%% Print out the DFA results

% Perform the manova on the Zscored values to facilitate interpretation
[d_ind, p_ind, stats] = manova1(ZAcoust, vocTypeCuts);

% Calculate and plot group stats
[mean_groups, std_groups, sem_groups]= grpstats(stats.canon,vocTypeCuts', {'mean', 'std', 'sem'} ); % groups along first three principal components

% Plot groups along first 3 canonicals
figure();

subplot(1,2,1);
plot(mean_groups(:,1),mean_groups(:,2), 'k+');
hold on;
for ia = 1:ngroups
    plot( mean_groups(ia,1) + (std_groups(ia,1).*cos(0:pi/10:2*pi)), mean_groups(ia,2) + (std_groups(ia,2).*sin(0:pi/10:2*pi)),'k');
    text(mean_groups(ia,1)+0.1,mean_groups(ia,2)-.1, name_grp{ia});
end
axis([-4 4 -4 4]);
axis square;

zscore_eigenvector1 = stats.eigenval(1)*stats.eigenvec(:,1);
[sorted_eigenvector1, sort_index] = sort(abs(zscore_eigenvector1), 1,'descend');
xlabel( sprintf('%.3f %s + %.3f %s + %.3f %s', zscore_eigenvector1(sort_index(1)), xtagPlot{sort_index(1)}, ...
    zscore_eigenvector1(sort_index(2)), xtagPlot{sort_index(2)}, ...
    zscore_eigenvector1(sort_index(3)), xtagPlot{sort_index(3)}) );
zscore_eigenvector2 = stats.eigenval(2)*stats.eigenvec(:,2);
[sorted_eigenvector2, sort_index] = sort(abs(zscore_eigenvector2), 1,'descend');
ylabel( sprintf('%.3f %s + %.3f %s + %.3f %s', zscore_eigenvector2(sort_index(1)), xtagPlot{sort_index(1)}, ...
    zscore_eigenvector2(sort_index(2)), xtagPlot{sort_index(2)}, ...
    zscore_eigenvector2(sort_index(3)), xtagPlot{sort_index(3)}) );
title('DFA1 vs DFA2');
hold off;

subplot(1,2,2);
plot(mean_groups(:,1),mean_groups(:,3), 'k+');
hold on;
for ia = 1:ngroups
    plot( mean_groups(ia,1) + (std_groups(ia,1).*cos(0:pi/10:2*pi)), mean_groups(ia,3) + (std_groups(ia,3).*sin(0:pi/10:2*pi)),'k');
    text(mean_groups(ia,1)+0.1,mean_groups(ia,3)-0.1, name_grp{ia});
end
axis([-4 4 -4 4]);
axis square;
xlabel( sprintf('%.3f %s + %.3f %s + %.3f %s', zscore_eigenvector1(sort_index(1)), xtagPlot{sort_index(1)}, ...
    zscore_eigenvector1(sort_index(2)), xtagPlot{sort_index(2)}, ...
    zscore_eigenvector1(sort_index(3)), xtagPlot{sort_index(3)}) );
zscore_eigenvector3 = stats.eigenval(3)*stats.eigenvec(:,3);
[sorted_eigenvector3, sort_index] = sort(abs(zscore_eigenvector3), 1,'descend');
ylabel( sprintf('%.3f %s + %.3f %s + %.3f %s', zscore_eigenvector3(sort_index(1)), xtagPlot{sort_index(1)}, ...
    zscore_eigenvector3(sort_index(2)), xtagPlot{sort_index(2)}, ...
    zscore_eigenvector3(sort_index(3)), xtagPlot{sort_index(3)}) );

hold off;


%     scatter3(mean_groups(:,1),mean_groups(:,2), mean_groups(:,3), 'filled');
%     for ia = 1:length(animal_name)
%         text(mean_groups(ia,1)+0.05,mean_groups(ia,2)+.02, mean_groups(ia,3), animal_info(ia).Name);
%     end
title('DFA1 vs DFA3');


% Make a table or statistical results
fprintf(1, 'Number of significant dimensions d = %d\n', d_ind);
for id = 1:d_ind
    zscore_eigenvectord = stats.eigenvec(:,id);
    [sorted_eigenvectord, sort_index] = sort(abs(zscore_eigenvectord), 1,'descend');
    fprintf(1, '%d\t%.3f\t%.1f\t%.1f\t%.3f(%s) + %.3f(%s) + %.3f(%s) + %.3f(%s)\n\t\t\t\t\t+%.3f(%s) + %.3f(%s) + %.3f(%s) + %.3f(%s)\n', id, p_ind(id), stats.eigenval(id)*100/sum(stats.eigenval), sum(stats.eigenval(1:id))*100/sum(stats.eigenval),...
        zscore_eigenvectord(sort_index(1)), xtagPlot{sort_index(1)}, ...
        zscore_eigenvectord(sort_index(2)), xtagPlot{sort_index(2)}, ...
        zscore_eigenvectord(sort_index(3)), xtagPlot{sort_index(3)}, ...
        zscore_eigenvectord(sort_index(4)), xtagPlot{sort_index(4)}, ...
        zscore_eigenvectord(sort_index(5)), xtagPlot{sort_index(5)}, ...
        zscore_eigenvectord(sort_index(6)), xtagPlot{sort_index(6)}, ...
        zscore_eigenvectord(sort_index(7)), xtagPlot{sort_index(7)}, ...
        zscore_eigenvectord(sort_index(8)), xtagPlot{sort_index(8)}) ;
    
end


% Another option for the DFA figure to match the Spectro Features
name_grp_plot = {'Be', 'LT', 'Tu', 'Th', 'Di', 'Ag', 'Wh', 'Ne', 'Te', 'DC', 'So'};

colorVals = { [0 230 255], [0 95 255], [255 200 65], [255 150 40], [255 105 15],...
    [255 0 0], [255 0 255], [255 100 255], [255 180 255], [140 100 185], [0 0 0]}; 

if (length(name_grp_plot) ~= ngroups)
    fprintf(1, 'Error: missmatch between the length of name_grp_plot and the number of groups\n');
end


figure();
nDF = 5;

for iDF=2:nDF
    %subplot(2,ceil((nDF-1)/2), iDF-1);
    subplot(1, nDF, iDF-1);
    for ig=1:ngroups
        for ig_ind=1:ngroups
            if strcmp(name_grp_plot{ig}, name_grp{ig_ind})
                break;
            end
        end
        
        plot(mean_groups(ig_ind,1), mean_groups(ig_ind,iDF), 's', 'MarkerSize', 10, ...
                'color', colorVals{ig}./255,'MarkerEdgeColor','k',...
                'MarkerFaceColor',colorVals{ig}./255);
        hold on;
    end
    xlabel('DF1');
    ylabel(sprintf('DF%d', iDF));
    axis([-5 5 -5 5]);
    axis square;
    if iDF == nDF
        legend(name_grp_plot);
    end
    hold off;
end





