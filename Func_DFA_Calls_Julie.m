function Func_DFA_Calls_Julie(dirName, vocCutsName, dbplot)
% Perform a regularized DFA based on Spectrograms for the semantic
% categories.  The data base in vocCutsName is composed of cut spectrograms
% and is make by LoopforSites in /auto/fdata/fet/julie/
% dbplot is the flag to turn plots on and off...

%% Read the data base produced by VocSectioningAmp.m 
load(fullfile(dirName,vocCutsName));

% remove whines and mlnoise from data
indExclude = find(strcmp(vocTypeCuts, 'Wh') | strcmp(vocTypeCuts, 'mlnoise') | strcmp(vocTypeCuts, 'Tu') );
if (~isempty(indExclude) )
    spectroCutsTot(indExclude,:) = [];
    birdNameCuts(indExclude) = [];
    soundCutsTot(indExclude,:) = [];
    vocTypeCuts(indExclude) = [];
    %stimNameCuts(indExclude) = [];
end

% Clean up data
vocTypes = unique(vocTypeCuts);
ngroups = length(vocTypes);

for ig = 1:ngroups
    indVoc = find(strcmp(vocTypeCuts, vocTypes{ig}));
    if (length(indVoc) <= 1)
        spectroCutsTot(indVoc,:) = [];
        birdNameCuts(indVoc) = [];
        soundCutsTot(indVoc,:) = [];
        vocTypeCuts(indVoc) = [];
        %stimNameCuts(indVoc) = [];
    end
end

% Just in case we deleted some categories...
vocTypes = unique(vocTypeCuts);
ngroups = length(vocTypes);
        
% chose to normalize or not...
normFlg = 0;
ncutsTot = size(spectroCutsTot, 1);

if normFlg % Normalize by peak
    for i=1:ncutsTot
        spectroCutsTot(i,:) = spectroCutsTot(i,:) - max(spectroCutsTot(i,:));
    end
end

% Set a 100 dB range threshold
maxAmp = max(max(spectroCutsTot));
minAmp = maxAmp - 100;
spectroCutsTot(spectroCutsTot<minAmp) = minAmp;

%% Spectrogram analysis: Perform a regularized PCA on the spectrograms
% At this point this is just the vanilla PCA

[Coeff, Score, latent] = princomp(spectroCutsTot, 'econ'); 
% Here Coeff are the PCs organized in columns and Score is the projection
% of the spectrogram on the PCs. One could get the Scores from the Coeffs.

mSpectro = mean(spectroCutsTot);
% xSpectro = spectroCutsTot - repmat(mSpectro, size(spectroCutsTot,1), 1);
% Score = Spectro*Coeff


% Clear the spectrograms and sounds to make space
% clear spectroCutsTot;
clear soundCutsTot;


%% Spectrogram Analysis: Display the first nb PCs as Spectrograms

if (dbplot)
    figure(1);
    nb = min(100, size(Coeff, 2)); % Show the first 100 PCs here.
    
    clear cmin cmax clims
    cmin = min(min(Coeff(:, 1:nb)));
    cmax = max(max(Coeff(:, 1:nb)));
    cabs = max(abs(cmin), abs(cmax));
    clims = [-cabs cabs];
    nf = length(fo);
    nt = length(to);
    
    
    for i=1:nb
        subplot(ceil(nb/10),10,i);
        PC_Spect = reshape(Coeff(:,i), nf, nt);
        imagesc(to, fo, PC_Spect, clims)
        title(sprintf('PCA%d', i));
        axis xy;
        axis off;
    end
    
    figure(2);
    nb = min(300, length(latent));  % Show the cumulative distribution up to 300 pcs.
    cumlatent = cumsum(latent);
    plot(cumlatent(1:nb)./sum(latent));
    xlabel('PCA Dimensions');
    ylabel('Cumulative Variance');
end



%%  DFA and RF with cross-validation both within and across birds


nPerm = 1000;  
% nbMax = min(200, fix(length(latent)./2));
nbMax = min(20, fix(length(latent)./2));

PCC_info = struct('nb', num2cell(10:10:nbMax), 'nvalid', 0, ...
    'PCC_Total_DFA', 0, 'PCC_M_DFA', 0, 'PCC_group_DFA', zeros(ngroups, ngroups), ...
    'PCC_Total_RF', 0, 'PCC_M_RF', 0, 'PCC_group_RF', zeros(ngroups, ngroups));


n_valid = fix(0.1*ncutsTot); % Save 10% of the data for cross-validation


for inb = 1:length(PCC_info)
    
    % Number of PCA used in the DFA
    nb = PCC_info(inb).nb;
    fprintf(1, 'Starting %d permutations for %d PCs...\n', nPerm, nb);
    
    % Allocate space for distance vector and confusion matrix
    ConfMat_DFA = zeros(ngroups, ngroups);
    
    for iboot=1:nPerm
        
        ind_valid = randsample(ncutsTot, n_valid);    % index of the validation calls
        
        % Separate data into fitting and validation
        X_valid = Score(ind_valid, 1:nb);
        X_fit = Score(:, 1:nb);
        X_fit(ind_valid, :) = [];
        
        % Similarly for the group labels.
        Group_valid = vocTypeCuts(ind_valid);
        Group_fit = vocTypeCuts;
        Group_fit(ind_valid) = [];
        
        % Perform the linear DFA using manova1 for the training set
        [nDF, p, stats] = manova1(X_fit, Group_fit);
        [mean_bgrp, sem_bgrp, meanbCI_grp, range_bgrp, name_bgrp] = grpstats(stats.canon(:,1:nDF),Group_fit', {'mean', 'sem', 'meanci', 'range', 'gname'});
        nbgroups = size(mean_bgrp,1);
        Dist = zeros(1, nbgroups);
        
        % Project the validation data set into the DFA.
        mean_X_fit = mean(X_fit);
        Xc = X_valid - repmat(mean_X_fit, size(X_valid,1), 1);
        Canon = Xc*stats.eigenvec(:, 1:nDF);
        
        % Use Euclidian Distances
        for i = 1:n_valid
            k_actual = 0;
            for j = 1:nbgroups
                Dist(j) = sqrt((Canon(i,:) - mean_bgrp(j,:))*(Canon(i,:) - mean_bgrp(j,:))');
                if strcmp(name_bgrp(j),Group_valid(i))
                    k_actual = j;
                end
            end
            if k_actual == 0   % The validation set has a type that is not in the training set
                continue;
            end
            k_guess = find(Dist == min(Dist), 1, 'first');
            
            % Just in case a group is missing find the index that corresponds
            % to the groups when all the data is taken into account.
            for j=1:ngroups
                if strcmp(vocTypes(j), name_bgrp(k_actual))
                    k_actual_all = j;
                    break;
                end
            end
            for j=1:ngroups
                if strcmp(vocTypes(j), name_bgrp(k_guess))
                    k_guess_all = j;
                    break;
                end
            end
            
            ConfMat_DFA(k_actual_all, k_guess_all) = ConfMat_DFA(k_actual_all, k_guess_all) + 1;
        end
        
        
        
    end
    
    PCC_Total = 100.0*sum(diag(ConfMat_DFA))./(n_valid*nPerm);
    PCC_group = zeros(ngroups, ngroups);
    for i = 1:ngroups
        for j = 1:ngroups
            PCC_group(i,j) = ConfMat_DFA(i,j) / sum(ConfMat_DFA(i, :), 2) * 100; % sum(.., 2) = somme par ligne
        end
    end
    PCC_M = mean(diag(PCC_group));
    
    % Store the information  
    PCC_info(inb).nvalid = n_valid*nPerm;
    PCC_info(inb).PCC_Total_DFA = PCC_Total;
    PCC_info(inb).PCC_M_DFA = PCC_M;
    PCC_info(inb).PCC_group_DFA = PCC_group;
  
    
    % Now do the Random Forest Classification
    ConfMat_RF = zeros(ngroups, ngroups);
    
    % THis worked on a prior version of Matlab... B = TreeBagger(300, Score(:, 1:nb), vocTypeCuts, 'OOBPred', 'on', 'priorprob', 'equal', 'MinLeaf', 5, 'NPrint', 10);
    B = TreeBagger(300, Score(:,1:nb), vocTypeCuts, 'FBoot', 1.0, 'OOBPred', 'on', 'MinLeaf', 5, 'NPrint', 100);
    
    Group_predict = oobPredict(B);   % This returns the predictions for the out of bag values.
    
    n_validRF = length(vocTypeCuts);   % this is the total number of observations
    
    for i = 1:n_validRF
        k_actual = find(strcmp(vocTypes, vocTypeCuts(i)));
        k_guess = find(strcmp(vocTypes, Group_predict(i)));
        ConfMat_RF(k_actual, k_guess) = ConfMat_RF(k_actual, k_guess) + 1;
    end
    
    PCC_Total = 100.0*sum(diag(ConfMat_RF))./(n_validRF);
    
    PCC_group = zeros(ngroups, ngroups);
    for i = 1:ngroups
        for j = 1:ngroups
            PCC_group(i,j) = ConfMat_RF(i,j) / sum(ConfMat_RF(i, :), 2) * 100;
        end
    end
    PCC_M = mean(diag(PCC_group));
    
    % Store the information   
    PCC_info(inb).PCC_Total_RF = PCC_Total;
    PCC_info(inb).PCC_M_RF = PCC_M;
    PCC_info(inb).PCC_group_RF = PCC_group;
        
end


% Display confusion Matrix
if (dbplot)
    figure(3);
    for inb = 1:length(PCC_info)
        
        subplot(2,length(PCC_info),inb);
        imagesc(PCC_info(inb).PCC_group_DFA,[0 100]);
        xlabel('Guess');
        ylabel('Actual');
        colormap(gray);
        %colorbar;
        title(sprintf('DFA %.1f%%(%.1f%%)', PCC_info(inb).PCC_Total_DFA, PCC_info(inb).PCC_M_DFA));
        set(gca(), 'Ytick', 1:ngroups);
        set(gca(), 'YTickLabel', vocTypes);
        set(gca(), 'Xtick', 1:ngroups);
        axis square;
        % set(gca(), 'XTickLabel', vocTypes);
        
        subplot(2,length(PCC_info),length(PCC_info)+inb);
        imagesc(PCC_info(inb).PCC_group_RF, [0 100]);
        xlabel('Guess');
        ylabel('Actual');
        colormap(gray);
        %colorbar;
        title(sprintf(' RF %.1f%%(%.1f%%)', PCC_info(inb).PCC_Total_RF, PCC_info(inb).PCC_M_RF));
        set(gca(), 'Ytick', 1:ngroups);
        set(gca(), 'YTickLabel', vocTypes);
        set(gca(), 'Xtick', 1:ngroups);
        axis square;
        % set(gca(), 'XTickLabel', vocTypes);
    end
    
    figure(4);
    plot([PCC_info.nb], [PCC_info.PCC_Total_DFA],'k');
    hold on;
    plot([PCC_info.nb], [PCC_info.PCC_M_DFA],'r');
    plot([PCC_info.nb], [PCC_info.PCC_Total_RF],'k--');
    plot([PCC_info.nb], [PCC_info.PCC_M_RF],'r--');
    xlabel('Number of PCs');
    ylabel('Correct Classification');
    legend('DFA all', 'DFA Type','RF all', 'RF Type');
    hold off;
end



%% Perform the logistic regression in a pairwise fashion
% Here we used the space obtained in the DFA...


nScores = size(Score,1);

nb = min(40, size(Score,2));     % should match the nb used to obtain the statsDFA used below
[nDF, p, statsDFA] = manova1(Score(:, 1:nb), vocTypeCuts);
%  Display the significant DFA in the spectrogram space
PC_DF = Coeff(:, 1:nb) * statsDFA.eigenvec(:, 1:nDF);


PC_LR = zeros(size(Coeff,1),ngroups);
PC_LR_Bias = zeros(1,ngroups);
PCCoeffLR = zeros(nScores,ngroups);

for ig = 1:ngroups
    yVocType = zeros(nScores, 1);
    yVocType(strcmp(vocTypeCuts, vocTypes{ig})) = 1;
    
    [BLR, devianceLR, statsLR] =  glmfit(statsDFA.canon(:,1:nDF), yVocType, 'binomial');
    PC_LR(:,ig) = Coeff(:, 1:nb) * (statsDFA.eigenvec(1:nb,1:nDF)*BLR(2:end, 1));
    PC_LR_Bias(ig) = BLR(1,1);
    xval = glmval(BLR, statsDFA.canon(:,1:nDF), 'identity');
    yval = glmval(BLR, statsDFA.canon(:,1:nDF), 'logit');
    PCCoeffLR(:,ig) = xval;
    
    if (dbplot)
        figure(5);
        subplot(ngroups, 1, ig);
        hold on;
        plot(xval(yVocType==1), yval(yVocType==1), 'xr');
        plot(xval(yVocType==0), yval(yVocType==0), 'xk');
        hold off;
        title(vocTypes{ig});
    end
end

if (dbplot)
    figure(6);
    nf = length(fo);
    nt = length(to);
    for ig = 1:ngroups
        clear cmin cmax clims
        cmin = min(PC_LR(:,ig));
        cmax = max(PC_LR(:,ig));
        cabs = max(abs(cmin), abs(cmax));
        clims = [-cabs cabs];
        subplot(1, ngroups, ig);
        imagesc(to,fo, reshape(PC_LR(:, ig), nf, nt), clims);
        axis xy;
        axis off;
        title(vocTypes{ig});
    end
end

%% Save the results from the DFA
%save(fullfile(dirName, sprintf('DFA%s', vocCutsName)),'normFlg', 'fo', 'to', 'PCC_info', 'ngroups', 'vocTypes', 'PC_LR', 'PCCoeffLR', 'birdNameCuts', 'vocTypeCuts', 'stimNameCuts');
save(fullfile(dirName, sprintf('DFA%s', vocCutsName)),'normFlg', 'fo', 'to', 'PCC_info', 'ngroups', 'vocTypes', 'PC_LR', 'PC_LR_Bias', 'PCCoeffLR', 'birdNameCuts', 'vocTypeCuts', 'mSpectro');



