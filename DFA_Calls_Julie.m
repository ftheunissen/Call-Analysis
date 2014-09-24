%% Perform a regularized DFA based on Spectrograms for all the zebra finch vocalizations

% Read the data base produced by VocSectioningAmp.m 
%load('/auto/fdata/fet/julie/FullVocalizationBank/vocCuts.mat');

load('/Users/frederictheunissen/Documents/Data/Julie/FullVocalizationBank/vocCuts.mat');

% Clean up the data
ind = find(strcmp(vocTypeCuts,'C-'));   % This corresponds to unknown-11
vocTypeCuts(ind) = [];     % Delete this bird because calls are going to be all mixed
spectroCutsTot(ind,:) = [];
birdNameCuts(ind) = [];
soundCutsTot(ind,:) = [];

ind = find(strcmp(vocTypeCuts,'WC'));   % These are copulation Whines.
vocTypeCuts(ind) = [];
spectroCutsTot(ind,:) = [];
birdNameCuts(ind) = [];
soundCutsTot(ind,:) = [];

ind = find(strcmp(vocTypeCuts,'-A'));
for i=1:length(ind)
   vocTypeCuts{ind(i)} = 'Ag';
   birdNameCuts{ind(i)}
end

ind = find(strcmp(birdNameCuts, 'HpiHpi4748'));
for i=1:length(ind)
    birdNameCuts{ind(i)} = 'HPiHPi4748';
end

% Read the Bird info file
fid = fopen('/Users/frederictheunissen/Documents/Data/Julie/FullVocalizationBank/Birds_List_Acoustic.txt', 'r');
birdInfo = textscan(fid, '%s %s %s %s %s');
nInfo = length(birdInfo{1});
fclose(fid);

% Check to see if we have info for all the birds
birdNames = unique(birdNameCuts);
nBirds = length(birdNames);

birdInfoInd = zeros(1, nBirds);
for ibird=1:nBirds    
    for iinfo=1:nInfo
       if (strcmp(birdInfo{1}(iinfo), birdNames{ibird}) )
           birdInfoInd(ibird) = iinfo;
           break;
       end
    end
    if birdInfoInd(ibird) == 0
        fprintf(1, 'Warning no information for bird %d %s\n', ibird, birdNames{ibird});
    else
        fprintf(1, 'Bird %d: Name %s Gender %s Age %s\n', ibird, birdInfo{1}{iinfo}, birdInfo{2}{iinfo}, birdInfo{3}{iinfo});
    end
        
end

notFoundInd = find(birdInfoInd == 0 );
for i=1:length(notFoundInd)
    fprintf(1, 'Warning no information for bird %s\n', birdNames{notFoundInd(i)});
end
    


%%

% set the order in which the call types should be displayed in confusion
% matrices
% name_grp_plot = {'Be', 'LT', 'Tu', 'Th', 'Di', 'Ag', 'Wh', 'Ne', 'Te', 'DC', 'So'};

% Check to see if there are sufficient calls in each group

vocTypes = unique(vocTypeCuts);   % This returns alphabetical but it is not the same as the order returned by grpstats above
ngroups = length(vocTypes);

% Print a little report
fprintf(1, 'vocCuts has %d calls from %d Call Types obtained in %d birds\n', length(vocTypeCuts), ngroups, nBirds);

for ig=1:ngroups
    indGrp = find(strcmp(vocTypeCuts,vocTypes{ig}));
    nGrp = length(indGrp);
    birdGrp = unique(birdNameCuts(indGrp));
    nBirdGrp = length(birdGrp);
    fprintf(1, '\t Call type %s: %d calls %d birds\n', vocTypes{ig}, nGrp, nBirdGrp);
    
    % Remove category if nBirdGrp < 3
    if (nBirdGrp < 3 )
        spectroCutsTot(indGrp,:) = [];
        birdNameCuts(indGrp) = [];
        soundCutsTot(indGrp,:) = [];
        vocTypeCuts(indGrp) = [];
    end
end

% We do it again, just in case we deleted some categories 
birdNames = unique(birdNameCuts);
nBirds = length(birdNames);
vocTypes = unique(vocTypeCuts);   % This returns alphabetical but it is not the same as the order returned by manova and grpstats below...
ngroups = length(vocTypes);
    

% chose to normalize or not...
normFlg = 0;
nSample = size(spectroCutsTot, 1);

if normFlg % Normalize by peak
    for i=1:nSample
        spectroCutsTot(i,:) = spectroCutsTot(i,:) - max(spectroCutsTot(i,:));
    end
end

% Set a 100 dB range threshold
maxAmp = max(max(spectroCutsTot));
minAmp = maxAmp - 100;
spectroCutsTot(spectroCutsTot<minAmp) = minAmp;


%% Calculate the Mel Cepstral Coefficients and the modulation power spectrum.

% Define parameters for Mel Cepstral Coefficients calculations
Tw = 25;                % analysis frame duration (ms)
Ts = 10;                % analysis frame shift (ms)
alpha = 0.97;           % preemphasis coefficient
M = 25;                 % number of filterbank channels 
C = 12;                 % number of cepstral coefficients
L = 22;                 % cepstral sine lifter parameter
LF = 500;               % lower frequency limit (Hz)
HF = 8000;              % upper frequency limit (Hz)
dbplot = 1;             % Set to 1 for plots and pause
dbres = 50;             % Resolution in dB for plots

nsounds = size(soundCutsTot, 1);      % Number of calls in the library
soundlen = size(soundCutsTot, 2);     % Length of sound in points

nf = length(fo);  % Number of frequency slices in our spectrogram
nt = length(to);  % Number of time slices in our spectrogram

% Define extra parameters for the modulation spectrum
HTW = 40;         % Maximum value for temporal modulation in Hz
HFW = 0.004;      % Maximum value for spectral modulation in cyc/Hz (s)

% Calculate the labels for temporal and spectral frequencies for the mod
% spectrum
fstep = fo(2);
for ib=1:ceil((nf+1)/2)
    dwf(ib)= (ib-1)*(1/(fstep*nf));
    if (ib > 1)
        dwf(nf-ib+2)=-dwf(ib);
    end
end
dwf = fftshift(dwf);
iwfl = find(dwf >=0, 1);
iwfh = find(dwf >= HFW, 1);

tstep = to(2);
for it=1:ceil((nt+1)/2)
    dwt(it) = (it-1)*(1/(tstep*nt));
    if (it > 1 )
        dwt(nt-it+2) = -dwt(it);
    end
end
dwt = fftshift(dwt);
iwtl = find(dwt >= -HTW, 1);
iwth = find(dwt >= HTW, 1);

dwf = dwf(iwfl:iwfh);
dwt = dwt(iwtl:iwth);
nwt = length(dwt);
% dwfRes = decimate(dwf, 4);
dwfRes = dwf;
nwfRes = length(dwfRes);

soundCutsMPS = zeros(nsounds, nwt*nwfRes);

% First call to get size of MFCCs
[ MFCCs, FBEs, FBEf, MAG, MAGf, frames ] = ...
    mfcc( soundCutsTot(1,:), samprate, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L );
lenMFCCs = size(MFCCs, 2);
soundCutsMFCCs = zeros(nsounds, C*lenMFCCs);
[ Nw, NF ] = size( frames );                % frame length and number of frames
time_frames = [0:NF-1]*Ts*0.001+0.5*Nw/samprate;  % time vector (s) for frames

if (dbplot)
    % Make a figure with right aspect ration and white background
    figure('Position', [30 30 400 1100], 'PaperPositionMode', 'auto', ...
        'color', 'w', 'PaperOrientation', 'portrait', 'Visible', 'on' );
    % Set colormap to seewave colors
    cmap = spec_cmap();
    colormap(cmap);
    
    % generate a 2d guasswin
    w1 = gausswin(11);
    n1 = length(w1);
    w2 = zeros(n1);
    for in=1:n1
        w2(in,:) = w1.*w1(in);
    end
    
end

for is = 7:nsounds

    % Calculate the Mel Frequency Cestral Coefficients or mfcc
    [ MFCCs, FBEs, FBEf, MAG, MAGf, frames ] = ...
        mfcc( soundCutsTot(is,:), samprate, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L );
    soundCutsMFCCs(is, :) = reshape(MFCCs(2:end,:), 1, C*lenMFCCs); 

    % Calculate the Modulation Power Spectrum
    soundSpect = reshape(spectroCutsTot(is,:), nf, nt);   % This is the spectrogram calculated in VocSectioningAmp
    soundSpect_floor = max(max(soundSpect))-dbres;
    soundSpect( soundSpect<soundSpect_floor ) = soundSpect_floor;   % Force a noise floor
    soundSpect = soundSpect - mean(mean(soundSpect));               %Subtract DC
    modSpect = fft2(soundSpect);                                    % calculate the 2D fft    
    ampModSpect = fftshift(abs(modSpect));                                    % calculate amplitude
    ampModSpect = ampModSpect(iwfl:iwfh, iwtl:iwth);
    
    % Resample the modulation Power Spectrum to match the MFCCs dimensions

    ampModSpectRes = zeros(nwfRes, nwt);
    minVal = max(min(min(ampModSpect),0.0001*max(max(ampModSpect))));  % 80 db Res
    for iwt=1:nwt
%         ampModSpectResampled = decimate(ampModSpect(:,iwt), 4);
        ampModSpectResampled = ampModSpect(:,iwt);
        ampModSpectResampled( ampModSpectResampled < minVal ) = minVal;  % Resampling introduces negative numbers so we fix to minimum
        ampModSpectRes(:,iwt) = ampModSpectResampled;   
    end
    logAmpModSpect = 20*log10(ampModSpectRes);      % The log of the resampled MPS is used for discrimination
    soundCutsMPS(is, :) = reshape(logAmpModSpect, 1, nwt*nwfRes);
           
    if (dbplot)
        % Generate data needed for plotting

        time = [ 0:soundlen-1 ]/samprate;           % time vector (s) for signal samples
        logFBEs = 20*log10( FBEs );                 % compute log FBEs for plotting
        logFBEs_floor = max(max(logFBEs))-dbres;         % get logFBE floor dbres dB below max
        logFBEs( logFBEs<logFBEs_floor ) = logFBEs_floor; % limit logFBE dynamic range
        logMAG = 20*log10( MAG );                 % compute log FBEs for plotting
        logMAG_floor = max(max(logMAG))-dbres;         % get logFBE floor dbres dB below max
        logMAG( logMAG<logMAG_floor ) = logMAG_floor; 
        
        
        subplot( 611 );
        plot( time, soundCutsTot(is,:), 'k' );
        xlim( [ min(time_frames) max(time_frames) ] );
        %xlabel( 'Time (s)' );
        %ylabel( 'Amplitude' );
        axis off;
        title(vocTypeCuts{is});

        subplot( 612 );

        imagesc(to, fo, soundSpect)
        axis( 'xy' );
        xlim( [ min(time_frames) max(time_frames) ] );
        ylim( [LF HF]);
        xlabel( 'Time (s)' );
        ylabel( 'Frequency (Hz)' );
        title( 'Spectrogram for PCA');
        
        subplot( 613 );
        imagesc( time_frames, MAGf, logMAG );
        axis( 'xy' );
        xlim( [ min(time_frames) max(time_frames) ] );
        ylim( [LF HF]);
        xlabel( 'Time (s)' );
        ylabel( 'Frequency (Hz)' );
        title( 'Spectrogram for Mel');
              
        subplot( 614 );
        imagesc( time_frames, log10(FBEf), logFBEs );
        axis( 'xy' );
        xlim( [ min(time_frames) max(time_frames) ] );
        ylim( [log10(LF) log10(HF)] );
        xlabel( 'Time (s)' );
        ylabel( 'Frequency (Log10 Hz)' );
        title( 'Mel Spectrogram');
        
        subplot( 615 );
        imagesc( time_frames, [1:C], MFCCs(2:end,:) ); % HTK's TARGETKIND: MFCC
        %imagesc( time_frames, [1:C+1], MFCCs );       % HTK's TARGETKIND: MFCC_0
        axis( 'xy' );
        xlim( [ min(time_frames) max(time_frames) ] );
        xlabel( 'Time (s)' );
        ylabel( 'Cepstrum index' );
        title( 'Mel frequency cepstrum' );
        
        subplot( 616 );
        logAmpModSpect_floor = max(max(logAmpModSpect))-dbres;
        ampModSpectPlot = logAmpModSpect;
        ampModSpectPlot( logAmpModSpect <logAmpModSpect_floor ) = logAmpModSpect_floor;
        
        imagesc(dwt, dwfRes.*1000, ampModSpectPlot);
        title('Modulation Power Spectrum');
        axis xy;
        xlabel('Temporal Modulation (Hz)');
        ylabel('Spectral Modulation (cyc/kHz)');
               
        pause();
    end
end

% Replace the nan by a -dbres floor.
soundCutsMFCCs(isnan(soundCutsMFCCs)==1) = -dbres;
  
    

%% Spectrogram analysis: Perform a regularized PCA on the spectrograms
% At this point this is just the vanilla PCA

clear soundCutsTot;


[Coeff, Score, latent] = princomp(spectroCutsTot, 'econ'); 
% Here Coeff are the PCs organized in columns and Score is the projection
% of the spectrogram on the PCs. One could get the Scores from the Coeffs.
% mSpectro = mean(spectroCutsTot);
% xSpectro = spectroCutsTot - repmat(mSpectro, size(spectroCutsTot,1), 1);
% Score = Spectro*Coeff

% We only save nb Scores and clear the spectrograms and sounds to make space
clear spectroCutsTot;
 
% if normFlg
%     save /Users/frederictheunissen/Documents/Data/Julie/Calls/vocPCANorm.mat
% else
%     save /Users/frederictheunissen/Documents/Data/Julie/Calls/vocPCA.mat
% end

if normFlg
    save -v7.3 /auto/fdata/fet/julie/FullVocalizationBank/vocPCANorm.mat
else
    save -v7.3 /auto/fdata/fet/julie/FullVocalizationBank/vocPCA.mat
end


%% Start here if you've already done the PCA, the Mel Spectrum and MPS calculation
%load /auto/fdata/fet/julie/FullVocalizationBank/vocPCA.mat
% load /Users/elie/Documents/MATLAB/calls_analysis/vocPCA.mat
%load /Users/frederictheunissen/Documents/Data/Julie/BanqueCrisNeuro/vocPCANorm.mat
load /Users/frederictheunissen/Documents/Data/Julie/FullVocalizationBank/vocPCA.mat

%% Spectrogram Analysis: Display the first nb PCs as Spectrograms
figure(1);
nb = 20; % Show the first 50 PCs here.

clear cmin cmax clims
cmin = min(min(Coeff(:, 1:nb)));
cmax = max(max(Coeff(:, 1:nb)));
cabs = max(abs(cmin), abs(cmax));
clims = [-cabs cabs];
nf = length(fo);
nt = length(to);

for i=1:nb
    subplot(ceil(nb/4),4,i);
    PC_Spect = reshape(Coeff(:,i), nf, nt);
    imagesc(to, fo, PC_Spect, clims)
    title(sprintf('%d', i));
    axis xy;
    axis off;
end

figure(2);
nb = 300;  % Show the cumulative distribution up to 300 pcs.
cumlatent = cumsum(latent);
plot(cumlatent(1:nb)./sum(latent));
xlabel('PCA Dimensions');
ylabel('Cumulative Variance');




%%  Spectrogram Analysis: Perform a boot-strap by systematically swapping birds

nPerm = 200;

% Perform an initinal DFA to get all the groups
nb = 50;
[nDF50, p, stats] = manova1(Score(:, 1:nb), vocTypeCuts);
[mean_grp, sem_grp, meanCI_grp, range_grp, name_grp] = grpstats(stats.canon(:,1:nDF50),vocTypeCuts', {'mean', 'sem', 'meanci', 'range', 'gname'});


nbVals = [10, 20, 30, 40, 50, 75, 100];
nnb = length(nbVals);

PCC_info_bird = struct('nb', nbVals, 'nvalid', zeros(nnb,1), ...
    'PCC_Total_DFA', zeros(nnb,1),  'PCC_group_DFA', zeros(nnb,ngroups) , ...
    'PCC_Total_DFA_CI', zeros(nnb,2),  'PCC_group_DFA_CI', zeros(nnb,ngroups, 2) , ...
    'Conf_DFA', zeros(nnb, ngroups, ngroups), ...
    'PCC_Total_RF', zeros(nnb, 1), 'PCC_group_RF', zeros(nnb,ngroups), ...
    'PCC_Total_RF_CI', zeros(nnb, 2), 'PCC_group_RF_CI', zeros(nnb, ngroups, 2), ...
    'Conf_RF', zeros(nnb, ngroups, ngroups), ...
    'PCC_Total_RFP', zeros(nnb, 1), 'PCC_group_RFP', zeros(nnb,ngroups), ...
    'PCC_Total_RFP_CI', zeros(nnb, 2), 'PCC_group_RFP_CI', zeros(nnb, ngroups, 2), ...
    'Conf_RFP', zeros(nnb, ngroups, ngroups));


for inb = 1:nnb
    
    % Number of PCA used in the DFA
    nb = PCC_info_bird.nb(inb);
    
    % Allocate space for distance vector and confusion matrix
    Dist = zeros(1, ngroups);
    ConfMat_DFA = zeros(ngroups, ngroups);
    ConfMat_RF = zeros(ngroups, ngroups);
    ConfMat_RFP = zeros(ngroups, ngroups);
    n_validTot = 0;
    
    fprintf(1, 'Starting PC %d (%d/%d)\n', nb, inb, nnb);
    for iperm=1:nPerm
        fprintf(1, '%d ', iperm);
        if ( mod(iperm, 10) == 0 )
            [PCC_Total_DFA, PCC_Total_DFA_CI]= binofit(sum(diag(ConfMat_DFA)), n_validTot);
            [PCC_Total_RF, PCC_Total_RF_CI]= binofit(sum(diag(ConfMat_RF)), n_validTot);
            [PCC_Total_RFP, PCC_Total_RFP_CI]= binofit(sum(diag(ConfMat_RFP)), n_validTot);
            
            PCC_group_DFA = zeros(ngroups, 1);
            PCC_group_RF = zeros(ngroups, 1);
            PCC_group_RFP = zeros(ngroups, 1);
            PCC_group_DFA_CI = zeros(ngroups, 2);
            PCC_group_RF_CI = zeros(ngroups, 2);
            PCC_group_RFP_CI = zeros(ngroups, 2);
            for i = 1:ngroups               
                [PCC_group_DFA(i), PCC_group_DFA_CI(i,:)] = binofit(ConfMat_DFA(i,i), sum(ConfMat_DFA(i, :)));
                [PCC_group_RF(i), PCC_group_RF_CI(i,:)] = binofit(ConfMat_RF(i,i), sum(ConfMat_RF(i, :)));   
                [PCC_group_RFP(i), PCC_group_RFP_CI(i,:)] = binofit(ConfMat_RFP(i,i), sum(ConfMat_RFP(i, :)));   
                if ( (sum(ConfMat_DFA(i, :)) ~= sum(ConfMat_RF(i, :))) | (sum(ConfMat_DFA(i, :)) ~= sum(ConfMat_RFP(i, :))) )
                    fprintf(1, 'Something is wrong: different number of validations trials found for DFA (%d), RF (%d) or RFP (%d) for %s\n', sum(ConfMat_DFA(i, :)), sum(ConfMat_RF(i, :)), sum(ConfMat_RFP(i, :)), name_grp{i});
                end
            end
            fprintf(1,'\n');
            fprintf(1, 'Interim Results : DFA (%.2f-%.2f) RF (%.2f-%.2f) RFP (%.2f-%.2f)\n', PCC_Total_DFA_CI(1)*100, PCC_Total_DFA_CI(2)*100, PCC_Total_RF_CI(1)*100, PCC_Total_RF_CI(2)*100, PCC_Total_RFP_CI(1)*100, PCC_Total_RFP_CI(2)*100);
            fprintf(1, '\t\t DFA Group Min %.2f RF Group Min %.2f RFP Group Min %.2f\n', min(PCC_group_DFA)*100, min(PCC_group_RF)*100, min(PCC_group_RFP)*100);
            fprintf(1, '\t\t DFA Group Max %.2f RF Group Max %.2f RFP Group Max %.2f\n', max(PCC_group_DFA)*100, max(PCC_group_RF)*100, max(PCC_group_RFP)*100);
            
            int_DFA = PCC_group_DFA_CI(:,2) - PCC_group_DFA_CI(:,1);
            int_RF = PCC_group_RF_CI(:,2) - PCC_group_RF_CI(:,1);
            int_RFP = PCC_group_RFP_CI(:,2) - PCC_group_RFP_CI(:,1);
            maxDFAInd = find(int_DFA == max(int_DFA));
            maxRFInd = find(int_RF == max(int_RF));
            maxRFPInd = find(int_RFP == max(int_RFP));
            fprintf(1, '\t\t DFA Err max %.2f for %s\n\t\tRF Err max %.2f for %s\n\t\tRFP Err max %.2f for %s\n', max(int_DFA)*100, name_grp{maxDFAInd}, max(int_RF)*100, name_grp{maxRFInd}, max(int_RFP)*100, name_grp{maxRFPInd});
        end
        
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
        
        % Project the validation data set into the DFA.
        mean_X_fit = mean(X_fit);
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
            k_guess = find(Dist == min(Dist), 1, 'first');
            
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
            
            ConfMat_DFA(k_actual_all, k_guess_all) = ConfMat_DFA(k_actual_all, k_guess_all) + 1;
        end
        
        
        
        % Repeat using a random forest classifier with and without equal
        % prior
        B = TreeBagger(200, X_fit, Group_fit, 'FBoot', 1.0, 'OOBPred', 'on', 'MinLeaf', 5, 'NPrint', 500); 
        BPrior = TreeBagger(200, X_fit, Group_fit, 'FBoot', 1.0, 'OOBPred', 'on', 'MinLeaf', 5, 'NPrint', 500, 'Prior', ones(1,ngroups).*(1/ngroups)); 
        
        Group_predict = predict(B, X_valid);   % This returns the predictions for the cross validation set.
        Group_predict_prior = predict(BPrior, X_valid); 
        
        for i = 1:n_valid
            k_actual = find(strcmp(name_grp,Group_valid(i)));
            k_guess = find(strcmp(name_grp,Group_predict(i)));
            ConfMat_RF(k_actual, k_guess) = ConfMat_RF(k_actual, k_guess) + 1;
            k_guess = find(strcmp(name_grp,Group_predict_prior(i)));
            ConfMat_RFP(k_actual, k_guess) = ConfMat_RFP(k_actual, k_guess) + 1;
        end       
        
        n_validTot = n_validTot + n_valid;
        
    end
    
    [PCC_Total_DFA, PCC_Total_DFA_CI]= binofit(sum(diag(ConfMat_DFA)), n_validTot);
    [PCC_Total_RF, PCC_Total_RF_CI]= binofit(sum(diag(ConfMat_RF)), n_validTot);
    [PCC_Total_RFP, PCC_Total_RFP_CI]= binofit(sum(diag(ConfMat_RFP)), n_validTot);
    
    PCC_group_DFA = zeros(ngroups, 1);
    PCC_group_RF = zeros(ngroups, 1);
    PCC_group_RFP = zeros(ngroups, 1);
    PCC_group_DFA_CI = zeros(ngroups, 2);
    PCC_group_RF_CI = zeros(ngroups, 2);
    PCC_group_RFP_CI = zeros(ngroups, 2);
    for i = 1:ngroups
        [PCC_group_DFA(i), PCC_group_DFA_CI(i,:)] = binofit(ConfMat_DFA(i,i), sum(ConfMat_DFA(i, :), 2));
        [PCC_group_RF(i), PCC_group_RF_CI(i,:)] = binofit(ConfMat_RF(i,i), sum(ConfMat_RF(i, :), 2));
        [PCC_group_RFP(i), PCC_group_RFP_CI(i,:)] = binofit(ConfMat_RFP(i,i), sum(ConfMat_RFP(i, :), 2));
    end
    fprintf(1,'\n');
    fprintf(1, 'Final Results : DFA (%.2f-%.2f) RF (%.2f-%.2f) RFP (%.2f-%.2f)\n', PCC_Total_DFA_CI(1)*100, PCC_Total_DFA_CI(2)*100, PCC_Total_RF_CI(1)*100, PCC_Total_RF_CI(2)*100, PCC_Total_RFP_CI(1)*100, PCC_Total_RFP_CI(2)*100);
    fprintf(1, '\t\t DFA Group Min %.2f RF Group Min %.2f RFP Group Min %.2f\n', min(PCC_group_DFA)*100, min(PCC_group_RF)*100, min(PCC_group_RFP)*100);
    fprintf(1, '\t\t DFA Group Max %.2f RF Group Max %.2f RFP Group Max %.2f\n', max(PCC_group_DFA)*100, max(PCC_group_RF)*100, max(PCC_group_RFP)*100);
    
    int_DFA = PCC_group_DFA_CI(:,2) - PCC_group_DFA_CI(:,1);
    int_RF = PCC_group_RF_CI(:,2) - PCC_group_RF_CI(:,1);
    int_RFP = PCC_group_RFP_CI(:,2) - PCC_group_RFP_CI(:,1);
    fprintf(1, '\t\t DFA Err max %.2f RF Err max %.2f RFP Err max %.2f\n', max(int_DFA)*100, max(int_RF)*100, max(int_RFP)*100);
    
    % Store the information
    PCC_info_bird.nvalid(inb) = n_validTot;
    PCC_info_bird.Conf_RF(inb,:,:) = ConfMat_RF;
    PCC_info_bird.Conf_RFP(inb,:,:) = ConfMat_RFP;
    PCC_info_bird.Conf_DFA(inb,:,:) = ConfMat_DFA;
    
    PCC_info_bird.PCC_Total_DFA(inb) = PCC_Total_DFA; 
    PCC_info_bird.PCC_Total_DFA_CI(inb,:) = PCC_Total_DFA_CI;
    PCC_info_bird.PCC_group_DFA(inb,:) = PCC_group_DFA; 
    PCC_info_bird.PCC_group_DFA_CI(inb,:,:) = PCC_group_DFA_CI;
    PCC_info_bird.PCC_Total_RF(inb) = PCC_Total_RF; 
    PCC_info_bird.PCC_Total_RF_CI(inb,:) = PCC_Total_RF_CI;
    PCC_info_bird.PCC_group_RF(inb,:) = PCC_group_RF;
    PCC_info_bird.PCC_group_RF_CI(inb,:,:) = PCC_group_RF_CI;
    PCC_info_bird.PCC_Total_RFP(inb) = PCC_Total_RFP; 
    PCC_info_bird.PCC_Total_RFP_CI(inb,:) = PCC_Total_RFP_CI;
    PCC_info_bird.PCC_group_RFP(inb,:) = PCC_group_RFP;
    PCC_info_bird.PCC_group_RFP_CI(inb,:,:) = PCC_group_RFP_CI;
    
end
% Display confusion Matrix

figure(8);
for inb = 1:nnb
    subplot(3,nnb,inb);
    imagesc(squeeze(PCC_info_bird.Conf_DFA(inb,:,:)));
    xlabel('Guess');
    ylabel('Actual');
    colormap(gray);
    colorbar;
    title(sprintf('Confusion Matrix DFA %.1f%%(%.1f%%) Correct', 100*PCC_info_bird.PCC_Total_DFA(inb), 100*mean(PCC_info_bird.PCC_group_DFA(inb,:))));
    set(gca(), 'Ytick', 1:ngroups);
    set(gca(), 'YTickLabel', name_grp);
    set(gca(), 'Xtick', 1:ngroups);
    set(gca(), 'XTickLabel', name_grp);
    
    subplot(3,nnb,nnb+inb);
    imagesc(squeeze(PCC_info_bird.Conf_RF(inb,:,:)));
    xlabel('Guess');
    ylabel('Actual');
    colormap(gray);
    colorbar;
    title(sprintf('Confusion Matrix RF %.1f%%(%.1f%%) Correct', 100*PCC_info_bird.PCC_Total_RF(inb), 100*mean(PCC_info_bird.PCC_group_RF(inb,:))));
    set(gca(), 'Ytick', 1:ngroups);
    set(gca(), 'YTickLabel', name_grp);
    set(gca(), 'Xtick', 1:ngroups);
    set(gca(), 'XTickLabel', name_grp);
    
    subplot(3,nnb,2*nnb+inb);
    imagesc(squeeze(PCC_info_bird.Conf_RFP(inb,:,:)));
    xlabel('Guess');
    ylabel('Actual');
    colormap(gray);
    colorbar;
    title(sprintf('Confusion Matrix RFP %.1f%%(%.1f%%) Correct', 100*PCC_info_bird.PCC_Total_RFP(inb), 100*mean(PCC_info_bird.PCC_group_RFP(inb,:))));
    set(gca(), 'Ytick', 1:ngroups);
    set(gca(), 'YTickLabel', name_grp);
    set(gca(), 'Xtick', 1:ngroups);
    set(gca(), 'XTickLabel', name_grp);
end

figure(9);
plot(PCC_info_bird.nb, 100*PCC_info_bird.PCC_Total_DFA,'k');
hold on;
plot(PCC_info_bird.nb, 100*mean(PCC_info_bird.PCC_group_DFA,2),'k--');
plot(PCC_info_bird.nb, 100*PCC_info_bird.PCC_Total_RF,'r');
plot(PCC_info_bird.nb, 100*mean(PCC_info_bird.PCC_group_RF, 2),'r--');
plot(PCC_info_bird.nb, 100*PCC_info_bird.PCC_Total_RFP,'b');
plot(PCC_info_bird.nb, 100*mean(PCC_info_bird.PCC_group_RFP, 2),'b--');
xlabel('Number of PCs');
ylabel('Correct Classification');
legend('DFA all', 'DFA Type','RF all', 'RF Type', 'RFP all', 'RFP Type');
hold off;

% save the DFA and RF for Spectrogram
if normFlg
    save vocTypeSpectroNorm.mat fo to PCC_info_bird ngroups name_grp
else
    save vocTypeSpectro.mat fo to PCC_info_bird ngroups name_grp
end

%% Spectrogram analysis: From these results we choose nb=40 PCs and make nice plots + save some data
% You can load vocTypeDFA.mat and vocPCA.mat or the normalized versions and and start here to make these
% final plots
%load /Users/frederictheunissen/Documents/Data/Julie/BanqueCrisNeuro/vocPCANorm.mat

% load '/auto/fdata/fet/julie/Acoustical Analysis/vocTypeSpectro.mat'
load '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/vocTypeSpectro.mat'
inb = 4;       % Corresponds to 50 PCs in PCC_info_bird
nb = 40;

name_grp_plot = {'Be', 'LT', 'Tu', 'Th', 'Di', 'Ag', 'Wh', 'Ne', 'Te', 'DC', 'So'};
colorVals = { [0 230 255], [0 95 255], [255 200 65], [255 150 40], [255 105 15],...
    [255 0 0], [255 0 255], [255 100 255], [255 180 255], [140 100 185], [0 0 0]}; 

if (length(name_grp_plot) ~= ngroups)
    fprintf(1, 'Error: missmatch between the length of name_grp_plot and the number of groups\n');
end

% A nice confusion matrix.

figure(10);
% first re-organize the confusion matrix so the call types are in the right
% order
subplot(1,2,1);
tosortMatrix = squeeze(PCC_info_bird.Conf_RFP(inb,:,:));

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

title(sprintf('RF (Equal Prior) Total = %.1f%% Type = %.1f%%', 100*PCC_info_bird.PCC_Total_RFP(inb), 100*mean(PCC_info_bird.PCC_group_RFP(inb,:))));
set(gca(), 'Ytick', 1:ngroups);
set(gca(), 'YTickLabel', name_grp_plot);
set(gca(), 'Xtick', 1:ngroups);
set(gca(), 'XTickLabel', name_grp_plot);


% first re-organize the confusion matrix so the call types are in the right
% order
subplot(1,2,2);
tosortMatrix = squeeze(PCC_info_bird.Conf_DFA(inb,:,:));

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

title(sprintf('DFA Total = %.1f%% Type = %.1f%%', 100*PCC_info_bird.PCC_Total_DFA(inb), 100*mean(PCC_info_bird.PCC_group_DFA(inb,:))));
set(gca(), 'Ytick', 1:ngroups);
set(gca(), 'YTickLabel', name_grp_plot);
set(gca(), 'Xtick', 1:ngroups);
set(gca(), 'XTickLabel', name_grp_plot);

figure(11);

[nDF, p, statsDFA] = manova1(Score(:, 1:nb), vocTypeCuts);
[mean_grp, std_grp, name_grp] = grpstats(statsDFA.canon(:,1:nDF),vocTypeCuts', {'mean', 'std', 'gname'});
fprintf(1, 'Number of significant dimensions d = %d\n', nDF);
for id = 1:nDF
    fprintf(1, '%d\t%.3f\t%.1f\t%.1f\n', id, p(id), statsDFA.eigenval(id)*100/sum(statsDFA.eigenval), sum(statsDFA.eigenval(1:id))*100/sum(statsDFA.eigenval)) ;    
end
    
%  Display the significant DFA in the spectrogram space
PC_DF = Coeff(:, 1:nb) * statsDFA.eigenvec(:, 1:nDF);

% Find color scale
clear cmin cmax clims
cmin = min(min(PC_DF(:, 1:nDF)));
cmax = max(max(PC_DF(:, 1:nDF)));
cabs = max(abs(cmin), abs(cmax));
clims = [-cabs cabs];
nf = length(fo);
nt = length(to);

nDF = 5;  % Let's go to 5...
for i=1:nDF
    subplot(1,nDF,i);
    PC_Spect = reshape(PC_DF(:,i), nf, nt);
    imagesc(to, fo, PC_Spect, clims)
    
    title(sprintf('DFA %d', i));
    
    axis xy;
    axis([0 0.350 0 8000]);
    if i ~= 1
        axis off;
    end
    
    if (i == 1)
        xlabel('Time (s)');
        ylabel('Frequency (Hz)');
    end
end

% Make DF1 vs DFX plots
figure(12);
    
nDF = 5;

for iDF=2:nDF
    %subplot(2,ceil((nDF-1)/2), iDF-1);
    subplot(1, nDF -1, iDF-1);
    for ig=1:ngroups
        for ig_ind=1:ngroups
            if strcmp(name_grp_plot{ig}, name_grp{ig_ind})
                break;
            end
        end
        
        plot(mean_grp(ig_ind,1), mean_grp(ig_ind,iDF), 's', 'MarkerSize', 10, ...
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

figure(13);   % Plot with error bars
subplot(1,2,1);
indPlot = zeros(1,ngroups);
bh = bar(1:ngroups, zeros(1, ngroups));
hold on;
for ig=1:ngroups
    for ig_ind=1:ngroups
        if strcmp(name_grp_plot{ig}, name_grp{ig_ind})
            indPlot(ig) = ig_ind;
            break;
        end
    end
    
    bar(ig, 100.*PCC_info_bird.PCC_group_RFP(inb,ig_ind), 'FaceColor', colorVals{ig}./255);
end
set(get(bh,'Parent'),'XTickLabel',name_grp_plot);
ylabel('Percent Correct Classification');
errorbar(1:ngroups,100.*PCC_info_bird.PCC_group_RFP(inb,indPlot),  ...
    100.*(PCC_info_bird.PCC_group_RFP(inb,indPlot)- squeeze(PCC_info_bird.PCC_group_RFP_CI(inb, indPlot, 1))),...
    100.*(squeeze(PCC_info_bird.PCC_group_RFP_CI(inb, indPlot, 2))-PCC_info_bird.PCC_group_RFP(inb,indPlot)), '+k');
plot([0 ngroups+1], [100/ngroups 100/ngroups], 'k--');
box off;
hold off;
axis([0 ngroups+1 0 100]);
title('Random Forest w Uniform Prior');

subplot(1,2,2);

bh = bar(1:ngroups, zeros(1, ngroups));
hold on;
for ig=1:ngroups    
    bar(ig, 100.*PCC_info_bird.PCC_group_DFA(inb,indPlot(ig)), 'FaceColor', colorVals{ig}./255);
end
set(get(bh,'Parent'),'XTickLabel',name_grp_plot);
ylabel('Percent Correct Classification');
errorbar(1:ngroups,100.*PCC_info_bird.PCC_group_DFA(inb,indPlot),  ...
    100.*(PCC_info_bird.PCC_group_DFA(inb,indPlot)- squeeze(PCC_info_bird.PCC_group_DFA_CI(inb, indPlot, 1))),...
    100.*(squeeze(PCC_info_bird.PCC_group_DFA_CI(inb, indPlot, 2))-PCC_info_bird.PCC_group_DFA(inb,indPlot)), '+k');
plot([0 ngroups+1], [100/ngroups 100/ngroups], 'k--');
box off;
hold off;
axis([0 ngroups+1 0 100]);
title('Discriminant Function Analysis');

%% Perform the logistic regression in a pairwise fashion
% Here we used the space obtained in the DFA...


nScores = size(Score,1);

nb = 40;     % should match the nb used to obtain the statsDFA used below
[nDF, p, statsDFA] = manova1(Score(:, 1:nb), vocTypeCuts);

PC_LR = zeros(size(Coeff,1),ngroups);
PC_LR_Bias = zeros(1,ngroups);
PCCoeffLR = zeros(nScores,ngroups);

for ig = 1:ngroups
    yVocType = zeros(nScores, 1);
    yVocType(strcmp(vocTypeCuts, vocTypes{ig})) = 1;
    
    
    [BLR, devianceLR, statsLR] =  glmfit(statsDFA.canon(:,1:nDF), yVocType, 'binomial');
    PC_LR(:,ig) = Coeff(:, 1:nb) * (statsDFA.eigenvec(1:nb,1:nDF)*BLR(2:end, 1));
    %
    %     [BLR, devianceLR, statsLR] =  glmfit(Score(:, 1:nb), yVocType, 'binomial');
    %     PC_LR(:,ig) = Coeff(:, 1:nb) *BLR(2:end, 1);
    
    PC_LR_Bias(ig) = BLR(1,1);
    
    xval = glmval(BLR, statsDFA.canon(:,1:nDF), 'identity');
    yval = glmval(BLR, statsDFA.canon(:,1:nDF), 'logit');
    
    %      xval = glmval(BLR, Score(:,1:nb), 'identity');
    %     yval = glmval(BLR, Score(:,1:nb), 'logit');
    
    PCCoeffLR(:,ig) = xval;
    
    correctDetection = 100.*(length(find(xval(yVocType==1) > 0))/length(xval(yVocType==1)));
    falsePositive = 100.*(length(find(xval(yVocType==0) > 0))/length(xval(yVocType==0)));
    
    figure(14);
    subplot(ngroups, 1, ig);
    hold on;
    plot(xval(yVocType==1), yval(yVocType==1), 'xr');
    plot(xval(yVocType==0), yval(yVocType==0), 'xk');
    axis([-10 10 0 1]);
    hold off;
    title(sprintf('%s : cd=%.1f %% fp=%.1f %%', vocTypes{ig}, correctDetection, falsePositive));
    
end


figure(15);
nf = length(fo);
nt = length(to);
for ig = 1:ngroups
    clear cmin cmax clims
    cmin = min(PC_LR(:,ig));
    cmax = max(PC_LR(:,ig));
    cabs = max(abs(cmin), abs(cmax));
    clims = [-cabs cabs];
    subplot(ngroups, 1, ig);
    imagesc(to,fo, reshape(PC_LR(:, ig), nf, nt), clims);
    axis xy;
    axis off;
    title(vocTypes{ig});
end

nPerm = 200;
% Repeat to obtain cross-validated
correctDFound = zeros(1, ngroups);
correctDPossible = zeros(1, ngroups);
falseDFound = zeros(1, ngroups);
falseDPossible = zeros(1, ngroups);

for ig = 1:ngroups
    yVocType = zeros(nScores, 1);
    yVocType(strcmp(vocTypeCuts, vocTypes{ig})) = 1;
    fprintf(1, 'Starting bootstrapped logistic for %s\n', vocTypes{ig});
    
    for iperm=1:nPerm
        ind_valid = [];
        for ig2 = 1:ngroups
            indGrp = find(strcmp(vocTypeCuts,vocTypes{ig2}));
            nGrp = length(indGrp);
            birdGrp = unique(birdNameCuts(indGrp));
            nBirdGrp = length(birdGrp);
            birdValid = randi(nBirdGrp, 1);
            indGrpValid = find(strcmp(vocTypeCuts,vocTypes{ig2}) & strcmp(birdNameCuts, birdGrp{birdValid}));
            ind_valid = [ind_valid indGrpValid];
        end
        
        % ind_valid = find(strcmp(birdNameCuts, birdNames{ibird}));    % index of the validation calls
        n_valid = length(ind_valid);
        
        % Separate data into fitting and validation
        X_valid = statsDFA.canon(ind_valid, 1:nDF);
        X_fit = statsDFA.canon(:,1:nDF);
        X_fit(ind_valid, :) = [];
        
        % Similarly for the group labels.
        Group_valid = yVocType(ind_valid);
        Group_fit = yVocType;
        Group_fit(ind_valid) = [];
        
        [BLR, devianceLR, statsLR] =  glmfit(X_fit, Group_fit, 'binomial');
        
        xval = glmval(BLR, X_valid, 'identity');
                
        correctDFound(ig) = correctDFound(ig) + length(find(xval(Group_valid==1) > 0));
        correctDPossible(ig) = correctDPossible(ig) + length(xval(Group_valid==1));
        falseDFound(ig) = falseDFound(ig) + length(find(xval(Group_valid==0) > 0));
        falseDPossible(ig) = falseDPossible(ig) + length(xval(Group_valid==0));       
        
    end
    fprintf(1,'CD = %.2f%% FP = %.2f%%\n', 100.0*(correctDFound(ig)/correctDPossible(ig)), 100.0*(falseDFound(ig)./falseDPossible(ig)));
end
        

%%  MEL Cepstrum analysis and MPS: Boot-strap to get cross-validated classification

birdNames = unique(birdNameCuts);
nBirds = length(birdNames);
vocTypes = unique(vocTypeCuts);   % This returns alphabetical 
name_grp = unique(vocTypeCuts, 'stable');  % This is the order returned by grpstats, manova, etcc
ngroups = length(vocTypes);

fprintf(1, 'Starting PCA for MFCCs\n');
[MELPC, CoeffMEL, latentMEL] = pca(soundCutsMFCCs);
cumlatentMEL = cumsum(latentMEL);

fprintf(1, 'Starting PCA for MPS\n');
[MPSPC, CoeffMPS, latentMPS] = pca(soundCutsMPS);
cumlatentMPS = cumsum(latentMPS);

% Perform the cross-validation
nPerm = 200;
nb = 40;   % Use 40 PC's
PCC_MM = struct( 'nvalid', 0, ...
    'MEL_Total_DFA', 0,  'MEL_group_DFA', zeros(1, ngroups) , ...
    'MEL_Total_DFA_CI', zeros(1,2),  'MEL_group_DFA_CI', zeros(ngroups, 2) , ...
    'MEL_Conf_DFA', zeros(ngroups, ngroups), ...
    'MPS_Total_DFA', 0,  'MPS_group_DFA', zeros(1, ngroups) , ...
    'MPS_Total_DFA_CI', zeros(1,2),  'MPS_group_DFA_CI', zeros(ngroups, 2) , ...
    'MPS_Conf_DFA', zeros(ngroups, ngroups), ...
    'MEL_Total_RFP', 0,  'MEL_group_RFP', zeros(1, ngroups) , ...
    'MEL_Total_RFP_CI', zeros(1,2),  'MEL_group_RFP_CI', zeros(ngroups, 2) , ...
    'MEL_Conf_RFP', zeros(ngroups, ngroups), ...
    'MPS_Total_RFP', 0,  'MPS_group_RFP', zeros(1, ngroups) , ...
    'MPS_Total_RFP_CI', zeros(1,2),  'MPS_group_RFP_CI', zeros(ngroups, 2) , ...
    'MPS_Conf_RFP', zeros(ngroups, ngroups) );

% Allocate space for distance vector and confusion matrix
Dist = zeros(1, ngroups);
n_validTot = 0;

for iperm=1:nPerm
    
    %

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
    % MFCC first
    % Separate data into fitting and validation
%     X_valid = soundCutsMFCCs(ind_valid, :);
%     X_fit = soundCutsMFCCs;
    X_valid = CoeffMEL(ind_valid, 1:nb);
    X_fit = CoeffMEL(:, 1:nb);
    X_fit(ind_valid, :) = [];
    
    % Similarly for the group labels.
    Group_valid = vocTypeCuts(ind_valid);
    Group_fit = vocTypeCuts;
    Group_fit(ind_valid) = [];
    
    % Perform the linear DFA using manova1 for the training set
    fprintf(1, '\t Starting DFA MEL\n');
    [nDF, p, stats] = manova1(X_fit, Group_fit);
    [mean_bgrp, sem_bgrp, meanbCI_grp, range_bgrp, name_bgrp] = grpstats(stats.canon(:,1:nDF),Group_fit', {'mean', 'sem', 'meanci', 'range', 'gname'});
    nbgroups = size(mean_bgrp,1);
    
    % Project the validation data set into the DFA.
    mean_X_fit = mean(X_fit);
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
        k_guess = find(Dist == min(Dist), 1, 'first');
        
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
        
        PCC_MM.MEL_Conf_DFA(k_actual_all, k_guess_all) = PCC_MM.MEL_Conf_DFA(k_actual_all, k_guess_all) + 1;
    end
    % Repeat using a random forest classifier with and without equal
    % prior
        fprintf(1, '\t Starting RF MEL\n');
    BPrior = TreeBagger(200, X_fit, Group_fit, 'FBoot', 1.0, 'OOBPred', 'on', 'MinLeaf', 5, 'NPrint', 500, 'Prior', ones(1,ngroups).*(1/ngroups));
    Group_predict_prior = predict(BPrior, X_valid);
    
    for i = 1:n_valid
        k_actual = find(strcmp(name_grp,Group_valid(i)));
        k_guess = find(strcmp(name_grp,Group_predict_prior(i)));
        PCC_MM.MEL_Conf_RFP(k_actual, k_guess) = PCC_MM.MEL_Conf_RFP(k_actual, k_guess) + 1;
    end
    
    % Now with the MPS
    % Separate data into fitting and validation
%     X_valid = soundCutsMPS(ind_valid, :);
%     X_fit = soundCutsMPS;
    X_valid = CoeffMPS(ind_valid, 1:nb);
    X_fit = CoeffMPS(:, 1:nb);
    X_fit(ind_valid, :) = [];
    
    
    % Perform the linear DFA using manova1 for the training set
        fprintf(1, '\t Starting DFA MPS\n');
    [nDF, p, stats] = manova1(X_fit, Group_fit);
    [mean_bgrp, sem_bgrp, meanbCI_grp, range_bgrp, name_bgrp] = grpstats(stats.canon(:,1:nDF),Group_fit', {'mean', 'sem', 'meanci', 'range', 'gname'});
    nbgroups = size(mean_bgrp,1);
    
    % Project the validation data set into the DFA.
    mean_X_fit = mean(X_fit);
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
        k_guess = find(Dist == min(Dist), 1, 'first');
        
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
        
        PCC_MM.MPS_Conf_DFA(k_actual_all, k_guess_all) = PCC_MM.MPS_Conf_DFA(k_actual_all, k_guess_all) + 1;
    end
    % Repeat using a random forest classifier with and without equal
    % prior
        fprintf(1, '\t Starting RF MPS\n');
    BPrior = TreeBagger(200, X_fit, Group_fit, 'FBoot', 1.0, 'OOBPred', 'on', 'MinLeaf', 5, 'NPrint', 500, 'Prior', ones(1,ngroups).*(1/ngroups));
    Group_predict_prior = predict(BPrior, X_valid);
    
    for i = 1:n_valid
        k_actual = find(strcmp(name_grp,Group_valid(i)));
        k_guess = find(strcmp(name_grp,Group_predict_prior(i)));
        PCC_MM.MPS_Conf_RFP(k_actual, k_guess) = PCC_MM.MPS_Conf_RFP(k_actual, k_guess) + 1;
    end
    
    n_validTot = n_validTot + n_valid;
    
end


[PCC_MM.MEL_Total_DFA, PCC_MM.MEL_Total_DFA_CI]= binofit(sum(diag(PCC_MM.MEL_Conf_DFA)), n_validTot);
[PCC_MM.MEL_Total_RFP, PCC_MM.MEL_Total_RFP_CI]= binofit(sum(diag(PCC_MM.MEL_Conf_RFP)), n_validTot);
[PCC_MM.MPS_Total_DFA, PCC_MM.MPS_Total_DFA_CI]= binofit(sum(diag(PCC_MM.MPS_Conf_DFA)), n_validTot);
[PCC_MM.MPS_Total_RFP, PCC_MM.MPS_Total_RFP_CI]= binofit(sum(diag(PCC_MM.MPS_Conf_RFP)), n_validTot);


for i = 1:ngroups
    [PCC_MM.MEL_group_DFA(i), PCC_MM.MEL_group_DFA_CI(i,:)] = binofit(PCC_MM.MEL_Conf_DFA(i,i), sum(PCC_MM.MEL_Conf_DFA(i, :), 2));
    [PCC_MM.MEL_group_RFP(i), PCC_MM.MEL_group_RFP_CI(i,:)] = binofit(PCC_MM.MEL_Conf_RFP(i,i), sum(PCC_MM.MEL_Conf_RFP(i, :), 2));
    [PCC_MM.MPS_group_DFA(i), PCC_MM.MPS_group_DFA_CI(i,:)] = binofit(PCC_MM.MPS_Conf_DFA(i,i), sum(PCC_MM.MPS_Conf_DFA(i, :), 2));
    [PCC_MM.MPS_group_RFP(i), PCC_MM.MPS_group_RFP_CI(i,:)] = binofit(PCC_MM.MPS_Conf_RFP(i,i), sum(PCC_MM.MPS_Conf_RFP(i, :), 2));
end
PCC_MM.nvalid = n_validTot;

fprintf(1,'\n');
fprintf(1, 'Final Results MEL : DFA (%.2f-%.2f) RFP (%.2f-%.2f)\n', mean(PCC_MM.MEL_group_DFA_CI(:,1))*100, mean(PCC_MM.MEL_group_DFA_CI(:,2))*100,  mean(PCC_MM.MEL_Total_group_CI(:,1))*100, mean(PCC_MM.MEL_Total_RFP_CI(:,2))*100);
fprintf(1, '\t\t DFA Group Min %.2f RFP Group Min %.2f\n', min(PCC_MM.MEL_group_DFA)*100, min(PCC_MM.MEL_group_RFP)*100);
fprintf(1, '\t\t DFA Group Max %.2f RFP Group Max %.2f\n', max(PCC_MM.MEL_group_DFA)*100, max(PCC_MM.MEL_group_RFP)*100);
fprintf(1, 'Final Results MPS : DFA (%.2f-%.2f) RFP (%.2f-%.2f)\n', PCC_MM.MPS_group_DFA_CI(1)*100, PCC_MM.MPS_group_DFA_CI(2)*100,  PCC_MM.MPS_Total_RFP_CI(1)*100, PCC_MM.MPS_Total_RFP_CI(2)*100);
fprintf(1, '\t\t DFA Group Min %.2f RFP Group Min %.2f\n', min(PCC_MM.MPS_group_DFA)*100, min(PCC_MM.MPS_group_RFP)*100);
fprintf(1, '\t\t DFA Group Max %.2f RFP Group Max %.2f\n', max(PCC_MM.MPS_group_DFA)*100, max(PCC_MM.MPS_group_RFP)*100);


% save the DFA on Mel cepstrum
if normFlg
    save vocTypeMELMPSNorm.mat C lenMFCCs time_frames nwfRes nwt dwt dwfRes PCC_MM ngroups name_grp
else
    save vocTypeMELMPS_PCA.mat C lenMFCCs time_frames nwfRes nwt dwt dwfRes PCC_MM ngroups name_grp
end

    

%% Display the MPS and MFCC results
% First calculate the DFA with all the dataset and plot the DF

figure(15);
% first re-organize the confusion matrix so the call types are in the right
% order
subplot(1,2,1);
tosortMatrix = PCC_MM.MEL_Conf_RFP;

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

title(sprintf('RF (Equal Prior) Total = %.1f%% Type = %.1f%%', 100*PCC_MM.MEL_Total_RFP, 100*mean(PCC_MM.MEL_group_RFP)));
set(gca(), 'Ytick', 1:ngroups);
set(gca(), 'YTickLabel', name_grp_plot);
set(gca(), 'Xtick', 1:ngroups);
set(gca(), 'XTickLabel', name_grp_plot);


% first re-organize the confusion matrix so the call types are in the right
% order
subplot(1,2,2);
tosortMatrix = PCC_MM.MEL_Conf_DFA;

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

title(sprintf('DFA Total = %.1f%% Type = %.1f%%', 100*PCC_MM.MEL_Total_DFA, 100*mean(PCC_MM.MEL_group_DFA)));
set(gca(), 'Ytick', 1:ngroups);
set(gca(), 'YTickLabel', name_grp_plot);
set(gca(), 'Xtick', 1:ngroups);
set(gca(), 'XTickLabel', name_grp_plot);

figure(16);
% first re-organize the confusion matrix so the call types are in the right
% order
subplot(1,2,1);
tosortMatrix = PCC_MM.MPS_Conf_RFP;

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

title(sprintf('RF (Equal Prior) Total = %.1f%% Type = %.1f%%', 100*PCC_MM.MPS_Total_RFP, 100*mean(PCC_MM.MPS_group_RFP)));
set(gca(), 'Ytick', 1:ngroups);
set(gca(), 'YTickLabel', name_grp_plot);
set(gca(), 'Xtick', 1:ngroups);
set(gca(), 'XTickLabel', name_grp_plot);


% first re-organize the confusion matrix so the call types are in the right
% order
subplot(1,2,2);
tosortMatrix = PCC_MM.MPS_Conf_DFA;

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

title(sprintf('DFA Total = %.1f%% Type = %.1f%%', 100*PCC_MM.MPS_Total_DFA, 100*mean(PCC_MM.MPS_group_DFA)));
set(gca(), 'Ytick', 1:ngroups);
set(gca(), 'YTickLabel', name_grp_plot);
set(gca(), 'Xtick', 1:ngroups);
set(gca(), 'XTickLabel', name_grp_plot);

%% Plot mean MPS for each call type

[meanMPS, name_grp] = grpstats(soundCutsMPS, vocTypeCuts', {'mean', 'gname'});

dbres = 80;

MPS_max = max(max(meanMPS));
MPS_floor = MPS_max-dbres;

figure(17)
for rr = 1:ngroups
    subplot(1,ngroups, rr);
    vocMPS = reshape(meanMPS(rr, :), nwfRes, nwt);
     
    imagesc(dwt, dwfRes.*1000, vocMPS, [MPS_floor MPS_max-5]);
    title(sprintf('%s', name_grp{rr}));
    axis xy;
    if rr == 1
    xlabel('Temporal Modulation (Hz)');
    ylabel('Spectral Modulation (cyc/kHz)');
    end
end

%%

[nDF, p, statsDFA] = manova1(soundCutsMFCCs, vocTypeCuts);
[mean_grp, std_grp, name_grp] = grpstats(statsDFA.canon(:,1:nDF),vocTypeCuts', {'mean', 'std', 'gname'});

    
%  Display the significant DFA
figure(13);
PC_DF = statsDFA.eigenvec(:, 1:nDF);

% Find color scale
clear cmin cmax clims
cmin = min(min(PC_DF(:, 1:nDF)));
cmax = max(max(PC_DF(:, 1:nDF)));
cabs = max(abs(cmin), abs(cmax));
clims = [-cabs cabs];
for i=1:nDF
    subplot(1,nDF,i);
    PC_Spect = reshape(PC_DF(:,i), C, lenMFCCs);
    imagesc(time_frames, [1 C], PC_Spect, clims)
    title(sprintf('Cepstrum DFA %d', i));    
    axis xy;
    if i ~= 1
        axis off;
    end   
    if (i == 1)
        xlabel( 'Time (s)' );
        ylabel( 'Cepstrum index' );
    end
end
% Display confusion Matrix
% First reorder the matrix
tosortMatrix = PCC_info_bird.PCC_group;
sortedMatrix = zeros(size(tosortMatrix));
for rr = 1:size(tosortMatrix,1)
    rInd = find(strcmp(name_grp_plot(rr), name_grp));
    for cc = 1:size(tosortMatrix,2)
        cInd = find(strcmp(name_grp_plot(cc), name_grp));
        sortedMatrix(rr,cc) = tosortMatrix(rInd, cInd);
    end
end
figure(14);
imagesc(sortedMatrix);
xlabel('Guess');
ylabel('Actual');
colormap(gray);
colorbar;
title(sprintf('Mel Cepstrum Confusion Matrix DFA %.1f%%(%.1f%%) Correct', PCC_info_bird.PCC_Total, PCC_info_bird.PCC_M));
set(gca(), 'Ytick', 1:ngroups);
set(gca(), 'YTickLabel', name_grp_plot);
set(gca(), 'Xtick', 1:ngroups);
set(gca(), 'XTickLabel', name_grp_plot);




