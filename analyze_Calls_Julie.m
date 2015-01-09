% Load up the call data base
% Read the data base produced by VocSectioningAmp.m
load /Users/frederictheunissen/Documents/Data/Julie/FullVocalizationBank/vocCuts.mat
%load /auto/fdata/fet/julie/FullVocalizationBank/vocCuts.mat

% set the order in which the call types should be displayed in confusion
% matrices
name_grp_plot = {'Be', 'LT', 'Tu', 'Th', 'Di', 'Ag', 'Wh', 'Ne', 'Te', 'DC', 'So'};

% Initialize some of the values
nsounds = size(soundCutsTot, 1);      % Number of calls in the library
soundlen = size(soundCutsTot, 2);     % Length of sound in points
DBNOISE = 50;
f_low = 250;
f_high = 10000;
plotme = 1;

nf = length(fo);  % Number of frequency slices in our spectrogram
nt = length(to);  % Number of time slices in our spectrogram



% Loop through all sounds to extract temporal and spectral parameters

for is=1:nsounds
    
%          if (~strcmp(vocTypeCuts{is},'DC') )   % Distance call is==7 is used as an example
%               continue;
%          end
%     
    fprintf(1,'Processing sound %d/%d\n', is, nsounds);
    soundIn  = soundCutsTot(is,:);
    rms = std(soundIn(soundIn~=0));
    soundSpect = reshape(spectroCutsTot(is,:), nf, nt);   % This is the spectrogram calculated in VocSectioningAmp
    
    % Plotting Variables
    t=1:soundlen;
    t = (t-1).*1000/samprate;
    
    % Calculate amplitude enveloppe
    ampEnv = enveloppe_estimator(soundIn, samprate, 20, 1000);
    tk=0:fix(soundlen*1000/samprate);
    
    if plotme    % Plot the oscilogram + spectrogram
        figure(1);
        subplot(4,1,1);
        
        plot(t,soundIn);
        xlabel('time (ms)');
        s_axis = axis;
        s_axis(1) = 0;
        s_axis(2) = t(end);
        axis(s_axis);
        
        
        % Plot the amplitude enveloppe        
        hold on;
        plot(tk, ampEnv, 'r');
        hold off;
        
        % Plot the spectrogram
        subplot(4,1,[2 3 4]);
        
        maxB = max(max(soundSpect));
        minB = maxB-DBNOISE;
        imagesc(to*1000,fo,soundSpect);          % to is in seconds
        axis xy;
        caxis('manual');
        caxis([minB maxB]);
        cmap = spec_cmap();
        colormap(cmap);
        
        %  Match the axis to oscillogram
        v_axis = axis;
        %v_axis(2)=4.0;
        v_axis(1) = s_axis(1);
        v_axis(2) = s_axis(2);
        v_axis(3)=f_low;
        v_axis(4)=f_high;
        axis(v_axis);
        xlabel('time (ms)'), ylabel('Frequency');
        %axis off;
    end
       
    % Calculate the fundamental
    [fund, sal, fund2, lenfund] = fundEstimator(soundIn, samprate, soundSpect, to, fo);
    meanfund = mean(fund(~isnan(fund)));
    meansal = mean(sal(~isnan(sal)));
    meanfund2 = mean(fund2(~isnan(fund2)));
    
    if (sum(~isnan(fund)) == 0)
        fund2prop = 0;
    else
        fund2prop = sum(~isnan(fund2))/sum(~isnan(fund));
    end
       
    % Plot the fundamental on the same figure
    if plotme
        figure(1);
        subplot(4,1,[2 3 4]);
        hold on;
        ph = plot(to.*1000.0, fund, 'k');
        set(ph, 'LineWidth', 3);
        ph = plot(to.*1000.0, fund2, 'g');
        set(ph, 'LineWidth', 3);
        hold off;
    end
    
    % Power spectrum
    window_len = 10.0;                 % Window length in ms
    nwindow = (1000.0.*length(soundIn)./samprate)./window_len;
    
    % Hs=spectrum.mtm(nwindow)
    Hs = spectrum.welch('Hann', 1024, 99);
    Hpsd = psd(Hs,soundIn,'Fs',samprate);
    
    % Find quartile power
    cum_power = cumsum(Hpsd.data);
    tot_power = sum(Hpsd.data);
    quartile_freq = zeros(1,3);
    quartile_values = [0.25, 0.5, 0.75];
    nfreqs = length(cum_power);
    iq = 1;
    for ifreq=1:nfreqs
        if (cum_power(ifreq) > quartile_values(iq)*tot_power)
            quartile_freq(iq) = ifreq;
            iq = iq+1;
            if (iq > 3)
                break;
            end
        end
    end
    
    % Apply LPC to extract formants
    A = lpc(soundIn, 8);    % 8 degree polynomial
    rts = roots(A);          % Find the roots of A
    rts = rts(imag(rts)>=0);  % Keep only half of them
    angz = atan2(imag(rts),real(rts));
    
    % Calculate the frequencies and the bandwidth of the formants
    [frqs,indices] = sort(angz.*(samprate/(2*pi)));
    bw = -1/2*(samprate/(2*pi))*log(abs(rts(indices)));
    
    % Keep formansts above 1000 Hz and with bandwidth < 1000
    nn = 0;
    formants = [];
    for kk = 1:length(frqs)
        if (frqs(kk) > 1000 && bw(kk) <1000)        
            nn = nn+1;
            formants(nn) = frqs(kk);
        end
    end
    
    
    % Calculation of spectral enveloppe - not the best way of doing it - better to fit data with Gaussian?
    %     indFit = find(Hpsd.Frequencies < 8000);
    %     pCoeff = polyfit(Hpsd.Frequencies(indFit), Hpsd.Data(indFit), 2);
    %     powFit = polyval(pCoeff, Hpsd.Frequencies(indFit));
    %     hold on;
    %     plot(Hpsd.Frequencies(indFit), powFit, 'r', 'LineWidth', 2);
    
    if plotme
        figure(2);
        
        % Plot Power Spectrum
        hp = plot(Hpsd.Frequencies, Hpsd.Data); 
        xlabel('Frequency Hz');
        ylabel('Power Linear');
        
        power_axis = axis();
        power_axis(1) = 0;
        power_axis(2) = f_high;
        axis(power_axis);
               
        hold on;
        for iq=1:3
            plot([Hpsd.Frequencies(quartile_freq(iq)) Hpsd.Frequencies(quartile_freq(iq))], [power_axis(3) power_axis(4)], 'k--');
        end
        for in=1:nn
            plot([formants(in) formants(in)], [power_axis(3) power_axis(4)], 'r--');
        end   
        
        hold off;
        
        figure(1);
        subplot(4,1,[2 3 4]);
        spectro_axis = axis();
        hold on;
        for in=1:nn
            plot([spectro_axis(1) spectro_axis(2)], [formants(in) formants(in)], 'r--');
        end 
        hold off;
    end
    
    % Find skewness, kurtosis and entropy for power spectrum below
    % f_high
    ind_fmax = nfreqs;
    for ifreq=1:nfreqs
        if (Hpsd.Frequencies(ifreq) > f_high )
            ind_fmax = ifreq;
            break;
        end
    end
    
    % Description of spectral enveloppe
    spectdata = Hpsd.data(1:ind_fmax);
    freqdata = Hpsd.Frequencies(1:ind_fmax);
    spectdata = spectdata./sum(spectdata);
    meanspect = sum(freqdata.*spectdata);
    stdspect = sqrt(sum(spectdata.*((freqdata-meanspect).^2)));
    skewspect = sum(spectdata.*(freqdata-meanspect).^3);
    skewspect = skewspect./(stdspect.^3);
    kurtosisspect = sum(spectdata.*(freqdata-meanspect).^4);
    kurtosisspect = kurtosisspect./(stdspect.^4);
    entropyspect = -sum(spectdata.*log2(spectdata))/log2(ind_fmax);
    
    % Repeat for temporal enveloppe
    ampdata = ampEnv;
    tdata = tk;
    ampdata = ampdata./sum(ampdata);
    meantime = sum(tdata.*ampdata);
    stdtime = sqrt(sum(ampdata.*((tdata-meantime).^2)));
    skewtime = sum(ampdata.*(tdata-meantime).^3);
    skewtime = skewtime./(stdtime.^3);
    kurtosistime = sum(ampdata.*(tdata-meantime).^4);
    kurtosistime = kurtosistime./(stdtime.^4);
    indpos = find(ampdata>0);
    entropytime = -sum(ampdata(indpos).*log2(ampdata(indpos)))/log2(length(indpos));
    
    
    % Print out results
    if plotme
        figure(3);
        clf(3);
        text(0.4, 1.0, sprintf('\\bf %d/%d: %s  %s\\rm', is, nsounds, birdNameCuts{is}, vocTypeCuts{is}));
        text(-0.1, 0.9, sprintf('Mean Fund = %.2f Hz Mean Saliency = %.2f Mean Fund2 = %.2f PF2 = %.2f%%', meanfund, meansal, meanfund2, fund2prop*100));
        text(-0.1, 0.85, sprintf('Max Fund = %.2f Hz, Min Fund = %.2f Hz, CV = %.2f', max(fund), min(fund), std(fund(~isnan(fund)))/meanfund));
        text(-0.1, 0.75, sprintf('Mean Spect = %.2f Hz, Std Spect= %.2f Hz', meanspect, stdspect));
        text(-0.1, 0.7, sprintf('\tSkew = %.2f, Kurtosis = %.2f Entropy=%.2f', skewspect, kurtosisspect, entropyspect));
        text(-0.1, 0.65, sprintf('Q1 F = %.2f Hz, Q2 F= %.2f Hz, Q3 F= %.2f Hz', Hpsd.Frequencies(quartile_freq(1)),Hpsd.Frequencies(quartile_freq(2)), Hpsd.Frequencies(quartile_freq(3)) ));
        text(-0.1, 0.55, sprintf('Mean Time = %.2f ms, Std Time= %.2f ms', meantime, stdtime));
        text(-0.1, 0.5, sprintf('\tSkew = %.2f, Kurtosis = %.2f Entropy=%.2f', skewtime, kurtosistime, entropytime));
        text(-0.1, 0.4, sprintf('\tRMS = %.2f, Max Amp = %.2f', rms, max(ampEnv)));
        
        axis off;
    end
    
    callAnalData(is).bird = birdNameCuts{is};
    callAnalData(is).type = vocTypeCuts{is};
    callAnalData(is).fund = meanfund;
    callAnalData(is).sal = meansal;
    callAnalData(is).fund2 = meanfund2;
    callAnalData(is).voice2percent = fund2prop*100;
    callAnalData(is).maxfund = max(fund);
    callAnalData(is).minfund = min(fund);
    callAnalData(is).cvfund = std(fund(~isnan(fund)))/meanfund;
    callAnalData(is).meanspect = meanspect;
    callAnalData(is).stdspect = stdspect;
    callAnalData(is).skewspect = skewspect;
    callAnalData(is).kurtosisspect = kurtosisspect;
    callAnalData(is).entropyspect = entropyspect;
    callAnalData(is).q1 = Hpsd.Frequencies(quartile_freq(1));
    callAnalData(is).q2 = Hpsd.Frequencies(quartile_freq(2));
    callAnalData(is).q3 = Hpsd.Frequencies(quartile_freq(3));
    callAnalData(is).meantime = meantime;
    callAnalData(is).stdtime = stdtime;
    callAnalData(is).skewtime = skewtime;
    callAnalData(is).kurtosistime = kurtosistime;
    callAnalData(is).entropytime = entropytime;
    callAnalData(is).psd = Hpsd;
    callAnalData(is).tAmp = ampEnv;
    callAnalData(is).rms = rms;
    callAnalData(is).maxAmp = max(ampEnv);
    
    if (nn > 0 )
        callAnalData(is).f1 = formants(1);
    else
        callAnalData(is).f1 = nan;
    end
    if (nn > 1 )
        callAnalData(is).f2 = formants(2);
    else
        callAnalData(is).f2 = nan;
    end
    if (nn > 2 )
        callAnalData(is).f3 = formants(3);
    else
        callAnalData(is).f3 = nan;
    end
    
    if plotme
        soundsc(soundIn, samprate);
        pause();
    end
    
end

input('Press Enter to save the data');
 cd('/Users/frederictheunissen/Documents/Data/Julie/FullVocalizationBank/');
%cd('/auto/fdata/fet/julie/FullVocalizationBank/');
save vocCutsAnalwFormants.mat callAnalData