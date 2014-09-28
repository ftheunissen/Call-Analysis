% Analyze the recording levels

recLevels = [60, 65, 67, 70, 75, 80, 85, 90, 95, 100];
soundStart = [436, 515, 442, 404, 312, 476, 286, 394, 310, 466];
% soundEnd = [711, 793, 714, 672, 581, 746, 553, 668, 577, 734];
soundEnd = soundStart + 270;
    
nlevels = length(recLevels);
cdSound = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/Recording Levels/';
% figure();

% Read the 100 sound for calibration.
fname = sprintf('140924-Reclevel%d.wav', recLevels(nlevels));
fullname = fullfile(cdSound, fname);
[soundCalibRef, fs] = audioread(fullname);

% Make stereo sounds - mono.
if size(soundCalibRef,2)==2
    soundCalibRef=(soundCalibRef(:,1) + soundCalibRef(:,2))/2;
end
soundCalibRef = soundCalibRef(soundStart(nlevels)*1000:soundEnd(nlevels)*1000);

% Make space for maxima
corrVal = zeros(1, nlevels);
autocorrVal = zeros(1, nlevels);


for i=1:nlevels

    figure(i);
    fname = sprintf('140924-Reclevel%d.wav', recLevels(i));
    fullname = fullfile(cdSound, fname);
    [soundCalib, fs] = audioread(fullname);
    
        % Make stereo sounds - mono.
    if size(soundCalib,2)==2
        soundCalib=(soundCalib(:,1) + soundCalib(:,2))/2;
    end
    
    %subplot(nlevels, 1, i);
    % plot(soundCalib(soundStart(i)*1000:soundEnd(i)*1000));
    
    % Perform the cross-correlation 
    [c, lags] = xcorr(soundCalib(soundStart(i)*1000:soundEnd(i)*1000), soundCalibRef, fix(0.3*fs), 'unbiased');
    plot(lags./fs, c);
    corrVal(i) = max(c);
    
    % Calculate the auto-correlation
    [a, lags] = xcorr(soundCalib(soundStart(i)*1000:soundEnd(i)*1000), soundCalib(soundStart(i)*1000:soundEnd(i)*1000), fix(0.3*fs), 'unbiased');
    autocorrVal(i) = max(a);
end

gain1 = corrVal./autocorrVal(nlevels);
gain2 = corrVal./autocorrVal;
gainavg = (gain1+1./gain2)./2;

corrValTheo = power(10, (recLevels - recLevels(nlevels))./50);

figure();
plot(recLevels, gainavg, 'r');
hold on;
plot(recLevels, corrValTheo, 'k--');
hold off;
legend('Gain Data', '25=10dB');
xlabel('Zoom Rec Level');
ylabel('Amplitude Gain');

    
   
    