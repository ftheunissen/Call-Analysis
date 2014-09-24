% Loop for sound files will make a vocCuts.mat file with 

%Set the parameters for sound filtering 
songfilt_args.f_low = 250;
songfilt_args.f_high = 12000;
songfilt_args.db_att = 0;
duration=0.2; % duration in s of cuts
pl = 1;


% Go to the place that has the information on the sound for each site
cd /auto/k6/julie/matfile
input_dir = pwd;
Subjects = dir(input_dir);

for ss=1:length(Subjects)
    Indiv=Subjects(ss).name;
    if length(Indiv)==11
        allFiles = dir(fullfile(input_dir, Indiv,'FirstVoc*.mat'));
        NF = length(allFiles);
        for nsite=1:8
            % find a matfile for each site
            Matsite='';
            for nf = 1:NF
                File_n=allFiles(nf).name;
                if str2num(File_n(14))==nsite
                    Matsite = allFiles(nf).name;
                    break
                end
            end
            if isempty(Matsite)
                fprintf(1, 'No file could be find for %s Site %d\n', Indiv, nsite);
            else
                % Retrieve the Matfile that contains the spectrograms
                MAT = load(fullfile(input_dir, Indiv,Matsite));
                 
                % Loop through all the wav files
                nstims = length(MAT.VocType);
                ncutsTot = 0;
                soundCutsTot = [];
                spectroCutsTot = [];
                
                
                for is=1:nstims
                    [sound_in samprate] = audioread(MAT.TDT_wavfiles{is});
                    sound_in = sound_in(MAT.WavIndices{is}(1):MAT.WavIndices{is}(2));
                    % Find the enveloppe
                    amp_env = enveloppe_estimator(sound_in, samprate, 20, samprate);
                    max_amp = max(amp_env);
                    
                    % Find the maxima above maxMinTh and minima below
                    max_min_ind = [];
                    j = 1;
                    max_min_ind(j) = -1;
                    maxMinTh = 0.1;
                    nt = length(amp_env);
                    for i=2:nt-1
                        if ( (amp_env(i) > maxMinTh*max_amp) && (amp_env(i-1) < amp_env(i)) && (amp_env(i) > amp_env(i+1)) )
                            j = j + 1;
                            max_min_ind(j) = i;
                        elseif ( (amp_env(i) <= maxMinTh*max_amp) && (amp_env(i) < amp_env(i-1)) && (amp_env(i) <= amp_env(i+1)) )
                            j = j + 1;
                            max_min_ind(j) = -i;
                        end
                    end
                    j=j+1;
                    max_min_ind(j) = -nt;
                    
                    
                    % Clean up the max/min by keeping more extreme ones in succesive pairs
                    i = 1;
                    while i < length(max_min_ind)
                        if (max_min_ind(i) > 0) && (max_min_ind(i+1) > 0)
                            if amp_env(max_min_ind(i)) > amp_env(max_min_ind(i+1))
                                max_min_ind(i+1) = [];
                            else
                                max_min_ind(i) = [];
                            end
                        elseif (max_min_ind(i) < 0) && (max_min_ind(i+1) < 0)
                            if amp_env(-max_min_ind(i)) > amp_env(-max_min_ind(i+1))
                                max_min_ind(i) = [];
                            else
                                max_min_ind(i+1) = [];
                            end
                        else
                            i=i+1;
                        end
                    end
                    
                    if pl
                        plot_sound_selection_amp(voctype, sound_in, samprate, amp_env, max_min_ind);
                    end
                    
                    [ncuts soundCuts spectroCuts to fo] = cut_sound_selection_amp(sound_in, samprate, amp_env, max_min_ind, duration, pl);
                end
                for ic=ncutsTot+1:ncutsTot+ncuts
                    birdNameCuts{ic} = birdname;
                    vocTypeCuts{ic} = voctype;
                end
                
                ncutsTot = ncutsTot + ncuts;
                soundCutsTot = vertcat(soundCutsTot, soundCuts);
                spectroCutsTot = vertcat(spectroCutsTot, spectroCuts);
                
                
    
    
           
%                 % save the output matrix
%                 Res.CORR=CORR;
%                 Res.VocTypeSel=VocTypeSel;
%                 Res.TDTwav=TDTwav;
%                 Res.StimIndices= StimIndices;
%                 if ismac()
%                     [status username] = system('who am i');
%                     if strcmp(strtok(username), 'frederictheunissen')
%                         if strncmp('/auto/fdata/solveig',stim_name, 19)
%                         elseif strncmp('/auto/fdata/julie',stim_name, 17)
%                             filename = fullfile('/Users','frederictheunissen','Documents','Data','Julie','matfile',Res.subject,['FirstVoc_' Res.Site '.mat']);
%                         end
%                     elseif strcmp(strtok(username), 'elie')
%                         filename = fullfile('/Users','elie','Documents','MATLAB','data','matfile',Res.subject,['FirstVoc_' Res.Site '.mat']);
%                     end
%                 else
%                     filename=fullfile('/auto','k6','julie','matfile',Indiv,sprintf('%s_CorrFirstVoc_Site%d.mat',Indiv,nsite));
%                 end
%                 save(filename, '-struct', 'Res');
%                 fprintf('saved data under: %s\n', filename);
            end
        end
    end
end
