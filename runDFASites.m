%% Run DFA analysis for all recorded sites

input_dir = '/auto/fdata/fet/julie/Acoustical Analysis';

Subjects = dir(input_dir);


for ss=1:length(Subjects)
      
    if Subjects(ss).isdir   % The folders have the names for each bird
        Indiv=Subjects(ss).name;
        fprintf(1, '\n%s:\n', Indiv);
        
        soundFiles = dir(strcat(input_dir, '/', Indiv,'/vocCuts*.mat'));      
        nSound = length(soundFiles);
        
        % if strcmp('WhiBlu5396M', Indiv)
        for is = 1:nSound
            fprintf(1, '\t%s\n', soundFiles(is).name);
            Func_DFA_Calls_Julie(strcat(input_dir, '/', Indiv, '/'), soundFiles(is).name, 0);
        end
        % end

    end
end

        