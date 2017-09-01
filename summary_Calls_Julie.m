%% Make a summary bar plot to show the performance of the different
% acoustical models

Ybar = zeros(4,2);   % Allocate space for summary data
YbarError = zeros(4,2,2); % And for the confidence intervals

load '/Users/frederictheunissen/Documents/Data/Julie/FullVocalizationBank/vocTypeSpectro.mat';
inb = 4;       % Corresponds to 40 PCs in PCC_info_bird
nb = 40;

Ybar(2,1) = 100*mean(PCC_info_bird.PCC_group_RFP(inb,:));
YbarError(2,1,1) = 100.*mean(PCC_info_bird.PCC_group_RFP(inb,:)- squeeze(PCC_info_bird.PCC_group_RFP_CI(inb, :, 1)));
YbarError(2,1,2) = 100.*mean(squeeze(PCC_info_bird.PCC_group_RFP_CI(inb, :, 2))-PCC_info_bird.PCC_group_RFP(inb,:));

Ybar(2,2) = 100*mean(PCC_info_bird.PCC_group_DFA(inb,:));
YbarError(2,2,1) = 100.*mean(PCC_info_bird.PCC_group_DFA(inb,:)- squeeze(PCC_info_bird.PCC_group_DFA_CI(inb, :, 1)));
YbarError(2,2,2) = 100.*mean(squeeze(PCC_info_bird.PCC_group_DFA_CI(inb, :, 2))-PCC_info_bird.PCC_group_DFA(inb,:));

load '/Users/frederictheunissen/Documents/Data/Julie/FullVocalizationBank/vocTypeAcoust.mat';

Ybar(1,1) = 100*mean(PCC_Acoust.group_RFP);
YbarError(1,1,1) = 100.*mean(PCC_Acoust.group_RFP - PCC_Acoust.group_RFP_CI( :, 1)');
YbarError(1,1,2) = 100.*mean(PCC_Acoust.group_RFP_CI( :, 2)' - PCC_Acoust.group_RFP);

Ybar(1,2) = 100*mean(PCC_Acoust.group_RFP);
YbarError(1,2,1) = 100.*mean(PCC_Acoust.group_DFA - PCC_Acoust.group_DFA_CI( :, 1)');
YbarError(1,2,2) = 100.*mean(PCC_Acoust.group_DFA_CI( :, 2)' - PCC_Acoust.group_DFA);

load '/Users/frederictheunissen/Documents/Data/Julie/FullVocalizationBank/vocTypeMELMPS_PCA.mat';

Ybar(3,1) = 100*mean(PCC_MM.MPS_group_RFP);
YbarError(3,1,1) = 100.*mean(PCC_MM.MPS_group_RFP - PCC_MM.MPS_group_RFP_CI( :, 1)');
YbarError(3,1,2) = 100.*mean(PCC_MM.MPS_group_RFP_CI( :, 2)' - PCC_MM.MPS_group_RFP);

Ybar(3,2) = 100*mean(PCC_MM.MPS_group_DFA);
YbarError(3,2,1) = 100.*mean(PCC_MM.MPS_group_DFA - PCC_MM.MPS_group_DFA_CI( :, 1)');
YbarError(3,2,2) = 100.*mean(PCC_MM.MPS_group_DFA_CI( :, 2)' - PCC_MM.MPS_group_DFA);

Ybar(4,1) = 100*mean(PCC_MM.MEL_group_RFP);
YbarError(4,1,1) = 100.*mean(PCC_MM.MEL_group_RFP - PCC_MM.MEL_group_RFP_CI( :, 1)');
YbarError(4,1,2) = 100.*mean(PCC_MM.MEL_group_RFP_CI( :, 2)' - PCC_MM.MEL_group_RFP);

Ybar(4,2) = 100*mean(PCC_MM.MEL_group_DFA);
YbarError(4,2,1) = 100.*mean(PCC_MM.MEL_group_DFA - PCC_MM.MEL_group_DFA_CI( :, 1)');
YbarError(4,2,2) = 100.*mean(PCC_MM.MEL_group_DFA_CI( :, 2)' - PCC_MM.MEL_group_DFA);

figure(1);
% first re-organize the confusion matrix so the call types are in the right
% order

bh = bar(Ybar);
set(get(bh(1),'Parent'),'XTickLabel',{'Env','Spect', 'MPS', 'MFCC'});
set(bh(1), 'FaceColor', [0.4 0.4 0.4]);
set(bh(2), 'FaceColor', [0.7 0.7 0.7]);

ylabel('Percent Correct Classification');
legend('RF', 'DFA');
hold on;

errorbar( get(bh(1), 'XData')-0.15 , Ybar(:,1),  squeeze(YbarError(:,1,1)), squeeze(YbarError(:,1,2)), '.k' );
errorbar( get(bh(2), 'XData')+0.15 , Ybar(:,2),  squeeze(YbarError(:,1,2)), squeeze(YbarError(:,1,2)), '.k' );

plot([0.5 4.5], [100/ngroups 100/ngroups], 'k--');

hold off;

%% Some summary data with merged categories

% PCC per category once again
pccCat = zeros(1, ngroups);
for i=1:ngroups
   pccCat(i) = PCC_Acoust.Conf_RFP(i,i)./sum(PCC_Acoust.Conf_RFP(i,:));
end

% Now we merge the confusion matrix
mergeId = [1 6];
confMatMerge = PCC_Acoust.Conf_RFP;

% merge columns
for i=1:ngroups
    confMatMerge(i, mergeId(1)) = confMatMerge(i, mergeId(1)) + confMatMerge(i, mergeId(2));
end
% merge rows
for i=1:ngroups
    confMatMerge(mergeId(1), i) = confMatMerge(mergeId(1), i) + confMatMerge(mergeId(2), i);
end

confMatMerge(:,mergeId(2)) = [];
confMatMerge(mergeId(2), :) = [];
name_grpMerge = name_grp;
name_grpMerge(mergeId(2)) = [];
ngroupsMerge = length(name_grpMerge);

mergeId = [7 8];
% merge columns
for i=1:ngroupsMerge
    confMatMerge(i, mergeId(1)) = confMatMerge(i, mergeId(1)) + confMatMerge(i, mergeId(2));
end
% merge rows
for i=1:ngroupsMerge
    confMatMerge(mergeId(1), i) = confMatMerge(mergeId(1), i) + confMatMerge(mergeId(2), i);
end

confMatMerge(:,mergeId(2)) = [];
confMatMerge(mergeId(2), :) = [];
name_grpMerge(mergeId(2)) = [];
ngroupsMerge = length(name_grpMerge);

% PCC per category once again
pccCatMerge = zeros(1, ngroupsMerge);
for i=1:ngroupsMerge
   pccCatMerge(i) = confMatMerge(i,i)./sum(confMatMerge(i,:));
end
    

%% Now make a sumary plot per call category
name_grp_plot = {'Be', 'LT', 'Tu', 'Th', 'Di', 'Ag', 'Wh', 'Ne', 'Te', 'DC', 'So'};
inb = 4;

colorVals = [ [0 230 255]; [0 95 255]; [255 200 65]; [255 150 40]; [255 105 15];...
    [255 0 0]; [255 180 255]; [255 100 255]; [140 100 185]; [100 50 200]; [100 100 100] ];

if (length(name_grp_plot) ~= ngroups)
    fprintf(1, 'Error: missmatch between the length of name_grp_plot and the number of groups\n');
end

figure(2);

indPlot = zeros(1,ngroups);
bh = bar(zeros(ngroups,4));

for ig=1:ngroups
    for ig_ind=1:ngroups
        if strcmp(name_grp_plot{ig}, name_grp{ig_ind})
            indPlot(ig_ind) = ig;
            break;
        end
    end
end

Y1 = 100*PCC_Acoust.group_DFA;
[Y1sorted, sortind] = sort(Y1);
   
hold on;
for ig=1:ngroups   
    Y = zeros(ngroups,4);
   
    Y(ig,1) = Y1sorted(ig);
    bar( Y, 'FaceColor', colorVals(indPlot(sortind(ig)),:)./255);
    Y = zeros(ngroups,4);
    Y(ig,2) = 100.0.*PCC_info_bird.PCC_group_DFA(inb, sortind(ig));
    bar(Y, 'FaceColor', colorVals(indPlot(sortind(ig)),:)./300);
    Y = zeros(ngroups,4);
    Y(ig,3) = 100.0.*PCC_MM.MPS_group_DFA(sortind(ig));
    bar(Y, 'FaceColor', colorVals(indPlot(sortind(ig)),:)./400);
    Y = zeros(ngroups,4);
    Y(ig,4) = 100.0.*PCC_MM.MEL_group_DFA(sortind(ig));
    bar(Y, 'FaceColor', colorVals(indPlot(sortind(ig)),:)./500);
end

% The error bars  
YbarError = zeros(2,ngroups);

Ybar = PCC_info_bird.PCC_group_DFA(inb,:);
YbarError(1,:) = 100.*(Ybar - squeeze(PCC_info_bird.PCC_group_DFA_CI(inb, :, 1)));
YbarError(2,:) = 100.*(squeeze(PCC_info_bird.PCC_group_DFA_CI(inb, :, 2))- Ybar);
errorbar(mean(get(get(bh(2), 'Children'), 'XData')) , 100.*Ybar(sortind),  YbarError(1,sortind), YbarError(2,sortind), '.k' );

Ybar = PCC_Acoust.group_DFA';
YbarError(1,:) = 100.*(Ybar - PCC_Acoust.group_DFA_CI(:, 1));
YbarError(2,:) = 100.*(PCC_Acoust.group_DFA_CI(:, 2) - Ybar);
errorbar(mean(get(get(bh(1), 'Children'), 'XData')) , 100.*Ybar(sortind),  YbarError(1,sortind), YbarError(2,sortind), '.k' );

Ybar = PCC_MM.MPS_group_DFA';
YbarError(1,:) = 100.*(Ybar - PCC_MM.MPS_group_DFA_CI(:, 1));
YbarError(2,:) = 100.*(PCC_MM.MPS_group_DFA_CI(:, 2) - Ybar);
errorbar(mean(get(get(bh(3), 'Children'), 'XData')) , 100.*Ybar(sortind),  YbarError(1,sortind), YbarError(2,sortind), '.k' );

Ybar = PCC_MM.MEL_group_DFA';
YbarError(1,:) = 100.*(Ybar - PCC_MM.MEL_group_DFA_CI(:, 1));
YbarError(2,:) = 100.*(PCC_MM.MEL_group_DFA_CI(:, 2) - Ybar);
errorbar(mean(get(get(bh(4), 'Children'), 'XData')) , 100.*Ybar(sortind),  YbarError(1,sortind), YbarError(2,sortind), '.k' );


% The chance level
plot([0 ngroups+1], [100/ngroups 100/ngroups], 'k--');

% Custom Legend
text(0.5, 85, 'Env', 'Color', [0 230 255]./255);
text(0.5, 80, 'Spect', 'Color', [0 230 255]./300);
text(0.5, 75, 'MPS', 'Color', [0 230 255]./400);
text(0.5, 70, 'MFCC', 'Color', [0 230 255]./500);

% Type some text
set(get(bh(1),'Parent'),'XTickLabel',name_grp(sortind));
ylabel('Percent Correct Classification');
xlabel('Call Type');



hold off;

%% Bar plot with DFA only

name_grp_plot = {'Be', 'LT', 'Tu', 'Th', 'Di', 'Ag', 'Wh', 'Ne', 'Te', 'DC', 'So'};
inb = 1;

colorVals = [ [0 230 255]; [0 95 255]; [255 200 65]; [255 150 40]; [255 105 15];...
    [255 0 0]; [255 180 255]; [255 100 255]; [140 100 185]; [100 50 200]; [100 100 100] ];

if (length(name_grp_plot) ~= ngroups)
    fprintf(1, 'Error: missmatch between the length of name_grp_plot and the number of groups\n');
end

figure(3);

indPlot = zeros(1,ngroups);
bh = bar(zeros(ngroups,1));

for ig=1:ngroups
    for ig_ind=1:ngroups
        if strcmp(name_grp_plot{ig}, name_grp{ig_ind})
            indPlot(ig_ind) = ig;
            break;
        end
    end
end

Y1 = 100*PCC_Acoust.group_DFA;
[Y1sorted, sortind] = sort(Y1);
   
hold on;
for ig=1:ngroups   
    Y = zeros(ngroups,1); 
    Y(ig,1) = Y1sorted(ig);
    bar( Y, 'FaceColor', colorVals(indPlot(sortind(ig)),:)./255);
end

% The error bars  
YbarError = zeros(2,ngroups);

Ybar = PCC_Acoust.group_DFA';
YbarError(1,:) = 100.*(Ybar - PCC_Acoust.group_DFA_CI(:, 1));
YbarError(2,:) = 100.*(PCC_Acoust.group_DFA_CI(:, 2) - Ybar);
errorbar( get(bh(1), 'XData') , 100.*Ybar(sortind),  YbarError(1,sortind), YbarError(2,sortind), '.k' );


% The chance level
plot([0 ngroups+1], [100/ngroups 100/ngroups], 'k--');

% Type some text
set(get(bh(1),'Parent'),'XTickLabel',name_grp(sortind));
ylabel('Percent Correct Classification');
xlabel('Call Type');


hold off;

