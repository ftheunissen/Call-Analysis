%  Check amplitude levels recorded in different conditions

fid = fopen('/Users/frederictheunissen/Documents/Data/Julie/FullVocalizationBank/ampcheck.txt', 'r');
ampData = textscan(fid, '%s %s %f %f');
ncalls = length(ampData{1});
fclose(fid);

ind_100 = find(ampData{3} == 100);
callType_100 = unique(ampData{2}(ind_100));
for ic=1:length(callType_100)
    fprintf(1, 'Number of %s recorded at 100: %d\n', callType_100{ic}, length(find(strcmp(ampData{2}(ind_100), callType_100{ic}))));
end

indBe_100 = find(strcmp(ampData{2}, 'Be') & ampData{3} == 100);
indBe_67 = find(strcmp(ampData{2}, 'Be') & ampData{3} == 67);

indDC_100 = find(strcmp(ampData{2}, 'DC') & ampData{3} == 100);
indDC_90 = find(strcmp(ampData{2}, 'DC') & ampData{3} == 90);

indTe_100 = find(strcmp(ampData{2}, 'Te') & ampData{3} == 100);
indTe_90 = find(strcmp(ampData{2}, 'Te') & ampData{3} == 90);

indNe_100 = find(strcmp(ampData{2}, 'Ne') & ampData{3} == 100);
indNe_90 = find(strcmp(ampData{2}, 'Ne') & ampData{3} == 90);



figure();
subplot(2, 1, 1);
h = histfit(ampData{4}(indBe_67), 30);
ax1 = axis();
title('Be Rec 67');
subplot(2, 1, 2);
histfit(ampData{4}(indBe_100), 20);
ax2 = axis();
axis([ax1(1) ax1(2) ax2(3) ax2(4)]);
title('Be Rec 100');
xlabel('RMS');

figure();
subplot(5, 1, 1);
h = histfit(ampData{4}(indDC_90), 30);
ax1 = axis();
title('DC Rec 90');
subplot(5, 1, 2);
histfit(ampData{4}(indDC_100), 20);
ax2 = axis();
title('DC Rec 100');
axis([ax1(1) ax1(2) ax2(3) ax2(4)]);
subplot(5,1,3);
corrValTheo = power(10, (90 - 100)./50);
histfit(ampData{4}(indDC_100)./corrValTheo, 20);
ax2 = axis();
title('DC Rec 100 (scaled 90)');
axis([ax1(1) ax1(2) ax2(3) ax2(4)]);
subplot(5,1,4);
corrValTheo = power(10, (80 - 100)./50);
histfit(ampData{4}(indDC_100)./corrValTheo, 20);
ax2 = axis();
title('DC Rec 100 (scaled 80)');
axis([ax1(1) ax1(2) ax2(3) ax2(4)]);
subplot(5,1,5);
corrValTheo = power(10, (70 - 100)./50);
histfit(ampData{4}(indDC_100)./corrValTheo, 20);
ax2 = axis();
title('DC Rec 100 (scaled 70)');
axis([ax1(1) ax1(2) ax2(3) ax2(4)]);
xlabel('RMS');

figure();
subplot(5, 1, 1);
h = histfit(ampData{4}(indTe_90), 30);
ax1 = axis();
title('Te Rec 90');
subplot(5, 1, 2);
histfit(ampData{4}(indTe_100), 20);
ax2 = axis();
title('Te Rec 100');
axis([ax1(1) ax1(2) ax2(3) ax2(4)]);
subplot(5,1,3);
corrValTheo = power(10, (90 - 100)./50);
histfit(ampData{4}(indTe_100)./corrValTheo, 20);
ax2 = axis();
title('Te Rec 100 (scaled 90)');
axis([ax1(1) ax1(2) ax2(3) ax2(4)]);
subplot(5,1,4);
corrValTheo = power(10, (80 - 100)./50);
histfit(ampData{4}(indTe_100)./corrValTheo, 20);
ax2 = axis();
title('Te Rec 100 (scaled 80)');
axis([ax1(1) ax1(2) ax2(3) ax2(4)]);
subplot(5,1,5);
corrValTheo = power(10, (70 - 100)./50);
histfit(ampData{4}(indTe_100)./corrValTheo, 20);
ax2 = axis();
title('Te Rec 100 (scaled 70)');
axis([ax1(1) ax1(2) ax2(3) ax2(4)]);
xlabel('RMS');

figure();
subplot(5, 1, 1);
h = histfit(ampData{4}(indNe_90), 30);
ax1 = axis();
title('Ne Rec 90');
subplot(5, 1, 2);
histfit(ampData{4}(indNe_100), 20);
ax2 = axis();
title('Ne Rec 100');
axis([ax1(1) ax1(2) ax2(3) ax2(4)]);
subplot(5,1,3);
corrValTheo = power(10, (90 - 100)./50);
histfit(ampData{4}(indNe_100)./corrValTheo, 20);
ax2 = axis();
title('Ne Rec 100 (scaled 90)');
axis([ax1(1) ax1(2) ax2(3) ax2(4)]);
subplot(5,1,4);
corrValTheo = power(10, (80 - 100)./50);
histfit(ampData{4}(indNe_100)./corrValTheo, 20);
ax2 = axis();
title('Ne Rec 100 (scaled 80)');
axis([ax1(1) ax1(2) ax2(3) ax2(4)]);
subplot(5,1,5);
corrValTheo = power(10, (70 - 100)./50);
histfit(ampData{4}(indNe_100)./corrValTheo, 20);
ax2 = axis();
title('Ne Rec 100 (scaled 70)');
axis([ax1(1) ax1(2) ax2(3) ax2(4)]);
xlabel('RMS');
