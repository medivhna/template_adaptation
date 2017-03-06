function [TPIR, CMC, FAR] = EvalTPIR(score0, galLabels, probLabels, farPoints, L)

if nargin < 3 || isempty(probLabels)
    galLabels = probLabels;
end

if ~iscolumn(galLabels)
    galLabels = galLabels';
end

if ~isrow(probLabels)
    probLabels = probLabels';
end

binaryLabels = bsxfun(@eq, probLabels, galLabels);

if ~isequal(size(score0), size(binaryLabels))
    error('The size of labels is not the same as the size of the score matrix.');
end

[score, sortedIndex] = sort(score0, 2, 'descend');
if L
    score = score(:, 1:L);
    labels = logical(zeros(size(score, 1), L));
    for r=1:size(score, 1)
        labels(r, :) = binaryLabels(r, sortedIndex(r, 1:L));
    end
else
    score = score;
    labels = logical(zeros(size(score)));
    for r=1:size(score, 1)
        labels(r, :) = binaryLabels(r, sortedIndex(r, :));
    end
end

genScore = score(labels); % Similarities of matched pairs
impScore = score(~labels); % Similarities of non-matched pairs

Nimp = length(impScore);

if nargin < 4 || isempty(farPoints)
    falseAlarms = 0 : Nimp;
else
    if any(farPoints < 0) || any(farPoints > 1)
        error('FAR should be in the range [0,1].');
    end
    falseAlarms = round(farPoints * Nimp); % False Alarm counts
end

impScore = sort(impScore, 'descend');

isZeroFAR = (falseAlarms == 0);
isOneFAR = (falseAlarms == Nimp);
thresholds = zeros(1, length(falseAlarms));
% Threshold with false alarm is the k-th non-matched similarity
thresholds(~isZeroFAR & ~isOneFAR) = impScore( falseAlarms(~isZeroFAR & ~isOneFAR) );

highGenScore = genScore(genScore > impScore(1)); % Matched similarities higher than highest non-matched.
if isempty(highGenScore)
    thresholds(isZeroFAR) = impScore(1) + sqrt(eps);
else
	% Threshold with no false alarm is the average similarity of minimum matched and maximum non-matched.
    thresholds(isZeroFAR) = ( impScore(1) + min(highGenScore) ) / 2; 
end

% Threshold with full false alarm is the minimum similarity of all.
thresholds(isOneFAR) = min(impScore(end), min(genScore)) - sqrt(eps);

if ~iscolumn(genScore)
    genScore = genScore';
end

if ~isrow(thresholds)
    thresholds = thresholds';
end

FAR = falseAlarms / Nimp;
TPIR = mean(bsxfun(@ge, genScore, thresholds));

%% get the matching rank of each probe
[score1, sortedIndex2] = sort(score0, 'descend'); % rank the score
labels = logical(zeros(size(score1)));
for r=1:size(score1, 2)
    labels(:, r) = binaryLabels(sortedIndex2(:, r), r);
end
score1(labels == false) = -Inf; % set scores of non-matches to -Inf
[~, maxIndex] = max(score1); % get the location of the maximum genuine score
%maxIndex(tscore == -Inf) = 0;
%[probRanks, ~] = bsxfun(@eq, 1:100, maxIndex); % get the matching rank of each probe, by finding the location of the matches in the sorted index


%% evaluate
%if ~iscolumn(probRanks)
%    probRanks = probRanks';
%end

T = bsxfun(@le, maxIndex', 1:100); % compare the probe matching ranks to the number of retrievals
CMC = squeeze( mean(T) ); % average over all probes 
