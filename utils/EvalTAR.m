function [TAR, FAR, thresholds] = EvalTAR(score, binaryLabels, farPoints)

if ~isequal(length(score), length(binaryLabels))
    error('The size of labels is not the same as the size of the score matrix.');
end

genScore = score(binaryLabels); % Similarities of matched pairs
impScore = score(~binaryLabels); % Similarities of non-matched pairs
clear score binaryLabels

Nimp = length(impScore);

if nargin < 3 || isempty(farPoints)
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
TAR = mean( bsxfun(@ge, genScore, thresholds) );