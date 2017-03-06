clear;clc;
addpath('utils')
addpath('utils/liblinear/matlab');
% Path settings
fea_path = '/tmp/fea/VGG_FACE';
name_train = 'IJBA_train_11.mat';
name_verify = 'IJBA_verify_11.mat';
csv_path = '/data/IJB_A/csv/11'; % 1:1 comparisons need to load the raw csv.
% Load features
load(fullfile(fea_path, name_train), 'train');
trainset = train; clear train; % Deal with the conflict with liblinear
load(fullfile(fea_path, name_verify), 'verify');

numdim = size(verify(1).fea, 1);

% Benchmark settings
veriFarPoints = [0, kron(10.^(-3:-1), 1:9), 1]; % FAR points for face verification ROC plot
reportVeriFar = [0.001 0.01 0.1]; % the FAR point for verification performance reporting
numVeriFarPoints = length(veriFarPoints);

numTrials = 1; % You can also implement more splits

TAR = zeros(numTrials, numVeriFarPoints); % verification rates of the 10 trials
veriFAR = zeros(numTrials, numVeriFarPoints); % verification false accept rates of the 10 trials

% Get the FAR or rank index where we report performance.
[~, veriFarIndex] = ismember(reportVeriFar, veriFarPoints);
    
matlabpool(12);
% Split test
for t = 1:numTrials
    % Template and media process
    train_temp_unique = unique(trainset(t).template);
    train_n = length(train_temp_unique);
    train_template(train_n).fea = [];
    train_label = zeros(train_n, 1);
    for j = 1:train_n
        train_temp_idx = find(trainset(t).template == train_temp_unique(j));
        train_media = trainset(t).media(train_temp_idx);
        train_media_unique = unique(train_media);

        for k = 1:length(train_media_unique)
            media_idx = find(train_media == train_media_unique(k));
            train_template(j).fea = [train_template(j).fea, mean(trainset(t).fea(:, train_temp_idx(media_idx)), 2)];
        end
        train_template(j).fea = normc(train_template(j).fea);
        train_label(j) = trainset(t).label(train_temp_idx(1));
    end
    
    verify_temp_unique = unique(verify(t).template);
    verify_n = length(verify_temp_unique);
    verify_template(verify_n).fea = [];
    verify_mean_fea = zeros(numdim, verify_n);
    verify_label = zeros(verify_n, 1);
    for j = 1:verify_n
        verify_temp_idx = find(verify(t).template == verify_temp_unique(j));
        verify_media = verify(t).media(verify_temp_idx);
        verify_media_unique = unique(verify_media);

        for k = 1:length(verify_media_unique)
            media_idx = find(verify_media == verify_media_unique(k));
            verify_template(j).fea = [verify_template(j).fea, mean(verify(t).fea(:, verify_temp_idx(media_idx)), 2)];
        end
        verify_template(j).fea = normc(verify_template(j).fea);
        verify_mean_fea(:, j) = mean(verify_template(j).fea, 2);
        verify_label(j) = verify(t).label(verify_temp_idx(1));
    end
    
    compare_csv = fullfile(csv_path, ...
                          ['split',num2str(t)], ...
                          ['verify_comparisons_',num2str(t),'.csv']);
    [comp1, comp2] = textread(compare_csv,'%n%n%*[^\n]','delimiter',',');
    n_pair = length(comp1);
    score = zeros(n_pair, 1);
    compLabel = logical(score);
    
    fprintf('Split %d: Data compared.\n', t);
    n_temp_train = length(train_label);
    
    tic;
    parfor k=1:n_pair
        fprintf('Split %d: Comparison %d\n', t, k);
        idx_p = find(verify_temp_unique == comp1(k));
        idx_q = find(verify_temp_unique == comp2(k));
        train_idx = randperm(n_temp_train);
        
        P_train = [verify_template(idx_p).fea, [train_template(train_idx).fea]];
        P_label=2*ones(size(P_train,2), 1); P_label(1:size(verify_template(idx_p).fea,2)) = 1;
        w1 = length(P_label) / (2 * size(verify_template(idx_p).fea, 2)); 
        w2 = length(P_label) / (2 * (length(P_label) - size(verify_template(idx_p).fea, 2)));
        option = ['-s 2 -c 10 -w1 ' num2str(w1), ' -w2 ', num2str(w2), ' -q'];
        model_P = train(P_label, sparse(P_train), option, 'col');
        
        Q_train = [verify_template(idx_q).fea, [train_template(train_idx).fea]];
        Q_label=2*ones(size(Q_train,2), 1); Q_label(1:size(verify_template(idx_q).fea,2)) = 1;
        w1 = length(Q_label) / (2 * size(verify_template(idx_q).fea, 2)); 
        w2 = length(Q_label) / (2 * (length(Q_label) - size(verify_template(idx_q).fea, 2)));
        option = ['-s 2 -c 10 -w1 ' num2str(w1), ' -w2 ', num2str(w2), ' -q'];
        model_Q = train(Q_label, sparse(Q_train), option, 'col');
        
        [~, ~, score_P] = predict(1, sparse(verify_mean_fea(:, idx_q)), model_P, '-q', 'col');
        [~, ~, score_Q] = predict(1, sparse(verify_mean_fea(:, idx_p)), model_Q, '-q', 'col');
        score(k) = (score_P + score_Q) / 2;
        
        compLabel(k) = (verify_label(idx_p) == verify_label(idx_q));
    end
    
    toc;
    % Evaluate the verification performance.
    [TAR(t,:), veriFAR(t,:)] = EvalTAR(score, compLabel, veriFarPoints);
    
    clear template
end
matlabpool close;

TAR(veriFarIndex)
