clear;clc;
addpath('utils')
addpath('utils/liblinear/matlab');
% Path setttings
fea_path = '/tmp/fea/VGG_FACE';
name_train = 'IJBA_train_1N.mat';
name_probe = 'IJBA_prob_1N.mat';
name_gallery = 'IJBA_gal_1N.mat';
% Load features
load(fullfile(fea_path, name_train), 'train'); 
trainset = train; clear train % Deal with the conflict with liblinear
load(fullfile(fea_path, name_probe), 'probe');
load(fullfile(fea_path, name_gallery), 'gal');

numdim = size(gal(1).fea, 1);

% Benchmark settings
veriFarPoints = [0, kron(10.^(-3:-1), 1:9), 1]; % FAR points for face verification ROC plot
reportVeriFar = [0.01 0.1]; % the FAR point for verification performance reporting
reportRank = [1 5 10];
    
numVeriFarPoints = length(veriFarPoints);

numTrials = 1; % You can also implement more splits
    
TPIR = zeros(numTrials, numVeriFarPoints); % verification rates of the 10 trials
veriFAR = zeros(numTrials, numVeriFarPoints); % verification false accept rates of the 10 trials
CMC = zeros(numTrials, 100);
    
[~, veriFarIndex] = ismember(reportVeriFar, veriFarPoints);

% Split test
for t = 1:numTrials    
    % Template and media process: train 
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
    
    % Template and media process: probe
    probe_temp_unique = unique(probe(t).template);
    probe_n = length(probe_temp_unique);
    probe_template(probe_n).fea = [];
    probe_mean_fea = zeros(numdim, probe_n);
    probe_label = zeros(probe_n, 1);
    for j = 1:probe_n
        %fprintf('Trail %d: Loading probe template %d/%d \n', t, j, probe_n);
        probe_temp_idx = find(probe(t).template == probe_temp_unique(j));
        probe_media = probe(t).media(probe_temp_idx);
        probe_media_unique = unique(probe_media);

        for k = 1:length(probe_media_unique)
            probe_media_idx = find(probe_media == probe_media_unique(k));
            probe_template(j).fea = [probe_template(j).fea, mean(probe(t).fea(:, probe_temp_idx(probe_media_idx)), 2)];
        end
        probe_template(j).fea = normc(probe_template(j).fea);
        probe_mean_fea(:, j) = mean(probe_template(j).fea, 2);
        probe_label(j) = probe(t).label(probe_temp_idx(1));
    end
    
    % Template and media process: gallery
    gal_temp_unique = unique(gal(t).template);
    gal_n = length(gal_temp_unique);
    gal_template(gal_n).fea = [];
    gal_mean_fea = zeros(numdim, gal_n);
    gal_label = zeros(gal_n, 1);
    for j = 1:gal_n
        %fprintf('Trail %d: Loading Gallery template %d/%d \n', t, j, gal_n);
        gal_temp_idx = find(gal(t).template == gal_temp_unique(j));
        gal_media = gal(t).media(gal_temp_idx);
        gal_media_unique = unique(gal_media);

        for k = 1:length(gal_media_unique)
            gal_media_idx = find(gal_media == gal_media_unique(k));
            gal_template(j).fea = [gal_template(j).fea, mean(gal(t).fea(:, gal_temp_idx(gal_media_idx)), 2)];
        end
        gal_template(j).fea = normc(gal_template(j).fea);
        gal_mean_fea(:, j) = mean(gal_template(j).fea, 2);
        gal_label(j) = gal(t).label(gal_temp_idx(1));
    end
    
    score = zeros(probe_n, gal_n);
    fprintf('Split %d: Data compared.\n', t);
    n_temp_train = length(train_label);
    
    for p=1:probe_n
        P_train=[probe_template(p).fea, [train_template(randperm(n_temp_train, 200)).fea]];
        P_label=2*ones(size(P_train,2), 1); P_label(1:size(probe_template(p).fea,2)) = 1;
        model_P = train(P_label, sparse(P_train), '-s 2 -c 10 -w1 400 -w2 0.5 -q', 'col');
        for x=1:gal_n
            tic;
            fprintf('Split %d: Probe %d/%d Gallery %d/%d   ',t, p,probe_n, x,gal_n);
            X_train=[gal_template(x).fea, gal_template(gal_label ~= gal_label(x)).fea];
            X_label=2*ones(size(X_train,2), 1); X_label(1:size(gal_template(x).fea,2)) = 1;
            model_X = train(X_label, sparse(X_train), '-s 2 -c 10 -w1 10 -w2 0.5 -q', 'col');

            [~, ~, score_P] = predict(1, sparse(gal_mean_fea(:, x)), model_P, '-q', 'col');
            [~, ~, score_X] = predict(1, sparse(probe_mean_fea(:, p)), model_X, '-q', 'col');
            score(p, x) = (score_P + score_X) / 2;
            toc;
        end
    end
    
    % Evaluate the verification performance.
    [TPIR(t,:), CMC(t, :), veriFAR(t,:)] = EvalTPIR(score', gal_label, probe_label, veriFarPoints, 30);
end

TPIR(veriFarIndex)
CMC(reportRank)