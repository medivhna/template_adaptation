clear;clc;
addpath('libsvm/matlab');
% Path setttings
root_path = '/ssd/IJB_A/csv';
fea_path = '/home/wangguanshuo/fea/IJB_A_sub/_iter_340000';
name_train_11 = 'IJBA_train_11.mat';
name_verify_11 = 'IJBA_verify_11.mat';
name_train_1N = 'IJBA_1N_train.mat';
name_probe_1N = 'IJBA_1N_prob.mat';
name_gallery_1N = 'IJBA_1N_gal.mat';


load(fullfile(fea_path, name_train_11), 'train');
for t = 1:10
    fprintf('Add template and media to train_11 split %d\n', t);
    csv_path = fullfile(root_path, '11', ...
                         ['split',num2str(t)], ...
                         ['train_',num2str(t),'.csv']);
    train(t).template = load_label(csv_path, 25, 1);
    train(t).media = load_label(csv_path, 25, 4); 
end
save(fullfile(fea_path, name_train_11), 'train');
clear train

load(fullfile(fea_path, name_verify_11), 'verify');
for t = 1:10
    fprintf('Add template and media to verify_11 split %d\n', t);
    csv_path = fullfile(root_path, '11', ...
                         ['split',num2str(t)], ...
                         ['verify_metadata_',num2str(t),'.csv']);   
    verify(t).template = load_label(csv_path, 25, 1);
    verify(t).media = load_label(csv_path, 25, 4); 
end
save(fullfile(fea_path, name_verify_11), 'verify');
clear verify
 
load(fullfile(fea_path, name_train_1N), 'train');
for t = 1:10
    fprintf('Add template and media to train_1N split %d\n', t);
    csv_path = fullfile(root_path, '1N', ...
                         ['split',num2str(t)], ...
                         ['train_',num2str(t),'.csv']);
    train(t).template = load_label(csv_path, 25, 1);
    train(t).media = load_label(csv_path, 25, 4); 
end
save(fullfile(fea_path, name_train_1N), 'train');
clear train

load(fullfile(fea_path, name_probe_1N), 'probe');
for t = 1:10
    fprintf('Add template and media to probe_1N split %d\n', t);
    csv_path = fullfile(root_path, '1N', ...
                         ['split',num2str(t)], ...
                         ['search_probe_',num2str(t),'.csv']);
    probe(t).template = load_label(csv_path, 25, 1);
    probe(t).media = load_label(csv_path, 25, 4); 
end
save(fullfile(fea_path, name_probe_1N), 'probe');
clear probe

load(fullfile(fea_path, name_gallery_1N), 'gal');
for t = 1:10
    fprintf('Add template and media to gallery_1N split %d\n', t);
    csv_path = fullfile(root_path, '1N', ...
                         ['split',num2str(t)], ...
                         ['search_gallery_',num2str(t),'.csv']);
    gal(t).template = load_label(csv_path, 25, 1);
    gal(t).media = load_label(csv_path, 25, 4); 
end
save(fullfile(fea_path, name_gallery_1N), 'gal');
clear gal