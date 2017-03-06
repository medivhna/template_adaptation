function label = load_label(filename, numcol, subject_idx)
fid = fopen(filename, 'r');
C = textscan(fid, repmat('%s', 1, numcol), 'delimiter', ',', 'CollectOutput', true);
C = C{1};
fclose(fid);

label = sscanf(sprintf('%s*', C{2:end, subject_idx}), '%f*');