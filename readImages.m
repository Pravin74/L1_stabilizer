function [ images_array,ss_l, ss_b ] = readImages( image_dir, max_val )
%readImages Summary
% Reads images in the give directory and returns a cell array.

% Read images
file_names = fullfile(image_dir, '*.png');
image_files = dir(file_names);

% Number of frames for testing
n = min([length(image_files) max_val]);

% Preallocation
images_array = cell(n, 1);

for k = 1:n
    base_name = image_files(k).name;
    full_name = fullfile(image_dir, base_name);
    fprintf(1, 'Now reading %s\n', full_name);
    im = imread(full_name);
    [ss_l,ss_b,~]=size(im);
    im = imresize(im, [ss_l ,ss_b]);
    images_array{k} = im;
    
end

end