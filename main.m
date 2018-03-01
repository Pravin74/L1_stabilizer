function im_size =main_stab(frame_dir,out_dir) 
    clear;
    % Parameters
    %frame_dir = '/home/shubham/Downloads/frames/';
    %out_dir = '/home/shubham/Downloads/frames_opt/';
    im_size = [360 640];
    crop_ratio = 0.8;
    image_files = dir(frame_dir);
    [num_frames,~]=size(image_files);
    %num_frames=1000;
    % Read Images
    im_array = readImages(frame_dir, num_frames);
    % Extract SIFT features
    [features, descriptors] = extractSIFT(im_array);
    % Get original camera path
    t_transforms = getTransforms(im_array, features, descriptors);
    tic
    %% Save and load variables to reduce re-run times
    %save('variables.mat', 'im_array', 't_transforms');
    %load('variables.mat');

    % Get optimized camera path
    n_transforms = optimizeTransforms(t_transforms, im_size);
    disp ('step 1 is done')
    % Plot camera paths
    %plotPath(t_transforms, n_transforms);

    % Apply new camera path and crop
    n_im_array = applyOptimizedTransforms(im_array, n_transforms);
    disp ('step 2 is done')
    % Save frames
    for k=1:num_frames
        file_name = fullfile(out_dir, sprintf('%d.png', k));
        imwrite(n_im_array{k}, file_name);
    end
    toc
end
%% create video from image sequence 
% shuttleVideo = VideoReader('/home/shubham/Downloads/OP01-R01-PastaSalad.mp4');

%imageNames = dir(fullfile('/home/shubham/Downloads/frames_opt/','*.png'));
%imageNames_orig = dir(fullfile('/home/shubham/Downloads/frames/','*.png'));

% imageNames = {imageNames.name}';
% 
% outputVideo = VideoWriter(fullfile('/home/shubham/Downloads/','video_1.avi'));
% outputVideo.FrameRate = 24;
% open(outputVideo)

%for ii = 1:length(imageNames)
%   img = imread(fullfile('/home/shubham/Downloads/frames_opt/',imageNames{ii}));
%   img_orig = imread(fullfile('/home/shubham/Downloads/frames_opt/',imageNames_orig{ii}));
%   imshowpair(img, img_orig, 'montage');

%end
% close(outputVideo)
