function flag_to_skip =main_stab(frame_dir,out_dir) 
    % Parameters
    %frame_dir = '/home/shubham/Downloads/frames/';
    %out_dir = '/home/shubham/Downloads/frames_opt/';
    crop_ratio = 0.8;
    image_files = dir(frame_dir);
    image_files=image_files(3:end);
    [num_frames,~]=size(image_files);
%   num_frames=10;
    % Read Images
    [im_array,ss_l, ss_b] = readImages(frame_dir, num_frames);
    im_size=[ss_l, ss_b];
    % Extract SIFT features
    [features, descriptors] = extractSIFT(im_array);
    % Get original camera path
    [flag_to_skip,t_transforms] = getTransforms(im_array, features, descriptors);
    
    %% Save and load variables to reduce re-run times
    %save('variables.mat', 'im_array', 't_transforms');
    %load('variables.mat');
    if (flag_to_skip==0)
        % Get optimized camera path
        n_transforms = optimizeTransforms(t_transforms, im_size);
        disp ('step 1 is done')
        % Plot camera paths
        %plotPath(t_transforms, n_transforms);

        % Apply new camera path and crop
        n_im_array = applyOptimizedTransforms(im_array, n_transforms,im_size);
        disp ('step 2 is done')
        % Save frames

        for k=1:num_frames
            file_name = fullfile(out_dir, sprintf('%.10d.png', k));
            imwrite(n_im_array{k}, file_name);
        end
    else
        for k=1:num_frames
            file_name = fullfile(out_dir, sprintf('%.10d.png', k));
            imwrite(im_array{k}, file_name);
        end
    end
end

