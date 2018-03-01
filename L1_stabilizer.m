clc;
clear;
run('~/vlfeat/toolbox/vl_setup.m');
path='/home/shubham/Egocentric/dataset/GTEA/pngs/';
path_destination='/home/shubham/Egocentric/dataset/GTEA/L1_stabilizer/';
videos=dir(path);
videos=videos(3:end);
[NumVideos,~]=size(videos);
% parfor i=1:NumVideos
%     frame_dir=strcat(path,videos(i).name,'/');
%     out_dir=strcat(path_destination, videos(i).name,'/');
%     im_size=main_stab(frame_dir, out_dir);
% end

%% create video from image sequence 
% for i =1:NumVideos
%     %shuttleVideo = VideoReader('/home/shubham/Downloads/OP01-R01-PastaSalad.mp4');
%     out_dir=strcat(path_destination, videos(i).name,'/');
%     original=strcat(path, videos(i).name,'/');
%     
%     imageNames = dir(fullfile(out_dir,'*.png'));
%     imageNames_org = dir(fullfile(original,'*.png'));
% 
%     imageNames = {imageNames.name}';
%     imageNames_org = {imageNames_org.name}';
% 
% 
%     outputVideo = VideoWriter(fullfile(out_dir,'video.avi'));
%     outputVideo_org = VideoWriter(fullfile(original,'video_org.avi'));
% 
%     outputVideo.FrameRate = 15;
%     outputVideo_org.FrameRate=15;
%     open(outputVideo)
%     open(outputVideo_org)
% 
%     for ii = 1:length(imageNames)
%       img = imread(fullfile(out_dir,imageNames{ii}));
%       img_org = imread(fullfile(original,imageNames_org{ii}));
%       writeVideo(outputVideo,img)
%       writeVideo(outputVideo_org,img_org)
%     end
%     close(outputVideo)
%     close(outputVideo_org)
% end