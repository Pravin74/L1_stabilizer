function [status,tform, inlierPoints1, inlierPoints2] ...
    = my_estimateGeometricTransform(matchedPoints1, matchedPoints2, ...
    transformType, varargin)
%estimateGeometricTransform Estimate geometric transformation from matching point pairs.
%   tform = estimateGeometricTransform(matchedPoints1,matchedPoints2,
%   transformType) returns a 2-D geometric transform which maps the
%   inliers in matchedPoints1 to the inliers in matchedPoints2.
%   matchedPoints1 and matchedPoints2 can be cornerPoints objects,
%   SURFPoints objects, MSERRegions objects, BRISKPoints objects, or M-by-2
%   matrices of [x,y] coordinates. transformType can be 'similarity',
%   'affine', or 'projective'. Outliers in matchedPoints1 and
%   matchedPoints2 are excluded by using the M-estimator SAmple Consensus
%   (MSAC) algorithm. The MSAC algorithm is a variant of the Random Sample
%   Consensus (RANSAC) algorithm. The returned tform is an affine2d object
%   if transformType is set to 'similarity' or 'affine', and is a
%   projective2d object otherwise.
%
%   [tform,inlierPoints1,inlierPoints2] = estimateGeometricTransform(...)
%   additionally returns the corresponding inlier points in inlierPoints1
%   and inlierPoints2.
%
%   [tform,inlierPoints1,inlierPoints2,status] =
%   estimateGeometricTransform(...) additionally returns a status code with
%   the following possible values:
% 
%     0: No error. 
%     1: matchedPoints1 and matchedPoints2 do not contain enough points.
%     2: Not enough inliers have been found.
%
%   When the STATUS output is not given, the function will throw an error
%   if matchedPoints1 and matchedPoints2 do not contain enough points or
%   if not enough inliers have been found.
%
%   [...] = estimateGeometricTransform(matchedPoints1,matchedPoints2, 
%   transformType,Name,Value) specifies additional
%   name-value pair arguments described below:
%
%   'MaxNumTrials'        A positive integer scalar specifying the maximum
%                         number of random trials for finding the inliers.
%                         Increasing this value will improve the robustness
%                         of the output at the expense of additional
%                         computation.
% 
%                         Default value: 1000
%  
%   'Confidence'          A numeric scalar, C, 0 < C < 100, specifying the
%                         desired confidence (in percentage) for finding
%                         the maximum number of inliers. Increasing this
%                         value will improve the robustness of the output
%                         at the expense of additional computation.
%
%                         Default value: 99
% 
%   'MaxDistance'         A positive numeric scalar specifying the maximum
%                         distance in pixels that a point can differ from
%                         the projection location of its associated point.
% 
%                         Default value: 1.5
% 
%   Class Support
%   -------------
%   matchedPoints1 and matchedPoints2 must be cornerPoints objects,
%   SURFPoints objects, MSERRegions objects, BRISKPoints objects, or M-by-2
%   matrices of [x,y] coordinates.
%
%   % EXAMPLE: Recover a transformed image using SURF feature points
%   % --------------------------------------------------------------
%   Iin  = imread('cameraman.tif'); imshow(Iin); title('Base image');
%   Iout = imresize(Iin, 0.7); Iout = imrotate(Iout, 31);
%   figure; imshow(Iout); title('Transformed image');
%  
%   % Detect and extract features from both images
%   ptsIn  = detectSURFFeatures(Iin);
%   ptsOut = detectSURFFeatures(Iout);
%   [featuresIn,   validPtsIn] = extractFeatures(Iin,  ptsIn);
%   [featuresOut, validPtsOut] = extractFeatures(Iout, ptsOut);
%  
%   % Match feature vectors
%   indexPairs = matchFeatures(featuresIn, featuresOut);
%   matchedPtsIn  = validPtsIn(indexPairs(:,1));
%   matchedPtsOut = validPtsOut(indexPairs(:,2));
%   figure; showMatchedFeatures(Iin,Iout,matchedPtsIn,matchedPtsOut);
%   title('Matched SURF points, including outliers');
%  
%   % Exclude the outliers and compute the transformation matrix
%   [tform,inlierPtsOut,inlierPtsIn] = estimateGeometricTransform(...
%        matchedPtsOut,matchedPtsIn,'similarity');
%   figure; showMatchedFeatures(Iin,Iout,inlierPtsIn,inlierPtsOut);
%   title('Matched inlier points');
%  
%   % Recover the original image Iin from Iout
%   outputView = imref2d(size(Iin));
%   Ir = imwarp(Iout, tform, 'OutputView', outputView);
%   figure; imshow(Ir); title('Recovered image');
%
% See also fitgeotrans, cornerPoints, SURFPoints, MSERRegions, BRISKPoints,
%          detectMinEigenFeatures, detectFASTFeatures, detectSURFFeatures,
%          detectMSERFeatures, detectBRISKFeatures, extractFeatures,
%          matchFeatures, imwarp

% References:
% [1] R. Hartley, A. Zisserman, "Multiple View Geometry in Computer
%     Vision," Cambridge University Press, 2003.
% [2] P. H. S. Torr and A. Zisserman, "MLESAC: A New Robust Estimator
%     with Application to Estimating Image Geometry," Computer Vision
%     and Image Understanding, 2000.

% Copyright  The MathWorks, Inc.

%#codegen
%#ok<*EMCA>
% List of status code
statusCode = struct(...
    'NoError',           int32(0),...
    'NotEnoughPts',      int32(1),...
    'NotEnoughInliers',  int32(2));

% Parse and check inputs
[points1, points2, maxNumTrials, confidence, maxDistance, sampleSize, ...
    status, classToUse] = parseInputs(statusCode, matchedPoints1, ...
    matchedPoints2, transformType, varargin{:});

% return identity matrix in case of failure
failedMatrix = eye([3,3], classToUse);

% Compute the geometric transformation
if status == statusCode.NoError
    [isFound, tmatrix, inliers] = msac(points1, points2, maxNumTrials, ...
        confidence, maxDistance, sampleSize, classToUse);
    if ~isFound
        status = statusCode.NotEnoughInliers;
    end
   
    % Do an extra check to verify the tform matrix. Check if matrix is 
    % singular or contains infs or nans.
    if isequal(det(tmatrix),0) || any(~isfinite(tmatrix(:)))
        status = statusCode.NotEnoughInliers;
        tmatrix = failedMatrix;
    end
else
    tmatrix = failedMatrix;
end

% Extract inlier points
if status == statusCode.NoError
    inlierPoints1 = matchedPoints1(inliers, :);
    inlierPoints2 = matchedPoints2(inliers, :);
else
    inlierPoints1 = matchedPoints1([]);
    inlierPoints2 = matchedPoints2([]);
    tmatrix = failedMatrix;
end

% Report runtime error if the status output is not requested
reportError = (nargout ~= 4);
if reportError
    checkRuntimeStatus(statusCode, status);
end

isSimilarityOrAffine = sampleSize < 4;

if isSimilarityOrAffine
    % Use the 3x2 affine2d syntax to have last column automatically
    % added to tform matrix. This prevents precision issues from
    % propagating downstream.
    tform = affine2d(tmatrix(:,1:2));
else % projective
    tform = projective2d(tmatrix);
end

%==========================================================================
% Check runtime status and report error if there is one
%==========================================================================
function checkRuntimeStatus(statusCode, status)
% if (status==statusCode.NotEnoughPts || status==statusCode.NotEnoughInliers)
%     flag_to_skip=1;
% end
coder.internal.errorIf(status==statusCode.NotEnoughPts, ...
    'vision:estimateGeometricTransform:notEnoughPts');

coder.internal.errorIf(status==statusCode.NotEnoughInliers, ...
    'vision:estimateGeometricTransform:notEnoughInliers');

%==========================================================================
% Parse and check inputs
%==========================================================================
function [points1, points2, maxNumTrials, confidence, maxDistance, ...
    sampleSize, status, classToUse] = parseInputs(statusCode, ...
    matchedPoints1, matchedPoints2, transform_type, varargin)

isSimulationMode = isempty(coder.target);
if isSimulationMode
    % Instantiate an input parser
    parser = inputParser;
    parser.FunctionName = 'estimateGeometricTransform';
    
    % Specify the optional parameters
    parser.addParameter('MaxNumTrials', 1000);
    parser.addParameter('Confidence',   99);
    parser.addParameter('MaxDistance',  1.5);
    
    % Parse and check optional parameters
    parser.parse(varargin{:});
    r = parser.Results;
    
    maxNumTrials = r.MaxNumTrials;
    confidence   = r.Confidence;
    maxDistance  = r.MaxDistance;
    
else
    % Instantiate an input parser
    parms = struct( ...
        'MaxNumTrials',       uint32(0), ...
        'Confidence',         uint32(0), ...
        'MaxDistance',        uint32(0));
    
    popt = struct( ...
        'CaseSensitivity', false, ...
        'StructExpand',    true, ...
        'PartialMatching', false);
    
    % Specify the optional parameters
    optarg       = eml_parse_parameter_inputs(parms, popt,...
        varargin{:});
    maxNumTrials = eml_get_parameter_value(optarg.MaxNumTrials,...
        1000, varargin{:});
    confidence   = eml_get_parameter_value(optarg.Confidence,...
        99, varargin{:});
    maxDistance  = eml_get_parameter_value(optarg.MaxDistance,...
        1.5, varargin{:});
end

% Check required parameters
sampleSize = checkTransformType(transform_type);

[points1, points2] = vision.internal.inputValidation.checkAndConvertMatchedPoints(...
    matchedPoints1, matchedPoints2, ...
    mfilename, 'MATCHEDPOINTS1', 'MATCHEDPOINTS2');

status  = checkPointsSize(statusCode, sampleSize, points1, points2);

% Check optional parameters
checkMaxNumTrials(maxNumTrials);
checkConfidence(confidence);
checkMaxDistance(maxDistance);

classToUse = getClassToUse(points1, points2);

maxNumTrials = int32(maxNumTrials);
confidence   = cast(confidence,  classToUse);
maxDistance  = cast(maxDistance, classToUse);
sampleSize   = cast(sampleSize,  classToUse);


%==========================================================================
function status = checkPointsSize(statusCode, sampleSize, points1, points2)

coder.internal.errorIf( size(points1,1) ~= size(points2,1), ...
    'vision:estimateGeometricTransform:numPtsMismatch');

coder.internal.errorIf( ~isequal(class(points1), class(points2)), ...
    'vision:estimateGeometricTransform:classPtsMismatch');

if size(points1,1) < sampleSize
    status = statusCode.NotEnoughPts;
else
    status = statusCode.NoError;
end

%==========================================================================
function r = checkMaxNumTrials(value)
validateattributes(value, {'numeric'}, ...
    {'scalar', 'nonsparse', 'real', 'integer', 'positive', 'finite'},...
    'estimateGeometricTransform', 'MaxNumTrials');
r = 1;

%========================================================================== 
function r = checkConfidence(value)
validateattributes(value, {'numeric'}, ...
    {'scalar', 'nonsparse', 'real', 'positive', 'finite', '<', 100},...
    'estimateGeometricTransform', 'Confidence');
r = 1;

%==========================================================================
function r = checkMaxDistance(value)
validateattributes(value, {'numeric'}, ...
    {'scalar', 'nonsparse', 'real', 'positive', 'finite'},...
    'estimateGeometricTransform', 'MaxDistance');
r = 1;

%==========================================================================
function sampleSize = checkTransformType(value)
list = {'similarity', 'affine', 'projective'};
validatestring(value, list, 'estimateGeometricTransform', ...
    'TransformType');

switch(lower(value(1)))
    case 's'
        sampleSize = 2;
    case 'a'
        sampleSize = 3;
    otherwise
        sampleSize = 4;
end

%==========================================================================
function c = getClassToUse(points1, points2)
if isa(points1, 'double') || isa(points2, 'double')
    c = 'double';
else
    c = 'single';
end

%==========================================================================
function flag = isTestingMode
isSimulationMode = isempty(coder.target);
coder.extrinsic('vision.internal.testEstimateGeometricTransform');
if isSimulationMode
    flag = vision.internal.testEstimateGeometricTransform;
else
    flag = eml_const(vision.internal.testEstimateGeometricTransform);
end

%==========================================================================
% Algorithms for computing the transformation matrix.
%==========================================================================
function T = computeSimilarity(points1, points2, classToUse)
numPts = size(points1, 1);
constraints = zeros(2*numPts, 5, classToUse);
constraints(1:2:2*numPts, :) = [-points1(:, 2), points1(:, 1), ...
    zeros(numPts, 1), -ones(numPts,1), points2(:,2)];
constraints(2:2:2*numPts, :) = [points1, ones(numPts,1), ...
    zeros(numPts, 1), -points2(:,1)];
[~, ~, V] = svd(constraints, 0);
h = V(:, end);
T = coder.nullcopy(eye(3, classToUse));
T(:, 1:2) = [h(1:3), [-h(2); h(1); h(4)]] / h(5);
T(:, 3)   = [0; 0; 1];

%==========================================================================
function T = computeAffine(points1, points2, classToUse)
numPts = size(points1, 1);
constraints = zeros(2*numPts, 7, classToUse);
constraints(1:2:2*numPts, :) = [zeros(numPts, 3), -points1, ...
    -ones(numPts,1), points2(:,2)];
constraints(2:2:2*numPts, :) = [points1, ones(numPts,1), ...
    zeros(numPts, 3), -points2(:,1)];
[~, ~, V] = svd(constraints, 0);
h = V(:, end);
T = coder.nullcopy(eye(3, classToUse));
T(:, 1:2) = reshape(h(1:6), [3,2]) / h(7);
T(:, 3)   = [0; 0; 1];

%==========================================================================
function T = computeProjective(points1, points2, classToUse)
numPts = size(points1, 1);
p1x = points1(:, 1);
p1y = points1(:, 2);
p2x = points2(:, 1);
p2y = points2(:, 2);
constraints = zeros(2*numPts, 9, classToUse);
constraints(1:2:2*numPts, :) = [zeros(numPts,3), -points1, ...
    -ones(numPts,1), p1x.*p2y, p1y.*p2y, p2y];
constraints(2:2:2*numPts, :) = [points1, ones(numPts,1), ...
    zeros(numPts,3), -p1x.*p2x, -p1y.*p2x, -p2x];
[~, ~, V] = svd(constraints, 0);
h = V(:, end);
T = reshape(h, [3,3]) / h(9);

%==========================================================================
function N = computeLoopNumber(sampleSize, confidence, pointNum, inlierNum)

pointNum = cast(pointNum, 'like', inlierNum);
inlierProbability = (inlierNum/pointNum)^sampleSize;

if inlierProbability < eps(class(inlierNum))
    N = intmax('int32');
else
    conf = cast(0.01, 'like', inlierNum) * confidence;
    one  = ones(1,    'like', inlierNum);
    num  = log10(one - conf);
    den  = log10(one - inlierProbability);
    N    = int32(ceil(num/den));
end  

%==========================================================================
function tform = computeTForm(sampleSize, points1, points2, indices, classToUse)

[samples1, normMatrix1] = ...
    vision.internal.normalizePoints(points1(indices, :)', 2, classToUse);
[samples2, normMatrix2] = ...
    vision.internal.normalizePoints(points2(indices, :)', 2, classToUse);

samples1 = samples1';
samples2 = samples2';

switch(sampleSize)
    case 2
        tform = computeSimilarity(samples1, samples2, classToUse);
    case 3
        tform = computeAffine(samples1, samples2, classToUse);
    otherwise % 4
        tform = computeProjective(samples1, samples2, classToUse);
        tform = tform / tform(end);
end
tform = normMatrix1' * (tform / normMatrix2');

%==========================================================================
function dis = evaluateTForm(sampleSize, threshold, tform, points1, points2)

pt = points1 * tform(1:2, 1:2);
pt = bsxfun(@plus, pt, tform(3,1:2));

if sampleSize == 4    
    denom = bsxfun(@plus, points1 * tform(1:2, 3), tform(3,3));
    
    tf = abs(denom) > eps(class(points1));
    
    pt(tf,:)  = bsxfun(@rdivide, pt(tf,:), denom(tf));   
    
    % Mark these points invalid by setting it to a location far away from
    % point2
    pt(~tf,:) = bsxfun(@plus, points2(~tf,:), threshold);
end

delta = pt - points2;
dis   = hypot(delta(:,1),delta(:,2));
dis(dis > threshold) = threshold;

%==========================================================================
function [isFound, tform, inliers] = msac(points1, points2, maxNumTrials, ...
        confidence, maxDistance, sampleSize, classToUse)

threshold = maxDistance;
numPts    = size(points1, 1);
idxTrial  = 1;
numTrials = int32(maxNumTrials);
maxDis    = cast(threshold * numPts, classToUse);
bestDis   = maxDis;
bestTForm = eye([3,3], classToUse);

% Create a random stream. It uses a fixed seed for the testing mode and a
% random seed for other mode.
if isTestingMode()
    rng('default');
end

points1 = cast(points1, classToUse);
points2 = cast(points2, classToUse);

while idxTrial <= numTrials
    indices = randperm(numPts, sampleSize);
    
    tform = computeTForm(sampleSize, points1, points2, indices, classToUse);    
    
    dis = evaluateTForm(sampleSize, threshold, tform, points1, points2);        
    
    accDis = sum(dis);
    
    if accDis < bestDis
        bestDis = accDis;
        bestTForm = tform;
        inlierNum = cast(sum(dis < threshold), classToUse);
        num = computeLoopNumber(sampleSize, confidence, numPts, inlierNum);
        numTrials = min(numTrials, num);
    end
    idxTrial = idxTrial + 1;
end

distances = evaluateTForm(sampleSize, threshold, bestTForm, points1, points2);
inliers = (distances < threshold);
isFound = (sum(inliers) >= sampleSize);

if isFound
    tform = computeTForm(sampleSize, points1, points2, inliers, classToUse);
    tform = tform / tform(3,3);
else
    tform = eye([3,3], classToUse);
end
