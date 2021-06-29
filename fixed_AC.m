
clc;    % Clear the command window.
clear all;
close all;
clearvars;
workspace;  % Make sure the workspace panel is showing.
format long g;
format compact;
fontSize = 20;

P_masks=dir('MASK');
I_pic=dir('PET');
b_box=dir('test_bbox');
Dice=[];
Dice1 = []; 
Dice2 = []; 
MeanInt=[];
MinInt=[];
MaxInt=[];
ModeInt=[];
BSlist=[];
JaccardScores = []; 
SizeV = []; 
Hausdorff = []; 
Name = [];
Structuralsimilarity = [];
ForelistS=[];
BacklistS=[];
Forelist=[];
Backlist=[];

%active contours
epsilon=0.01;
num_it = 20;
rad=7.5; %2,9, 29 and 19 are also OK.
alpha=0.01;
Dil=5;%dilate the ITM mask for AC
Thresh=0.215;

leng=length(b_box);

for i = 4:leng
%get the folder name   
B=b_box(i).name;
FileB=load(fullfile('test_bbox', B));
underlineLocationB = strfind(B, '_');
patient=B(1:underlineLocationB-1);
slice=B(underlineLocationB+1:end-4);
slice=str2num(slice);
% slice=double(slice);
maskfile= append(char(patient), '_MASK.nii');
petfile= append(char(patient), '_PET.nii');
%importing files from folders
FileP=niftiread(fullfile('MASK', maskfile));
FileI=niftiread(fullfile('PET', petfile));

%Indiviual slices
Pfile = FileP(:,:,slice);
grayPET = FileI(:,:,slice);

% grayPET= flipdim(grayPET ,2);
% grayPET = imtranslate(grayPET,[6, 0]);

factor = size(grayPET);
fx=factor(1);
fy=factor(2);

centerx = FileB(1,2)*fx;
centery = FileB(1,3)*fy;
wid=FileB(1,4)*fx;
hei=FileB(1,5)*fy;
% centerx = FileB(3)*fx;
% centery = FileB(2)*fy;
% wid=FileB(5)*fx;
% hei=FileB(4)*fy;

x1=centerx-wid/2;
y1=centery-hei/2;

s1=[x1 y1 wid hei]; %size of cropping
s=s1;
% s=[x1+5 y1 wid hei]; %size of cropping
% s=[centerx centery wid hei]; 

grayImage=im2double(grayPET);
grayImage = imcrop(grayImage,s);
normImage = mat2gray(grayImage);
TRY=im2uint8(normImage);
%filtering and enhancement 
grayPET=imgaussfilt(TRY,1.5); 
%remove background
se = strel('disk',5);
background = imopen(TRY,se);
grayPET=imsubtract (TRY,background);
grayImage=grayPET;%TRY;

grayPImage = imcrop(Pfile,s1);

% sim=dice(grayPImage1,grayPImage2);
% grayPImage=imrotate(grayPImage,180);

[pixelCount, grayLevels] = imhist(TRY);
Id = im2double(TRY);
Imax = max(Id(:));
Imean = mean(Id(:));
Imin = min(Id(:));
Imode = mode(Id(:));
BS = Imode/Imax; %background to source ratio

MeanInt=[MeanInt;Imean];
MinInt=[MinInt;Imin];
MaxInt=[MaxInt;Imax];
ModeInt=[ModeInt;Imode];
BSlist=[BSlist;BS];

%Training data for forground
T_f = 0.5*(min(grayImage(:)) + max(grayImage(:))); %%%%%%%%%%%%%%%%%
% T_f = (0.11+Imode)*(min(grayImage(:)) + max(grayImage(:))); %%%%%%%%%%%%%%%%%
g = grayImage >= T_f; %ITM lesion

%%%physcian mask
grayPImage =logical(grayPImage);
brightPixels = grayPImage >= 0.5;
nBP = sum(brightPixels(:));
volumeP = nBP*4.0728*4.0728*3*0.001;

pbi=grayPImage;%im2bw(grayPImage); %should be binary Physican mask as binary 

newMask = imdilate(g, true(Dil)); %dilate the ITM output!!!

mask_init  = zeros(size(grayImage));
mask_init(find(newMask)) = 1; %creation of inital contour
filter=medfilt2(grayImage);
filter=im2double(filter);
seg1 = local_AC_MS(filter,mask_init,rad,alpha,num_it,epsilon);
maxGrayLevel = max(grayImage(:));
[labeledImage, numberOfBlobs] = bwlabel(g);% find roughly how many lesions in image
j=1;
init=zeros(size(grayImage));
init(find(g))=1;
seg2=init;
d1 = bwdist(seg1) - bwdist(~seg1); % dt is the distance transform
d2 = bwdist(seg2) - bwdist(~seg2);   % ~ is the logical negation
seg = (d1+d2) > 0;   % output
seg = ~seg;

similarity =  dice(pbi,seg); %sum(andImage) / sum(orImage); returns scalar only if images are binary

Dice = [Dice; dice(pbi,seg)];
JaccardScores = [JaccardScores;jaccard(pbi,seg)]; %similarity./(2-similarity); returns scalar only if images are binary
Hausdorff = [Hausdorff;HausdorffDist(pbi,seg)];
Structuralsimilarity=[Structuralsimilarity;ssim(double(pbi),double(seg))];


% %declare figure
% f=figure;
% imshow(grayImage,[],'InitialMagnification', 2000);
% hold on
% %plot contours
% plot(x1,y1,'r','LineWidth', 2);
% hold on
% plot(x1,y1,'g','LineWidth', 2);
% hold on
% [B,L] = bwboundaries(seg,'noholes');
% [S,R] = bwboundaries(pbi,'noholes');
% for k = 1:length(B)
%    boundary_res = B{k};
%    plot(boundary_res(:,2), boundary_res(:,1), 'r', 'LineWidth', 2)
%    hold on;
% end
% for k = 1:length(S)
%    boundary_res = S{k};
%    plot(boundary_res(:,2), boundary_res(:,1), 'g', 'LineWidth', 2)
% end
% title(['Dice Index = ' num2str(similarity)])           
% legend('Hybrid Method','Physician Mask')
% saveas(f,justname,'jpeg')
dicerevised=Dice(Dice~=0);

end

DICE=mean(Dice)
DICErevised=mean(dicerevised)
JaccardScores = JaccardScores;
Hausdorff = Hausdorff;
Structuralsimilarity=Structuralsimilarity;



