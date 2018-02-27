function [srad, sang, S] = specxture(f);
% define function
% specxture is name of function
% [srad, sang, S] are the return values of specxture function
% call with specxture(img)

%SPECXTURE Computes spectral texture of an image.
%   [SRAD, SANG, S] = SPECXTURE(F) computes SRAD, the spectral energy
%   distribution as a function of radius from the center of the
%   spectrum, SANG, the spectral energy distribution as a function of
%   angle for 0 to 180 degrees in increments of 1 degree, and S =
%   log(1 + spectrum of f), normalized to the range [0, 1]. The
%   maximum value of radius is min(M,N), where M and N are the number
%   of rows and columns of image (region) f. Thus, SRAD is a row
%   vector of length = (min(M, N)/2) - 1; and SANG is a row vector of
%   length 180.

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.7 $  $Date: 2003/11/21 14:48:47 $

% Obtain the centered spectrum, S, of f. The variables of S are 
% (u, v), running from 1:M and 1:N, with the center (zero frequency)
% at [M/2 + 1, N/2 + 1] (see Chapter 4). 

% doing fft2, matlab returns an array of the same size, each element in the resulting array is a complex number (a,b)
% shift switches the first and third quadrant, and the second and fourth
% TODO: kernel for fft shift
S = fftshift(fft2(f)); 
% magnitude, sqrt(re^2 + im^2) of all elements
% TODO: kernel for magnitude
S = abs(S);
% return a 1x2 matrix, with size of matrix
[M, N] = size(S); 
% store the middle of frequency domain
x0 = M/2 + 1;
y0 = N/2 + 1;

% Maximum radius that guarantees a circle centered at (x0, y0) that
% does not exceed the boundaries of S. 
% a radius around the middle, half of min dim, is integer
rmax = min(M, N)/2 - 1; 

% Compute srad.
% return 1 by rmax array (one row)
srad = zeros(1, rmax);
% store value S(x0,y0), i.e., zero frequency magnitude, in first array entry
srad(1) = S(x0, y0);
% traverse the the rest of srad, rmax times
% calculate 
for r = 2:rmax
   % in xc, which is a column array, are the cartesian x coordinates of a half circle stored, with radius r and center x0,y0. in yc the y coordinates. in one degree incremenents, from 91 to 270.	
   [xc, yc] = halfcircle(r, x0, y0);
   % sub2ind(arraySize, rowsub, colsub, ...)
   % sub2ind([M, N], xc, yc) creates linear index
   % https://www.ini.uzh.ch/~ppyk/BasicsOfInstrumentation/matlab_help/ref/sub2ind.html	
   % does 180 sub2inds 
   % take indices (xc,yc) = (row,col), and make them a linear index by calc. M*yc + xc
   % make 180 such linear indices. now 180 indices present
   % get S([ ..180 linear indices...]) and sum them together, store in srad(r)	
   % sum on circular arc	 
   srad(r) = sum(S(sub2ind(size(S), xc, yc)));
end

% Compute sang.
% radius is rmax. get cart. coord. of half circle only for radius rmax
[xc, yc] = halfcircle(rmax, x0, y0);
% create row array with length xc, 180 elements 
sang = zeros(1, length(xc));
% iterate 180 times
for a = 1:length(xc)
   % coordinates for line segment. from (x0,y0) to (xc(a), yc(a)) 	
   [xr, yr] = radial(x0, y0, xc(a), yc(a));
   % coordinates of line segment, convert to index in S, and sum all those values
   % sum on straight line segment
   sang(a) = sum(S(sub2ind(size(S), xr, yr)));
end

% Output the log of the spectrum for easier viewing, scaled to the
% range [0, 1].
S = mat2gray(log(1 + S));
    
%-------------------------------------------------------------------%
function [xc, yc] = halfcircle(r, x0, y0)
%   Computes the integer coordinates of a half circle of radius r and
%   center at (x0,y0) using one degree increments. 
%
%   Goes from 91 to 270 because we want the half circle to be in the
%   region defined by top right and top left quadrants, in the
%   standard image coordinates. 

% create vector with numbers [91, 92, 93, ..., 270]
theta=91:270;
% mult every element in vector by pi/180, in rad
theta = theta*pi/180;
% polar coordinates (angle, radius) to (x,y)
[xc, yc] = pol2cart(theta, r);
% round each element of xc to nearest integer, then transpose and add offset
xc = round(xc)' + x0; % Column vector.
yc = round(yc)' + y0;

%-------------------------------------------------------------------%
function [xr, yr] = radial(x0, y0, x, y);
%   Computes the coordinates of a straight line segment extending
%   from (x0, y0) to (x, y). 
%
%   Based on function intline.m.  xr and yr are returned as column
%   vectors.  

[xr, yr] = intline(x0, x, y0, y);
