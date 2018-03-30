u=1;
%create 19345x21 (rows x columns) matrix
f = zeros(19346,21);  %<<--
% r x c
%%%SRAD
% c=1: img nr beginning with 0
% c=2: max value of srad
% c=3: location of max value of srad
% c=4: mean of srad
% c=5: variance of srad
% c=6: argument of distance of C and mean
%%%SANG
% c=7: max value of sang
% c=8: location of max value of sang
% c=9: mean of sang
% c=10: variance sang
% c=11: argument of distance of C and mean
%%%WAVELET
% c=12: Ea
% c=13:15: Eh
% c=16:18: Ev
% c=19:21: Ed
fileID = fopen('FreqFeatures_Train.txt','w'); %<<--
fprintf(fileID,'Image_ID srad_max srad_maxLoc srad_mean srad_var srad_dist sang_max sang_maxLoc sang_mean sang_var sang_dist wave_1 wave_2 wave_3 wave_4 wave_5 wave_6 wave_7 wave_8 wave_9 wave_10\n');

for e=0:19345 %<<--
% u=1, increase in every iteration
        %Image number
        f(u,1)=e;
        num = num2str(e);
        %num = num2str(e,'%05d');
        name = sprintf('%s.png',num);
        I = imread(name);
        disp(name);

        % Spectral features (Fourier spectrum)
        %calculate FFT, srad, sang
        [srad, sang, S] = specxture(I);
        %imshow(S);
        %%%%%%%%%%%%%%%%%%%%%%%%%%srad
        %store indices of largest element in srad in I. C has the value
        [C,I] = max([srad]);
        %max
        f(u,2) = C;
        %max location
        f(u,3) = I;
        %mean
        m = mean([srad]);
        f(u,4) = m;
        %variance
        f(u,5) = var([srad]);
        %distance
        f(u,6) = abs((m-C));

        %%%%%%%%%%%%%%%%%%%%%%%%%%sang
        [C,I] = max([sang]);
        %max
        f(u,7) = C;
        %max location
        f(u,8) = I;
        %mean
        m = mean([sang]);
        f(u,9) = m;
        %variance
        f(u,10) = var([sang]);
        %distance
        f(u,11) = abs((m-C));

        % Wavelets features
        I_w = imread(name);
        [C,S] = wavedec2(I_w,3,'haar');
        [Ea,Eh,Ev,Ed] = WENERGY2(C,S);
        f(u,12)=Ea;
        f(u,13:15)=Eh;
        f(u,16:18)=Ev;
        f(u,19:21)=Ed;

        fprintf(fileID,'%d %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e\n',f(u,:));

        u=u+1;
end;
fclose(fileID);
%         writetable(f,'FreqFeatures_Train.csv','Delimiter',' ');
       % writetable(f,'FreqFeatures_Train.csv');
%         xlswrite('FreqFeatures_Train.xlsx',f);
xlswrite('FreqFeatures_Train.xls',f);
xlswrite('FreqFeatures_Train.csv',f);
