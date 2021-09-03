%%%%% RH/Ta/WS case from SFP weather station dataset %%%

% Allison Goodwell
% April 2021
% use data from WRR TIPNet papers (Goodwell and Kumar 2017a,b)

% focus on information from Ta and WS to RH
% in 5-day moving windows, plot info components versus I(Ta;WS)

clear all
close all
clc

filenamesave = 'Results082021_SImodels';
load('modcollect')

opt = 1; %0 for 9 models in paper, 1 for 6 models in SI

if opt ==1
    RH_mod = RH_modbest; %SI analysis of additional machine learning models
    nmodels = 6;
else
    nmodels = 9; %regular analysis of 9 models
end

%need to normalize RH_mod to lie between 0 and 1 (like obs RH time-series)
for m =1:nmodels
  RHmodvect = RH_mod{m};
  maxval = max(RHmodvect);
  minval = min(RHmodvect);
  RHmod_norm = (RHmodvect-minval)./(maxval-minval);
  RH_mod{m}=RHmod_norm; %re-define model as normalized values  
end


%%

disp('doing IT calcs for windows...')

%change to a moving window of ndays
ndays = 5;
seglen = 60*24*ndays; 
nsegs = floor(length(Ta)/(60*24))-ndays;

start_ind = 1;
end_ind = seglen;


ctsig = 0;
ctnonsig=0;
for i = 1:nsegs
    fprintf('window %d of %d \n',i,nsegs)
    Ta_seg = Ta(start_ind:end_ind);
    RH_seg = RH(start_ind:end_ind);
    WS_seg = WS(start_ind:end_ind);
    
    pdfseg = compute_pdf([Ta_seg WS_seg RH_seg],15);
    infoseg{i} = compute_info_measures(pdfseg);
    
    
    %statistical significance testing for total info
    for n = 1:100
        Ta_shuff = randsample(Ta_seg,length(Ta_seg));
        WS_shuff = randsample(WS_seg,length(WS_seg));
        
        pdfshuff = compute_pdf([Ta_shuff WS_shuff RH_seg],15);
        infoshuff = compute_info_measures(pdfshuff);
        Ishuff(n) = infoshuff.Itot;
        Icond1shuff(n) = infoshuff.I_x1ycond;
        Icond2shuff(n) = infoshuff.I_x2ycond;
        
    end
    Icrit = mean(Ishuff)+3.*std(Ishuff);
    Icrit1 = mean(Icond1shuff)+2.*std(Icond1shuff);
    Icrit2 = mean(Icond2shuff)+2.*std(Icond2shuff);
    
    if infoseg{i}.Itot < Icrit %nothing is stat sig
       R(i) =0;
       S(i) =0;
       UTa(i) =0;
       UWS(i) =0;
       Itot(i)=0;
       ctnonsig = ctnonsig +1;
    elseif infoseg{i}.I_x1ycond < Icrit1 %still need to do stat sig testing for conditional info, just weaker source
        %for this case, first source provides no U, and no S
        S(i) = 0;
        UTa(i) =0;
        R(i) = infoseg{i}.I_x1y;
        UWS(i) = infoseg{i}.I_x2y - R(i);
        Itot(i) = infoseg{i}.I_x2y;
     
    elseif infoseg{i}.I_x2ycond < Icrit2
        S(i) = 0;
        UWS(i) =0;
        R(i) = infoseg{i}.I_x2y;
        UTa(i) = infoseg{i}.I_x1y - R(i);
        Itot(i) = infoseg{i}.I_x1y;
        
    else %all info segments exist
        
        R(i) = infoseg{i}.R;
        S(i) = infoseg{i}.S;
        UTa(i) = infoseg{i}.U1;
        UWS(i) = infoseg{i}.U2;
        Itot(i) = infoseg{i}.Itot;
         
        ctsig = ctsig +1;
    end
     
   
    Ixy(i) = infoseg{i}.I_x1x2; %source dependency I(Ta,WS) for window
    Hz(i) = infoseg{i}.Hy;
    

        for m =1:nmodels %go through all model cases for modeled SUR, info

            RH_segmod = RH_mod{m}(start_ind:end_ind);

            %compute RMSE of RH for this segment
            sq_error = (RH_segmod - RH_seg).^2;
            RMSE_test(i,m) = sqrt(mean(sq_error));

            %compute predictive performance I(RH;RH_mod)/H(RH);

            pdfRHmodRH = compute_pdf([RH_seg, RH_segmod],15);
            infoRHmodRH = compute_info_measures(pdfRHmodRH);

            MInorm_RHmodRH(i,m) = infoRHmodRH.I/infoRHmodRH.Hx1;


            %compute info partitioning and total info
            pdfseg = compute_pdf([Ta_seg WS_seg RH_segmod],15);
            infosegmod{i,m} = compute_info_measures(pdfseg);


            for n = 1:100 %stat sig testing
                Ta_shuff = randsample(Ta_seg,length(Ta_seg));
                WS_shuff = randsample(WS_seg,length(WS_seg));

                pdfshuff = compute_pdf([Ta_shuff WS_shuff RH_segmod],15);
                infoshuff = compute_info_measures(pdfshuff);
                Ishuff(n) = infoshuff.Itot;
                Icond1shuff(n) = infoshuff.I_x1ycond;
                Icond2shuff(n) = infoshuff.I_x2ycond;

            end
            Icrit = mean(Ishuff)+3.*std(Ishuff);
            Icrit1 = mean(Icond1shuff)+2.*std(Icond1shuff);
            Icrit2 = mean(Icond2shuff)+2.*std(Icond2shuff);

            if infosegmod{i,m}.Itot < Icrit
              %all info components are zero  
                Rmod(i,m) = 0;
                Smod(i,m) = 0;
                UTamod(i,m) = 0;
                UWSmod(i,m) = 0;
                Itotmod(i,m) = 0;

            elseif infosegmod{i,m}.I_x1ycond < Icrit1
                %no U from first source, no S
                Smod(i,m)=0;
                UTamod(i,m)=0;
                Rmod(i,m) = infosegmod{i,m}.I_x1y;
                UWSmod(i,m) = infosegmod{i,m}.I_x2y-Rmod(i,m);
                Itotmod(i,m) = infosegmod{i,m}.I_x2y;

            elseif infosegmod{i,m}.I_x2ycond < Icrit2
                %no U from second source, no S
                Smod(i,m)=0;
                UWSmod(i,m)=0;
                Rmod(i,m) = infosegmod{i,m}.I_x2y;
                UTamod(i,m) = infosegmod{i,m}.I_x1y-Rmod(i,m);
                Itotmod(i,m) = infosegmod{i,m}.I_x1y;

            else %all info components exist, compute

            Rmod(i,m) = infosegmod{i,m}.R;
            Smod(i,m) = infosegmod{i,m}.S;
            UTamod(i,m) = infosegmod{i,m}.U1;
            UWSmod(i,m) = infosegmod{i,m}.U2;
            Itotmod(i,m) = infosegmod{i,m}.Itot;

            end


            Hzmod(i,m) = infosegmod{i,m}.Hy;

        end
        
    start_ind = start_ind + 60*24; %move up window by a day
    end_ind = start_ind + seglen -1;
end


save(filenamesave)

disp('done, run Plot code to see figs')




%%



