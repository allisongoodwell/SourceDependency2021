%%%%%%%%%%%%%%%%
% Allison Goodwell, 2021
% generated data models, this code reproduces Figure 3 in manuscript
% goal: for model Z = f(X,Y), with some different configs (addition,
% multiply, proportion)
% see how SUR quantities vary with source dependencies
%%%%%%%%%%%%%%%%

clear all
close all
clc

step = .1; %step size for a (for weighted average example)
ct=1;

avals = 0:step:1;
ct_tot = length(avals);
N_data=20000; %total values to generate
ndata = 10000; %size of dataset to get pdfs - must be smaller than N_data

%%
noise_vals = 0:.01:1; %noise level in X and Y relationship
a_mean =0:.1:1;  %mean of a value (weighting)

ncases = length(noise_vals)*length(a_mean);
%%

X = rand(N_data,1).*10;
Y = rand(N_data,1).*10;


ct=1;

for n = 1:length(noise_vals)
    
    randvect = rand(length(X),1);
    randvect(randvect<noise_vals(n))=0; %leave value of Y alone
    randvect(randvect>0)=1; %indices of Y values to be changed to match X
    X_replace=X;
    new_Yvalues=Y;
    X_replace(randvect==0)=0;
    new_Yvalues(randvect==1)=0;
    new_Yvalues = new_Yvalues + X_replace;
    
    Ynew = new_Yvalues;
    
    Xnew = X;
    
    %normalize sources to 0-1 values
    Xnew = (Xnew-min(Xnew))./(range(Xnew));
    Ynew = (Ynew-min(Ynew))./(range(Ynew));
    
    
    %now determine values of a for weighted average, to obtain Z value
    for a = 1:length(a_mean)
        
        %ready to compute Z, resulting pdf, and information measures
        Z1 = a_mean(a).*Xnew + (1-a_mean(a)).*Ynew;
        
        %normalize Z1 to 0-1 range for pdf
        Z1 = (Z1-min(Z1))./(range(Z1));
        
        pdf_3d = compute_pdf([Xnew Ynew Z1],15);
        infoXY_add{n,a} = compute_info_measures(pdf_3d);
        data_add_save{n,a}=[Xnew Ynew Z1];
        
        if a ==1 %multiplication and division tests (no weighting here)
            Z2 = Xnew.*(Ynew);
            
            %normalize Z2 to 0-1 range
            Z2 = (Z2-min(Z2))./range(Z2);
            pdf_3d = compute_pdf([Xnew Ynew Z2],15);
            infoXY_mult{n} = compute_info_measures(pdf_3d);
            data_mult_save{n}=[Xnew Ynew Z2];
            
            
            Z3 = Xnew./(Xnew+Ynew);
            Z3 = (Z3-min(Z3))./range(Z3);
            pdf_3d = compute_pdf([Xnew Ynew Z3],15);
            infoXY_div{n} = compute_info_measures(pdf_3d);
            data_div_save{n}=[Xnew Ynew Z3];
        end
    end
end



%% now, we can plot different slices of info values (S, R, U1, U2)
%1) for constant a value, no correlation between X and a: relationship
%between XY dependency and S,U,R for different a values, for 2 slopes

figure(1)

var_a_ind = 1; %a is a constant
xa_ind = 1;    %no correlation between X and a
cvect = jet(length(a_mean));

infostruct = infoXY_add;
for a = 1:length(a_mean)
        
        ct=1;        
            for n = 1:length(noise_vals)
                
                
                info_val = infostruct{n,a};
                
                Itot(ct) = info_val.Itot;
                Ixy(ct) = info_val.I_x1x2;
                S(ct) = info_val.S;
                Ux(ct) = info_val.U1;
                Uy(ct)= info_val.U2;
                R(ct)= info_val.R;
                ct=ct+1;
            end
               
        normval=Itot;
        %normval=1;
        
        subplot(5,2,1)
        title('Z = aX + (1-a)Y')
        hh(a) = plot(Ixy,Itot,'.','Color',cvect(a,:));
        hold on
        ylim([0 4])
        ylabel('I(X,Y;Z) (bits)')
        set(gca,'Xticklabels',[])
        
        
        subplot(5,2,3)
        hold on
        plot(Ixy,S./normval,'.','Color',cvect(a,:))
        ylim([0 1])
        ylabel('S (bits/bit)')

        set(gca,'Xticklabels',[])
        

        subplot(5,2,5)
        hold on
        plot(Ixy,R./normval,'.','Color',cvect(a,:))
        ylim([0 1])
        set(gca,'Xticklabels',[])
        ylabel('R (bits/bit)')
        
   
        subplot(5,2,7)
        hold on
        plot(Ixy,Ux./normval,'.','Color',cvect(a,:))
        ylim([0 1])
        set(gca,'Xticklabels',[])
        ylabel('U_x (bits/bit)')
               
        subplot(5,2,9)
        hold on
        plot(Ixy,Uy./normval,'.','Color',cvect(a,:))
        ylim([0 1])
        xlabel('I(X;Y)')
        ylabel('U_y (bits/bit)')
        
        
end
    
subplot(5,2,1)
legend(hh,{'a=0','a=0.1','a=0.2','a=0.3','a=0.4','a=0.5',...
    'a=0.6','a=0.7','a=0.8','a=0.9','a=1.0'})

%%

cvect = [1 0 0; 0 0 1];
for c = 1:2
    

    if c==1
        infostruct = infoXY_mult;
        ct2=0;
    else
        infostruct = infoXY_div;
        ct2=4;
    end
    
        ct=1;


            for n = 1:length(noise_vals)
                
                
                info_val = infostruct{n};
                
                Itot(ct) = info_val.Itot;
                Ixy(ct) = info_val.I_x1x2;
                S(ct) = info_val.S;
                Ux(ct) = info_val.U1;
                Uy(ct)= info_val.U2;
                R(ct)= info_val.R;
                Rmax(ct) = info_val.Rmax;
                Rmin(ct) = info_val.Rmin;
                ct=ct+1;
            end

        
        normval=Itot;

        subplot(5,2,2)
        hold on
        plot(Ixy,Itot,'.','Color',cvect(c,:))
        title('Z = XY, Z = X/(X+Y)')

        set(gca,'Xticklabels',[])
        set(gca,'Yticklabels',[])
        
        subplot(5,2,4)
        hold on
        hh(c) = plot(Ixy,S./normval,'.','Color',cvect(c,:));
        ylim([0 1])
        set(gca,'Xticklabels',[])
        set(gca,'Yticklabels',[])
  
        subplot(5,2,6)
        hold on
        plot(Ixy,R./normval,'.','Color',cvect(c,:))
        plot(Ixy,Rmin./normval,'--','Color',cvect(c,:))
        plot(Ixy,Rmax./normval,'--','Color',cvect(c,:))
        alpha(.25)
        ylim([0 1])

        set(gca,'Xticklabels',[])
        set(gca,'Yticklabels',[])
 
        subplot(5,2,8)
        hold on
        plot(Ixy,Ux./normval,'.','Color',cvect(c,:))
        
        ylim([0 1])

        set(gca,'Xticklabels',[])
        set(gca,'Yticklabels',[])
               
        subplot(5,2,10)
        hold on
        plot(Ixy,Uy./normval,'.','Color',cvect(c,:))
        ylim([0 1])
        xlabel('I(X;Y)')

        set(gca,'Yticklabels',[])
             
end

subplot(5,2,2)
legend(hh,{'Mult','Div'})
  
