%%%%% RH/Ta/WS case from SFP dataset %%%

%Allison Goodwell
%April 2021
%use data from WRR TIPNet papers (Goodwell and Kumar 2017a,b)

%focus on information from Ta and WS to RH
%in several-hour segments, plot info components versus I(Ta;WS)

%This is the code to plot results

clear all
close all
clc

%load('Results082021_mainmodels') % load main results file
load('Results082021_SImodels') % load SI results file
opt = 1; %0 for 9 models in paper, 1 for 6 models in SI

if opt ==1
    RH_mod = RH_modbest; %SI analysis of additional machine learning models
    nmodels = 6;
    modnames ={'Lin Ta, WS', 'Lin Int', 'TreeBoost','GPR','Lin SVM','Gauss SVM'};
    cvect = jet(nmodels);
    %want to make the yellow a little darker:
    cvect(5,:) = [.9 .9 0];
    cvect(1,:) = [0 0 0]; %regular linear regression - show as black
else
    nmodels = 9; %regular analysis of 9 models
    modnames = {'1: Ta','2: WS', '3: Ta*WS ', '4: Ta,WS', '5: int','6: tree','7: Ta/WS','8: Ta+WS','9: Ta-WS',};
    cvect = jet(nmodels);
    %want to make the yellow a little darker:
    cvect(7,:) = [.9 .9 0];
    cvect(4,:) = [0 0 0]; %regular linear regression - show as black
end

%%
close all

figure(1)

hold on


n = 60*12;

hrvect = (1:n)./60;


for m =1:nmodels  
plot(hrvect,RH_mod{1,m}(1:n),'Color',cvect(m,:),'Linewidth',1.5)
end

figure(1)
plot(hrvect,RH2(1:n),'--k','Linewidth',2)

legend([modnames 'Obs'],'Location','EastOutside')
xlabel('Hour')
ylabel('RH')
%%
figure(2)

ct =1;
ha = tight_subplot(6,nmodels,[.01 .01],[.08 .05],[.08 .01]);

for m = 1:nmodels
    
%subplot(6,nmodels,ct)
axes(ha(ct))
plot(Ixy,1-MInorm_RHmodRH(:,m),'.r')
ylim([.5 1])
xlim([0 .75])
hold on
set(gca,'Xticklabel',[])
if m>1
set(gca,'Yticklabel',[])
else
    ylabel('A_P')
end
title(modnames{m})
    
%subplot(6,nmodels,ct+nmodels)
axes(ha(ct+nmodels))
plot(Ixy,Itot./Hz,'.k')
hold on
plot(Ixy,Itotmod(:,m)/Hzmod(:,m),'.r')
ylim([0 1])
xlim([0 .75])
set(gca,'Xticklabel',[])
if m>1
set(gca,'Yticklabel',[])
else
ylabel('I_{tot}')
end
    
%subplot(6,nmodels,ct+nmodels*2)
axes(ha(ct+nmodels*2))
plot(Ixy,S./Itot,'.k')
hold on
plot(Ixy,Smod(:,m)./Itotmod(:,m),'.r')
ylim([0 1])
xlim([0 .75])
set(gca,'Xticklabel',[])
if m>1
set(gca,'Yticklabel',[])
else
ylabel('S fraction')
end

%subplot(6,nmodels,ct+nmodels*3)
axes(ha(ct+nmodels*3))
plot(Ixy,R./Itot,'.k')
hold on
plot(Ixy,Rmod(:,m)./Itotmod(:,m),'.r')
ylim([0 1])
xlim([0 .75])
set(gca,'Xticklabel',[])
if m>1
set(gca,'Yticklabel',[])
else
ylabel('R fraction')
end

%subplot(6,nmodels,ct+nmodels*4)
axes(ha(ct+nmodels*4))
plot(Ixy,UTa./Itot,'.k')
hold on
plot(Ixy,UTamod(:,m)./Itotmod(:,m),'.r')
ylim([0 1])
xlim([0 .75])
set(gca,'Xticklabel',[])
if m>1
set(gca,'Yticklabel',[])
else
ylabel('U_{Ta} fraction')
end

%subplot(6,nmodels,ct+nmodels*5)
axes(ha(ct+nmodels*5))
plot(Ixy,UWS./Itot,'.k')
hold on
plot(Ixy,UWSmod(:,m)./Itotmod(:,m),'.r')
ylim([0 1])
xlim([0 .75])
if m>1
set(gca,'Yticklabel',[])
else
ylabel('U_{WS} fraction')
end
xlabel('I(Ta;WS)')

ct=ct+1;
end


%plot model predictive performance vs measures of functional performance
%for different source correlation categories


Ixyvals = prctile(Ixy,0:25:100); %categories of Ixy values
msize = 10:10:50;

for c=1:1 %c=1:3 to show plots for night and day trained models also...

for i =1:(size(Ixyvals,2)-1)
    
    inds = find(Ixy>= Ixyvals(i) & Ixy <Ixyvals(i+1));
    MInorm_RHmodRH_vect =  MInorm_RHmodRH(inds,:);
    Itot_vect = Itot(inds);
    Itotmod_vect = Itotmod(inds,:);
    S_vect = S(inds);
    R_vect = R(inds);
    UTa_vect = UTa(inds);
    UWS_vect = UWS(inds);
    Smod_vect = Smod(inds,:);
    Rmod_vect = Rmod(inds,:);
    UTamod_vect = UTamod(inds,:);
    UWSmod_vect = UWSmod(inds,:);
    
    
    for m =1:nmodels
        
        mean_predictive(m,i) = 1-mean(MInorm_RHmodRH_vect(:,m));
        mean_Itot(m,i) = mean((Itotmod_vect(:,m)-Itot_vect')./Itot_vect');
        
        %relative differences between normalized information components
        mean_S(m,i) = mean((-S_vect./Itot_vect)' + Smod_vect(:,m)./Itotmod_vect(:,m),'omitnan');
        mean_R(m,i) = mean((-R_vect./Itot_vect)' + Rmod_vect(:,m)./Itotmod_vect(:,m),'omitnan');
        mean_Uta(m,i) = mean((-UTa_vect./Itot_vect)' + UTamod_vect(:,m)./Itotmod_vect(:,m),'omitnan');
        mean_Uws(m,i) = mean((-UWS_vect./Itot_vect)' + UWSmod_vect(:,m)./Itotmod_vect(:,m),'omitnan');
        
        %abs value sum of all info components
        overall_funct =  [abs(mean_S(m,i)), abs(mean_R(m,i)), abs(mean_Uta(m,i)), abs(mean_Uws(m,i))];
        overall_weights = [mean(S_vect./Itot_vect), mean(R_vect./Itot_vect), mean(UTa_vect./Itot_vect), mean(UWS_vect./Itot_vect)];
        %overall_functional(m,i) = 1 - sum(overall_funct .* overall_weights);
        overall_functional(m,i) = sum(overall_funct)./2;
        
        figure(3)
        subplot(1,2,2) %total information from Ta, RH to RH vs RHmod
        plot(mean_Itot(m,i),mean_predictive(m,i),'Marker','.','Color',cvect(m,:),'MarkerSize',msize(i));
        hold on
        xlabel('A_{f,Itot}: I_{tot, mod}-I_{tot}')
        ylabel('A_p')
        set(gca,'Yticklabel',[])
        
        subplot(1,2,1)
        hh(m)=plot(overall_functional(m,i), mean_predictive(m,i),'Marker','.','Color',cvect(m,:),'MarkerSize',msize(i));
        hold on
        xlabel('A_{f,p}')
        ylabel('A_p')
        ylim([.75 1])
        xlim([0 1])
        
        figure(4)
        subplot(1,4,1)
        h2(m)=plot(mean_S(m,i),mean_predictive(m,i),'Marker','.','Color',cvect(m,:),'MarkerSize',msize(i));
        hold on
        xlim([-.4 .4])
        line([0 0],[.75 1])
        xlabel('A_{f,S}: S_{mod}-S')
        
        subplot(1,4,2)
        plot(mean_R(m,i),mean_predictive(m,i),'Marker','.','Color',cvect(m,:),'MarkerSize',msize(i))
        hold on
        xlim([-.2 .2])
        line([0 0],[.75 1])
        xlabel('A_{f,R}: R_{mod}-R')
        set(gca,'Yticklabel',[])
        
        subplot(1,4,3)
        plot(mean_Uta(m,i),mean_predictive(m,i),'Marker','.','Color',cvect(m,:),'MarkerSize',msize(i))
        hold on
        xlim([-.8 .8])
        line([0 0],[.75 1])
        xlabel('A_{f,U_{Ta}}: U_{Ta,mod}-U_{Ta}')
        set(gca,'Yticklabel',[])
        
        subplot(1,4,4)
        plot(mean_Uws(m,i),mean_predictive(m,i),'Marker','.','Color',cvect(m,:),'MarkerSize',msize(i))
        hold on
        xlim([-.8 .8])
        line([0 0],[.75 1])
        xlabel('A_{f,U_{WS}}: U_{WS,mod}-U_{WS}')
        set(gca,'Yticklabel',[])
       
       
    end
end

for m=1:nmodels
    
    figure(3)
    subplot(1,2,2) %total information from Ta, RH to RH vs RHmod
    plot(mean_Itot(m,:),mean_predictive(m,:),'Color',cvect(m,:));
    hold on
    
    subplot(1,2,1)
    plot(overall_functional(m,:), mean_predictive(m,:),'Color',cvect(m,:))
    hold on

    figure(4)
    subplot(1,4,1)
    plot(mean_S(m,:),mean_predictive(m,:),'Color',cvect(m,:))
    hold on

    subplot(1,4,2)
    plot(mean_R(m,:),mean_predictive(m,:),'Color',cvect(m,:))
    hold on

    
    subplot(1,4,3)
    plot(mean_Uta(m,:),mean_predictive(m,:),'Color',cvect(m,:))
    hold on

    
    subplot(1,4,4)
    plot(mean_Uws(m,:),mean_predictive(m,:),'Color',cvect(m,:))
    hold on

    

end

end


legend(hh,modnames)

legend(h2,modnames)
