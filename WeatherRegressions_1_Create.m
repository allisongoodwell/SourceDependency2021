
% Create models from 1-minute weather station data
% This code runs models for the main text (9 models) and the SI (4
% additional machine learning models)

% saved as modcollect.mat and modcollect_bestonly.mat
% to be loaded in subsequent file

load SFP2_AllData

disp('preprocessing')

Ta = OrigData.data_network(:,4);
WS = OrigData.data_network(:,2);
RH = OrigData.data_network(:,1);
Rg = OrigData.data_network(:,5);
Yr = OrigData.Year + OrigData.data_raw_DOY./365;

ind1 = find(isnan(Ta) | isnan(WS) | isnan(RH));
Ta(ind1)=[];
WS(ind1)=[];
RH(ind1)=[];
Yr(ind1)=[];
Rg(ind1)=[];

%need to filter diurnal cycle from Ta and RH....
[Tafilt,~,~] = ButterFiltFun('hp',Ta,'lambdac',60*6);
[RHfilt,~,~] = ButterFiltFun('hp',RH,'lambdac',60*6);

%get rid of extreme outliers in data
ind1 = find(Tafilt> prctile(Tafilt,99.95) | Tafilt < prctile(Tafilt,0.05));
ind2 = find(WS > 15 | WS< 0);
ind3 = find(RHfilt> prctile(RHfilt,99.95) | RHfilt < prctile(RHfilt,0.05));

inds = [ind1' ind2' ind3'];
inds = unique(inds);
Tafilt(inds)=[];
WS(inds)=[];
RHfilt(inds)=[];
Yr(inds)=[];
Rg(inds)=[];

Ta2 = (Tafilt-min(Tafilt))./range(Tafilt);
WS2 = (WS-min(WS))./range(WS);
RH2 = (RHfilt-min(RHfilt))./range(RHfilt);

Ta = Ta2;
RH = RH2;
WS = WS2;


% save some data that will be used to train the models...can uncomment to
% generate new training datasets
% for i =1:3
% [Tatrain, idx]=datasample(Ta,50000);
% WStrain = WS(idx);
% RHtrain = RH(idx);
% 
% traindata{i} = [Tatrain WStrain RHtrain];
% trainvarnames{i} = {'Ta','WS','RH'};
% end
% 
% save('traindata.mat','traindata','trainvarnames')

%can also define training data sets based day and night
% Ta_night = Ta(Rg==0);
% Ta_day = Ta(Rg>10);
% WS_night = WS(Rg==0);
% WS_day = WS(Rg>10);
% RH_night = RH(Rg==0);
% RH_day = RH(Rg>10);
% 
% [Tatrain_night, idx]=datasample(Ta_night,50000);
% WStrain_night = WS_night(idx);
% RHtrain_night = RH_night(idx);
% 
% [Tatrain_day, idx]=datasample(Ta_day,50000);
% WStrain_day = WS_day(idx);
% RHtrain_day = RH_day(idx);
% 
% traindata_night = [Tatrain_night WStrain_night RHtrain_night];
% traindata_day = [Tatrain_day WStrain_day RHtrain_day];


figure(1)
subplot(3,1,1)
plot(Yr,Ta)
title('normalized, filtered air temperature (Ta)')
set(gca,'Xticklabel',[])
xlim([min(Yr) max(Yr)])

subplot(3,1,2)
plot(Yr,WS)
title('normalized wind speed (WS)')
set(gca,'Xticklabel',[])
xlim([min(Yr) max(Yr)])

subplot(3,1,3)
plot(Yr, RH)
set(gca,'Xticklabel',[])
xlim([min(Yr) max(Yr)])
title('normalized, filtered relative humidity (RH)')


%% train models

load traindata

traindat = traindata{1}; %[Ta, WS, RH] training data

%models generated from regressionLearner in Matlab - 
%here, just re-training them
disp('basic models training...')
modcollect{1} = trainRegressionModelLin1source([traindat(:,1) traindat(:,3)]); %Ta only
modcollect{2}= trainRegressionModelLin1source([traindat(:,2) traindat(:,3)]); %WS only
modcollect{3} = trainRegressionModelLin1source([traindat(:,1).*traindat(:,2) traindat(:,3)]); %Ta*WS
modcollect{4} = trainRegressionModelLin2source(traindat); %Ta and WS linear
modcollect{5} = trainRegressionModelLinInt(traindat); %linear with interaction term
modcollect{6} = trainRegressionModelMedTree(traindat); %tree model

divtrain = traindat(:,1)./traindat(:,2); %get rid of extreme high values (inf) due to division
divtrain(divtrain>100)=100;
modcollect{7} = trainRegressionModelLin1source([divtrain traindat(:,3)]); %division model
modcollect{8} = trainRegressionModelLin1source([traindat(:,1)+traindat(:,2) traindat(:,3)]); %addition
modcollect{9} = trainRegressionModelLin1source([traindat(:,1)-traindat(:,2) traindat(:,3)]); %subtraction

traindat = traindat(1:10000,:); %shorten for faster training of models for SI

disp('machine learning models training...')
modcollect_bestonly{1} = trainRegressionModelLin2source(traindat); %Ta and WS linear
modcollect_bestonly{2} = trainRegressionModelLinInt(traindat);
modcollect_bestonly{3} = trainRegressionModelTreeBoost(traindat);
modcollect_bestonly{4} = trainRegressionModelGPR(traindat);
modcollect_bestonly{5} = trainRegressionModelLinSVM(traindat);
modcollect_bestonly{6} = trainRegressionModelGaussMedSVM(traindat);

nmodelsbest = 6;
nmodels = 9;
%%

disp('now applying models')



%save RMSE for all training datasets for each model

traindat = traindata{1};
  
rhobs = traindat(:,3);
    
rhmod{1} = modcollect{1}.predictFcn([traindat(:,1)]); %Ta only
rhmod{2} = modcollect{2}.predictFcn([traindat(:,2)]);
rhmod{3} = modcollect{3}.predictFcn([traindat(:,1).*traindat(:,2)]);
rhmod{4} = modcollect{4}.predictFcn([traindat(:,1), traindat(:,2)]);
rhmod{5} = modcollect{5}.predictFcn([traindat(:,1), traindat(:,2)]);
rhmod{6} = modcollect{6}.predictFcn([traindat(:,1), traindat(:,2)]);

divtrain = traindat(:,1)./traindat(:,2);
divtrain(divtrain>100)=100;
rhmod{7} = modcollect{7}.predictFcn([divtrain]);
rhmod{8} = modcollect{8}.predictFcn([traindat(:,1)+traindat(:,2)]);
rhmod{9} = modcollect{9}.predictFcn([traindat(:,1)-traindat(:,2)]);

traindat = traindata{1};
rhobs = traindat(:,3);

rhmod_best{1} = modcollect_bestonly{1}.predictFcn([traindat(:,1), traindat(:,2)]);
rhmod_best{2} = modcollect_bestonly{2}.predictFcn([traindat(:,1), traindat(:,2)]);
rhmod_best{3} = modcollect_bestonly{3}.predictFcn([traindat(:,1), traindat(:,2)]);
rhmod_best{4} = modcollect_bestonly{4}.predictFcn([traindat(:,1), traindat(:,2)]);
rhmod_best{5} = modcollect_bestonly{5}.predictFcn([traindat(:,1), traindat(:,2)]);
rhmod_best{6} = modcollect_bestonly{6}.predictFcn([traindat(:,1), traindat(:,2)]);

  
for m =1:nmodels
    sq_errs = (rhobs - rhmod{m}).^2;
    RMSE_train(m) = sqrt(mean(sq_errs)); 
end

for m =1:nmodelsbest
    sq_errs = (rhobs - rhmod_best{m}).^2;
    RMSE_train_best(m) = sqrt(mean(sq_errs)); 
end
    

disp('applying basic models to whole dataset')


RH_mod{1} = modcollect{1}.predictFcn([Ta]);
RH_mod{2}= modcollect{2}.predictFcn([WS]);
RH_mod{3} = modcollect{3}.predictFcn([Ta.*WS]);
RH_mod{4} = modcollect{4}.predictFcn([Ta WS]);
RH_mod{5} = modcollect{5}.predictFcn([Ta WS]);
RH_mod{6}= modcollect{6}.predictFcn([Ta WS]);
divall = Ta./WS;
divall(divall>100)=100;
RH_mod{7} = modcollect{7}.predictFcn([divall]);
RH_mod{8} = modcollect{8}.predictFcn([Ta+WS]);
RH_mod{9} = modcollect{9}.predictFcn([Ta-WS]);


disp('applying machine learning models to whole dataset')
RH_modbest{1} = modcollect_bestonly{1}.predictFcn([Ta WS]);
RH_modbest{2} = modcollect_bestonly{2}.predictFcn([Ta WS]);
RH_modbest{3} = modcollect_bestonly{3}.predictFcn([Ta WS]);
RH_modbest{4} = modcollect_bestonly{4}.predictFcn([Ta WS]);
RH_modbest{5} = modcollect_bestonly{5}.predictFcn([Ta WS]);
RH_modbest{6} = modcollect_bestonly{6}.predictFcn([Ta WS]);

save('modcollect')

disp('done, run IT Analysis code for info measures')

