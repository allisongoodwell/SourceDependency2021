%%%%%%%%%%%%%%%%
% Allison Goodwell, 2021
% goal: create a bunch of binary source distributions, and model mappings
% to a binary output, find SUR for each source/model combo
% this represents almost the full ``functional range'' for a given model
% This code reproduces Figure 2 in manuscript
%%%%%%%%%%%%%%%%

clear all
close all
clc

step = .025; %larger step size = coarser pdfs, faster

ct=1;
%create binary 2d pdfs....all combinations of 4 prob values, must sum to 1
for px0y0=0:step:1        
    for px1y0 = 0:step:(1-px0y0)            
        for px0y1=0:step:(1-px1y0-px0y0)                
            px1y1 = 1-px0y0-px1y0-px0y1;

            %list of pdf values
            matrix(:,ct)=[px0y0 px1y0 px0y1 px1y1];

            %determinant: ad-bc
            determinant(ct)=px0y0*px1y1-px1y0*px0y1;

            %prob of 0 vs 1
            px0(ct) = px0y0 + px0y1;
            py0(ct) = px1y0 + px0y0;

            ux = 1-px0(ct);
            uy = 1-py0(ct);

            %covariance
            covar(ct)=px0y0*(0-ux)*(0-uy) + px0y1*(0-ux)*(1-uy) +...
                px1y0*(1-ux)*(0-uy) + px1y1*(1-ux)*(1-uy);
            ct=ct+1;

        end           
    end
end

matrix(matrix<step/2)=0;

%create model maps: (x,y) pair could map to 1st or 2nd bin
ct=1;
z1=1;
for z2 = 1:2
    for z3 = 1:2
        for z4=1:2
            model(:,ct)=[z1 z2 z3 z4];
            ct=ct+1;
        end
    end
end


%% create 3d pdf p(x,y,z) and info measures
ct=1;
for f = 1:size(matrix,2)
    
    xy_pdf = matrix(:,f);
    
    for m = 1:size(model,2)
        
        map = model(:,m);
        
        pdf = zeros(2,2,2);
        
        %fill pdf with values according to model map
        pdf(1,1,map(1))=xy_pdf(1);
        pdf(2,1,map(2))=xy_pdf(2);
        pdf(1,2,map(3))=xy_pdf(3);
        pdf(2,2,map(4))=xy_pdf(4);
        
        map2d(:,:,m) = [map(1), map(3); map(2),map(4)]-1;
       
        %compute information measures for each pdf
        info = compute_info_measures(pdf);
        
        S(f,m)=info.S;
        U1(f,m)=info.U1;
        U2(f,m)=info.U2;
        R(f,m)=info.R;
        Itot(f,m)=info.Itot;
        Ixy(f,m)=info.I_x1x2;
        Ixz(f,m) = info.I_x1y;
        Iyz(f,m) = info.I_x2y;
        Hx(f,m)=info.Hx1;
        Hy(f,m)=info.Hx2;
        
        ct=ct+1;
        
        if mod(ct,10000)==0
            disp('progress:')
            disp(ct/(size(model,2)*size(matrix,2)))
        end
        
    end
end

%%
%plot Ixy versus source covariance
% figure(22)
% plot(Ixy(:,1),covar,'*b')
% xlabel('MI')
% ylabel('cov')

figure(2)
ct =1;
for m=2:size(model,2)
    subplot(1,7,ct)
    imagesc(map2d(:,:,m))
    xticks([1, 2])
    set(gca,'Xticklabels',[0,1])
    axis square
    yticks([1,2])
    if m==2
    set(gca,'Yticklabels',[0,1])
    else
    set(gca,'Yticklabels',[])
    end
    ct=ct+1;
end

%%
%plot S-U-R versus source dependency for different binary models
%skip first model (trivial case)

modnames = {'B1','B2','B3','B4','B5','B6','B7'};

figure(1)
ct=1;
graycol = [.7 .7 .7];

for m=2:size(model,2)
    
MIvect = Ixy(:,m);
Svect = S(:,m)./Itot(:,m);
Rvect = R(:,m)./Itot(:,m);
U1vect = U1(:,m)./Itot(:,m);
U2vect = U2(:,m)./Itot(:,m);
H1vect = Hx(:,m);
Hdiffvect = Hx(:,m)-Hy(:,m);
inds = 1:length(MIvect);

subplot(5,7,ct)
hold on
plot(MIvect(inds),Itot(:,m),'.','Color',graycol)
  
subplot(5,7,ct+7)
hold on
plot(MIvect(inds),Svect(inds),'.','Color',graycol)
ylim([0 1])

subplot(5,7,ct+7*2)
hold on
plot(MIvect(inds),Rvect(inds),'.','Color',graycol)
ylim([0 1])

subplot(5,7,ct+7*3)
hold on
plot(MIvect(inds),U1vect(inds),'.','Color',graycol)
ylim([0 1])

subplot(5,7,ct+7*4)
hold on
plot(MIvect(inds),U2vect(inds),'.','Color',graycol)
ylim([0 1])


    subplot(5,7,ct)
    title(modnames{ct})
   
    subplot(5,7,ct+7*4)
    xlabel('I(X;Y)')

ct=ct+1;


end


ct=1;
for m=2:size(model,2)
    
MIvect = Ixy(:,m);
Svect = S(:,m)./Itot(:,m);
Rvect = R(:,m)./Itot(:,m);
U1vect = U1(:,m)./Itot(:,m);
U2vect = U2(:,m)./Itot(:,m);
H1vect = Hx(:,m);
Hdiffvect = Hx(:,m)-Hy(:,m);
inds = 1:length(MIvect);


for cases = 1:3
    
    if cases ==1
        inds = find(covar<-.2);
        col = 'b';
        
    elseif cases ==2
        inds = find(covar > -.001 & covar < 0.001);
        col = 'g';
        
    elseif cases ==3
        inds = find(covar>.2);
        col = 'r';
    end

subplot(5,7,ct)
hold on
plot(MIvect(inds),Itot(inds,m),'.','Color',col)
ylim([0 1])

set(gca,'Xticklabels',[])
if m>2
    set(gca,'Yticklabels',[])
end
   
    
subplot(5,7,ct+7)
hold on
plot(MIvect(inds),Svect(inds),'.','Color',col)
ylim([0 1])
set(gca,'Xticklabels',[])
if m>2
    set(gca,'Yticklabels',[])
end


subplot(5,7,ct+7*2)
hold on
plot(MIvect(inds),Rvect(inds),'.','Color',col)
ylim([0 1])
set(gca,'Xticklabels',[])
if m>2
    set(gca,'Yticklabels',[])
end

subplot(5,7,ct+7*3)
hold on
plot(MIvect(inds),U1vect(inds),'.','Color',col)
ylim([0 1])
set(gca,'Xticklabels',[])
if m>2
    set(gca,'Yticklabels',[])
end


subplot(5,7,ct+7*4)
hold on
plot(MIvect(inds),U2vect(inds),'.','Color',col)
ylim([0 1])

if m>2
    set(gca,'Yticklabels',[])
end

end
ct=ct+1;

end



