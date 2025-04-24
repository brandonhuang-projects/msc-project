function [TiMaxBurst, PkMaxBurst, MeanBurst, TiLow, TiHigh, Area, TEST, PeakIndex]= PeaksBeta_V2(Count, PoisLamda, thresh, time, Buff)

% Data
s = Count';
s = s + rand(1, length(s))*1e-9;
Data = s;

if Buff.On == 1
    %%%% fill
    g = s;
    g(g< PoisLamda) = 0; % thresh 3STD or PoisLamda
    g(g > 0) = 1;

    for j = 1:Buff.Numb
        r = strfind(g, [1 1 1 zeros(1,j) 1]);
        for n = 1:j
            g(r+3+n-1) = 1;
        end
        r = strfind(g, [1 zeros(1,j) 1 1 1]);

        for n = 1:j
            g(r+1 + n - 1) = 1;
        end
    end

    s(g==0) = 0;
    %%%%%%
else
    s(s< PoisLamda) = 0; % thresh 3STD or PoisLamda
end



% ix=(1:length(s)).'; % index of bins


p=abs(sign(s));  %ones for value > 0

ps=[p(1) diff(p)]; % 1 0 0 0 -1 etc.... BW = 5

ps(ps~=1)=0; % only ones left
ps=cumsum(ps);
ps=ps.*p; % 1 1 1 for burst 1 , 2 2 2 for burst 2 etc...

% ...remove zeros from index, counts, and burst
s=s(p~=0).';
% g=g(p~=0).';
p=ps(ps~=0).';
%%%%%%

[m] =accumarray(p,s,[max(p) 1],@max); % SLOW

%%
              
                FndAboveThresh = find(m >= thresh);
                ind_new = ismember(ps, FndAboveThresh);
                
                s2=Data(ind_new);
                time=time(ind_new);
                
                ind_new2 =[ind_new(1) diff(ind_new)];
                ind_new2(ind_new2~=1)=0;
                
                ind_start = ind_new2;
               
                
                ind_new2=cumsum(ind_new2);
                ind_new2=ind_new2.*ind_new; % 1 1 1 for burst 1 , 2 2 2 for burst 2 etc...
                ind_new3=ind_new2(ind_new2~=0).';

                first =accumarray(ind_new3,s2,[max(ind_new3) 1],@(x)x(1));
                last =accumarray(ind_new3,s2,[max(ind_new3) 1],@(x)x(end));
                MeanBurst=accumarray(ind_new3,s2,[max(ind_new3) 1],@mean);
                SUM =accumarray(ind_new3,s2,[max(ind_new3) 1],@sum); 

                

                TEST = accumarray(ind_new3,s2,[max(ind_new3) 1], @(x) {x}); % SLOW

                
                Output = accumarray(ind_new3,1:numel(s2),[max(ind_new3) 1],@(x) {findTime(x, s2, time)});
                Output = cell2mat(Output);
                
%                 first = Output(:,3);
%                 last = Output(:,4);
%                 MeanBurst = Output(:,5);
%                 SUM = Output(:,6);
                Area = (2*SUM -first - last)/2;
                
                
                TiMaxBurst = Output(:,2);
                PkMaxBurst = m(FndAboveThresh);

                TiLow = accumarray(ind_new3,time,[max(ind_new3) 1],@(x)x(1));
                TiHigh = accumarray(ind_new3,time,[max(ind_new3) 1],@(x)x(end));
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                % not correct index

                ind_start = find(ind_start ==1)' + Output(:,1) - 1; % start location of each event + relative loc of peakmax
                PeakIndex = ind_start;
%%


end

    function ix = findTime(indx, s2, time)
        [m2,ix_loc] = max(s2(indx));
        ix(1,2) = time(indx(ix_loc));
        
        ix(1,1) = ix_loc;
        
%         ix(1,3) = s2(indx(1));
%         ix(1,4) = s2(indx(2));
%         
%         ix(1,5) = mean(s2(indx));
%         ix(1,6) = sum(s2(indx));
    end