function [TiMaxBurst, PkMaxBurst, MeanBurst, TiLow, TiHigh, Area, TEST, PeakIndex] = FindPeaks_V2(CountBase, PoisLamda, thresh, time, T_res, FullWidthHM, WidthLimit, CurrentLimit, FileCondition, Buff)

%%%%%%%%%%%%%%%%%%%%%%% Peak Location

% whos
tic
[TiMaxBurst, PkMaxBurst, MeanBurst, TiLow, TiHigh, Area, TEST, PeakIndex]= PeaksBeta_V2(CountBase, PoisLamda, thresh, time, Buff);
toc

%%%%%%%%%%%%%%% width has to be greater than 0
ValZero = find((TiHigh - TiLow) > 0);
TiMaxBurst = TiMaxBurst(ValZero);
PkMaxBurst = PkMaxBurst(ValZero);
MeanBurst = MeanBurst(ValZero);
TiLow = TiLow(ValZero);
TiHigh = TiHigh(ValZero);
Area = Area(ValZero);
TEST = TEST(ValZero);
PeakIndex = PeakIndex(ValZero);
%%%%%%%%%%%%%%%


if WidthLimit.Low_on == 1
    ValC = find((TiHigh - TiLow) >= WidthLimit.Low_width);
    TiMaxBurst = TiMaxBurst(ValC);
    PkMaxBurst = PkMaxBurst(ValC);
    MeanBurst = MeanBurst(ValC);
    TiLow = TiLow(ValC);
    TiHigh = TiHigh(ValC);
    Area = Area(ValC);
    
    TEST = TEST(ValC);
    PeakIndex = PeakIndex(ValC);
end

if WidthLimit.Upper_on == 1
    ValC = find((TiHigh - TiLow) <= WidthLimit.Upper_width);
    TiMaxBurst = TiMaxBurst(ValC);
    PkMaxBurst = PkMaxBurst(ValC);
    MeanBurst = MeanBurst(ValC);
    TiLow = TiLow(ValC);
    TiHigh = TiHigh(ValC);
    Area = Area(ValC);
    
    TEST = TEST(ValC);
    PeakIndex = PeakIndex(ValC);
end

if CurrentLimit.on == 1
    % PkMaxBurst
    ValD = find((PkMaxBurst <= CurrentLimit.Value/1000)); % convert to nA
    TiMaxBurst = TiMaxBurst(ValD);
    PkMaxBurst = PkMaxBurst(ValD);
    MeanBurst = MeanBurst(ValD);
    TiLow = TiLow(ValD);
    TiHigh = TiHigh(ValD);
    Area = Area(ValD);
    
    TEST = TEST(ValD);
    PeakIndex = PeakIndex(ValD);
end



PeakIndexRaw = round(TiMaxBurst/T_res);


%%%%% full width half max of bursts

if FullWidthHM.On ==  1
    for j= 1:length(TEST)

        X = TEST{j,1};
        MX(j,1) = max(X);
        %
        MX_Loc(j,1) = find(X == MX(j,1));
        
        
        Y(j,1) = MX(j,1)/FullWidthHM.factor; % 2 for FWHM
        
        [d1 p1(j,1)] = min(abs(X(1:MX_Loc(j,1)) - Y(j,1)));
        [d2 p2(j,1)] = min(abs(X(MX_Loc(j,1):end) - Y(j,1)));
        
        p3(j,1) = p2(j,1)+MX_Loc(j,1);
        
        if p3(end) > length(X)
            p3(end) = length(X);
            
        end
        
        PEAKS_FWHM{j,1} = X(p1(end):p3(end));
        
        AreaFWHM(j,1) = (2*sum(X(p1(end):p3(end))) - X(p1(end)) - X(p3(end)))/2;
        
    end
    
    TEST =   PEAKS_FWHM;
    
    Area = AreaFWHM;
    
    
    
    WidthFWHM = (p3-p1)*T_res;
    
    %
    TiLowFWHM = TiMaxBurst+(p1 - MX_Loc)*T_res;
    TiHighFWHM = TiMaxBurst+(p3-MX_Loc)*T_res;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    TiLow = TiLowFWHM;
    TiHigh = TiHighFWHM;
    
    
    PeakIndex = (PeakIndex - MX_Loc + (p3 - p1)/2+1);
    %
    
    
end

if WidthLimit.Low_on == 1
    ValC = find((TiHigh - TiLow) >= WidthLimit.Low_width);
    TiMaxBurst = TiMaxBurst(ValC);
    PkMaxBurst = PkMaxBurst(ValC);
    MeanBurst = MeanBurst(ValC);
    TiLow = TiLow(ValC);
    TiHigh = TiHigh(ValC);
    Area = Area(ValC);
    
    TEST = TEST(ValC);
    PeakIndex = PeakIndex(ValC);
end