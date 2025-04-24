% Matlab FindPeaks_V2 file needed (along with PeaksBeta_V2)
%[app.TiMaxBurst, app.PkMaxBurst, app.MeanBurst, app.TiLow, app.TiHigh, app.Area, app.Event_all, PeakIndex] = FindPeaks_V2(app.CountBase, app.PoisLamda, app.thresh, app.Time_2, app.T_res, FullWidthHM, WidthLimit, CurrentLimit, app.FileCondition, Buff);

[app.TiMaxBurst, app.PkMaxBurst, app.MeanBurst, app.TiLow, app.TiHigh, app.Area, app.Event_all, PeakIndex] = ...
                runProcessor('peakfinder', app.CountBase, app.PoisLamda, app.thresh, app.Time_2, app.T_res, FullWidthHM, WidthLimit, CurrentLimit, app.FileCondition, Buff);
 


%%%%



%  assignin('base','Event', app.Event);



 
% [app.TiMaxBurst, app.PkMaxBurst, app.MeanBurst, app.TiLow, app.TiHigh, app.Area, app.Event_all, PeakIndex] = runProcessor('peakfinder', app.CountBase, app.PoisLamda, app.thresh, app.Time_2, app.T_res, FullWidthHM.On, FullWidthHM.factor, WidthLimit.Low_on, WidthLimit.Low_width, WidthLimit.Upper_on, WidthLimit.Upper_width, CurrentLimit.on, CurrentLimit.Value, app.FileCondition, Buff.On, Buff.Numb);