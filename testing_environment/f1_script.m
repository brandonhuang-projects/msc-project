app.MovMean_Const.Value = 10; app.TrackBaselinePlot = matlab.ui.control.UIAxes;

app.Base = movmean(app.Count_2, app.MovMean_Const.Value);   % Count_2 = y data, MovMean_Const = number of bins to use for moving mean

app.Base = app.Base - 0.75*std(app.Base); % lower moving mean by standard deviation

app.CountBase = (app.Count_2-app.Base); % subtract tracked baseline from data

app.CountBase(app.CountBase <=0) = 0; % if value less than 0 make equal to zero
 
% plot x axis and y axis

%plot(app.TrackBaselinePlot, app.Time_2, app.CountBase);
plot(app.Time_2, app.CountBase);

ylabel(app.TrackBaselinePlot, 'Current (nA)');xlabel(app.TrackBaselinePlot, 'Time (s)');