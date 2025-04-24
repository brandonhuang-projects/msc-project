

%app.CountBase = app.Count_2;

 
% Threshold matlab file needed to determine Poisson Fit

%[app.thresh, app.PoisLamda, app.StepVec, app.HistThresh, app.PoisFit] = Threshold(app.CountBase, ThresholdVar{1,2}, ThresholdVar{2,2}, ThrStep);

% runProcessor('threshold',app.CountBase, ThresholdVar{1,2}, ThresholdVar{2,2})

% plot data

[app.thresh, app.PoisLamda, app.StepVec, app.HistThresh, app.PoisFit] = runProcessor('threshold',app.CountBase, ThresholdVar{1,2}, ThresholdVar{2,2}, ThrStep);