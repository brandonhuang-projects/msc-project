%addpath 'C:\Users\brand\OneDrive\UNI\Chem\Year 5\MSc\Scripts'
addpath 'C:\Users\brand\AppData\Roaming\MathWorks\MATLAB Add-Ons\Apps\TheNanoporeAppLt'

%% Variables to Initialize:

% app.Resample.Value
% app.FWHM_CheckBox.Value
% app.FWHMDropDown.Value
% app.LWidthLimit_CheckBox.Value
% app.LWidthLimit.Value = 0.0005
% app.UWidthLimit_CheckBox.Value
% app.UWidthLimit.Value
% app.UCurrentLimit_CheckBox.Value
% app.CurrentBelowVal.Value
% app.Buffer_CheckBox.Value 
% app.BufferBinsNo.Value 

% app.CountBase 
% app.PoisLamda
% app.thresh
% app.Time_2
% app.T_res

% app.FileCondition
% app.Peaks_Plot

% app.BaselineLengthIndividual.Value

% app.Count_3
% app.Time_3


%% Pre-function Setup

[data, app.Resample.Value, h]  = abfload('C:\Users\brand\OneDrive\UNI\Chem\Year 5\MSc\Scripts\test.abf');

app.Resample.Value = 100000;

app.Count_2 = data(:,1); s = size(data(:,1)); app.Time_2 = linspace(0,s(1),s(1));
app.Count_3 = data(:,2); s_3 = size(data(:,2)); app.Time_3 = linspace(0,s_3(1),s_3(1));

app.MovMean_Const.Value = 10;

app.Base = movmean(app.Count_2, app.MovMean_Const.Value);   % Count_2 = y data, MovMean_Const = number of bins to use for moving mean
app.Base = app.Base - 0.75*std(app.Base); % lower moving mean by standard deviation
app.CountBase = (app.Count_2 - app.Base); % subtract tracked baseline from data
app.CountBase(app.CountBase <=0) = 0; % if value less than 0 make equal to zero

ThresholdVar = {'std', 7; 'SO', 1.8; 'Overide', false; 'Overide Step', 0.01; 'Overide Max', 0.5};
ThrStep.Overide = ThresholdVar{3,2};
ThrStep.Step    = ThresholdVar{4,2};
ThrStep.MX      = ThresholdVar{5,2};

[app.thresh, app.PoisLamda, app.StepVec, app.HistThresh, app.PoisFit] = Threshold(app.CountBase, ThresholdVar{1,2}, ThresholdVar{2,2}, ThrStep);

try
    app.FileCondition = double(ismember(FileSeries(end-3:end), {'.Gn', '.Rd', 'csv'}));
catch
    app.FileCondition = false;
end

app.FWHM_CheckBox.Value = true;
app.FWHMDropDown.Value = '2';

app.LWidthLimit_CheckBox.Value = 0; % == 1 dependance 
app.LWidthLimit.Value = 0.0005;

app.UWidthLimit_CheckBox.Value = 0; % == 1 dependance 
app.UWidthLimit.Value = 0.01;

app.UCurrentLimit_CheckBox.Value = 0; % == 1 dependance 
app.CurrentBelowVal.Value = 50;

app.Buffer_CheckBox.Value = 0;
app.BufferBinsNo.Value = 5;

app.Peaks_Plot = matlab.ui.control.UIAxes;
app.BaselineLengthIndividual.Value = 75;

BaselineLength = 100; 

app.T_res = app.Resample.Value; app.T_res = app.T_res*1e-6;  % time resolution of data
FullWidthHM.On = app.FWHM_CheckBox.Value;
FullWidthHM.factor = str2double(app.FWHMDropDown.Value);
 
WidthLimit.Low_on  = app.LWidthLimit_CheckBox.Value;
WidthLimit.Low_width = app.LWidthLimit.Value;
WidthLimit.Upper_on  = app.UWidthLimit_CheckBox.Value;
WidthLimit.Upper_width = app.UWidthLimit.Value;


CurrentLimit.on  = app.UCurrentLimit_CheckBox.Value;
CurrentLimit.Value = app.CurrentBelowVal.Value;
        

%profile on
Buff.On = app.Buffer_CheckBox.Value;
Buff.Numb = app.BufferBinsNo.Value;
 

app.T_res = app.Resample.Value; app.T_res = app.T_res*1e-6;  % time resolution of data
FullWidthHM.On = app.FWHM_CheckBox.Value;
FullWidthHM.factor = str2double(app.FWHMDropDown.Value);
 
WidthLimit.Low_on  = app.LWidthLimit_CheckBox.Value;
WidthLimit.Low_width = app.LWidthLimit.Value;
WidthLimit.Upper_on  = app.UWidthLimit_CheckBox.Value;
WidthLimit.Upper_width = app.UWidthLimit.Value;


CurrentLimit.on  = app.UCurrentLimit_CheckBox.Value;
CurrentLimit.Value = app.CurrentBelowVal.Value;
        

%profile on
Buff.On = app.Buffer_CheckBox.Value;
Buff.Numb = app.BufferBinsNo.Value;