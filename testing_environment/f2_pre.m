addpath 'C:\Users\brand\OneDrive\UNI\Chem\Year 5\MSc\Scripts'
addpath 'C:\Users\brand\AppData\Roaming\MathWorks\MATLAB Add-Ons\Apps\TheNanoporeAppLt'

%% Variables to Initialize:

% app.CountBase
% app.ThresholdPlot
% app.TrackBaselinePlot

%% Pre-function Setup

data = abfload('C:\Users\brand\OneDrive\UNI\Chem\Year 5\MSc\Scripts\test.abf');

app.Count_2 = data(:,1); s_2 = size(data(:,1)); app.Time_2 = linspace(0,s_2(1),s_2(1));

app.MovMean_Const.Value = 10;

app.Base = movmean(app.Count_2, app.MovMean_Const.Value);   % Count_2 = y data, MovMean_Const = number of bins to use for moving mean

app.Base = app.Base - 0.75*std(app.Base); % lower moving mean by standard deviation

app.CountBase = (app.Count_2 - app.Base); % subtract tracked baseline from data

app.CountBase(app.CountBase <=0) = 0; % if value less than 0 make equal to zero

ThresholdVar = {'std', 7; 'SO', 1.8; 'Overide', false; 'Overide Step', 0.01; 'Overide Max', 0.5};

ThrStep.Overide = ThresholdVar{3,2};

ThrStep.Step    = ThresholdVar{4,2};

ThrStep.MX      = ThresholdVar{5,2};


% Plots

app.ThresholdPlot = matlab.ui.control.UIAxes;

app.TrackBaselinePlot = matlab.ui.control.UIAxes;

ThresholdVar = {'std', 7; 'SO', 1.8; 'Overide', false; 'Overide Step', 0.01; 'Overide Max', 0.5};

ThrStep.Overide = ThresholdVar{3,2};

ThrStep.Step    = ThresholdVar{4,2};

ThrStep.MX      = ThresholdVar{5,2};