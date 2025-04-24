
%% Variables to Initialize:

% app.Count_2
% app.Time_2

%% Pre-function Setup

[data, app.Resample.Value, h]  = abfload('C:\Users\brand\OneDrive\UNI\Chem\Year 5\MSc\Scripts\test.abf');

app.Count_2 = data(:,1); s = size(data(:,1)); app.Time_2 = linspace(0,s(1),s(1));

app.T_res = app.Resample.Value;

T_Resample = 7;

Fact = app.T_res/T_Resample;
[Num, Denom] = rat(Fact);