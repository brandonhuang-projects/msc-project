
%% Variables to Initialize:

% app.Count_2
% app.Time_2
% app.BaseVar
% app.Baselinevars.Data

%% Pre-function Setup

data = abfload('C:\Users\brand\OneDrive\UNI\Chem\Year 5\MSc\Scripts\test.abf');

app.Count_2 = data(:,1); s = size(data(:,1)); app.Time_2 = linspace(0,s(1),s(1));

app.BaselineVars.Data = {'Step', 3; 'Lambda', 10000000; 'p', 0.01; 'Order', 2};
BaseVar = app.BaselineVars.Data;