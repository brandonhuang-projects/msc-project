

Pad = 1e1;


%app.Count_2 = runProcessor('resample',app.Count_2, Num, Denom, Pad);


app.Count_2 = [repmat(mean(app.Count_2(1:10)), Pad, 1); app.Count_2; repmat(mean(app.Count_2(end-10:end)), Pad, 1)];
app.Count_2 = resample(app.Count_2, Num, Denom);
app.Count_2 = app.Count_2(Pad+1:end-Pad);
