addpath(fullfile(pwd,'matpy_shmem'))

func_names = initProcessor();

data1 = rand(1, 1000);          
data2 = int32(rand(1, 100));

tic
[data2, mean, data3] = runProcessor('test', data1, data2);
toc 