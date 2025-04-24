function outcome = shmemStatus()
    addpath(fullfile(fileparts(mfilename('fullpath')),'shared_memory_processor'))
    func_names = initProcessor();
    
    data1 = "Hello, Shared Memory!";       % String
    data2 = [1.1, 2.2, 3.3, 4.4];            % Numeric double array (np.float64)
    data3 = int8([1, -2, 3, -4]);            % int8 array
    data4 = int16([100, -200, 300, -400]);    % int16 array
    data5 = uint8([10, 20, 30, 40]);         % uint8 array
    data6 = [1+2i, 3+4i, 5+6i];              % Complex double array
    data7 = 3.1415926535;
    data8 = [1,2,3,4,5];
    data9 = struct('a', 10, 'b', 'hello', 'c', [1,2,3], 'd', true);
    
    [out1, out2, out3, out4, out5, out6, out7, out8, out9] = runProcessor('status', ...
        data1, data2, data3, data4, data5, data6, data7, data8, data9);
    
    check = {'HELLO, SHARED MEMORY!', 2.75, -2, 300, 100, ([1-2i, 3-4i, 5-6i].'), 3.14, ...
        {[1;2];[3;4;5]}, struct('a', 11, 'b', 'HELLO', 'c', 6, 'd', false);};
    
    outcome = isequal(check,{out1, out2, out3, out4, out5, out6, out7, out8, out9});