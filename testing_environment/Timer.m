%addpath 'C:\Users\brand\AppData\Roaming\MathWorks\MATLAB Add-Ons\Apps\TheNanoporeAppLt'
%addpath 'C:\Users\brand\OneDrive\UNI\Chem\Year 5\MSc\Scripts'

function Timer(pre_script, main_script, post_script, repeat)
    % Function to test the timing of a main script with setup and cleanup.
    % 
    % Inputs:
    %   pre_script   
    %   main_script 
    %   post_script 
    %   repeat      

    temp_workspace_file = 'temp_workspace.mat'; 
    
    % Run pre-setup 
    if ~isempty(pre_script)
        fprintf('-- Running pre-setup script: %s\n', pre_script);
        run(pre_script);

        % Save workspace
        save(temp_workspace_file);
    end

    fprintf('\n')

    timings = zeros(1, repeat, 'double');
    for i = 1:repeat

        fprintf('-- Timing main script: %s (Run %d of %d)\n', main_script, i, repeat);
        
        % Reload workspace
        load(temp_workspace_file);
        
        t0 = tic;
        run(main_script);
        timings(i) = toc(t0);
    end
    
    fprintf('\n');
    
    % Calculate mean and standard deviation
    mean_time = mean(timings);
    std_dev_time = std(timings);

    % Display the timing result
    fprintf('Elapsed time (MATLAB): %.6f s ± %.6f s (mean ± std. dev. of %d runs)\n', ...
        mean_time, std_dev_time, repeat);

    % Run post-cleanup
    if ~isempty(post_script)
        run(post_script);
    end

end