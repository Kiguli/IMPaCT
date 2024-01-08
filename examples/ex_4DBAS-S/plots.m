controller = h5read('controller.h5','/dataset');

A = [0.6682, 0.0, 0.02632, 0.0; 0.0, 0.6830, 0.0, 0.02096; 1.0005, 0.0,  -0.000499, 0.0; 0.0, 0.8004, 0.0, 0.1996];
 B = [0.1320; 0.1402; 0.0; 0.0];
 Q = [3.4378; 2.9272; 13.0207; 10.4166];

%% Test Run
init_x = [19, 20,36,36]; %1,-8
init_u = [20];
trace = [init_x, init_u];
for t= 1:11
    x = trace(end, 1:4);
    u = getInput(controller, x);
    trace = [trace; dynamics(x, u), u];
end

trace = trace (1:11,1:4);

time = 0:(size(trace, 1)-1);

for i = 1:size(trace, 2)
    subplot(2, 2, i); % 2x2 grid, select ith subplot
    plot(time, trace(:, i), '-'); % Plot data for the ith parameter
    %title(['x', num2str(i)]); % Set subplot title
    xlabel('Time'); % X-axis label
    ylabel(['x', num2str(i)]); % Y-axis label
    grid on; % Show grid
end



%% functions
function u = getInput(controller,x)
% Calculate the Euclidean distances between each row of the array and the given value
distances = sqrt(sum((controller(:,1:4) - x).^2, 2));

% Find the index of the row with the minimum distance
[~, closestRow] = min(distances);
u = controller(closestRow, 5);
end

function xx = dynamics(x,u)
sigma = [0.0774,0.0774,0.3872,0.3098];
 A = [0.6682, 0.0, 0.02632, 0.0; 0.0, 0.6830, 0.0, 0.02096; 1.0005, 0.0,  -0.000499, 0.0; 0.0, 0.8004, 0.0, 0.1996];
 B = [0.1320; 0.1402; 0.0; 0.0];
 Q = [3.4378; 2.9272; 13.0207; 10.4166];
    
xx = A*x.' + B*u + Q;
xx(1) = xx(1) + normrnd(0,sigma(1));
xx(2) = xx(2) + normrnd(0,sigma(2));
xx(3) = xx(3) + normrnd(0,sigma(3));
xx(4) = xx(4) + normrnd(0,sigma(4));
xx = xx.';
end