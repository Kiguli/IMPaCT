controller = h5read('controller.h5','/dataset');

%% Test Run
init_x = [21	21 21]; %1,-8
init_u = [0,0];
trace = [init_x, init_u];
for t= 1:10
    x = trace(end, 1:3);
    u = getInput(controller, x);
    trace = [trace; dynamics(x, u), u];
end

trace = trace (1:11,1:3);

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
distances = sqrt(sum((controller(:,1:3) - x).^2, 2));

% Find the index of the row with the minimum distance
[~, closestRow] = min(distances);
u = controller(closestRow, 4:5);
end

function xx = dynamics(x,u)
sigma = [1/50,1/50,1/50];
eta = 0.20; beta = 0.022; gamma = 0.05; a = 1.0 - 2.0*0.20 - 0.022; T_h = 50.0; T_e = -1.0;
xx(1) = (a - gamma*u(1))*x(1) + eta*(x(2) + x(3)) + gamma*T_h*u(1) + beta*T_e;
xx(2) = a*x(2) + eta*(x(1) + x(3)) + beta*T_e;
xx(3) = (a - gamma*u(2))*x(3) + eta*(x(1) + x(2)) + gamma*T_h*u(2) + beta*T_e;

xx(1) = xx(1) + normrnd(0,sigma(1));
xx(2) = xx(2) + normrnd(0,sigma(2));
xx(3) = xx(3) + normrnd(0,sigma(3));
end