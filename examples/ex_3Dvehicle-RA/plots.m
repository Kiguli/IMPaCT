clear
close all
clc
%%
ss = h5read('ss.h5','/dataset');
target = h5read('ts.h5','/dataset');
avoid = h5read('as.h5','/dataset');
is = h5read('is.h5','/dataset');
controller = h5read('controller.h5','/dataset');

boxSize = 0.5;

%% Test Run
init_x = [3 -3 0.6];
init_u = [4, 0.4];
trace = [init_x, init_u];
while ~((trace(end, 1) >= -5.75-boxSize/2 && trace(end, 1) <= 0.25+boxSize/2) && (trace(end, 2) >= -0.25-boxSize/2 && trace(end, 2) <= 5.75+boxSize/2))
%for t= 1:5
    x = trace(end, 1:3);
    u = getInput(controller, x);
    trace = [trace; dynamics(x, u), u];
end

%% plot the state space with target set and avoid set
figure
plot(ss(:,1),ss(:,2),'.',"Color", "blue")
hold on
x = target(:,1);
y = target(:,2);
% Plotting boxes around each point
for i = 1:length(x)
    rectangle('Position', [x(i)-boxSize/2, y(i)-boxSize/2, boxSize, boxSize], 'FaceColor', 'g');
end
hold on
x = avoid(:,1);
y = avoid(:,2);
% Plotting boxes around each point
for i = 1:length(x)
    rectangle('Position', [x(i)-boxSize/2, y(i)-boxSize/2, boxSize, boxSize], 'FaceColor', 'r');
end

hold on
% Plot arrows connecting each row
for i = 1:size(trace, 1)-1
    x = trace(i, 1);
    y = trace(i, 2);
    dx = trace(i+1, 1) - x;
    dy = trace(i+1, 2) - y;
    quiver(x, y, dx, dy, 0,'LineWidth', 4);
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

Ts = 0.1;
   
f = [0,0,0];
    
xx = x;
f(1) = u(1)*cos(atan((tan(u(2))/2.0))+xx(3))/cos(atan((tan(u(2)/2.0))));
f(2) = u(1)*sin(atan((tan(u(2))/2.0))+xx(3))/cos(atan((tan(u(2))/2.0)));
f(3) = u(1)*tan(u(2));
        
xx = xx + Ts*f;

sigma = [sqrt(1/1.5),sqrt(1/1.5),sqrt(1/1.5)];
xx(1) = xx(1) + normrnd(0,sigma(1));
xx(2) = xx(2) + normrnd(0,sigma(2));
xx(3) = xx(3) + normrnd(0,sigma(3));
end
