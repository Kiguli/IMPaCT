clear
close all
clc

%% read in files
ss = h5read('ss.h5','/dataset');
target = h5read('ts.h5','/dataset');
avoid = h5read('as.h5','/dataset');
is = h5read('is.h5','/dataset');
controller = h5read('controller.h5','/dataset');

%% figure 1 - state space with target/avoid regions
boxSize = 1; % Eta_x

%plot the state space with target set and avoid set
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

%% Figure 2 - Plot lower bound satisfaction probabilities
C = [controller(:,1:2), controller(:,5)];
X = C(:,1) ; Y = C(:,2) ; Z = C(:,3);
dt = delaunayTriangulation(X,Y);
Tri = dt.ConnectivityList;

figure
trisurf(Tri, X, Y, Z, 'FaceColor', 'interp'); 
colormap('cool');
colorbar;
hold on;

% Plotting boxes around each point (target)
for i = 1:length(target)
    x = target(i, 1);
    y = target(i, 2);
    z = 1;
    patch([x-boxSize x+boxSize x+boxSize x-boxSize],[y-boxSize y-boxSize y+boxSize y+boxSize],[z z z z],'g');
end

% Plotting boxes around each point (avoid)
for i = 1:length(avoid)
    x = avoid(i, 1);
    y = avoid(i, 2);
    z = 1;
    patch([x-boxSize x+boxSize x+boxSize x-boxSize],[y-boxSize y-boxSize y+boxSize y+boxSize],[z z z z],'r');
end

%% Figure 3 - Plot uppoer bound satisfaction probabilities
C = [controller(:,1:2), controller(:,6)];
X = C(:,1) ; Y = C(:,2) ; Z = C(:,3);
dt = delaunayTriangulation(X,Y);
Tri = dt.ConnectivityList;

figure
trisurf(Tri, X, Y, Z, 'FaceColor', 'interp'); 
colormap('cool');
colorbar;
hold on;

% Plotting boxes around each point (target)
for i = 1:length(target)
    x = target(i, 1);
    y = target(i, 2);
    z = 1;
    patch([x-boxSize x+boxSize x+boxSize x-boxSize],[y-boxSize y-boxSize y+boxSize y+boxSize],[z z z z],'g');
end

% Plotting boxes around each point (avoid)
for i = 1:length(avoid)
    x = avoid(i, 1);
    y = avoid(i, 2);
    z = 1;
    patch([x-boxSize x+boxSize x+boxSize x-boxSize],[y-boxSize y-boxSize y+boxSize y+boxSize],[z z z z],'r');
end
%% Figure 4 - Run Trace of Controller
init_x = [-6, -6];
init_u = [0, 0];
trace = [init_x, init_u];
while ~((trace(end, 1) >= 5 && trace(end, 1) <= 8) && (trace(end, 2) >= 5 && trace(end, 2) <= 8))
%for t= 1:5
    x = trace(end, 1:2);
    u = getInput(controller, x);
    trace = [trace; dynamics(x, u), u];
end

%plot the state space with target set and avoid set
figure
plot(ss(:,1),ss(:,2),'.',"Color", "blue")
hold on
x = target(:,1);
y = target(:,2);
% Plotting boxes around each point (target)
for i = 1:length(x)
    rectangle('Position', [x(i)-boxSize/2, y(i)-boxSize/2, boxSize, boxSize], 'FaceColor', 'g');
end

hold on
x = avoid(:,1);
y = avoid(:,2);
% Plotting boxes around each point (avoid)
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
distances = sqrt(sum((controller(:,1:2) - x).^2, 2));
% Find the index of the row with the minimum distance
[~, closestRow] = min(distances);
u = controller(closestRow, 3:4);
end

function xx = dynamics(x,u)
sigma = [1.3333,1.3333];
xx(1) = x(1) + 2 * u(1) * cos(u(2)) + normrnd(0,sigma(1));
xx(2) = x(2) + 2 * u(1) * sin(u(2)) + normrnd(0,sigma(2));
end