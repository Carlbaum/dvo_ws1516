close all

% Load trajectory data
fid = fopen('../data/DataSet Freiburg/traj.txt','r');
data = textscan(fid, '%f%f%f%f%f%f%f%f');
fclose(fid);
traj = zeros(3, size(data{1}, 1));
traj(1,:) = data{2}';
traj(2,:) = data{3}';
traj(3,:) = data{4}';

% Load groundtruth data
fid = fopen('../data/DataSet Freiburg/groundtruth.txt','r');
data = textscan(fid, '%f%f%f%f%f%f%f%f');
fclose(fid);
groundtruth = zeros(3, size(data{1}, 1));
groundtruth(1,:) = data{2}';
groundtruth(2,:) = data{3}';
groundtruth(3,:) = data{4}';

% Plot
hold on
plot3(traj(1,:), traj(2,:), traj(3,:), 'b-');
plot3(groundtruth(1,:), groundtruth(2,:), groundtruth(3,:), 'r-');
set(gca, 'Projection', 'perspective');
axis off vis3d
hold off