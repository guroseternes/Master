function plotDataFromGPU(Gt, formation_name, file_name1, file_name2, file_name3)
close all

path = strcat('./SimulationData/ResultData/',formation_name, '/');
file_name1 = strcat(path, file_name1);
file_name2 = strcat(path, file_name2);
file_name3 = strcat(path, file_name3);
        
%% IMPORT DATA FROM MATLAB
% FILE 1
file = fopen(file_name1);
line1 = fgets(file);
[nx_ny] = sscanf(line1,'nx: %i ny: %i');
nx = nx_ny(1);
ny = nx_ny(2);

[x, y, values] = textread(file_name1,'%f%f%f','headerlines', 1);
length(values);
size = nx*ny;
x = x(1:nx);
y = x;

currentValues1 = values; 
currentValuesMatrix1 = (reshape(currentValues1,nx,ny));
%currentValuesMatrix1(currentValuesMatrix1==0) = NaN;

% FILE 2
file = fopen(file_name2);
line1 = fgets(file);
[nx_ny] = sscanf(line1,'nx: %i ny: %i');
nx = nx_ny(1);
ny = nx_ny(2);

[x, y, values] = textread(file_name2,'%f%f%f','headerlines', 1);
length(values);
size = nx*ny;
x = x(1:nx);
y = x;
currentValues2 = values;
currentValuesMatrix2 = (reshape(currentValues2,nx,ny));

% FILE 3
file = fopen(file_name3);
line1 = fgets(file);
[nx_ny] = sscanf(line1,'nx: %i ny: %i');
nx = nx_ny(1);
ny = nx_ny(2);

[x, y, values] = textread(file_name3,'%f%f%f','headerlines', 1);
length(values);
size = nx*ny;
x = x(1:nx);
y = x;
currentValues3 = values;
currentValuesMatrix3 = (reshape(currentValues3,nx,ny));

    figure
    surf(currentValuesMatrix1', 'EdgeColor', 'none');
    view(2)
    colorbar
    set(gca,'FontSize',16)

    figure
    surf(currentValuesMatrix2', 'EdgeColor', 'none');
    view(2)
    colorbar
    set(gca,'FontSize',16)
    
    figure
    surf(currentValuesMatrix3', 'EdgeColor', 'none');
    view(2)
    colorbar
    set(gca,'FontSize',16)
    
end