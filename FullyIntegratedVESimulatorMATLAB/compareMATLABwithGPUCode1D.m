function compareMATLABwithGPUCode1D(forComparison, Gt, rock2D, filename, preComp, sol1, sol2)
%close all

%% First make a rectangular 2D carthesian grid of the data
ij = Gt.cells.ij;
len = length(ij(:,1)); 

xdim = max(ij(:,1)) - min (ij(:,1)) + 1;
ydim = max(ij(:,2)) - min (ij(:,2)) + 1;

ij(:,1) = ij(:,1)-min (ij(:,1))+1;
ij(:,2) = ij(:,2)-min (ij(:,2))+1;

cells = Gt.cells;

mob_c = zeros(xdim, ydim);
mob_b = zeros(xdim, ydim);

dmob_c = zeros(xdim, ydim);
dmob_b = NaN*ones(xdim, ydim);
perm = zeros(xdim, ydim);
dz = zeros(xdim, ydim);
gflux = zeros(xdim, ydim);
b = ones(xdim, ydim)*NaN;
gvec = zeros(xdim, ydim);
zdiff = zeros(xdim, ydim);
flux = zeros(xdim, ydim);
rawFlux = zeros(xdim, ydim);
fw_face = ones(xdim, ydim)*NaN;
h_new = ones(xdim, ydim)*NaN;
face_mob = ones(xdim, ydim)*NaN;
face_mob_b = ones(xdim, ydim)*NaN;
h_old = zeros(xdim, ydim);
vol_new = ones(xdim, ydim)*NaN;
ff_north = zeros(xdim, ydim);
kr = zeros(xdim, ydim);
H = zeros(xdim,ydim);
q = zeros(xdim,ydim);
z = ones(xdim,ydim);
pv = ones(xdim, ydim)*NaN;
poro = ones(xdim,ydim)*NaN;
perm = ones(xdim,ydim)*NaN;

forComparison.q = full(forComparison.q);

for k = 1:length(ij)
    % Find indexes
    i = ij(k,1);
    j = ij(k,2);
    current_faces = Gt.cells.faces(Gt.cells.facePos(k):Gt.cells.facePos(k+1)-1,1);
    for l=1:4
        current_face = current_faces(l);
        f=Gt.faces.neighbors(current_face,:);
        if (f(1) == k && diff(f) == 1)
            east_face = current_face;
        end
        if (f(1) == k && diff(f) > 1)
            north_face = current_face;
        end
    end
    
    poro(i,j) = rock2D.poro(k);
    perm(i,j) = rock2D.perm(k);
    
    face_mob_b(i,j) = forComparison.facemob_b(east_face);
    perm(i,j) = rock2D.perm(k);
    mob_c(i,j) = forComparison.mob(k,1);
    
    pv(i,j) = preComp.pv(k);
    
    ff_east(i,j) = forComparison.ff(east_face);
    ff_north(i,j) = forComparison.ff(north_face);
    
    vol_new(i,j) = forComparison.vol_new(k);
    z(i,j) = Gt.cells.z(k);
    dmob_c(i,j) = forComparison.dmob(k,1);
    dmob_b(i,j) = forComparison.dmob(k,2);
    dz(i,j) = forComparison.dz(k);
%    kr(i,j) = forComparison.kr(k);
    gflux(i,j) = forComparison.gflux(current_faces(3)); %*mob_b(i,j);
    b(i,j) = forComparison.b(current_faces(3));
    gvec(i,j) = forComparison.g_vec(current_faces(3));
    zdiff(i,j) = forComparison.zdiff(current_faces(3));
    face_mob(i,j) = forComparison.facemob(east_face);

    rawFlux(i,j) =  sol1.flux(east_face);%forComparison.rawFlux(current_faces(3));
    q(i,j) = forComparison.q(k);
    H(i,j) = Gt.cells.H(k);
    h_new(i,j) = sol2.h(k);
    h_old(i,j) = sol1.h(k);
    mob_b(i,j) = forComparison.mob(k,2);
    fw_face(i,j) = forComparison.fw_face(current_faces(3));

end

file = fopen(filename);
 line1 = fgets(file);
 [nx_ny] = sscanf(line1,'nx: %i ny: %i');
nx = nx_ny(1);
ny = nx_ny(2);

[x, y, values] = textread(filename,'%f%f%f','headerlines', 1);

length(values);

size = nx*ny;

x = x(1:nx);
y = x;

currentValues1 = values;    
currentValuesMatrix1 = (reshape(currentValues1,nx,ny));
currentValuesMatrix1(isnan(h_new)) = NaN;
%currentValuesMatrix1(currentValuesMatrix1 == 0) = NaN;

file = fopen('toMATLAB1.txt');
 line1 = fgets(file);
 [nx_ny] = sscanf(line1,'nx: %i ny: %i');
nx = nx_ny(1);
ny = nx_ny(2);

[x, y, values] = textread('toMATLAB1.txt','%f%f%f','headerlines', 1);

length(values);

size = nx*ny;

x = x(1:nx);
y = x;

currentValues2 = values;    
currentValuesMatrix2 = (reshape(currentValues2,nx,ny));
currentValuesMatrix2(isnan(h_new)) = NaN;
%currentValuesMatrix1(currentValuesMatrix1 == 0) = NaN;
%currentValuesMatrix2(currentValuesMatrix2 == 0) = NaN;

saving = false; 

surf(currentValuesMatrix2', 'EdgeColor', 'none');
view(2)
colorbar

   download = load('johansen_independent_coats_500_years_Beta_04_mob3_satu.mat')
   matlab_matrix = download.matlab_matrix;
matlab_matrix = -(matlab_matrix.*H)-z; %./H*(1-0.1);
length(currentValuesMatrix1)
 GPU_matrix = -currentValuesMatrix2(1:end,1:end)-z;%./H;

           
    figure
    diffi = (GPU_matrix);
    x1= 52;
    lim1 = 48;
    lim2 = 65;
    plot(lim1:lim2,diffi(x1,lim1:lim2), 'LineWidth', 2);
    set(gca,'FontSize',16)
    hold on
    plot(lim1:lim2, matlab_matrix(x1,lim1:lim2), 'LineWidth', 2);
    hold on 
    plot(lim1:lim2,-z(x1,lim1:lim2),'w', 'LineWidth', 2);
    legend('h linear mobility', 'h sharp interface')
    hold on
    plot(lim1:lim2,-z(x1,lim1:lim2),'Color', [0.87 0.49 0], 'LineWidth', 1.2);
    hold on
    thick = -z-H;
    plot(lim1:lim2,thick(x1,lim1:lim2), 'Color', [0.87 0.49 0], 'LineWidth', 1.2);
    xlim([lim1, lim2])

    figure
    lim2 = 60;
    lim1 = 40;
    y1= 58
    plot(lim1:lim2,diffi(lim1:lim2,y1), 'LineWidth', 2.5);
    hold on
    plot(lim1:lim2,matlab_matrix(lim1:lim2,y1), 'LineWidth', 2.5);
    hold on;
    legend('h cubic mobility', 'h sharp interface')
    plot(lim1:lim2,-z(lim1:lim2,y1),'w', 'LineWidth', 2.5);
     plot(lim1:lim2,-z(lim1:lim2,y1), 'Color', [0.87 0.49 0], 'LineWidth', 1.2);
    set(gca,'FontSize',17)
    hold on
    %plot(40:lim2,thick(40:65,y1), 'Color', [0.87 0.49 0], 'LineWidth', 1.2);
    %caxis([mini maxi]);
    view(2)
    
end
