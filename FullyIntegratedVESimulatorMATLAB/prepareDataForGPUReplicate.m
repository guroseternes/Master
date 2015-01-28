function [east_face_index, north_face_index] = prepareDataForGPUAbsPerformance(sol, Gt, rock, rock2D, fluidVE, bcVE, WVE, preComp,filenameData, filenameDim, filenameActive)
%close all
gravity = 9.80665
zdim = 100;
border = 1;
tiling = 4;


%% First make a rectangular 2D carthesian grid of the data
ij = Gt.cells.ij;
len = length(ij(:,1)); 

xdim = max(ij(:,1)) - min (ij(:,1)) + 1;
ydim = max(ij(:,2)) - min (ij(:,2)) + 1;

ij(:,1) = ij(:,1)-min (ij(:,1))+1;
ij(:,2) = ij(:,2)-min (ij(:,2))+1;

is_int   = all(double(Gt.faces.neighbors) > 0, 2);

H = zeros(xdim, ydim);
h = zeros(xdim, ydim);

[xx yy] = size(H);
z = zeros(xdim, ydim);
pv = zeros(xdim, ydim);
active_east= zeros(xdim, ydim);
active_north= zeros(xdim, ydim);
averageperm = zeros(xdim, ydim);
averagepermFromRock2D = zeros(xdim, ydim);
east_flux = zeros(xdim + 2*border, ydim + 2*border);
north_flux = zeros(xdim + 2*border, ydim + 2*border);
source = zeros(xdim, ydim);
east_grav_old = zeros(xdim, ydim);
north_grav_old = zeros(xdim, ydim);
east_grav = zeros(xdim, ydim);
north_grav = zeros(xdim, ydim);
north_K_face = zeros(xdim, ydim);
east_K_face = zeros(xdim, ydim);

normal_z = zeros(xdim, ydim); 
perm3D = zeros(xdim, ydim, zdim+1);
poro3D = zeros(xdim, ydim, zdim+1);

is_int = all(double(Gt.faces.neighbors) > 0, 2);
cells = Gt.cells;
max_height = max(cells.H);
dz = max_height/zdim;
perm_values = rock.perm(:,1);
poro_values = rock.poro;
allCellsVec = -1*ones([Gt.faces.num, 1]);
allCellsVec(is_int) = 0;



for k = 1:len
    % Find indexes
    i = ij(k,1);
    j = ij(k,2);
    current_faces = Gt.cells.faces(Gt.cells.facePos(k):Gt.cells.facePos(k+1)-1,1);
    
    % Put H,z and h into a 2D cartesian matrix
    H(i,j) = cells.H(k);
    z(i,j) = cells.z(k);
    h(i,j) = sol.h(k);
    
    normal_z(i,j) = preComp.n_z(k);
    pv(i,j) = preComp.pv(k);
    
    % Get permeabilities and porosities in 3D matrix
    col_z_values =  [0; Gt.columns.z(cells.columnPos(k):cells.columnPos(k+1)-1,:)];
    col_dz_values =  [0; Gt.columns.dz(cells.columnPos(k):cells.columnPos(k+1)-1,:)];
    cellIndexes = Gt.columns.cells(cells.columnPos(k):cells.columnPos(k+1)-1,:);
    col_perm_values = 0;
    col_poro_vlues = 0;
    for l=1:length(cellIndexes)
        col_perm_values(l) = perm_values(cellIndexes(l));        
        col_poro_values(l) = poro_values(cellIndexes(l));
    end
    col_perm_values = col_perm_values';
    col_poro_values = col_poro_values';
    z_intervals = [0:dz:cells.H(k)];
    if z_intervals(end) < cells.H(k)
        z_intervals = [z_intervals cells.H(k)];
    end
    perm = z_intervals*0;
    poro = z_intervals*0;
    for l=1:length(col_perm_values)
        from = col_z_values(l);
        to = col_z_values(l+1)+0.00001;
        I = (z_intervals>=from&z_intervals<to);
        perm = I*col_perm_values(l) + perm;
        poro = I*col_poro_values(l) + poro;
    end
    averagepermFromRock2D(i,j) = rock2D.perm(k);
    averageperm(i,j) = sum(perm(2:end).*diff(z_intervals))/cells.H(k);
    perm3D(i,j,1:length(perm)) = perm;
    variance_of_perm(i,j)= var(perm);
    poro3D(i,j,1:length(poro)) = rock2D.poro(k);%poro;
    
    %%Get north and south fluxes
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
    
    east_face_index(k) = east_face;
    north_face_index(k) = north_face;
    
    east_flux(i+1,j+1) = sol.flux(east_face);
    north_flux(i+1,j+1) = sol.flux(north_face);
    east_grav(i,j) = preComp.g_cell(east_face);
    north_grav(i,j) = preComp.g_cell(north_face);
    east_K_face(i,j) = preComp.K_cell(east_face);
    north_K_face(i,j) = preComp.K_cell(north_face); 
    activeNorth = allCellsVec(north_face);
    activeEast = allCellsVec(east_face);
    active_east(i,j) = activeEast;
    active_north(i,j) = activeNorth;
      
    if (i==1)
        east_flux(i,j+1) = -sol.flux(current_faces(1));
    end
    if (j==1)
        north_flux(i+1,j) = -sol.flux(current_faces(2));
    end
    
end

poro3D = repmat(poro3D,tiling);
perm3D = repmat(perm3D, tiling);

poro3D = permute(poro3D, [3 2 1]); 
perm3D = permute(perm3D, [3 2 1]); 
%surf(active_east')
view(2)
z=single(z);
z = repmat(z,tiling);
H=single(H);
H = repmat(H,tiling);

h=single(h);
h = repmat(h,tiling);
normal_z = single(normal_z);
normal_z = repmat(normal_z,tiling);
poro3D = single(poro3D);
perm3D = single(perm3D);
pv = single(pv);
pv = repmat(pv,tiling);
north_flux = zeros(xdim*tiling+2, ydim*tiling+2);
north_flux = single(north_flux);
east_flux = zeros(xdim*tiling+2, ydim*tiling+2);
east_flux = single(east_flux);
source = single(source);
source = repmat(source,tiling);
north_grav = single(north_grav);
north_grav = repmat(north_grav,tiling);
east_grav = single(east_grav);
east_grav = repmat(east_grav,tiling)
north_K_face = single(north_K_face);
north_K_face = repmat(north_K_face, tiling);
east_K_face = single(east_K_face);
east_K_face = repmat(east_K_face,tiling);
zdim = single(zdim);
dz = single(dz);
active_east = single(active_east);
active_east = repmat(active_east,tiling);
active_north = single(active_north);
active_north = repmat(active_north,tiling);
xdim = single(xdim)*tiling;
zdim = single(zdim);
ydim = single(ydim)*tiling;

save(filenameData,'H', 'z', 'h','normal_z','perm3D','poro3D','pv','north_flux', 'east_flux', 'north_grav', 'east_grav', 'north_K_face', 'east_K_face', 'dz'); %%, 'max_height');        
save(filenameDim, 'xdim', 'ydim','zdim');
save(filenameActive, 'active_east', 'active_north');
save('source', 'source');


end