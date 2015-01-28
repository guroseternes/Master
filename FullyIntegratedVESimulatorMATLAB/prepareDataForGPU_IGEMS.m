function [east_face_index, north_face_index] = prepareDataForGPU_IGEMS(sol, Gt, rock, fluidVE, bcVE, WVE, preComp, filenameData, filenameDim, filenameActive)
%close all
gravity = 9.80665
zdim = 100;
border = 1;

%% First make a rectangular 2D carthesian grid of the data
ij = Gt.cells.ij;
len = length(ij(:,1)); 

xdim = max(ij(:,1)) - min (ij(:,1)) + 1
ydim = max(ij(:,2)) - min (ij(:,2)) + 1

ij(:,1) = ij(:,1)-min (ij(:,1))+1;
ij(:,2) = ij(:,2)-min (ij(:,2))+1;

is_int   = all(double(Gt.faces.neighbors) > 0, 2);

H = zeros(xdim, ydim);
[xx yy] = size(H);
h = zeros(xdim, ydim);
z = zeros(xdim, ydim);
pv = zeros(xdim, ydim);
active_east= zeros(xdim, ydim);
active_north= zeros(xdim, ydim);
averageperm = zeros(xdim, ydim);

east_flux = zeros(xdim + 2*border, ydim + 2*border);
north_flux = zeros(xdim + 2*border, ydim + 2*border);

north_face_index = zeros(len,1);
east_face_index = zeros(len,1);

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
%    source(i,j) = q(k);

    normal_z(i,j) = preComp.n_z(k);
    pv(i,j) = preComp.pv(k);
    
    % Get permeabilities and porosities in 3D matrix
    
    
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
    
    
%     len = length(rock.perm3D(i,j,:))
%     if len > 1
%         perm_dz = H(i,j)/(len);   
%         col_z_values =  [0:perm_dz:H];
%         col_perm_values = rock.perm3D(i,j,:);
%     else 
%         col_z_values = H;
%         col_perm_values = rock.perm3D(i,j,1);
%     end
%     col_perm_values = col_perm_values';
    z_intervals = [0:dz:cells.H(k)];
    if z_intervals(end) < cells.H(k)
      z_intervals = [z_intervals cells.H(k)];
    end
%     perm = z_intervals*0;
%     poro = z_intervals*0;
%     for l=1:length(col_perm_values)
%         from = col_z_values(l);
%         to = col_z_values(l+1)+0.00001;
%         I = (z_intervals>=from&z_intervals<to);
%         perm = I*col_perm_values(l) + perm;
%     end
    
    perm = z_intervals*rock.perm;
    poro = z_intervals*rock.poro;
    averageperm(i,j) = sum(perm(2:end).*diff(z_intervals))/cells.H(k);
    perm3D(i,j,1:length(perm)) = perm;
    poro3D(i,j,1:length(poro)) = poro;%poro*ones(length(poro),1)*poro2D(i,j);
    
    
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

poro3D = permute(poro3D, [3 2 1]); 
perm3D = permute(perm3D, [3 2 1]); 
%surf(active_east')
view(2)
z=single(z);
H=single(H);

h=single(h);
normal_z = single(normal_z);
poro3D = single(poro3D);
perm3D = single(perm3D);
pv = single(pv);
north_flux = single(north_flux);
east_flux = single(east_flux);
north_grav = single(north_grav);
east_grav = single(east_grav);
north_K_face = single(north_K_face);
east_K_face = single(east_K_face);
zdim = single(zdim);
dz = single(dz);
active_east = single(active_east);
active_north = single(active_north);
xdim = single(xdim);
zdim = single(zdim);
ydim = single(ydim);

save(filenameData,'H', 'z', 'h','normal_z','perm3D','poro3D','pv','north_flux', 'east_flux', 'north_grav', 'east_grav', 'north_K_face', 'east_K_face', 'dz'); %%, 'max_height');        
save(filenameDim, 'xdim', 'ydim','zdim');
save(filenameActive, 'active_east', 'active_north');


end
    