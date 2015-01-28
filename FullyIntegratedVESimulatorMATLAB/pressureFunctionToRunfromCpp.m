function [sources, east_flux, north_flux] = pressureFunctionToRunfromCpp(h_matrix, variables, open_well)

    SVE = variables.SVE;
    rock = variables.rock;
    fluidVE = variables.fluidVE;
    bcVE = variables.bcVE;
    WVE = variables.WVE; 
    gravity on
        
    ij = variables.Gt.cells.ij;
    nCells = variables.Gt.cells.num;
    len = length(ij(:,1)); 
    border = 1;

     xdim = max(ij(:,1)) - min (ij(:,1)) + 1;
     ydim = max(ij(:,2)) - min (ij(:,2)) + 1;

    ij(:,1) = ij(:,1)-min (ij(:,1))+1;
    ij(:,2) = ij(:,2)-min (ij(:,2))+1;
    
    h_cell_format = zeros(nCells,1);
    
    idx = sub2ind(size(h_matrix), ij(:,1), ij(:,2))
    h_cell_format(1:len) = h_matrix(idx);
  
    sol.h = h_cell_format;
    sol.h_max = h_cell_format;
     
    if open_well == false
        bcVE = [];
        WVE = [];
    end
    sol1 = solveIncompFlowVE(variables.sol, variables.Gt, SVE, rock, fluidVE, ...
     'bc', bcVE, 'wells', WVE);
    
    is_int   = all(double(variables.Gt.faces.neighbors) > 0, 2);
    q = full(sources_cpp(variables.Gt, sol1, is_int, 'bc', bcVE, 'wells', WVE));
 
    east_flux = zeros(xdim + 2*border, ydim + 2*border);
    north_flux = zeros(xdim + 2*border, ydim + 2*border);
    sources = zeros(xdim, ydim);

    idx2 = sub2ind(size(east_flux), ij(:,1)+1, ij(:,2)+1);
    east_flux(idx2) = sol1.flux(variables.east_face_index);
    north_flux(idx2) = sol1.flux(variables.north_face_index);
    sources(idx) = q;
  
%     for k = 1:len
%         % Find indexes
%         i = ij(k,1);
%         j = ij(k,2);
%         current_faces = Gt.cells.faces(Gt.cells.facePos(k):Gt.cells.facePos(k+1)-1,1);
%         for l=1:4
%             current_face = current_faces(l);
%                 f=Gt.faces.neighbors(current_face,:);
%             if (f(1) == k && diff(f) == 1)
%                 east_face = current_face;
%             end
%             if (f(1) == k && diff(f) > 1)
%                 north_face = current_face;
%             end
%         end
%         north_flux(i+1,j+1) = sol1.flux(north_face);
%         east_flux(i+1,j+1) = sol1.flux(east_face);
%         sources(i,j) = q(k);
%         if (i==1)
%             east_flux(i,j+1) = -sol1.flux(current_faces(1));
%         end
%         if (j==1)
%             north_flux(i+1,j) = -sol1.flux(current_faces(2));
%         end
%     end

    %filename =['/home/guro/cuda-workspace/ExplicitTransportSolverGPU/fluxes.mat'];
    sources = single(sources);
    north_flux = single(north_flux);
    east_flux = single(east_flux);

end

  
  
  
  
  