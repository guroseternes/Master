%% File for preparing IGEMS for GPU

%% Set fluid properties and hydrostatic pressure
muw = 0.30860;  rhow = 975.86; sw    = 0.1;
muc = 0.056641; rhoc = 686.54; srco2 = 0.2;
kwm = [0.2142 0.85];

mu  = [muc  muw ] .* centi*poise;
rho = [rhoc rhow] .* kilogram/meter^3;

% Define reservoar top 
topPos = 300*meter;
topPressure = 300*barsa;

% Turn on gravity 
gravity on;
grav = gravity();

% Pressure function
pressure = @(z) topPressure + rho(2)*(z - topPos)*grav(3);

rate = 1.0e6*meter^3/year;

%% Read and prepare grid structure
% The project developed both full 3D grid saved in the ECLIPSE format and
% surfaces saved in the IRAP formate.  The ECLIPSE files are huge (588 MB)
% and reading and processing them typically requires a computer with at
% least 12 GB memory. Here, we will therefore only use the surface grid.
Gt = topSurfaceGrid( readIGEMSIRAP('OSSUP1', 3, 'coarse', [3 3]) );

rock.poro = 0.25;
rock.perm = 500*(1/1000)*(9.869233*10^(-13));
rock2D.poro = rock.poro*ones(Gt.cells.num,1); 
rock2D.perm = rock.perm*ones(Gt.cells.num,1);

fluidVE = initVEFluidHForm(Gt, 'mu' , [muc muw], ...
                             'rho', [rhoc rhow], ...
                             'sr', srco2, 'sw', sw, 'kwm', kwm);
wellPos = [15000, 15000];
si = findEnclosingCell(Gt, wellPos);

WVE = addWell([], Gt, rock2D, si,...
   'Type', 'rate', 'Val', rate, 'comp_i', [1,0], 'name', 'Igems field'); 
WVE.h = Gt.cells.H(WVE.cells);                           %#ok
WVE.dZ = Gt.cells.H(WVE.cells)*0.0;     

%rock.poro =  ones(Gt.)

bnd = boundaryFaces(Gt);
bcVE = addBC([], bnd, 'pressure', pressure(Gt.faces.z(bnd)), 'sat', [0 1]);
bcVE = rmfield(bcVE,'sat');
bcVE.h = zeros(size(bcVE.face));

% Convert to 2D wells
%WVE = convertwellsVE_s(W, G, Gt, rock2D,'ip_tpf');


%%  Set up initial reservoir conditions
% The initial pressure is set to hydrostatic pressure. Setup and plot.
SVE = computeMimeticIPVE(Gt, rock2D, 'Innerproduct','ip_simple');
preComp = initTransportVE(Gt, rock2D);

%sol = initResSolVE_s(Gt, pressure(Gt.cells.z), 0);
sol.wellSol = initWellSol(WVE, 0);
sol = initResSolVE(Gt, 0, 0);
%sol.wellSol = initWellSol(W, 300*barsa());
sol.s = height2Sat(sol, Gt, fluidVE);


grav = [0, 0, 9.806649999999999];

%% Get a sub-index
ij = Gt.cells.ij;
nCells = Gt.cells.num;
len = length(ij(:,1)); 
border = 1;

xdim = max(ij(:,1)) - min (ij(:,1)) + 1;
ydim = max(ij(:,2)) - min (ij(:,2)) + 1;

ij(:,1) = ij(:,1)-min (ij(:,1))+1;
ij(:,2) = ij(:,2)-min (ij(:,2))+1;
    
 [east_face_index, north_face_index] = prepareDataForGPU_IGEMS(sol, Gt, rock, fluidVE, bcVE, WVE, preComp, ...
    './IGEMS_Data/igemsdata', './IGEMS_Data/igems_dimensions', './IGEMS_Data/igems_active_cells');
 save('./IGEMS_Data/variablesForRunningPressureSolver.mat','grav', 'sol', 'Gt', 'SVE', 'rock', 'fluidVE', ...
     'bcVE', 'WVE', 'east_face_index','north_face_index');
