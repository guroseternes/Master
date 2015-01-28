%% Simulate long term migration on the Utsira formation
% This example demonstrates simulation of long-term migration of CO2 in the
% Utsira formation using incompressible flow and Dirichlet pressure
% boundary conditions. CO2 is injected in the cell nearest to the Sleipner
% field where CO2 injection is ongoing.

%% Set up fluid properties and hydrostatic pressure
% We define approximate hydrostatic pressure for a set of z values to use
% as initial values:
%
% $$ P = P_0 + rho_{water}\cdot \delta h g_z $$
%
% At the same time, we define the physical properties of CO2 and water at
% our reference pressure.
%
% We also define a coarsening of the full Utsira grid. The full grid
% contains a fairly large number of cells, so we coarse by a factor 3. This
% can be changed to a higher  or lower number depending on the available
% processing power and the patience of the user.
%% Display header
clc;
disp('================================================================');
disp('   Preparing data from the Utsira formation');
disp('================================================================');
disp('');


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

% Default viewing angle
v = [-80, 80];
 
% Coarsening factor
nc = 2;

%% Set up injection parameters
% We inject a yearly amount of 1 Mt/year (which is approximately the same
% as the current injection at Sleipner) for a period of 100 years, followed
% by 4900 years of migration.  During injection the timesteps are smaller
% than during migration. 

% ~ 1Mt annual injection
rate = 20e9*kilogram/(year*rhoc*kilogram*meter^3);
%rate = 1e9*kilogram/(year*rhoc*kilogram*meter^3);

%% Set up a grid corresponding to the Utsira formation
% The Sleipner field has a history of CO2 injection. It is embedded in the
% Utsira formation. We will demonstrate how the larger formation grids can
% be used to simulate long term migration.

%[grids, info, petrodata] = getAtlasGridWithInterpolation('Utsirafm', 'nz', 1, 'coarsening', nc);
[grids, info, petrodata] = getAtlasGrid('Utsirafm', 'nz', 1, 'coarsening', nc);

% Store heightmap data for contour plots
info = info(cellfun(@(x) strcmpi(x.variant, 'top'), info));
info = info{1};
clearvars info

G = processGRDECL(grids{1});

clearvars grids{1}
% Depending on the coarsening factor, occasionally very small subsets of
% cells may become disconnected. We guard against this by taking only the
% first grid returned by processGRDECL,  which is guaranteed by
% processGRDECL to be the one with the most cells.
G = G(1);
try
    % Try accelerated geometry calculations.
    G = mcomputeGeometry(G);
catch ex %#ok
    % Fall back to pure MATLAB code.
    G = computeGeometry(G);
end

[Gt, G] = topSurfaceGrid(G);


%% Set up rock properties and compute transmissibilities
% We use the averaged values for porosity and permeability as given in the
% Atlas tables. Since cellwise data is not present, we assume to averaged
% values to be valid everywhere.
pd = petrodata{1};

rock.poro = repmat(pd.avgporo, G.cells.num, 1);
rock.perm = repmat(pd.avgperm, G.cells.num, 1);
rock2D    = averageRock(rock, Gt);

%% Set up fluid
fluidVE = initVEFluidHForm(Gt, 'mu' , [muc muw], ...
                             'rho', [rhoc rhow], ...
                             'sr', srco2, 'sw', sw, 'kwm', kwm);
                            
%% Set up well and boundary conditions
% This example is using an incompressible model for both rock and fluid. If
% we assume no-flow on the boundary, this will result in zero flow from a
% single injection well. However, this can be compensated if we use the
% physical understanding of the problem to set appropriate boundary
% conditions: The Utsira formation is enormous compared to the volume of
% the injected CO2. Thus, it is impossible that the injection will change
% the composition of the formation significantly. We therefore assume that
% the boundary conditions can be set equal to hydrostatic pressure to drive
% flow.

% Approximate position of the Sleipner field
sleipnerPos = [438514, 6472100];

% Approximate position of additional wells
Pos2 = [449314,6474100];
Pos3 = [459314,6594100];
Pos4 = [479514,6794100];
Pos5 = [473514,6714100];
Pos6 = [481614,6614100];
Pos7 = [491614,6614100];
Pos8 = [491614,6574210];
Pos9 = [475614,6474210];
Pos10 = [449314,6514100];
Pos11 = [459314,6514100];
Pos12 = [471314,6574100];
Pos13 = [491314,6674100];
Pos14 = [473514,6774100];
Pos15 = [483514,6734100];

% Find the cell nearest to the well positions
si = findEnclosingCell(Gt, sleipnerPos);
si2 = findEnclosingCell(Gt, Pos2);
si3 = findEnclosingCell(Gt, Pos3);
si4 = findEnclosingCell(Gt, Pos4);
si5 = findEnclosingCell(Gt, Pos5);
si6 = findEnclosingCell(Gt, Pos6);
si7 = findEnclosingCell(Gt, Pos7);
si8 = findEnclosingCell(Gt, Pos8);
si9 = findEnclosingCell(Gt, Pos9);
si10 = findEnclosingCell(Gt, Pos10);
si11 = findEnclosingCell(Gt, Pos11);
si12 = findEnclosingCell(Gt, Pos12);
si13 = findEnclosingCell(Gt, Pos13);
si14 = findEnclosingCell(Gt, Pos14);
si15 = findEnclosingCell(Gt, Pos15);

% Add the injector wells for the CO2
W = addWell([], G, rock, si,...
   'Type', 'rate', 'Val', rate, 'comp_i', [1,0], 'name', 'Sleipner field');
W = addWell(W, G, rock, si2,...
   'Type', 'rate', 'Val', rate, 'comp_i', [1,0], 'name', 'Sleipner field');
W = addWell(W, G, rock, si3,...
   'Type', 'rate', 'Val', rate, 'comp_i', [1,0], 'name', 'Sleipner field');
W = addWell(W, G, rock, si4,...
   'Type', 'rate', 'Val', rate, 'comp_i', [1,0], 'name', 'Sleipner field');
W = addWell(W, G, rock, si5,...
   'Type', 'rate', 'Val', rate, 'comp_i', [1,0], 'name', 'Sleipner field');
W = addWell(W, G, rock, si6,...
   'Type', 'rate', 'Val', rate, 'comp_i', [1,0], 'name', 'Sleipner field');
W = addWell(W, G, rock, si7,...
   'Type', 'rate', 'Val', rate, 'comp_i', [1,0], 'name', 'Sleipner field');
W = addWell(W, G, rock, si8,...
   'Type', 'rate', 'Val', rate, 'comp_i', [1,0], 'name', 'Sleipner field');
W = addWell(W, G, rock, si9,...
   'Type', 'rate', 'Val', rate, 'comp_i', [1,0], 'name', 'Sleipner field');
W = addWell(W, G, rock, si10,...
   'Type', 'rate', 'Val', rate, 'comp_i', [1,0], 'name', 'Sleipner field');
W = addWell(W, G, rock, si11,...
   'Type', 'rate', 'Val', rate, 'comp_i', [1,0], 'name', 'Sleipner field');
W = addWell(W, G, rock, si12,...
   'Type', 'rate', 'Val', rate, 'comp_i', [1,0], 'name', 'Sleipner field');
W = addWell(W, G, rock, si13,...
   'Type', 'rate', 'Val', rate, 'comp_i', [1,0], 'name', 'Sleipner field');
W = addWell(W, G, rock, si14,...
   'Type', 'rate', 'Val', rate, 'comp_i', [1,0], 'name', 'Sleipner field');
W = addWell(W, G, rock, si15,...
   'Type', 'rate', 'Val', rate, 'comp_i', [1,0], 'name', 'Sleipner field');

% Add pressure boundary 
bnd = boundaryFaces(Gt);
bcVE = addBC([], bnd, 'pressure', pressure(Gt.faces.z(bnd)), 'sat', [0 1]);
bcVE = rmfield(bcVE,'sat');
bcVE.h = zeros(size(bcVE.face));

% Convert to 2D wells
WVE = convertwellsVE(W, G, Gt, rock2D);

clearvars G 

%%  Set up initial reservoir conditions
% The initial pressure is set to hydrostatic pressure. Setup and plot.
SVE = computeMimeticIPVE(Gt, rock2D, 'Innerproduct','ip_simple');
preComp = initTransportVE(Gt, rock2D);

sol.wellSol = initWellSol(W, 0);
sol = initResSolVE(Gt, 0, 0);
sol.s = height2Sat(sol, Gt, fluidVE);


grav = [0, 0, 9.806649999999999];
 
%% Convert MATLAB data to a suitable format for the GPU solver after injection phase

[east_face_index, north_face_index] = prepareDataForGPU(sol, Gt, rock, rock2D, fluidVE, bcVE, WVE, preComp, ...
    './SimulationData/FormationData/Utsira/data', './SimulationData/FormationData/Utsira/dimensions',...
    './SimulationData/FormationData/Utsira/active_cells');

 save('./SimulationData/FormationData/Utsira/variablesForRunningPressureSolver.mat','grav', 'sol', 'Gt', 'SVE', 'rock', 'fluidVE', ...
     'bcVE', 'WVE', 'east_face_index', 'north_face_index');


