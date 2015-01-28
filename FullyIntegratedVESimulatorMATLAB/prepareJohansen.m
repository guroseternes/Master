%% Vertical-Averaged Simulation of the Johansen Formation
% The Johansen formation is a candidate site for large-scale CO2 storage
% offshore the south-west coast of Norway. In the following, we will use a
% simple vertically averaged model to simulate the early-stage migration of
% a CO2 plume injected from a single well positioned near the main fault in
% the formation. The formation is described by a geological model that has
% been developed based on available seismic and well data. A more thorough
% presentation of the geological model can be found in the script
% <matlab:edit('showJohansen.m') showJohansen.m>
%
% The data files
% necessary to run the example can be downloaded from the
% <http://www.sintef.no/Projectweb/MatMorA/Downloads/Johansen/ MatMoRA
% website>.
moduleCheck('mimetic');

%% Display header
clc;
disp('================================================================');
disp('   Preparing data form the Johansen formation');
disp('================================================================');
disp('');

%% Input data and construct grid models
% We use a sector model in given in the Eclipse input format (GRDECL). The
% model has five vertical layers in the Johansen formation and five shale
% layers above and one below in the Dunhil and Amundsen formations. The
% shale layers are removed and we construct the 2D VE grid of the top
% surface, assuming that the major fault is sealing, and identify all outer
% boundaries that are open to flow. Store grid and rock structures to file
% to avoid time-consuming processing.
[G, Gt, rock, rock2D, bcIxVE] = makeJohansenVEgrid();

%% Set time and fluid parameters
gravity on
T1 = 0*year();

% Fluid data at p = 300 bar
muw = 0.30860;  rhow = 975.86; sw    = 0.1;
muc = 0.056641; rhoc = 686.54; srco2 = 0.2;
kwm = [0.2142 0.85];

fluidVE = initVEFluidHForm(Gt, 'mu' , [muc muw] .* centi*poise, ...
                             'rho', [rhoc rhow] .* kilogram/meter^3, ...
                             'sr', srco2, 'sw', sw, 'kwm', kwm);

%% Set well and boundary conditions
% We use one well placed in the center of the model, perforated in layer 6.
% Injection rate is 1.4e4 m^3/day of supercritical CO2. Hydrostatic
% boundary conditions are specified on all outer boundaries that are not in
% contact with the shales; the latter are assumed to be no-flow boundaries.

% Set well in 3D model
wellIx = [51, 51, 6, 6];
rate = 1.4e4*meter^3/day;
W = verticalWell([], G, rock, wellIx(1), wellIx(2), wellIx(3):wellIx(4),...
   'Type', 'rate', 'Val', rate, 'Radius', 0.1, 'comp_i', [1,0], 'name', 'I');

% Well and BC in 2D model
WVE = convertwellsVE(W, G, Gt, rock2D);

bcVE = addBC([], bcIxVE, 'pressure', Gt.faces.z(bcIxVE)*rhow*norm(gravity));
bcVE = rmfield(bcVE,'sat');
bcVE.h = zeros(size(bcVE.face));

%% Prepare simulations
% Compute inner products and instantiate solution structure
% For the transport simulation, the default choice is to use a
% C-accelerated solver, but if this does not work, we use the standard
% solver from MRST.
SVE = computeMimeticIPVE(Gt, rock2D, 'Innerproduct','ip_simple');
preComp = initTransportVE(Gt, rock2D);
sol = initResSolVE(Gt, 0, 0);
sol.wellSol = initWellSol(W, 300*barsa());
sol.s = height2Sat(sol, Gt, fluidVE);
% select transport solver
try
   mtransportVE;
   cpp_accel = true;
catch me
   d = fileparts(mfilename('fullpath'));
   disp('mex-file for C++ acceleration not found');
   disp(['See ', fullfile(mrstPath('co2lab'),'ve','VEmex','README'),...
      ' for building instructions']);
   disp('Using matlab ve-transport');
   cpp_accel = false;
end

sol.h_max = sol.h;
sol = solveIncompFlowVE(sol, Gt, SVE, rock, fluidVE, ...
  'bc', bcVE, 'wells', WVE);  

% Reconstruct 'saturation' defined as s=h/H, where h is the height of
% the CO2 plume and H is the total height of the formation
sol.s = height2Sat(sol, Gt, fluidVE);
assert( max(sol.s(:,1))<1+eps && min(sol.s(:,1))>-eps );
   
% delete C++ simulator
if cpp_accel, mtransportVE(); end

%% Convert MATLAB data to a suitable format for the GPU solver after injection phase
[east_face_index, north_face_index] = prepareDataForGPU(sol, Gt, rock, rock2D, fluidVE, bcVE, WVE, preComp, ...
    './SimulationData/FormationData/Johansen/data', './SimulationData/FormationData/Johansen/dimensions', ...
    './SimulationData/FormationData/Johansen/active_cells');

 save('./SimulationData/FormationData/Johansen/variablesForRunningPressureSolver.mat','grav', 'sol', 'Gt', 'SVE', 'rock', 'fluidVE', ...
     'bcVE', 'WVE','east_face_index', 'north_face_index');
