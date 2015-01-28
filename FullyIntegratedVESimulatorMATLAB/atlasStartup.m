moduleCheck('co2lab', 'deckformat');

fprintf('Loading atlas data (this may take a few minutes)..');
[grdecls, rawdata] = getAtlasGrid(); %#ok
fprintf('done\n');

%% Description of raw data
% Show the raw data. Each dataset contains four fields:
% - Name, which is the name of the formation
% - Variant: Either thickness or height, indicating wether the dataset
% represents height data or thickness data.
% - Data: The actual datasets as a matrix.
% - Meta: Metadata. The most interesting field here is the
%   xllcorner/yllcorner variable which indicates the position in ED50
%   datum space.

fprintf('\nRaw data:\n')
fprintf('----------------------------------------------------------------\n');
for i=1:numel(rawdata);
    rd = rawdata{i};
    fprintf('Dataset %-2i is %-12s (%-9s). Resolution: %4i meters\n', ...
            i, rd.name, rd.variant,  rd.meta.cellsize)
end
fprintf('----------------------------------------------------------------\n');

% Store names for convenience
names = cellfun(@(x) x.name, rawdata, 'UniformOutput', false)';

%% Show the data directly: Utsira formation
% The datasets are perfectly usable for visualization on their own. To see
% this, we find the datasets corresponding to the Utsira formation and plot
% both the thickness and the heightmap.
%
% Note that the datasets are not entirely equal: Some sections are not
% included in the thickness map and vice versa. In addition to this, the
% coordinates are not always overlapping, making interpolation neccessary.
%
% Some formations are only provided as thickness maps; These are processed
% by sampling the relevant part of the Jurassic formation for top surface
% structure.

utsira_rd = rawdata(strcmpi(names, 'Utsirafm'));
clf;
for i = 1:numel(utsira_rd)
    urd = utsira_rd{i};
    
    subplot(2,1,i)
    surf(urd.data)
    
    title([urd.name ' ' urd.variant])
    shading interp
    view(0,90)
    axis tight off
end


%% Visualize all the formations
% We then visualize the formations along with a map of Norway and point
% plots of all production wells in the Norwegian Continental Shelf.
%
% The well data comes from the Norwegian Petroleum Directorate and can be
% found in more detail at http://factpages.npd.no/factpages/.
%
% The map of Norway comes from The Norwegian Mapping and Cadastre Authority
% and can be found at http://www.kartverket.no/. Note that the map is only
% provided for scale and rough positioning - no claims are made regarding
% the accuracy in relation the subsea reservoirs.
%
% To visualize the formations, we load a 5x5 coarsened version of each data
% set and use this to create a simple volumetric model that represents
% approximately the outline of each formation. More details about how to
% create 3D grid models are given in the script 'modelsFromAtlas.m'

grdecls = getAtlasGrid('coarsening', 1);
ng = numel(grdecls);

grids = cell(ng,1);
gd = processGRDECL(grdecls{1});
grid = computeGeometry(gd(1));
G = grid
convertDataToCppGrid(G, 'heightAndTopSurface.mat');
