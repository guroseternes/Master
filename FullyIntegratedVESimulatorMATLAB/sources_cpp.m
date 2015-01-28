function q = sources_cpp(G_top, state, is_int,  varargin)
% Contributions from wells, sources and boundary conditions
   opt = struct('wells'    , [],          ...
                'src'      , [],          ...
                'bc'       , []);

   opt = merge_options(opt, varargin{:});
   %-----------------------------------------------------------------------
   % External sources/sinks (e.g. wells and BC's) ------------------------
   %
   qi = [];  % Cells to which sources are connected
   qh = [];  % Actual strength of source term (in m^3/s).

   if ~isempty(opt.wells),
      assert (isfield(state, 'wellSol'));
      [ii, h] = contrib_wells(opt.wells, state.wellSol,G_top);
      qi = [qi; ii];
      qh = [qh; h];
   end

   if ~isempty(opt.src), assert (~isempty(opt.src.h))
      [ii, h] = contrib_src(opt.src,G_top);
      qi = [qi; ii];
      qh = [qh; h];
   end

   if ~isempty(opt.bc), assert (~isempty(opt.bc.h))
      [ii, h] = contrib_bc(G_top, state, opt.bc, is_int);
      qi = [qi; ii];
      qh = [qh; h];
   end

   %-----------------------------------------------------------------------
   % Assemble final phase flux and source contributions in SPARSE format -
   %
   q  = sparse(qi, 1, qh, G_top.cells.num, 1);
end

%--------------------------------------------------------------------------
function [qi, qh] = contrib_wells(W, wellSol,g_top)
   % Contributions from wells as defined by 'addWell'.

   nperf = cellfun(@numel, { W.cells }) .';

   qi = vertcat(W.cells);
   qh = vertcat(wellSol.flux);

   % Injection perforations have positive flux (into reservoir).
   %
   %assert(nperf==1);
   comp      = rldecode(vertcat(W.h), nperf);
   inj_p     = qh > 0;
   qh(inj_p) = qh(inj_p).*comp(inj_p)./g_top.cells.H(qi(inj_p));
end

%--------------------------------------------------------------------------

function [qi, qh] = contrib_src(src,g_top)
   % Contributions from explicit sources defined by (e.g.) 'addSource'.

   qi = src.cell;
   qh = src.rate;

   % Injection sources have positive rate into reservoir.
   %
   in = find(src.rate > 0);
   if ~isempty(in),
      qh(in) = qh(in) .* src.h(in,1)./g_top.cells.H(qi(in));
   end
end

%--------------------------------------------------------------------------

function [qi, qh] = contrib_bc(G, resSol, bc, is_int)
   % Contributions from boundary conditions as defined by 'addBC'.

   qh = zeros([G.faces.num, 1]);
   dF = false([G.faces.num, 1]);

   isDir = strcmp('pressure', bc.type);
   isNeu = strcmp('flux',     bc.type);

   dF(bc.face(isDir))      = true;
   cfIx                    = dF(G.cells.faces(:,1));

   cflux = faceFlux2cellFlux(G, resSol.flux);
   qh(G.cells.faces(cfIx,1)) = -cflux(cfIx);
   qh(bc.face(isNeu))        = bc.value(isNeu);

   % Injection BC's have positive rate (flux) into reservoir.
   %
   is_inj = qh > 0;
   if any(is_inj),
      qh(is_inj) = qh(is_inj) .* bc.h(is_inj(bc.face), 1);
   end

   is_outer = ~is_int;

   qi = sum(G.faces.neighbors(is_outer,:), 2);
   qh = qh(is_outer);
end