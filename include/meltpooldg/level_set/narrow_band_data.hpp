#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/linear_algebra/linear_solver_data.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>
#include <meltpooldg/utilities/enum.hpp>

#include <vector>

namespace MeltPoolDG::LevelSet
{
  BETTER_ENUM(NarrowBandType,
              char,
              // band of the cut cells and their face neighbors, whose level set is reconstructed
              // by the extension from the cut cells in every time step
              extension,
              // band of the cut cells and several layers of vertex neighbors, which is advected;
              // the level set of the layers is only reconstructed by geometric redistancing for
              // newly added cells and after the reinitialization of the cut cells
              advected_layers,
              // level set only on the cut cells; semi-Lagrangian advection onto the cut cells and
              // their vertex neighbors (candidate cells), without extension
              semi_lagrangian)

  BETTER_ENUM(NarrowBandAdvectionVelocity,
              char,
              // the given (bulk) velocity u
              bulk,
              // extension velocity u(y): the given velocity at the closest point y = x - phi grad
              // phi / |grad phi|^2 on the zero level set, which is constant along the normals and
              // keeps the level set of the band a signed distance
              extension)

  BETTER_ENUM(NarrowBandLimiterCells,
              char,
              // no limiter
              none,
              // all cells of the band (cut cells and their neighbors or layers)
              all_band_cells,
              // all cells of the band, which are not cut at the beginning of the time step
              non_cut_cells)

  BETTER_ENUM(NarrowBandNeighborActivation,
              char,
              // all face neighbors of the cut cells
              all_face_neighbors,
              // only the face neighbors, into which the velocity points across the face shared
              // with a cut cell (cells, into which the interface can move)
              downstream_face_neighbors)

  BETTER_ENUM(NarrowBandViscosityCells,
              char,
              // all cells of the band
              all_band_cells,
              // all cells of the band, which are not cut at the beginning of the time step
              non_cut_cells)

  BETTER_ENUM(NarrowBandLimiterType,
              char,
              // local maximum principle: the level set remains within [m, M] of the domain of
              // dependence
              maximum_principle,
              // sign preservation: cells, whose domain of dependence has a definite sign, keep this
              // sign
              sign)

  BETTER_ENUM(NarrowBandLimiterBounds,
              char,
              // the cell and all its face neighbors in the band
              face_neighbors,
              // the cell and its face neighbors in the band across inflow faces
              upwind_face_neighbors,
              // the cell and the strips of width |u.n| dt of its face neighbors in the band along
              // the inflow faces (domain of dependence of the time step)
              upwind_strips)

  BETTER_ENUM(NarrowBandExtensionType,
              char,
              // global face-based ghost-penalty extension (polynomial extrapolation)
              ghost_penalty,
              // closest-point extension: first-order Taylor expansion from the closest point on
              // the discrete interface
              closest_point)

  BETTER_ENUM(NarrowBandReinitializationFaces,
              char,
              // SIPG face terms on all faces between two cut cells
              cut_cell_faces,
              // SIPG face terms only on the faces between two cut cells, which are crossed by the
              // interface
              interface_faces)

  BETTER_ENUM(NarrowBandReinitializationInterface,
              char,
              // zero level set of the level set before the reinitialization
              zero_level_set,
              // points of the interface, which are seeded on the initial zero level set and
              // advected with the velocity (front tracking, see NarrowBandInterfacePoints)
              advected_points)

  BETTER_ENUM(NarrowBandUnresolvedCells,
              char,
              // reinitialization with the zero level set of the advected level set in the penalty
              // term (cut cells without usable points)
              zero_level_set,
              // L2-projection of the signed distance to the advected points evaluated at
              // sub-cell quadrature points (see
              // NarrowBandInterfacePoints::project_signed_distance())
              subcell_distance)

  BETTER_ENUM(NarrowBandSubcellCoupling,
              char,
              // the projection replaces the reinitialized level set in the unresolved cells after
              // the reinitialization
              overwrite,
              // the projection is computed before the reinitialization, which keeps it as
              // prescribed values in the unresolved cells and couples the adjacent cells to them
              // by the SIPG face terms
              prescribed,
              // as prescribed, but without the face terms between the projected and the other
              // cells (natural boundary condition of the other cells at these faces)
              excluded)

  BETTER_ENUM(NarrowBandSubcellTransfer,
              char,
              // L2-projection of the signed distance evaluated at the points of the iterated Gauss
              // quadrature (resolves the kink of the distance inside the cells)
              projection,
              // nodal interpolation: the signed distance evaluated at the nodes of the cells
              // (continuous level set space only)
              interpolation)

  BETTER_ENUM(NarrowBandLevelSetSpace,
              char,
              // discontinuous level set (FE_DGQ) on the cut cells
              discontinuous,
              // continuous level set (FE_Q) on the cut cells: reinitialization and subcell distance
              // in the continuous space (see NarrowBandContinuousReinitialization)
              continuous)

  BETTER_ENUM(NarrowBandPointInsertion,
              char,
              // new points in the gaps of the point set from the zero level set of the advected
              // level set (before the reinitialization)
              advected_level_set,
              // new points in the gaps of the point set from the zero level set of the level set
              // at the end of the time step (after the reinitialization, the subcell distance and
              // the closed-interface correction)
              final_level_set)

  BETTER_ENUM(NarrowBandSubcyclingCharacteristics,
              char,
              // backward integration of the characteristics from the current time to the time of
              // the reference over all time steps since the reference (requires the velocity at
              // all these times)
              integration,
              // characteristic map: the displacement since the reference is stored as a continuous
              // field on the candidate cells of the reference and composed in each time step with
              // the backward characteristic of the time step (requires only the velocity of the
              // time step, see NarrowBandCharacteristicMap)
              map)

  BETTER_ENUM(NarrowBandRedistancing,
              char,
              // first-order Taylor expansion phi(y) + grad phi(y) . (x - y) from the closest point
              // y on the discrete interface
              taylor_expansion,
              // signed distance s_K |x - y| to the closest point y on the discrete interface, where
              // the side s_K of the cell is not taken from the cut cells but from the tracked sign
              // of the cell (it only changes, if the interface passes through the cell)
              signed_distance,
              // as signed_distance, but scaled with the gradient magnitude at the closest point,
              // s_K |grad phi(y)| |x - y| (the magnitude of the Taylor expansion, consistent with
              // a level set of the cut cells, which is not a signed distance function)
              scaled_signed_distance)

  BETTER_ENUM(NarrowBandLayerRedistancing,
              char,
              // the layers of the advected band are redistanced after each reinitialization
              after_reinitialization,
              // the layers of the advected band are redistanced in every time step
              every_time_step)

  BETTER_ENUM(NarrowBandGhostPenaltyFaces,
              char,
              // all interior faces of the narrow band with at least one adjacent neighbor cell,
              // i.e., faces between a cut cell and a neighbor cell and faces between two neighbor
              // cells (global coupling of the neighbor cells)
              all_band_faces,
              // only faces between a cut cell and a neighbor cell (decoupled, cell-local
              // extension)
              cut_cell_faces)

  /**
   * @brief Parameters of the narrow-band DG level-set method, where the level-set field is only
   * defined on the cells intersected by the interface (cut cells) and their face neighbors.
   *
   * For the band type NarrowBandType::extension, one time step consists of
   * 1. the extension of the level-set field from the cut cells to their face neighbors by a
   *    global face-based ghost-penalty procedure or a closest-point extension,
   * 2. the advection of the extended level-set field on the narrow band with an explicit upwind
   *    DG scheme,
   * 3. the determination of the new cut cells and, if requested, the elliptic reinitialization
   *    on the new cut cells.
   *
   * For the band type NarrowBandType::advected_layers, the band consists of the cut cells and
   * several layers of vertex neighbors, which are advected together with the cut cells. The level
   * set of the layers is only reconstructed by the geometric redistancing (closest point on the
   * discrete interface) for newly added cells and after the elliptic reinitialization of the cut
   * cells.
   */
  template <typename number = double>
  struct NarrowBandData
  {
    NarrowBandData();

    /// Type of the narrow band.
    NarrowBandType type = NarrowBandType::extension;

    /// Number of layers of vertex neighbors around the cut cells (advected band, at least one).
    unsigned int n_layers = 2;

    /// Parameters of the extension from the cut cells to the neighbor cells.
    struct Extension
    {
      /// Type of the extension.
      NarrowBandExtensionType type = NarrowBandExtensionType::ghost_penalty;

      /// Dimensionless ghost-penalty parameters gamma_k of the jump of the k-th normal derivative,
      /// k = 0, ..., p. If less than p+1 values are given, the last value is used for all
      /// remaining k. The default emphasizes the continuity of the values across the faces,
      /// since the upwind flux of the advection only sees the jump of the values at the inflow
      /// faces of the cut cells.
      std::vector<number> ghost_penalty_parameters = {100.0, 1.0};

      /// Set of faces on which the ghost-penalty terms are evaluated.
      NarrowBandGhostPenaltyFaces ghost_penalty_faces = NarrowBandGhostPenaltyFaces::all_band_faces;

      /// Tolerance of the sign limiter for the extended level-set field on the neighbor cells,
      /// relative to the cell size h. Neighbor cells do not contain the interface by definition.
      /// If the extended level set attains values of the opposite sign of its cell mean with a
      /// magnitude larger than tolerance * h in a neighbor cell, e.g., due to the polynomial
      /// extrapolation near under-resolved interface tips, its deviation from the cell mean is
      /// scaled such that it does not change its sign. Otherwise, such spurious zero crossings
      /// create spurious cut cells. A negative value disables the limiter.
      number sign_limiter_tolerance = 0.05;

      /// Sign clamp of the extrapolated level set: each extended (or geometrically redistanced)
      /// cell is assigned to the side of the interface, on which it is located (from the traces
      /// of the adjacent cut cells or the closest point on the interface). Nodal values of the
      /// opposite sign (or zero) are set to -epsilon * h on the negative side and to +epsilon * h
      /// on the positive side, where h is the cell size. Thus, extrapolated values never
      /// introduce a new interface. A value of zero disables the sign clamp.
      number sign_clamp_epsilon = 0.0;

      /// Minimum volume fraction |K cap {phi > 0}| / |K| of the positive phase of a cut cell K, for
      /// which K serves as source of the extension. For the ghost-penalty extension, the
      /// ghost-penalty faces between cut cells with a smaller volume fraction (small cut cells)
      /// and neighbor cells are not evaluated, such that the neighbor cells are extended from the
      /// other cut cells via the faces to these cut cells and to other neighbor cells. If a
      /// neighbor cell is not connected to a cut cell with a sufficient volume fraction, the faces
      /// to the small cut cells are kept. For the closest-point extension, small cut cells are
      /// only considered in the closest-point search, if no other cut cells are available. All
      /// neighbor cells remain active. A value of zero uses all cut cells as sources.
      number minimum_source_volume_fraction = 0.0;

      /// Geometric redistancing of the cells of the advected band
      /// (NarrowBandType::advected_layers), i.e., of the newly added cells and of all cells of the
      /// layers after the reinitialization: first-order Taylor expansion from the closest point
      /// (taylor_expansion) or signed distance to the closest point with the tracked sign of the
      /// cell (signed_distance). In the latter case, the sign of each cell, which is not cut, is
      /// tracked over time; the redistancing never changes the sign of a cell and, thus, never
      /// creates an interface, even if the polynomial of the cut cell at the closest point is
      /// under-resolved (e.g., at the tip of a thin filament): if the polynomial of a redistanced
      /// cell changes its sign, it is shifted such that the minimum of |phi| is
      /// sign_clamp_epsilon * h (if sign_clamp_epsilon is zero: 1/10 of the undershoot, such that
      /// the modification vanishes with the undershoot). Cells without a sign change are not
      /// modified, such that the interface can approach and enter them.
      NarrowBandRedistancing redistancing = NarrowBandRedistancing::taylor_expansion;

      /// Redistancing of all cells of the layers of the advected band: after each reinitialization
      /// of the cut cells or in every time step. In the latter case, the reinitialization can be
      /// performed less often (e.g., NarrowBandData::reinitialization_displacement), while the
      /// layers, which provide the inflow data of the cut cells, do not degrade by the inflow
      /// condition at the outer boundary of the band.
      NarrowBandLayerRedistancing layer_redistancing =
        NarrowBandLayerRedistancing::after_reinitialization;

      /// Advected band with tracked signs: a cell, which is not cut, but at whose vertices or faces
      /// an adjacent cut cell has the other sign (the interface passes through a corner or along
      /// a face of the cell, which the discontinuous level set of the cell itself does not
      /// resolve), is redistanced by the Taylor expansion from the closest point on the interface
      /// of the cut cells (see NarrowBand::find_clipped_cells()). If the quadrature generator
      /// finds no interface in it afterwards, it is excluded from the redistancing of the layers in
      /// this time step. Cells cleared by the thin feature removal are excluded, until they are
      /// cut again.
      bool cut_clipped_cells = false;

      /// Linear solver for the extension problem (ghost-penalty extension).
      LinearSolverData<number> linear_solver;

      /**
       * @brief Return the ghost-penalty parameter for the jump of the k-th normal derivative.
       */
      number
      get_ghost_penalty_parameter(const unsigned int k) const;
    } extension;

    /// Time integrator for the advection on the narrow band.
    TimeIntegration::TimeIntegratorData<number> time_integrator{
      TimeIntegration::TimeIntegratorSchemes::SSP_RK_stage_3_order_3};

    /// Velocity of the DG advection on the band: the given (bulk) velocity or the extension
    /// velocity u(y(x)), i.e., the given velocity at the closest point y(x) = x - phi grad phi /
    /// |grad phi|^2 of x on the zero level set, evaluated with the current level set at each stage
    /// of the time integration. The extension velocity is constant along the normals of the
    /// interface; thus, the advection keeps the level set of the band a signed distance (it does
    /// not compress or stretch the level set in the band), while the interface moves with the given
    /// velocity.
    NarrowBandAdvectionVelocity advection_velocity = NarrowBandAdvectionVelocity::bulk;

    /// Bound-preserving limiter of the advection: after each stage of the SSP Runge-Kutta scheme,
    /// the level set of each limited cell is scaled around its cell mean, such that its values at
    /// the Gauss-Lobatto points (p+2 per direction) remain within the bounds given by the minimum
    /// m and maximum M of the level set at the beginning of the time step on the domain of
    /// dependence of the cell (Zhang & Shu, 2010).
    struct AdvectionLimiter
    {
      /// Cells, which are limited.
      NarrowBandLimiterCells cells = NarrowBandLimiterCells::none;

      /// Type of the bounds: local maximum principle [m - r h, M + r h] or sign preservation
      /// (m > 0: phi >= min(m, epsilon h); M < 0: phi <= max(M, -epsilon h)).
      NarrowBandLimiterType type = NarrowBandLimiterType::sign;

      /// Approximation of the domain of dependence of a cell.
      NarrowBandLimiterBounds bounds = NarrowBandLimiterBounds::upwind_strips;

      /// Relaxation of the bounds of the local maximum principle relative to the cell size h:
      /// [m - r h, M + r h].
      number relaxation = 0.0;

      /// Minimum distance epsilon h (relative to the cell size h) of the level set from zero,
      /// which is enforced by the sign-preserving limiter, unless the domain of dependence attains
      /// values closer to zero.
      number sign_epsilon = 0.01;
    } advection_limiter;

    /// Artificial viscosity of the advection in the cells flagged by the modal smoothness
    /// indicator of Persson & Peraire (2006): the term div(epsilon grad phi) is added to the
    /// advection equation (SIPG discretization on the band, natural boundary condition at the outer
    /// boundary of the band). The viscosity epsilon_K is constant per cell and determined at the
    /// beginning of each time step from the indicator S_K = log10(E_p / E_var), where E_p is the
    /// energy of the highest Legendre modes (maximum degree p in any direction) and E_var the
    /// energy of all modes except the cell mean, as
    /// epsilon_K = epsilon_0 for S_K > s_0 + kappa,
    /// epsilon_K = epsilon_0 / 2 (1 + sin(pi (S_K - s_0) / (2 kappa))) for |S_K - s_0| <= kappa,
    /// and zero otherwise, with epsilon_0 = c h_K max_K |u| / p and s_0 = -4 log10(p) + offset.
    /// The viscous term is integrated explicitly, i.e., the factor c is limited by the stability
    /// of the explicit time integration.
    struct ArtificialViscosity
    {
      /// Factor c of the maximum viscosity; zero disables the artificial viscosity.
      number factor = 0.0;

      /// Offset of the threshold s_0 = -4 log10(p) + offset of the indicator.
      number indicator_offset = 0.0;

      /// Half width kappa of the transition of the viscosity around the threshold.
      number indicator_width = 0.5;

      /// Cells, in which the artificial viscosity is applied.
      NarrowBandViscosityCells cells = NarrowBandViscosityCells::all_band_cells;
    } artificial_viscosity;

    /// Removal of thin features, which cannot be resolved by the mesh (DG advection of the band,
    /// i.e., NarrowBandType::extension and NarrowBandType::advected_layers): after the advection,
    /// a cell K of the band is cleared from a phase, if the thickness 2 |K cap phase| / |Gamma cap
    /// K| of the phase in K is below thickness * h (h: cell size) and if two boundaries of the
    /// phase in K (branches of the interface or faces, at which the face neighbor has the other
    /// sign) of opposite orientation enclose a region of this phase and are closer than thickness *
    /// h, or if the phase does not continue into any face neighbor of K (e.g., the remainder of a
    /// filament, whose neighboring part has been removed); the part of each face in the phase is
    /// determined by a face quadrature. The nodal values of the removed phase are set to the other
    /// phase. A cell is not cleared, if a face neighbor would have the other
    /// sign in the whole cell afterwards, since the interface would lie on the shared face (of two
    /// face neighbors, which would be cleared to different phases, only the one with the thinner
    /// removed phase is cleared). Thus, a thin filament disappears cell by cell from its tip once
    /// it is thinner than thickness * h, instead of creating spurious interfaces in the
    /// reinitialization and the redistancing of the under-resolved cells (see
    /// NarrowBand::remove_thin_features()).
    struct ThinFeatureRemoval
    {
      /// Thickness relative to the cell size, below which a thin feature is removed. Zero
      /// disables the removal.
      number thickness = 0.;

      /// Minimum magnitude epsilon * h of the level set in a cleared cell.
      number epsilon = 0.05;
    } thin_feature_removal;

    /**
     * Geometric reconstruction of the advected band (type advected_layers), which replaces the
     * elliptic reinitialization, the removal of thin features, the clipped cells and the
     * redistancing of the layers: signed distance to the discrete interface (zero level sets of
     * the cells and faces, at which the traces of the two cells have opposite signs),
     * morphological opening of both phases with the radius thickness * h / 2 (features thinner
     * than thickness * h disappear; smaller changes than a quarter of the thickness, e.g., rounded
     * corners, are ignored), and cell-wise projection onto the polynomials, which keeps the
     * remaining interface by a penalty term (see NarrowBandGeometricReconstruction).
     */
    struct GeometricReconstruction
    {
      /// Enable the geometric reconstruction.
      bool enable = false;

      /// Thickness relative to the cell size, below which a feature is removed by the opening.
      number thickness = 0.8;

      /// Penalty gamma of the interface term gamma h ||phi||^2 on the remaining interface.
      number penalty = 100.;

      /// Displacement (sum of max |u| dt), after which the reconstruction is performed; zero:
      /// in every time step. In the other time steps, the band is only advected.
      number displacement = 0.;
    } geometric_reconstruction;

    /// Face neighbors of the cut cells, which are activated as neighbor cells of the band
    /// (extension band). For downstream_face_neighbors, only the face neighbors, into which the
    /// velocity points across a face shared with a cut cell (at the beginning or the end of the
    /// next time step), are activated, i.e., the cells into which the interface can move. The
    /// level set is only extended to these cells. At the faces of the cut cells to inactive
    /// upstream cells, the upwind flux uses the interior trace of the cut cell.
    NarrowBandNeighborActivation neighbor_activation =
      NarrowBandNeighborActivation::all_face_neighbors;

    /// Minimum volume fraction |K cap {phi > 0}| / |K| of a cut cell K, for which its face
    /// neighbors are activated as neighbor cells of the band. Face neighbors of cut cells with a
    /// smaller volume fraction of the positive phase remain inactive (FE_Nothing), unless they are
    /// face neighbors of another cut cell with a sufficient volume fraction. Thus, the level set
    /// is only extended from cut cells, in which the positive phase occupies a sufficiently large
    /// part of the cell. A value of zero activates all face neighbors of the cut cells.
    number neighbor_activation_volume_fraction = 0.0;

    /// Set of faces on which the SIPG terms of the elliptic reinitialization on the cut cells are
    /// evaluated.
    NarrowBandReinitializationFaces reinitialization_faces =
      NarrowBandReinitializationFaces::cut_cell_faces;

    /// Finite element space of the level set on the cut cells (semi-Lagrangian band).
    NarrowBandLevelSetSpace level_set_space = NarrowBandLevelSetSpace::discontinuous;

    /// Continuous level set space: closed-interface correction after each time step, i.e., the
    /// level set takes the sign of the neighbor cell on all faces between the cut cells and the
    /// cells, which are not cut (see NarrowBandContinuousReinitialization::close_interface()).
    bool closed_interface_correction = true;

    /// Closed-interface correction: cells, which are not cut, behind faces, on which the level set
    /// does not have their sign, are added to the cut cells (with the level set of the
    /// semi-Lagrangian step on their other nodes), before the remaining faces are lifted.
    bool closed_interface_activation = true;

    /// Closed-interface correction: minimum of the level set relative to the cell size on a
    /// lifted face (distance of the corrected interface from the face).
    number closed_interface_offset = 0.01;

    /// Semi-Lagrangian band: sub-cycling of the level-set update. If positive, the level set of a
    /// time step is evaluated from the last reference level set at the foot points of the nodes
    /// at the time of the reference (see level_set_update_characteristics), and the
    /// reinitialization, the insertion of new interface points and the update of the reference
    /// are only performed, once the maximum displacement of the nodes since the reference exceeds
    /// this value times the cell size. Thus, the number of these operations per unit time does not
    /// depend on the time-step size. Zero: update in every time step.
    number level_set_update_cfl = 0.;

    /// Sub-cycling: computation of the foot points of the nodes at the time of the reference,
    /// either by the backward integration of the characteristics over all time steps since the
    /// reference or by the characteristic map, which is composed in each time step with the
    /// backward characteristic of the time step (only the velocity of the current time step is
    /// required, see NarrowBandCharacteristicMap).
    NarrowBandSubcyclingCharacteristics level_set_update_characteristics =
      NarrowBandSubcyclingCharacteristics::integration;

    /// Interface, at which the reinitialization on the cut cells enforces phi = 0 weakly
    /// (interface penalty term): the zero level set of the level set before the reinitialization
    /// or points of the interface, which are seeded on the initial zero level set and advected
    /// with the velocity independently of the level set (front tracking with local seeding in
    /// the gaps of the point set and merging in cells with too many points, see
    /// NarrowBandInterfacePoints). In the latter case, the zero level set is used in the cut
    /// cells without usable points.
    NarrowBandReinitializationInterface reinitialization_interface =
      NarrowBandReinitializationInterface::zero_level_set;

    /// Reinitialize the level-set field on the cut cells every n-th time step. A value of zero
    /// disables the frequency-based reinitialization.
    unsigned int reinitialization_frequency = 1;

    /// Reinitialize the level-set field on the cut cells, if the eikonal error
    /// sqrt(int_cut (|grad phi| - 1)^2 dx / |cut cells|) exceeds this value. A value of zero
    /// disables the error-based reinitialization.
    number reinitialization_eikonal_threshold = 0.0;

    /// Reinitialize the level-set field on the cut cells, once the displacement of the interface
    /// since the last reinitialization, estimated by the sum of max |u| dt over the time steps
    /// (maximum of the velocity on the band), reaches this length. The number of
    /// reinitializations then depends on the distance travelled by the interface, but neither on
    /// the time step size nor on the cell size. A value of zero disables the displacement-based
    /// reinitialization.
    number reinitialization_displacement = 0.0;

    /// Parameters of the advected interface points (reinitialization interface =
    /// advected_points, see NarrowBandInterfacePoints).
    struct AdvectedPoints
    {
      /// Distance relative to the cell size, below which points of opposite orientation are not
      /// used (unresolved interface, e.g., both sides of a thin filament). Smaller values use the
      /// points down to thinner features; in the remaining regions, the zero level set of the
      /// advected level set is used.
      number unresolved_distance = 2.0;

      /// Points, at which the magnitude of the advected level set exceeds this value times the
      /// cell size, are removed (parts of the interface, which the level set has lost).
      number support_tolerance = 0.2;

      /// Merge distance c_m relative to the cell size of the merge and break-up rules for
      /// topology changes (e.g., merging droplets, a neck, which pinches off): points of two parts
      /// of the interface, which have collided within the time step or whose gap (merge rule) or
      /// part of the phase (break-up rule) between them is thinner than about c_m h, and points,
      /// which have crossed a line of converging characteristics, are removed (zero: no rules,
      /// see NarrowBandInterfacePoints).
      number merge_distance = 0.05;

      /// Number n of Gauss points per direction of the surface quadrature, at which the points are
      /// seeded on the initial zero level set and inserted into gaps of the point set (0: n = p +
      /// 1). It also sets the minimum distance h / n of an inserted point to all points and the
      /// maximum number 2 n^(dim-1) of points per cell before merging.
      unsigned int points_per_direction = 0;

      /**
       * @brief Number of Gauss points per direction of the interface points for the polynomial
       * degree @p fe_degree of the level set (points_per_direction, or p + 1 if it is zero).
       */
      unsigned int
      get_points_per_direction(const unsigned int fe_degree) const
      {
        return points_per_direction > 0 ? points_per_direction : fe_degree + 1;
      }

      /// Level set, from whose zero level set new points are inserted into the gaps of the point
      /// set: the advected level set or the level set at the end of the time step (after the
      /// reinitialization and the closed-interface correction).
      NarrowBandPointInsertion point_insertion = NarrowBandPointInsertion::final_level_set;

      /// Conservative insertion: the weights of the cells with new points are scaled, such that
      /// the sum of the weights of each cell is conserved (see
      /// NarrowBandInterfacePoints::conserve_cell_weights()).
      bool conservative_insertion = true;

      /// Treatment of the cut cells with unresolved points (points of opposite orientation closer
      /// than the unresolved distance) and of the cut cells within the buffer around them.
      NarrowBandUnresolvedCells unresolved_cells = NarrowBandUnresolvedCells::zero_level_set;

      /// Subcell distance: distance relative to the cell size, within which the cells around
      /// unresolved points are projected (buffer).
      number subcell_buffer = 1.0;

      /// Subcell distance: number of subdivisions per direction of the iterated Gauss quadrature
      /// of the L2-projection.
      unsigned int subcell_subdivisions = 4;

      /// Subcell distance: coupling of the projected cells to the reinitialization of the other
      /// cut cells.
      NarrowBandSubcellCoupling subcell_coupling = NarrowBandSubcellCoupling::overwrite;

      /// Subcell distance: representation of the signed distance in the finite element space,
      /// L2-projection with the iterated Gauss quadrature or nodal interpolation (continuous
      /// level set space only).
      NarrowBandSubcellTransfer subcell_transfer = NarrowBandSubcellTransfer::projection;

      /// Subcell distance: in a cell, whose points all have the same orientation, the distance is
      /// only computed to the points of this orientation (branch of the cell).
      bool subcell_orientation_filter = false;

      /// Subcell distance: dimensionless penalty parameter of the zero level set at the points of
      /// a projected cell with one branch of the interface (pinned projection, zero: off).
      number subcell_pinning_penalty = 0.;
    } advected_points;

    /// Continuous level-set field on the cut cells, which is computed after each time step by the
    /// constrained L2-projection of the DG level set, and the normal vector and the curvature
    /// computed from it (see NarrowBandContinuousLevelSet).
    struct ContinuousLevelSet
    {
      /// Compute the continuous level set, the normal vector and the curvature.
      bool enable = false;

      /// Dimensionless factor c of the interface penalty gamma_K = c h_K of the projection, which
      /// pins the zero level set of the continuous level set to the zero level set of the DG
      /// level set. A value of zero results in the plain L2-projection.
      number interface_penalty = 100.0;

      /// Enforce the signs of the cells, which are not cut, on the continuous level set, i.e., its
      /// zero level set does not leave the cut cells (requires NarrowBandType::semi_lagrangian, see
      /// NarrowBandContinuousLevelSet).
      bool sign_constraints = false;

      /// Magnitude c of the constrained nodal values s c h_K (s: sign of the adjacent cell, which
      /// is not cut) relative to the cell size.
      number sign_constraint_offset = 0.01;

      /// Linear solver of the projection.
      LinearSolverData<number> linear_solver;
    } continuous_level_set;

    /**
     * @brief Add the parameters to the parameter handler.
     *
     * @param prm The parameter handler to which the parameters are added.
     */
    void
    add_parameters(dealii::ParameterHandler &prm);

    /**
     * @brief Check the parameters for validity.
     *
     * @param fe_degree Polynomial degree of the level-set field.
     */
    void
    check_input_parameters(const unsigned int fe_degree) const;
  };
} // namespace MeltPoolDG::LevelSet
