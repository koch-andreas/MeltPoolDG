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

    /// Number of layers of vertex neighbors around the cut cells (advected band).
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

    /// Reinitialize the level-set field on the cut cells every n-th time step. A value of zero
    /// disables the frequency-based reinitialization.
    unsigned int reinitialization_frequency = 1;

    /// Reinitialize the level-set field on the cut cells, if the eikonal error
    /// sqrt(int_cut (|grad phi| - 1)^2 dx / |cut cells|) exceeds this value. A value of zero
    /// disables the error-based reinitialization.
    number reinitialization_eikonal_threshold = 0.0;

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
