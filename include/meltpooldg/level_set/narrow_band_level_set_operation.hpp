#pragma once

#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>

#include <deal.II/hp/fe_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/level_set/curvature_data.hpp>
#include <meltpooldg/level_set/narrow_band_advection_DG_operator.hpp>
#include <meltpooldg/level_set/narrow_band_bound_preserving_limiter.hpp>
#include <meltpooldg/level_set/narrow_band_closest_point_extension.hpp>
#include <meltpooldg/level_set/narrow_band_continuous_level_set.hpp>
#include <meltpooldg/level_set/narrow_band_data.hpp>
#include <meltpooldg/level_set/narrow_band_ghost_penalty_extension.hpp>
#include <meltpooldg/level_set/narrow_band_semi_lagrangian_advection.hpp>
#include <meltpooldg/level_set/normal_vector_data.hpp>
#include <meltpooldg/level_set/reinitialization_data.hpp>
#include <meltpooldg/level_set/reinitialization_elliptic_operation.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/time_integration/solution_history.hpp>
#include <meltpooldg/time_integration/time_integrator_base.hpp>

#include <functional>
#include <memory>
#include <vector>

namespace MeltPoolDG::LevelSet
{
  /**
   * @brief Narrow-band DG level-set method, where the level-set field is primarily defined on the
   * cells intersected by the interface (cut cells) and advected on the narrow band of the cut
   * cells and their face neighbors.
   *
   * The level-set field lives on two hp-DoFHandlers (see NarrowBand::ActiveFEIndex): the cut
   * DoFHandler (FE_DGQ on the cut cells $\mathcal{T}^\Gamma$) and the band DoFHandler (FE_DGQ on
   * the narrow band $\mathcal{T}^B = \mathcal{T}^\Gamma \cup \mathcal{T}^N$, where
   * $\mathcal{T}^N$ are the face neighbors of the cut cells). Both DoFHandlers are rebuilt
   * whenever the cut cells change. One time step $t^n \to t^{n+1}$ consists of
   *
   * 1. S4: the advection of the extended level-set field on the band
   *    $\mathcal{T}^{B,n}$ with an explicit upwind DG scheme (see
   *    NarrowBandAdvectionDGOperator),
   * 2. S1: the determination of the new cut cells $\mathcal{T}^{\Gamma,n+1} \subset
   *    \mathcal{T}^{B,n}$ and of the new band; the DoF values of the new cut cells are transferred
   *    cell-wise (the DoFHandlers and the matrix-free data structures are only rebuilt if the
   *    cut cells have changed). Optionally, only the face neighbors of cut cells with a
   *    sufficient volume fraction of the positive phase are activated (see
   *    NarrowBandData::neighbor_activation_volume_fraction) and/or only the face neighbors, into
   *    which the velocity points from the cut cells (see NarrowBandData::neighbor_activation; the
   *    DoFHandlers are then rebuilt in every time step),
   * 3. S5: the elliptic reinitialization on the new cut cells (only SIPG coupling between cut
   *    cells, optionally only across the faces crossed by the interface, see
   *    ReinitializationEllipticOperatorFixedPointIntersectedCells), if the reinitialization
   *    criterion is met,
   * 4. S2: the extension of the level-set field from the new cut cells to the new neighbor cells,
   *    either by the global face-based ghost-penalty extension (see
   *    NarrowBandGhostPenaltyExtension) or by the closest-point extension (see
   *    NarrowBandClosestPointExtension).
   *
   * For the semi-Lagrangian band (NarrowBandType::semi_lagrangian), the level set is only stored
   * on the cut cells. In each time step, it is evaluated on the cut cells and their vertex
   * neighbors (candidate cells) by the semi-Lagrangian advection (see
   * NarrowBandSemiLagrangianAdvection), which replaces steps 1 and 4; the new cut cells are
   * determined from the candidate cells (step 2) and reinitialized (step 3).
   *
   * For the advected band (NarrowBandType::advected_layers), the band consists of the cut cells
   * and NarrowBandData::n_layers layers of vertex neighbors, which are advected together with
   * the cut cells (step 4 is omitted). Whenever the cut cells change, the band is rebuilt: the
   * advected level set of the cells of the old band is transferred cell-wise and newly added
   * cells, which are at least two layers away from the cut cells if the interface moves less than
   * one cell per time step, are initialized by geometric redistancing (closest point on the
   * discrete interface, see NarrowBandClosestPointExtension::redistance()). After the elliptic
   * reinitialization of the cut cells, all cells of the layers are redistanced geometrically.
   *
   * Optionally, an artificial viscosity is added to the advection in the cells flagged by a modal
   * smoothness indicator (see NarrowBandData::ArtificialViscosity and
   * NarrowBandAdvectionDGOperator::compute_artificial_viscosity()).
   *
   * Optionally, the advection is limited after each stage of the SSP Runge-Kutta scheme by a
   * bound-preserving limiter (local maximum principle or sign preservation, see
   * NarrowBandBoundPreservingLimiter).
   *
   * Optionally (NarrowBandData::ContinuousLevelSet), a continuous level-set field is computed on
   * the cut cells at the end of each time step and after setting the initial condition by the
   * constrained L2-projection of the DG level set (S6), and the normal vector and the curvature
   * are computed from it (S7, see NarrowBandContinuousLevelSet). The continuous level set does
   * not feed back into the DG level set.
   *
   * The velocity extension (S3) is not part of this class. The advection velocity is prescribed
   * by a function.
   *
   * The DoFHandlers are attached to the given ScratchData object, which is rebuilt whenever the
   * cut cells change. All DoFHandlers (band, cut and, optionally, the continuous fields) share the
   * same active FE indices, as required by MatrixFree for several hp-DoFHandlers.
   *
   * @note Only triangulations without hanging nodes and without periodic boundaries are
   * supported. The time step size has to satisfy the CFL condition such that the interface does
   * not leave the band within one time step.
   */
  template <int dim, typename number>
  class NarrowBandLevelSetOperation
  {
  public:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    /**
     * Level-set function evaluated at a point of a given cell.
     */
    using CellwiseFunction =
      std::function<number(const dealii::CellAccessor<dim, dim> &, const dealii::Point<dim> &)>;

    /**
     * @brief Constructor.
     *
     * @param scratch_data Scratch data, to which the DoFHandlers and the quadrature rule are
     * attached. It is rebuilt whenever the narrow band changes. No other hp-DoFHandler may be
     * attached (except the DoFHandler of the continuous level set, which is attached by this
     * class), since MatrixFree requires the same active FE indices for all hp-DoFHandlers.
     * @param triangulation Triangulation.
     * @param narrow_band_data Parameters of the narrow-band method.
     * @param reinit_data Parameters of the elliptic reinitialization on the cut cells.
     * @param fe_degree Polynomial degree of the level-set field.
     * @param velocity_function Advection velocity.
     * @param normal_vector_data Parameters of the normal vector of the continuous level set
     * (NarrowBandData::ContinuousLevelSet). If nullptr, the default parameters without damping
     * (filter parameter zero) are used.
     * @param curvature_data Parameters of the curvature of the continuous level set. If nullptr,
     * the default parameters without damping (filter parameter zero) are used.
     */
    NarrowBandLevelSetOperation(ScratchData<dim, dim, number>                &scratch_data,
                                const dealii::Triangulation<dim>             &triangulation,
                                const NarrowBandData<number>                 &narrow_band_data,
                                const ReinitializationData<number>           &reinit_data,
                                const unsigned int                            fe_degree,
                                const std::shared_ptr<dealii::Function<dim>> &velocity_function,
                                const NormalVectorData<number> *normal_vector_data = nullptr,
                                const CurvatureData<number>    *curvature_data     = nullptr);

    /**
     * @brief Set the initial condition.
     *
     * The candidates of the cut cells are determined from the analytical function with
     * dealii::NonMatching::MeshClassifier. The function is interpolated onto the band of these
     * candidates, and the cut cells are determined from the interpolated field. Finally, the
     * level-set field is interpolated onto the cut cells, optionally reinitialized and extended to
     * the neighbor cells.
     *
     * @param level_set_function Initial level-set function.
     * @param do_reinitialize Reinitialize the initial level-set field on the cut cells.
     * @param cellwise_level_set_function Optional level-set function, which is evaluated with the
     * cell and is interpolated onto the cut cells instead of @p level_set_function (the cut cells
     * and the signs of the other cells are still determined from @p level_set_function). It allows
     * for an initial level set that jumps across the faces between the cut cells, e.g., the signed
     * distance function of the interface branch of each cut cell.
     */
    void
    set_initial_condition(const dealii::Function<dim> &level_set_function,
                          const bool                   do_reinitialize             = false,
                          const CellwiseFunction      &cellwise_level_set_function = {});

    /**
     * @brief Advance the level-set field by one time step.
     *
     * @param time Time at the beginning of the time step.
     * @param time_step Time step size.
     * @param time_step_number Number of the time step (starting at one), used for the
     * frequency-based reinitialization.
     */
    void
    advance_time_step(const number       time,
                      const number       time_step,
                      const unsigned int time_step_number);

    /**
     * @brief Level-set field on the cut cells (cut DoFHandler).
     */
    const VectorType &
    get_level_set() const;

    /**
     * @brief Extended level-set field on the narrow band (band DoFHandler).
     */
    const VectorType &
    get_level_set_band() const;

    /**
     * @brief Cut DoFHandler.
     */
    const dealii::DoFHandler<dim> &
    get_dof_handler_cut() const;

    /**
     * @brief Band DoFHandler.
     */
    const dealii::DoFHandler<dim> &
    get_dof_handler_band() const;

    /**
     * @brief Continuous level set, normal vector and curvature on the cut cells (nullptr, if
     * NarrowBandData::ContinuousLevelSet::enable is false).
     */
    const NarrowBandContinuousLevelSet<dim, number> *
    get_continuous_level_set() const;

    /**
     * @brief Attach the level-set field on the cut cells ("level_set") and the extended level-set
     * field on the band ("level_set_band") to the output. If enabled, the continuous level set
     * ("level_set_cg"), the normal vector ("normal") and the curvature ("curvature") are attached
     * as well.
     */
    void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const;

    /**
     * @brief Compute the CFL number of the advection on the current band.
     *
     * @param time Time at which the velocity is evaluated.
     * @param time_step Time step size.
     */
    number
    compute_cfl_number(const number time, const number time_step) const;

  private:
    /**
     * @brief Distribute the DoFs of the cut and the band DoFHandler for the given cut cells,
     * rebuild the matrix-free data structures and reinitialize all vectors and operators.
     *
     * @param is_cut_cell Flags indexed by the active cell index, which are true for the locally
     * owned cut cells.
     * @param is_small_cut_cell_in Flags indexed by the active cell index, which are true for the
     * locally owned cut cells, whose face neighbors are not activated (see
     * NarrowBand::distribute_dofs()). If empty, all face neighbors of the cut cells are activated.
     *
     * For the downstream neighbor activation (NarrowBandData::neighbor_activation), the face
     * neighbors are only activated across the faces, across which the velocity points into the
     * neighbor at the times neighbor_activation_times.
     */
    void
    setup_dof_system(const std::vector<bool> &is_cut_cell,
                     const std::vector<bool> &is_small_cut_cell_in = {});

    /**
     * @brief Determine the locally owned cut cells, whose volume fraction of the positive phase is
     * below NarrowBandData::neighbor_activation_volume_fraction.
     *
     * @param is_cut_cell Flags of the locally owned cut cells.
     * @param volume_fractions Volume fractions of the positive phase of the locally owned cut
     * cells (see NarrowBand::compute_cut_cells()).
     *
     * @return Flags indexed by the active cell index; empty if all face neighbors of the cut
     * cells are activated.
     */
    std::vector<bool>
    compute_small_cut_cells(const std::vector<bool>   &is_cut_cell,
                            const std::vector<number> &volume_fractions) const;

    /**
     * @brief Elliptic reinitialization of the level-set field on the cut cells (S5).
     */
    void
    reinitialize();

    /**
     * @brief Copy the level-set field of the cut cells to the band and extend it to the
     * neighbor cells (S2).
     */
    void
    extend();

    /**
     * @brief Return true for the advected band (NarrowBandType::advected_layers).
     */
    bool
    is_advected_band() const;

    /**
     * @brief Semi-Lagrangian band: update the signs of the locally owned candidate cells, which are
     * not cut, from the cell means of the level set on the band, and communicate the signs to the
     * ghost cells. If @p function is given, the signs of all locally owned cells are initialized
     * from its values at the cell centers.
     *
     * @param is_cut_cell Flags of the locally owned cut cells.
     * @param function Level-set function for the initialization (optional).
     */
    void
    update_cell_signs(const std::vector<bool> &is_cut_cell, const dealii::Function<dim> *function);

    /**
     * @brief Return true for the semi-Lagrangian band (NarrowBandType::semi_lagrangian).
     */
    bool
    is_semi_lagrangian() const;

    /**
     * @brief Return true for the bands of the cut cells and layers of vertex neighbors (advected
     * band with NarrowBandData::n_layers layers, semi-Lagrangian band with one layer of candidate
     * cells).
     */
    bool
    is_layered_band() const;

    /**
     * @brief Advected band: copy the (reinitialized) level-set field of the cut cells to the band
     * and redistance all cells of the layers geometrically (see
     * NarrowBandClosestPointExtension::redistance()).
     */
    void
    redistance_layers();

    /**
     * @brief Print the size of the narrow band.
     */
    void
    print_band_statistics() const;

    /**
     * @brief Compute the continuous level set, the normal vector and the curvature from the
     * level-set field on the cut cells (S6, S7), if enabled.
     */
    void
    update_continuous_level_set();

    /// Scratch data.
    ScratchData<dim, dim, number> &scratch_data;

    /// Parameters of the narrow-band method.
    const NarrowBandData<number> &narrow_band_data;

    /// Parameters of the reinitialization.
    const ReinitializationData<number> &reinit_data;

    /// Polynomial degree of the level-set field.
    const unsigned int fe_degree;

    /// FE collections of the band and the cut DoFHandler.
    const dealii::hp::FECollection<dim> fe_collection_band;
    const dealii::hp::FECollection<dim> fe_collection_cut;

    /// DoFHandlers and (empty) constraints of the band and the cut cells.
    dealii::DoFHandler<dim>           dof_handler_band;
    dealii::DoFHandler<dim>           dof_handler_cut;
    dealii::AffineConstraints<number> constraints_band;
    dealii::AffineConstraints<number> constraints_cut;

    /// Indices of the DoFHandlers and the quadrature rule in the scratch data.
    unsigned int band_dof_idx;
    unsigned int cut_dof_idx;
    unsigned int quad_idx;

    /// Level-set field on the cut cells.
    VectorType level_set;

    /// Flags of the locally owned cut cells, whose face neighbors are not activated (indexed by
    /// the active cell index; empty if all face neighbors of the cut cells are activated).
    std::vector<bool> is_small_cut_cell;

    /// Times at which the velocity is evaluated for the downstream neighbor activation (beginning
    /// and end of the next time step).
    std::vector<double> neighbor_activation_times;

    /// Level-set field on the band; the current solution of the solution history.
    TimeIntegration::SolutionHistory<VectorType> solution_history_band;

    /// Elliptic reinitialization on the cut cells.
    std::unique_ptr<ReinitializationEllipticOperation<dim, number>> reinit_operation;

    /// Ghost-penalty extension.
    const NarrowBandGhostPenaltyExtension<dim, number> ghost_penalty_extension;

    /// Closest-point extension.
    const NarrowBandClosestPointExtension<dim, number> closest_point_extension;

    /// Semi-Lagrangian advection (NarrowBandType::semi_lagrangian).
    const NarrowBandSemiLagrangianAdvection<dim, number> semi_lagrangian_advection;

    /// Semi-Lagrangian band: signs (+1, -1) of the level set on the locally owned and ghost cells,
    /// which are not cut, indexed by the active cell index (0: unknown). The sign of a cell only
    /// changes if the interface passes through it, i.e., the signs are updated on the candidate
    /// cells in each time step.
    std::vector<int> cell_signs;

    /// Advection operator on the band.
    NarrowBandAdvectionDGOperator<dim, number> advection_operator;

    /// Advection velocity.
    std::shared_ptr<dealii::Function<dim>> velocity_function;

    /// Bound-preserving limiter of the advection.
    NarrowBandBoundPreservingLimiter<dim, number> advection_limiter;

    /// Artificial viscosity of the advection (constant per cell, on the band DoFHandler).
    VectorType artificial_viscosity;

    /// Explicit time integrator of the advection.
    std::unique_ptr<TimeIntegration::TimeIntegratorBase<number>> time_integrator;

    /// Continuous level set, normal vector and curvature on the cut cells (optional).
    std::unique_ptr<NarrowBandContinuousLevelSet<dim, number>> continuous_level_set;
  };
} // namespace MeltPoolDG::LevelSet
