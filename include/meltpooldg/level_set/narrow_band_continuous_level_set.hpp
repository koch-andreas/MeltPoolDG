#pragma once

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/hp/fe_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/matrix_free/fe_point_evaluation.h>

#include <meltpooldg/core/operator_base.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/level_set/curvature_data.hpp>
#include <meltpooldg/level_set/narrow_band_data.hpp>
#include <meltpooldg/level_set/normal_vector_data.hpp>
#include <meltpooldg/linear_algebra/preconditioner.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>

#include <memory>

namespace MeltPoolDG::LevelSet
{
  /**
   * @brief Matrix-free operator of a weighted mass matrix for a continuous scalar field $u$ on the
   * cut cells of the narrow band,
   *
   * $a(u, w) = (u, w)_{\Omega_\Gamma} + \sum_{K \in \mathcal{T}^\Gamma} \eta_K (\nabla u, \nabla
   * w)_K + \sum_{K \in \mathcal{T}^\Gamma} \gamma_K (u, w)_{\Gamma \cap K}$,
   *
   * with the damping $\eta_K = c_\eta h_K^2$ (Helmholtz filter as in NormalVectorOperation and
   * CurvatureOperation) and the interface penalty $\gamma_K = c_\gamma h_K$, where $h_K$ is the
   * cell size and $\Gamma$ the interface described by the surface quadrature of
   * @p mapping_info_surface. Here, $\Omega_\Gamma$ denotes the union of the cut cells
   * $\mathcal{T}^\Gamma$. The boundary of $\Omega_\Gamma$ is a natural boundary.
   *
   * The field lives on an hp-DoFHandler with FE_Q elements on the cut cells (active FE index
   * NarrowBand::ActiveFEIndex::cut) and FE_Nothing elements (non-dominating) on all other cells,
   * i.e., the field is continuous across the faces between two cut cells. For a block vector, the
   * operator is applied to each block, e.g., to the components of a vector field.
   *
   * Optionally, a set of DoFs is constrained (set_constrained_dofs()): the operator then acts as
   * $\begin{pmatrix} A_{II} & 0 \\ 0 & I \end{pmatrix}$, i.e., as the operator on the free DoFs
   * $I$ and as the identity on the constrained DoFs, which remains symmetric and positive definite.
   * The values of the constrained DoFs are prescribed by the right-hand side (elimination of
   * inhomogeneous Dirichlet conditions without rebuilding the matrix-free data structures).
   */
  template <int dim, typename number>
  class NarrowBandContinuousMassOperator : public OperatorMatrixFree<dim, number>
  {
    using OperatorMatrixFree<dim, number>::vmult;
    using OperatorMatrixFree<dim, number>::compute_inverse_diagonal_from_matrixfree;

  public:
    using VectorType          = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType     = dealii::LinearAlgebra::distributed::BlockVector<number>;
    using VectorizedArrayType = dealii::VectorizedArray<number>;
    using MappingInfoType     = CutUtil::MappingInfoType<dim, number>;
    using PointEvaluationType = dealii::FEPointEvaluation<1, dim, dim, VectorizedArrayType>;

    /**
     * @brief Constructor.
     *
     * @param scratch_data Scratch data.
     * @param dof_idx Index of the hp-DoFHandler of the continuous field in @p scratch_data.
     * @param quad_idx Index of the quadrature rule in @p scratch_data.
     * @param damping_factor Factor $c_\eta$ of the damping $\eta_K = c_\eta h_K^2$.
     * @param penalty_factor Factor $c_\gamma$ of the interface penalty $\gamma_K = c_\gamma h_K$.
     * @param mapping_info_surface Surface quadrature of the interface $\Gamma$ (required if
     * @p penalty_factor is not zero).
     */
    NarrowBandContinuousMassOperator(const ScratchData<dim, dim, number> &scratch_data,
                                     const unsigned int                   dof_idx,
                                     const unsigned int                   quad_idx,
                                     const number                         damping_factor,
                                     const number                         penalty_factor,
                                     const MappingInfoType *mapping_info_surface = nullptr);

    /**
     * @brief Update the internal data structures after the DoFs have been distributed and the
     * matrix-free data structures have been rebuilt.
     */
    void
    reinit() final;

    /**
     * @brief Apply the operator.
     */
    void
    vmult(VectorType &dst, const VectorType &src) const final;

    /**
     * @brief Apply the operator to each block.
     */
    void
    vmult(BlockVectorType &dst, const BlockVectorType &src) const final;

    /**
     * @brief Compute the sparse matrix of the operator (for matrix-based preconditioners).
     */
    void
    compute_system_matrix_from_matrixfree(
      dealii::TrilinosWrappers::SparseMatrix &system_matrix) const final;

    /**
     * @brief Compute the inverse diagonal of the operator (Jacobi preconditioner).
     */
    void
    compute_inverse_diagonal_from_matrixfree(VectorType &diagonal) const final;

    /**
     * @brief Compute the inverse diagonal of the operator for each block (Jacobi preconditioner).
     */
    void
    compute_inverse_diagonal_from_matrixfree(BlockVectorType &diagonal) const final;

    /**
     * @brief Constrain the DoFs with nonzero entries in @p mask (locally owned entries): the
     * operator acts as the identity on these DoFs and ignores their values on the free DoFs. The
     * system matrix of matrix-based preconditioners is computed with @p matrix_constraints, which
     * have to contain the constrained DoFs (locally relevant) as homogeneous constraints. Both
     * pointers have to remain valid while the DoFs are constrained; call with nullptr to remove
     * the constraints.
     */
    void
    set_constrained_dofs(const VectorType                        *mask,
                         const dealii::AffineConstraints<number> *matrix_constraints);

    /**
     * @brief Return true, if the range of cell batches belongs to the cut cells.
     */
    bool
    is_cut_cell_range(const std::pair<unsigned int, unsigned int> &cell_range) const;

    /**
     * @brief Create a cell integrator for the cut cells.
     */
    FECellIntegrator<dim, 1, number>
    create_cell_integrator() const;

  private:
    /**
     * @brief Apply the operator without the constrained DoFs.
     */
    void
    vmult_unconstrained(VectorType &dst, const VectorType &src) const;

    /**
     * @brief Apply the cell contribution to the DoF values of @p phi (input: source values,
     * output: integrated values).
     */
    void
    cell_operation(FECellIntegrator<dim, 1, number>           &phi,
                   PointEvaluationType                        *point_eval,
                   dealii::AlignedVector<VectorizedArrayType> &penalty_buffer) const;

    /**
     * @brief Create the point evaluator for the interface penalty (nullptr without penalty).
     */
    std::unique_ptr<PointEvaluationType>
    create_point_evaluator() const;

    /// Scratch data.
    const ScratchData<dim, dim, number> &scratch_data;

    /// Index of the quadrature rule.
    const unsigned int quad_idx;

    /// Factor of the damping.
    const number damping_factor;

    /// Factor of the interface penalty.
    const number penalty_factor;

    /// Surface quadrature of the interface.
    const MappingInfoType *mapping_info_surface;

    /// FE of the cut cells; required by FEPointEvaluation.
    std::unique_ptr<dealii::FE_Q<dim>> fe_point;

    /// Number of DoFs per cut cell.
    unsigned int n_dofs_per_cell = 0;

    /// Damping eta_K and interface penalty gamma_K per cell batch.
    dealii::AlignedVector<VectorizedArrayType> damping;
    dealii::AlignedVector<VectorizedArrayType> penalty;

    /// Optional constrained DoFs (nonzero entries) and the constraints of the system matrix.
    const VectorType                        *constrained_dof_mask = nullptr;
    const dealii::AffineConstraints<number> *matrix_constraints   = nullptr;

    /// Source vector with zero values on the constrained DoFs.
    mutable VectorType src_free;
  };

  /**
   * @brief Continuous level-set field, normal vector and curvature on the cut cells of the narrow
   * band (steps S6 and S7 of NarrowBandLevelSetOperation).
   *
   * The DG level-set field $\phi_h$ of the narrow band is discontinuous across the faces of the
   * cut cells. Thus, its zero level set $\Gamma_h$ is not closed and the quadrature rules of
   * adjacent cut cells do not match at their common faces. This class computes a continuous
   * level-set field $\psi_h \in V_h = \{ v \in C^0(\Omega_\Gamma) : v|_K \in Q_p(K), K \in
   * \mathcal{T}^\Gamma \}$ on the union $\Omega_\Gamma$ of the cut cells by the constrained
   * $L^2$-projection
   *
   * $\psi_h = \arg\min_{v \in V_h} \|v - \phi_h\|^2_{\Omega_\Gamma} + \sum_K \gamma_K
   * \|v\|^2_{\Gamma_h \cap K}$, i.e., $(\psi_h, w)_{\Omega_\Gamma} + \sum_K \gamma_K (\psi_h,
   * w)_{\Gamma_h \cap K} = (\phi_h, w)_{\Omega_\Gamma}$ for all $w \in V_h$,
   *
   * with the interface penalty $\gamma_K = c_\gamma h_K$ (NarrowBandData::ContinuousLevelSet).
   * The penalty term pins the zero level set of $\psi_h$ to the zero level set $\Gamma_h$ of the
   * DG field in a least-squares sense; it does not guarantee that the interface does not move,
   * since a continuous field cannot vanish on the discontinuous $\Gamma_h$ in general. The
   * maximum interface shift $\max_{\Gamma_h} |\psi_h| / |\nabla \psi_h|$ and the numbers of
   * linear iterations of the projection, the normal vector and the curvature are reported
   * (verbosity level 1). The continuous level set does not feed back into the DG level set.
   *
   * From $\psi_h$, the normal vector and the curvature are computed in $V_h$ with the Helmholtz
   * filters of NormalVectorOperation and CurvatureOperation (damping $\eta_K = c h_K^2$ with the
   * filter parameters of NormalVectorData and CurvatureData):
   *
   * - normal vector: $(\boldsymbol{n}_h, \boldsymbol{w})_{\Omega_\Gamma} + \sum_K \eta^n_K
   *   (\nabla \boldsymbol{n}_h, \nabla \boldsymbol{w})_K = (\nabla \psi_h,
   *   \boldsymbol{w})_{\Omega_\Gamma}$, followed by the normalization of the nodal values,
   *   i.e., $\boldsymbol{n}_h$ is the interpolant of the unit normal pointing towards increasing
   *   level-set values,
   * - curvature: $(\kappa_h, w)_{\Omega_\Gamma} + \sum_K \eta^\kappa_K (\nabla \kappa_h, \nabla
   *   w)_K = -(\nabla \cdot \boldsymbol{n}_h, w)_{\Omega_\Gamma}$, i.e., $\kappa = -\nabla \cdot
   *   \boldsymbol{n}$ (positive for a circle with positive level-set values inside, as in
   *   MeltPoolDG). Since $\boldsymbol{n}_h$ is continuous, the volume integral is the weak
   *   divergence.
   *
   * Since $\psi_h$ is only defined on the cut cells, its zero level set is closed only if it does
   * not leave $\Omega_\Gamma$. Where the DG level set jumps near a vertex of the boundary of
   * $\Omega_\Gamma$, the projection may take the wrong sign at the nodes shared with a cell, which
   * is not cut, so that the zero level set of $\psi_h$ cuts the corner of this cell. With the sign
   * constraints (NarrowBandData::ContinuousLevelSet::sign_constraints), $\psi_h$ is required to
   * have the sign $s_{K'}$ of each cell $K'$, which is not cut, on the closure of $K'$ (the signs
   * of the semi-Lagrangian band, see compute()). After the projection, the regions of the cut
   * cells shared with such cells (vertices, edges and faces) are checked for nodal values with
   * $s_{K'} \psi_j \le 0$ and for sign changes between the nodes (on a lattice of $4p + 1$ points
   * per direction). In a region with a violation, all nodal values are constrained (frozen), such
   * that the next projection cannot change the region: to $\psi_j = s_{K'} \max(s_{K'} \psi_j, c
   * h_K)$ (sign constraint offset $c$), and, if the sign still changes between the nodes, the
   * nodal values in the interior of the region are raised to the smallest value $s_{K'} M$, $M =
   * c h_K 2^k$, which removes the sign change, but at most to the maximum of the vertex values (for
   * $p = 2$, this guarantees the sign by the nonnegative Bernstein coefficients). Then, the
   * projection is solved again with these constraints (elimination in
   * NarrowBandContinuousMassOperator). The check and the constrained projection are repeated at
   * most five times. The constraints only change $\psi_h$ locally at the boundary of
   * $\Omega_\Gamma$; the numbers of constrained nodal values, of constrained projections and of
   * remaining violations are reported (verbosity level 1).
   *
   * The boundary of $\Omega_\Gamma$ (faces to cells, which are not cut) is a natural boundary in
   * all three problems. The curvature statistics (mean, minimum and maximum) at the quadrature
   * points of the zero level set of $\psi_h$ are reported (verbosity level 1).
   *
   * The continuous fields live on an hp-DoFHandler with the same active FE indices as the cut
   * DoFHandler of the narrow band (FE_Q on the cut cells, non-dominating FE_Nothing otherwise),
   * which is attached to the scratch data of the narrow band. The normal vector is stored as a
   * block vector with one block per component.
   *
   * Of NormalVectorData and CurvatureData, only the filter parameters, the linear solvers, the
   * verbosity levels and CurvatureData::enable are used.
   *
   * @note The cut cells form a band of one to two cells, whose boundary consists of cell faces
   * (staircase) with a natural boundary condition. The damping $\eta_K = c h_K^2$ acts across the
   * whole band: e.g., the natural boundary condition of the normal vector suppresses its variation
   * along the interface within a layer of the width of the band. Thus, the damping biases the
   * normal vector and the curvature independently of $h$ (circle, $p = 2$: about 13 % for the
   * curvature with $c = 0.5$ for the normal vector, about 1 % with $c = 0.5$ for the curvature
   * only). Without damping (filter parameters zero, the default of the mp-narrow-band
   * application), the curvature on the interface converges with the order $p - 1$.
   */
  template <int dim, typename number>
  class NarrowBandContinuousLevelSet
  {
  public:
    using VectorType      = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;
    using MappingInfoType = CutUtil::MappingInfoType<dim, number>;

    /**
     * @brief Diagnostics of the last call of compute() (reduced over all processes).
     */
    struct Statistics
    {
      /// Maximum shift max |psi_h| / |grad psi_h| at the quadrature points of the zero level set
      /// of the DG level set.
      number max_interface_shift = 0.;

      /// Maximum interface shift relative to the cell size.
      number max_relative_interface_shift = 0.;

      /// Measure of the zero level set of the continuous level set.
      number interface_measure = 0.;

      /// Mean, minimum and maximum of the curvature at the quadrature points of the zero level
      /// set of the continuous level set.
      number mean_curvature = 0.;
      number min_curvature  = 0.;
      number max_curvature  = 0.;

      /// Number of iterations of the linear solvers (projection: sum over the constrained
      /// projections).
      unsigned int n_iterations_projection = 0;
      unsigned int n_iterations_normal     = 0;
      unsigned int n_iterations_curvature  = 0;

      /// Sign constraints: number of constrained nodal values, number of projections with
      /// constraints and number of remaining violations (nodal values or faces with the wrong
      /// sign).
      unsigned int n_sign_constrained_dofs  = 0;
      unsigned int n_sign_constraint_solves = 0;
      unsigned int n_sign_violations        = 0;
    };

    /**
     * @brief Constructor. Attaches the hp-DoFHandler of the continuous fields to @p scratch_data.
     *
     * @param scratch_data Scratch data of the narrow band.
     * @param data Parameters of the continuous level set.
     * @param normal_vector_data Parameters of the normal vector.
     * @param curvature_data Parameters of the curvature.
     * @param fe_collection_cut FE collection of the cut DoFHandler of the narrow band.
     * @param fe_degree Polynomial degree of the level-set field.
     * @param cut_dof_idx Index of the cut DoFHandler in @p scratch_data.
     * @param quad_idx Index of the quadrature rule in @p scratch_data.
     */
    NarrowBandContinuousLevelSet(ScratchData<dim, dim, number> &scratch_data,
                                 const typename NarrowBandData<number>::ContinuousLevelSet &data,
                                 const NormalVectorData<number>      &normal_vector_data,
                                 const CurvatureData<number>         &curvature_data,
                                 const dealii::hp::FECollection<dim> &fe_collection_cut,
                                 const unsigned int                   fe_degree,
                                 const unsigned int                   cut_dof_idx,
                                 const unsigned int                   quad_idx);

    /**
     * @brief Distribute the DoFs of the continuous fields with the active FE indices of the cut
     * DoFHandler. Has to be called after the DoFs of the cut DoFHandler have been distributed and
     * before the scratch data is rebuilt.
     */
    void
    distribute_dofs(const dealii::DoFHandler<dim> &dof_handler_cut);

    /**
     * @brief Reinitialize the vectors, operators and preconditioners after the scratch data has
     * been rebuilt.
     */
    void
    reinit();

    /**
     * @brief Compute the continuous level set, the normal vector and the curvature from the DG
     * level-set field on the cut cells.
     *
     * @param level_set_dg DG level-set field on the cut DoFHandler.
     * @param cell_signs Signs (+1, -1, 0: unknown) of the cells, which are not cut, indexed by
     * the active cell index (at least on the locally owned and ghost cells); required for the sign
     * constraints.
     */
    void
    compute(const VectorType &level_set_dg, const std::vector<int> *cell_signs = nullptr);

    /**
     * @brief Continuous level-set field.
     */
    const VectorType &
    get_level_set() const;

    /**
     * @brief Unit normal vector (one block per component).
     */
    const BlockVectorType &
    get_normal_vector() const;

    /**
     * @brief Curvature.
     */
    const VectorType &
    get_curvature() const;

    /**
     * @brief hp-DoFHandler of the continuous fields.
     */
    const dealii::DoFHandler<dim> &
    get_dof_handler() const;

    /**
     * @brief Index of the hp-DoFHandler of the continuous fields in the scratch data.
     */
    unsigned int
    get_dof_index() const;

    /**
     * @brief Diagnostics of the last call of compute().
     */
    const Statistics &
    get_statistics() const;

    /**
     * @brief Attach the continuous level set ("level_set_cg"), the normal vector ("normal", vector
     * field) and the curvature ("curvature") to the output.
     */
    void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const;

  private:
    /**
     * @brief Constrained L2-projection of the DG level set.
     */
    void
    project(const VectorType &level_set_dg);

    /**
     * @brief Sign constraints: set up the regions, in which the continuous level set has to have
     * the sign of a cell, which is not cut, check the projection and repeat it with constraints.
     */
    void
    enforce_sign_constraints(const std::vector<int> &cell_signs);

    /**
     * @brief Sign constraints: check the continuous level set in the regions and compute the
     * required nodal values (maximum over all regions and processes) in @p required_positive and
     * @p required_negative (magnitudes). Returns the number of violations (summed over all
     * processes).
     */
    unsigned int
    check_sign_constraints(VectorType &required_positive, VectorType &required_negative) const;

    /**
     * @brief Sign constraints: projection with the constrained nodal values.
     */
    void
    project_with_sign_constraints();

    /**
     * @brief Maximum interface shift at the quadrature points of the DG interface.
     */
    void
    compute_interface_shift(const VectorType &level_set_dg);

    /**
     * @brief Normal vector from the continuous level set.
     */
    void
    compute_normal_vector();

    /**
     * @brief Curvature from the normal vector.
     */
    void
    compute_curvature();

    /**
     * @brief Curvature statistics at the quadrature points of the continuous interface.
     */
    void
    compute_curvature_statistics();

    /**
     * @brief Print the diagnostics.
     */
    void
    print_statistics() const;

    /// Scratch data.
    ScratchData<dim, dim, number> &scratch_data;

    /// Parameters.
    const typename NarrowBandData<number>::ContinuousLevelSet data;
    const NormalVectorData<number>                            normal_vector_data;
    const CurvatureData<number>                               curvature_data;

    /// Polynomial degree.
    const unsigned int fe_degree;

    /// Index of the cut DoFHandler (DG level set) and of the quadrature rule.
    const unsigned int cut_dof_idx;
    const unsigned int quad_idx;

    /// FE collection of the continuous fields (FE_Q on the cut cells).
    const dealii::hp::FECollection<dim> fe_collection;

    /// FE collection of the vector-valued output of the normal vector.
    const dealii::hp::FECollection<dim> fe_collection_vector;

    /// hp-DoFHandler and (hanging-node) constraints of the continuous fields.
    dealii::DoFHandler<dim>           dof_handler;
    dealii::AffineConstraints<number> constraints;

    /// Index of the hp-DoFHandler in the scratch data.
    const unsigned int dof_idx;

    /// Element of the cut cells; required by FEPointEvaluation.
    const dealii::FE_Q<dim> fe_point;

    /// Surface quadrature of the zero level set of the DG level set (interface penalty).
    MappingInfoType mapping_info_surface_dg;

    /// Operators of the projection, the normal vector and the curvature.
    NarrowBandContinuousMassOperator<dim, number> projection_operator;
    NarrowBandContinuousMassOperator<dim, number> normal_operator;
    NarrowBandContinuousMassOperator<dim, number> curvature_operator;

    /// Preconditioners.
    Preconditioner<dim, VectorType, number>      projection_preconditioner;
    Preconditioner<dim, BlockVectorType, number> normal_preconditioner;
    Preconditioner<dim, VectorType, number>      curvature_preconditioner;

    /// Continuous fields and right-hand sides.
    VectorType      level_set;
    VectorType      rhs;
    BlockVectorType normal_vector;
    BlockVectorType rhs_normal;
    VectorType      curvature;

    /// Diagnostics.
    Statistics statistics;

    /// Sign constraints: region of a locally owned cut cell (vertex, edge or face), which is
    /// shared with a cell, which is not cut, with the sign of this cell, the cell-local indices of
    /// the nodes and the indices of the sample points in the region.
    struct SignRegion
    {
      int                       sign = 0;
      std::vector<unsigned int> dofs;
      std::vector<unsigned int> samples;
    };

    /// Sign constraints: locally owned cut cell with its regions.
    struct SignCell
    {
      std::vector<dealii::types::global_dof_index> dof_indices;
      number                                       offset = 0.;
      std::vector<SignRegion>                      regions;
    };

    /// Sign constraints: cut cells with regions, the sample points on the boundary of the unit
    /// cell and the values of the shape functions at the sample points.
    std::vector<SignCell>            sign_cells;
    std::vector<dealii::Point<dim>>  sign_sample_points;
    std::vector<std::vector<number>> sign_sample_shape_values;

    /// Sign constraints: constrained DoFs (nonzero entries), prescribed values, right-hand side
    /// of the constrained projection and constraints of the system matrix.
    VectorType                        sign_mask;
    VectorType                        sign_values;
    VectorType                        rhs_constrained;
    dealii::AffineConstraints<number> sign_matrix_constraints;

    /// DoFHandler and vector of the vector-valued output of the normal vector.
    mutable dealii::DoFHandler<dim> dof_handler_vector;
    mutable VectorType              normal_vector_output;
  };
} // namespace MeltPoolDG::LevelSet
