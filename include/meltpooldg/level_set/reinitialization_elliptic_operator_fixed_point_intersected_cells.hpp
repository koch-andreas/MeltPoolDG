#pragma once

#include <deal.II/base/aligned_vector.h>

#include <deal.II/fe/fe_dgq.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/core/operator_base.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/level_set/reinitialization_data.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>

#include <memory>
#include <utility>

namespace MeltPoolDG::LevelSet
{
  /**
   * @brief Matrix-free operator for the fixed-point iteration of the elliptic level-set
   * reinitialization with a discontinuous Galerkin discretization, where the level-set field is
   * only defined on the cells intersected by the zero level-set iso-surface.
   *
   * The level-set field is expected to live on an hp-DoFHandler with FE_DGQ elements on the
   * intersected cells and FE_Nothing elements on all other cells, following the cell category
   * numbering of CutUtil::CellCategory (i.e., the FE index CutUtil::CellCategory::intersected
   * holds the FE_DGQ element). In the following, the union of the intersected cells is denoted as
   * $\Omega_\Gamma$, the faces shared by two intersected cells as $\mathcal{F}_\Gamma$ and the
   * zero level-set iso-surface of the initial level-set field $\phi^0$ as $\Gamma$.
   *
   * Following Basting & Kuzmin (2013) and Adams et al. (2019), the reinitialized level-set field
   * minimizes $\int_{\Omega_\Gamma} (|\nabla\phi| - 1)^2 \, \mathrm{d}x$, while the position of the
   * zero level-set is enforced weakly by a penalty term on $\Gamma$. The resulting nonlinear
   * problem is solved by the fixed-point iteration: find $\phi^{k+1}$ such that
   *
   * $a(\phi^{k+1}, v) = l(\phi^k; v)$ for all test functions $v$,
   *
   * with the symmetric interior penalty (SIPG) bilinear form
   *
   * $a(\phi, v) = (\nabla \phi, \nabla v)_{\Omega_\Gamma}
   *    - (\{\nabla \phi\} \cdot n, [v])_{\mathcal{F}_\Gamma}
   *    - ([\phi], \{\nabla v\} \cdot n)_{\mathcal{F}_\Gamma}
   *    + \sigma ([\phi], [v])_{\mathcal{F}_\Gamma}
   *    + \gamma (\phi, v)_\Gamma$
   *
   * and the linear form
   *
   * $l(\phi^k; v) = (g(|\nabla\phi^k|) \nabla\phi^k, \nabla v)_{\Omega_\Gamma}
   *    - (\{g(|\nabla\phi^k|) \nabla\phi^k\} \cdot n, [v])_{\mathcal{F}_\Gamma}$,
   *
   * where $g(s) = 1/s$ for $s > 1$ and $g(s) = 2 - s$ otherwise.
   *
   * The faces between intersected and non-intersected cells as well as the domain boundary are
   * treated as part of the boundary of $\Omega_\Gamma$, where the natural boundary condition of
   * the minimization problem applies. Thus, no face integrals are evaluated on these faces.
   *
   * @note Only the cells and faces of $\Omega_\Gamma$ are visited during the operator
   * evaluation. Since the vectors of the hp-DoFHandler only contain the DoFs of the intersected
   * cells, the cost of the reinitialization scales with the size of the interface.
   *
   * @tparam dim    Spatial dimension.
   * @tparam number Scalar number type.
   */
  template <int dim, typename number>
  class ReinitializationEllipticOperatorFixedPointIntersectedCells
    : public OperatorMatrixFree<dim, number>
  {
    using OperatorMatrixFree<dim, number>::vmult;
    using OperatorMatrixFree<dim, number>::create_rhs;
    using OperatorMatrixFree<dim, number>::compute_inverse_diagonal_from_matrixfree;

  public:
    using VectorType          = dealii::LinearAlgebra::distributed::Vector<number>;
    using VectorizedArrayType = dealii::VectorizedArray<number>;
    using MappingInfoType     = CutUtil::MappingInfoType<dim, number>;
    using PointEvaluationType = dealii::FEPointEvaluation<1, dim, dim, VectorizedArrayType>;

    /**
     * @brief Constructor.
     *
     * @param scratch_data_in Reference to the used ScratchData object.
     * @param reinit_data_in Reference to the object for reinitialization-specific data.
     * @param ls_dof_idx_in Index of the hp-DoFHandler of the level-set field in
     * @p scratch_data_in.
     * @param reinit_quad_idx_in Index of the used quadrature object in @p scratch_data_in.
     * @param mapping_info_surface_in Mapping information for the interface surface.
     */
    ReinitializationEllipticOperatorFixedPointIntersectedCells(
      const ScratchData<dim, dim, number> &scratch_data_in,
      const ReinitializationData<number>  &reinit_data_in,
      const unsigned int                   ls_dof_idx_in,
      const unsigned int                   reinit_quad_idx_in,
      const MappingInfoType               &mapping_info_surface_in);

    /**
     * @brief Apply the operator, i.e., the left-hand side of the fixed-point iteration, in a
     * matrix-free fashion.
     *
     * @param dst Output vector.
     * @param src Input vector.
     */
    void
    vmult(VectorType &dst, const VectorType &src) const final;

    /**
     * @brief Create the right-hand side vector of the fixed-point iteration.
     *
     * @param dst Right-hand side vector.
     * @param level_set_old Level-set field of the previous fixed-point iteration.
     */
    void
    create_rhs(VectorType &dst, const VectorType &level_set_old) const final;

    /**
     * @brief Reinitialize the internal data structures, e.g., after the DoF layout has changed.
     *
     * This function checks that the DoFHandler of the level-set field holds an FE_DGQ element
     * on the intersected cells and FE_Nothing elements otherwise and computes the interior
     * penalty parameter.
     */
    void
    reinit() final;

    /**
     * @brief Compute the system matrix from the matrix-free operator. Used by matrix-based
     * preconditioners.
     *
     * @param system_matrix Output sparse matrix.
     */
    void
    compute_system_matrix_from_matrixfree(
      dealii::TrilinosWrappers::SparseMatrix &system_matrix) const final;

    /**
     * @brief Compute the inverse diagonal of the system matrix. Used by the Jacobi preconditioner.
     *
     * @param diagonal Output vector containing the inverse diagonal entries.
     */
    void
    compute_inverse_diagonal_from_matrixfree(VectorType &diagonal) const final;

    /**
     * @brief Return the polynomial degree of the level-set field on the intersected cells.
     */
    unsigned int
    get_fe_degree() const;

  private:
    /**
     * @brief Check whether a range of cell batches belongs to the intersected cells.
     */
    bool
    is_intersected_cell_range(const std::pair<unsigned int, unsigned int> &cell_range) const;

    /**
     * @brief Check whether both adjacent cells of a range of face batches are intersected cells.
     */
    bool
    is_intersected_face_range(const std::pair<unsigned int, unsigned int> &face_range) const;

    /**
     * @brief Create a cell integrator for the intersected cells.
     */
    FECellIntegrator<dim, 1, number>
    create_cell_integrator() const;

    /**
     * @brief Create a face integrator for faces between two intersected cells.
     *
     * @param is_interior_face Select the interior (minus) or exterior (plus) side of the face.
     */
    FEFaceIntegrator<dim, 1, number>
    create_face_integrator(const bool is_interior_face) const;

    /**
     * @brief Create the point evaluator for the surface integrals over the interface.
     */
    PointEvaluationType
    create_point_evaluator() const;

    /**
     * @brief Apply the cell contribution of the left-hand side, i.e., the Laplace operator and the
     * interface penalty term, to the DoF values stored in @p phi.
     *
     * On input, the DoF values of @p phi contain the (unevaluated) source values; on output, they
     * contain the integrated values.
     *
     * @param phi Cell integrator of the level-set field.
     * @param point_eval Point evaluator for the surface integral over the interface.
     * @param penalty_buffer Scratch buffer for the contribution of the interface penalty term.
     */
    void
    lhs_cell_operation(FECellIntegrator<dim, 1, number>           &phi,
                       PointEvaluationType                        &point_eval,
                       dealii::AlignedVector<VectorizedArrayType> &penalty_buffer) const;

    /**
     * @brief Evaluate the SIPG contribution of a face between two intersected cells at the
     * quadrature points. The face integrators need to be evaluated (values and gradients) before
     * and integrated (values and gradients) after a call to this function.
     *
     * @param phi_m Face integrator on the interior side of the face.
     * @param phi_p Face integrator on the exterior side of the face.
     */
    void
    lhs_face_operation(FEFaceIntegrator<dim, 1, number> &phi_m,
                       FEFaceIntegrator<dim, 1, number> &phi_p) const;

    /**
     * @brief Coefficient $g(|\nabla\phi|)$ of the right-hand side.
     *
     * @param gradient_norm Norm of the level-set gradient.
     */
    static VectorizedArrayType
    rhs_coefficient(const VectorizedArrayType &gradient_norm);

    /// Reference to scratch data containing mesh, geometry, and FE evaluation utilities.
    const ScratchData<dim, dim, number> &scratch_data;

    /// Reference to the reinitialization parameters.
    const ReinitializationData<number> &reinit_data;

    /// Quadrature index for the cell and face integrals.
    const unsigned int reinit_quad_idx;

    /// Mapping information for the surface integrals over the interface.
    const MappingInfoType &mapping_info_surface;

    /// Polynomial degree of the level-set field on the intersected cells.
    unsigned int fe_degree = 0;

    /// FE of the intersected cells; required by FEPointEvaluation.
    std::unique_ptr<dealii::FE_DGQ<dim>> fe_point;

    /// Number of DoFs per intersected cell.
    unsigned int n_dofs_per_cell = 0;

    /// Interior penalty parameter of the SIPG discretization.
    number interior_penalty = 0.0;
  };
} // namespace MeltPoolDG::LevelSet
