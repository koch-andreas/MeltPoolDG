#pragma once

#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/compressible_flow/convective_kernels.hpp>
#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/dg_operator_base.hpp>
#include <meltpooldg/compressible_flow/operation_scratch_data.hpp>
#include <meltpooldg/compressible_flow/viscous_kernels.hpp>
#include <meltpooldg/time_integration/bdf_time_integration.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>


namespace MeltPoolDG::CompressibleFlow
{
  /**
   * @brief Operator for the matrix-free evaluation of a compressible single-phase flow cutDG
   * formulation for implicit time integration.
   *
   * @tparam dim Dimension of the considered simulation case.
   * @tparam number Floating point format type.
   * @tparam is_viscous Indicates whether the flow is viscous.
   */
  template <int dim, typename number, bool is_viscous = true>
  class DGOperatorImplicit final : public DGOperatorBase<dim, number>
  {
  public:
    using VectorType                 = dealii::LinearAlgebra::distributed::Vector<number>;
    using ConservedVariables         = ConservedVariablesType<dim, number>;
    using ConservedVariablesGradient = ConservedVariablesGradientType<dim, number>;

    /**
     * @brief Constructor.
     *
     * @param flow_scratch_data Reference to the flow scratch data object (usually owned by the
     * corresponding operation class).
     */
    explicit DGOperatorImplicit(OperationScratchData<dim, number> &flow_scratch_data);

    /**
     * @brief Reinitialize the internal data structures.
     *
     * The reinitialization includes setting a new required size for the solution history object
     * according to the demands of the used time integrator.
     */
    void
    reinit() override;

    /**
     * @brief Advances solver by a single time step.
     *
     * This function performs a single implicit time step of size @p time_step starting from the
     * solution at time @p time.
     *
     * @note The function does not take care about updating the solution history object or similar
     * operations which are not directly related to the integration. It **only** advances the
     * solution by a single time step starting from the current solution in the solution history
     * object of the @ref flow_scratch_data object.
     */
    void
    advance_time_step(number time, number time_step) override;

    void
    add_external_force(
      std::shared_ptr<ExternalFlowForce<dim, number>>         external_force_residuum,
      std::shared_ptr<ExternalFlowForceJacobian<dim, number>> external_force_jacobian) override;

    /**
     * @brief Compute the matrix representation of the Jacobian.
     *
     * @param sparse_matrix Spars matrix in which the resulting matrix representation is stored.
     */
    void
    compute_system_matrix_from_matrixfree(
      dealii::TrilinosWrappers::SparseMatrix &sparse_matrix) const;

    /**
     * @brief Compute the inverse elements of the diagonal of the Jacobian.
     *
     * @param diagonal Vector in which the inverse elements of the diagonal are stored.
     */
    void
    compute_inverse_diagonal_from_matrixfree(VectorType &diagonal) const;

    /**
     * @brief Compute the result of J*x, where J is the Jacobian.
     *
     * The method on how to compute/approximate the Jacobian is defined by the user in the
     * compressible flow data.
     *
     * @param src Source vector x with which the Jacobian gets multiplied.
     * @param dst Location at which the result of J*x is stored.
     *
     * @throws Exception if the layout of the two given vectors @p src and @p dst are not identical.
     *
     * @note This function assumes that the function set_stage_constants() has been called in advance.
     */
    void
    apply_jacobian(number            time_step,
                   VectorType       &dst,
                   const VectorType &src) const; // TODO

    /**
     * @brief Compute the negative residual.
     *
     * Compute the negative residual, i.e. -(y'-F(y)) where y' is the temporal derivative of the
     * primary variables and F is the sum of all fluxes occuring in the compressible Navier-Stokes
     * equations (right-hand side).
     *
     * @param current_time Current physical time.
     * @param src Current solution vector used to compute the residual.
     * @param dst Vector in which the residual is stored.
     *
     * @throws Assert if the layout of the two given vectors @p src and @p dst are not identical.
     *
     * @note This function assumes that the function set_stage_constants() has been called in advance.
     */
    void
    compute_residual(number            current_time,
                     number            time_step,
                     const VectorType &src,
                     VectorType       &dst,
                     const VectorType &old_solution) const; // TODO


    /**
     * @brief Local cell operations at the given quadrature point for computing the Jacobian.
     *
     * @param delta_phi Cell integrator for the change in the primary variables. Quadrature point
     * distributions are added to this integrator.
     * @param phi Cell integrator for the primary variables.
     * @param q_index Quadrature point index.
     */
    void
    local_cell_jacobian_kernel(
      FECellIntegrator<dim, dim + 2, number>                      &delta_phi,
      const FECellIntegrator<dim, dim + 2, number>                &phi,
      unsigned int                                                 q_index,
      std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> cell_iterators) const;

    /**
     * @brief Local face operations at the given quadrature point for computing the Jacobian.
     *
     * @param delta_phi_m Face integrator for the change in the primary variables on the inner face.
     * Quadrature point distributions are added to this integrator.
     * @param delta_phi_p Face integrator for the change in the primary variables on the outer face.
     * Quadrature point distributions are added to this integrator.
     * @param phi_m Cell integrator for the primary varibales on the inner face.
     * @param phi_p Cell integrator for the primary varibales on the outer face.
     * @param q_index Quadrature point index.
     */
    void
    local_face_jacobian_kernel(FEFaceIntegrator<dim, dim + 2, number>       &delta_phi_m,
                               FEFaceIntegrator<dim, dim + 2, number>       &delta_phi_p,
                               const FEFaceIntegrator<dim, dim + 2, number> &phi_m,
                               const FEFaceIntegrator<dim, dim + 2, number> &phi_p,
                               unsigned int                                  q_index) const;

    /**
     * @brief Local boundary face operations at the given quadrature point for computing the Jacobian.
     *
     * @param delta_phi_m Face integrator for the change in the primary variables on the inner face.
     * Quadrature point distributions are added to this integrator.
     * @param phi_m Cell integrator for the primary varibales on the inner face.
     * @param q_index Quadrature point index.
     */
    void
    local_boundary_face_jacobian_kernel(FEFaceIntegrator<dim, dim + 2, number>       &delta_phi_m,
                                        const FEFaceIntegrator<dim, dim + 2, number> &phi_m,
                                        unsigned int q_index) const;

  private:
    /// This vector is used to compute the temporal derivative y' in the residual computation by
    /// y'=(current_solution-old_solution)/dt. We do not take the old_solution directly from
    /// solution_history as providing the option to pass a (fictional) old solution can have
    /// performance advantages, e.g. in the bdf time integration. It can be set by the function
    /// set_stage_constants().
    mutable const VectorType *time_integrator_old_solution = nullptr;

    /// This vector is used in the approximation of the jacobian by finite differences. Here the
    /// vector stores the residual with a disturbed input.
    mutable VectorType disturbed_residual;

    /// Current time step size. This needs to be stored as this value is required by the local cell
    /// appliers.
    mutable number current_time_step;

    mutable number current_time = 0.;

    /// Scratch data for compressible flows
    OperationScratchData<dim, number> &flow_scratch_data;

    /// Time integrator class used for the time integration.
    TimeIntegration::BDFIntegrator<dim, number> time_integrator;

    /// Object for the convective term evaluations
    ConvectiveKernels<dim, number> convective_terms;

    /// Object for the viscous term evaluations
    ViscousKernels<dim, number> viscous_terms;

    /// This set of pointers may hold a list of external fluid force contributions to the residuum
    /// (e.g., gravity, or user-defined source terms)
    std::vector<std::shared_ptr<ExternalFlowForce<dim, number>>> external_forces_residual;

    /// This set of pointers may hold a list of external fluid force contributions to the jacobian
    /// (e.g., gravity, or user-defined source terms)
    std::vector<std::shared_ptr<ExternalFlowForceJacobian<dim, number>>> external_forces_jacobian;

    /**
     * @brief Compute the result of J*x, where J is the Jacobian computed analytically.
     *
     * @param src Source vector x with which the Jacobian gets multiplied.
     * @param dst Location at which the result of J*x is stored.
     *
     * @throws Exception if the layout of the two given vectors @p src and @p dst are not identical.
     */
    void
    apply_jacobian_analytic(const VectorType &src, VectorType &dst) const;

    /**
     * @brief Compute the result of J*x, where J is the Jacobian approximated by finite differences.
     *
     * @param src Source vector x with which the Jacobian gets multiplied.
     * @param dst Location at which the result of J*x is stored.
     *
     * @throws Exception if the layout of the two given vectors @p src and @p dst are not identical.
     */
    void
    apply_jacobian_finite_differences(const VectorType &src, VectorType &dst) const;

    /**
     * @brief The local cell applier computing the residual contribution of the cell.
     *
     * @param matrix_free Matrix free object on which the applier works on.
     * @param dst Destination vector to which the result is added.
     * @param src Current solution.
     * @param cell_range Cell range which is considered in the applier.
     */
    void
    local_cell_residual(const dealii::MatrixFree<dim, number>                    &matrix_free,
                        dealii::LinearAlgebra::distributed::Vector<number>       &dst,
                        const dealii::LinearAlgebra::distributed::Vector<number> &src,
                        const std::pair<unsigned int, unsigned int>              &cell_range) const;

    /**
     * @brief The local cell applier computing the residual contribution of the inner faces.
     *
     * @param matrix_free Matrix free object on which the applier works on.
     * @param dst Destination vector to which the result is added.
     * @param src Current solution.
     * @param face_range Face range which is considered in the applier.
     */
    void
    local_face_residual(const dealii::MatrixFree<dim, number>                    &matrix_free,
                        dealii::LinearAlgebra::distributed::Vector<number>       &dst,
                        const dealii::LinearAlgebra::distributed::Vector<number> &src,
                        const std::pair<unsigned int, unsigned int>              &face_range) const;

    /**
     * @brief The local cell applier computing the residual contribution of the boundary faces.
     *
     * @param matrix_free Matrix free object on which the applier works on.
     * @param dst Destination vector to which the result is added.
     * @param src Current solution.
     * @param face_range Face range which is considered in the applier.
     */
    void
    local_boundary_face_residual(const dealii::MatrixFree<dim, number>              &matrix_free,
                                 dealii::LinearAlgebra::distributed::Vector<number> &dst,
                                 const dealii::LinearAlgebra::distributed::Vector<number> &src,
                                 const std::pair<unsigned int, unsigned int> &face_range) const;

    /**
     * @brief Computes the cell contribution of the Jacobian.
     *
     * Computes the contribution of the cells to the product of the Jacobian and the provided source
     * vector for a specified cell range. The current solution of the primary variables, required
     * for the Jacobian computation, is retrieved from the solution_history object.
     *
     * @param matrix_free The matrix-free object utilized by the applier.
     * @param dst The destination vector to which the computed result is added.
     * @param src The source vector to be multiplied with the Jacobian.
     * @param cell_range The range of cells considered by the applier.
     */
    void
    local_cell_jacobian(const dealii::MatrixFree<dim, number>       &matrix_free,
                        VectorType                                  &dst,
                        const VectorType                            &src,
                        const std::pair<unsigned int, unsigned int> &cell_range) const;

    /**
     * @brief Computes the inner face contribution of the Jacobian.
     *
     * Computes the contribution of the inner faces to the product of the Jacobian and the provided
     * source vector for a specified face range. The current solution of the primary variables,
     * required for the Jacobian computation, is retrieved from the solution_history object.
     *
     * @param matrix_free The matrix-free object utilized by the applier.
     * @param dst The destination vector to which the computed result is added.
     * @param src The source vector to be multiplied with the Jacobian.
     * @param face_range The range of faces considered by the applier.
     */
    void
    local_face_jacobian(const dealii::MatrixFree<dim, number>       &matrix_free,
                        VectorType                                  &dst,
                        const VectorType                            &src,
                        const std::pair<unsigned int, unsigned int> &face_range) const;

    /**
     * @brief Computes the boundary face contribution of the Jacobian.
     *
     * Computes the contribution of the boundary faces to the product of the Jacobian and the
     * provided source vector for a specified face range. The current solution of the primary
     * variables, required for the Jacobian computation, is retrieved from the solution_history
     * object.
     *
     * @param matrix_free The matrix-free object utilized by the applier.
     * @param dst The destination vector to which the computed result is added.
     * @param src The source vector to be multiplied with the Jacobian.
     * @param face_range The range of faces considered by the applier.
     */
    void
    local_boundary_face_jacobian(const dealii::MatrixFree<dim, number>              &matrix_free,
                                 dealii::LinearAlgebra::distributed::Vector<number> &dst,
                                 const dealii::LinearAlgebra::distributed::Vector<number> &src,
                                 const std::pair<unsigned int, unsigned int> &face_range) const;
  };
} // namespace MeltPoolDG::CompressibleFlow
