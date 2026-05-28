/**
 * @brief This operation solves the compressible Navier-Stokes equations, comprising
 * the primary variables
 *  - density (ρ)
 *  - momentum (ρ u)
 *  - volume-specific energy (ρ E)
 *
 * It is an extension of deal.II step-67 and is based on the paper
 *
 * Fehn, N., Wall, W. A., & Kronbichler, M. (2019). A matrix‐free high‐order
 * discontinuous Galerkin compressible Navier‐Stokes solver: A performance
 * comparison of compressible and incompressible formulations for turbulent
 * incompressible flows. International Journal for Numerical Methods in Fluids,
 * 89(3), 71-102.
 */

#pragma once

#include <deal.II/base/function.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/compressible_flow/dg_operator_base.hpp>
#include <meltpooldg/compressible_flow/operation_data.hpp>
#include <meltpooldg/compressible_flow/operation_scratch_data.hpp>
#include <meltpooldg/compressible_flow/output_post_processor.hpp>
#include <meltpooldg/compressible_flow/utils.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/time_integration/solution_history.hpp>
#include <meltpooldg/time_integration/time_integrator_base.hpp>

#include <memory>

namespace MeltPoolDG::CompressibleFlow
{
  /**
   * @brief Operation that performs a full time step for the compressible Navier-Stokes.
   */
  template <int dim, typename number, int n_species = 1>
  class DGOperation
  {
  public:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    /**
     * @brief Constructor.
     *
     * Initializes all internal data structures required to simulate compressible Navier-Stokes
     * flows.
     *
     * @param scratch_data Reference to the used ScratchData object.
     * @param flow_data Reference to the compressible flow data struct used.
     * @param material_data_in Reference to the material data struct.
     * @param flow_dof_idx Index of the used dof handler in @p scratch_data_in.
     * @param flow_quad_idx Index of the used quadrature object in @p scratch_data_in.
     * @param external_forces Pointer to a struct implementing external forces acting on the fluid.
     */
    explicit DGOperation(const ScratchData<dim, dim, number> &scratch_data,
                         const OperationData<number>         &flow_data,
                         const MaterialPhaseData<number>     &material_data_in,
                         unsigned int                         flow_dof_idx  = 0,
                         unsigned int                         flow_quad_idx = 0);

    /**
     * @brief Set up the required internal data structures.
     *
     * After a call to this function the solve() function of the class can be utilized.
     */
    void
    reinit();

    /**
     * @brief Solves the compressible Navier-Stokes equations for a single time step.
     *
     * @param current_time Current time at t^n.
     * @param time_step Current time step size.
     */
    void
    solve(const number current_time, const number time_step);

    /**
     * @brief Distribute the degrees of freedom to the passed dof handler object.
     *
     * @param dof_handler Dof handler object used for the compressible flow solver.
     */
    void
    distribute_dofs(dealii::DoFHandler<dim> &dof_handler) const;

    /**
     * @brief Set the boundary conditions.
     *
     * @param simulation_case dealii::Pointer to the considered simulation case class.
     * @param operation_name String for the name of the considered operation.
     *
     * @note The function simply passes the parameters to the set_boundary_conditions function in the
     * CompressibleFlowBoundaryConditions class.
     */
    void
    set_boundary_conditions(const std::shared_ptr<SimulationCaseBase<dim, number>> &simulation_case,
                            const std::string                                      &operation_name);

    /**
     * @brief Set a body force, e.g. gravity, specified by the passed function.
     *
     * @param body_force_in Function specifying the body force.
     *
     * @note The function simply passes the parameters to the corresponding operator function.
     */
    void
    set_body_force(std::unique_ptr<dealii::Function<dim>> body_force_in);

    void
    add_external_force(
      std::shared_ptr<ExternalFlowForce<dim, number, n_species>>         external_force_residuum,
      std::shared_ptr<ExternalFlowForceJacobian<dim, number, n_species>> external_force_jacobian);

    /**
     * @brief Compute the maximum time step size.
     *
     * The maximum time step size arises from the convective and viscous time step limits.
     * Optionally, it is printed to the console.
     *
     * @param do_print If true, the time step limit is printed to the console.
     *
     * @return The computed maximum time step size.
     */
    number
    compute_time_step_size(bool do_print = false) const;

    /**
     * @brief Estimate the eigenvalues of the Jacobian matrix.
     *
     * @param time_step The current time step size.
     * @param max_eigenvalues The maximum number of eigenvalues to estimate.
     * @return A vector of complex eigenvalues.
     */
    std::vector<std::complex<number>>
    estimate_jacobian_eigenvalues(const number time_step) const;

    /**
     * @brief Set the solution vector to the passed initial flow field state.
     *
     * @param function Initial condition of the flow field.
     */
    void
    set_initial_condition(const dealii::Function<dim> &function);

    /**
     * @brief Attach the solution to the passed data out object.
     *
     * @param data_out Object to which the solution vector is attached.
     */
    void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const;

    /**
     * @brief Constant getter function for the current solution vector.
     */
    const VectorType &
    get_solution() const;

    /**
     * @brief Getter function for the current solution vector.
     */
    VectorType &
    get_solution();

    /**
     * @brief Constant getter function for the DoFHandler.
     */
    const dealii::DoFHandler<dim> &
    get_dof_handler() const;

  private:
    /// Scratch data for compressible flows
    OperationScratchData<dim, number> flow_scratch_data;

    /// Compressible flow operator object
    std::unique_ptr<DGOperatorBase<dim, number, n_species>> comp_flow_operator;

    /// Object containing the data post processor for the different output options
    OutputManager<dim, number> output_manager;

    /**
     * @brief Compute the convective time step limit for the current mesh and flow field.
     *
     * @return Maximum convective time step size.
     */
    number
    compute_convective_time_step_limit() const;

    /**
     * @brief Compute the minimum density currently occurring in the flow field.
     *
     * @return Minimum density.
     */
    number
    compute_minimum_density() const;

    /**
     * @brief Set up the operator to suit the specified time integration scheme.
     *
     * @param external_forces Pointer to a struct implementing external forces acting on the fluid.
     */
    void
    setup_operator();
  };


  //! inlined functions
  template <int dim, typename number, int n_species>
  const dealii::LinearAlgebra::distributed::Vector<number> &
  DGOperation<dim, number, n_species>::get_solution() const
  {
    return flow_scratch_data.solution_history.get_current_solution();
  }

  template <int dim, typename number, int n_species>
  dealii::LinearAlgebra::distributed::Vector<number> &
  DGOperation<dim, number, n_species>::get_solution()
  {
    return flow_scratch_data.solution_history.get_current_solution();
  }

  template <int dim, typename number, int n_species>
  const dealii::DoFHandler<dim> &
  DGOperation<dim, number, n_species>::get_dof_handler() const
  {
    return flow_scratch_data.scratch_data.get_dof_handler(flow_scratch_data.dof_idx);
  }

} // namespace MeltPoolDG::CompressibleFlow
