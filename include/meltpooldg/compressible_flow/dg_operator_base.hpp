#pragma once

#include <meltpooldg/compressible_flow/utils.hpp>

#include <memory>

namespace MeltPoolDG::CompressibleFlow
{
  /**
   * @brief Interface of the compressible flow operator interacting with the compressible flow
   * operation.
   */
  template <int dim, typename number, int n_species = 1>
  class DGOperatorBase
  {
  public:
    virtual ~DGOperatorBase() = default;

    /**
     * @brief Advances solver by a single time step.
     *
     * This function performs a single time step of size @p time_step starting from the solution
     * at time @p time. For a detailed description refer to the documentation of the derived
     * classes.
     */
    virtual void
    advance_time_step(number time, number time_step) = 0;

    virtual void vmult(dealii::LinearAlgebra::distributed::Vector<number> &dst,
                   const dealii::LinearAlgebra::distributed::Vector<number> &src) const
    {}

    virtual void
    compute_residual(number            current_time,
                     number            time_step,
                     const dealii::LinearAlgebra::distributed::Vector<number> &src,
                     dealii::LinearAlgebra::distributed::Vector<number>       &dst,
                     const dealii::LinearAlgebra::distributed::Vector<number> &old_solution) const
    {}

    virtual void
    set_eigenvalue_flag()
    {}

    virtual void
    reset_eigenvalue_flag()
    {}


    /**
     * @brief Reinit the operator.
     *
     * This function is intended to be called every time at which the data structure needs to be
     * updated.
     */
    virtual void
    reinit() = 0;

    virtual void
    add_external_force(
      std::shared_ptr<ExternalFlowForce<dim, number, n_species>> external_force_residuum,
      std::shared_ptr<ExternalFlowForceJacobian<dim, number, n_species>>
        external_force_jacobian) = 0;
  };
} // namespace MeltPoolDG::CompressibleFlow
