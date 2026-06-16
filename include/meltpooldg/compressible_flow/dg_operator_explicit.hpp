#pragma once

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/compressible_flow/convective_kernels.hpp>
#include <meltpooldg/compressible_flow/dg_operator_base.hpp>
#include <meltpooldg/compressible_flow/kernels_n_species.hpp>
#include <meltpooldg/compressible_flow/operation_scratch_data.hpp>
#include <meltpooldg/compressible_flow/utils.hpp>
#include <meltpooldg/compressible_flow/viscous_kernels.hpp>
#include <meltpooldg/time_integration/explicit_low_storage_runge_kutta_integrator.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>
#include <meltpooldg/utilities/dg_generic_convection_diffusion_worker.hpp>

#include <functional>
#include <memory>
#include <utility>

namespace MeltPoolDG::CompressibleFlow
{
  /**
   * @brief Operator for the matrix-free evaluation of a compressible single-phase flow cutDG
   * formulation for explicit time integration.
   *
   * @tparam dim Dimension of the considered simulation case.
   * @tparam number Floating point format type.
   * @tparam n_species The number of species in a multi-component case. The default value is 1 for a single-component flow.
   */
  template <int dim, typename number, int n_species = 1>
  class DGOperatorExplicit final : public DGOperatorBase<dim, number, n_species>
  {
  public:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    using ConservedVariables         = ConservedVariablesType<dim, number, n_species>;
    using ConservedVariablesGradient = ConservedVariablesGradientType<dim, number, n_species>;

    using FlowFluxType   = FluxType<dim, number, n_species>;
    using FlowSourceType = SourceType<dim, number, n_species>;

    using ConvectiveKernel =
      NSpeciesConvectiveFlux<dim, n_species, number, ConservedVariables, FlowFluxType>;
    using DiffusiveKernel = NSpeciesDiffusiveFlux<dim,
                                                  n_species,
                                                  number,
                                                  ConservedVariables,
                                                  ConservedVariablesGradient,
                                                  FlowFluxType>;

    using ConvectionDiffusionOperator =
      Utils::DGConvectionDiffusionOperator<dim, number, ConvectiveKernel, DiffusiveKernel>;

    using ConvectionOperator = Utils::DGConvectionOperator<dim, number, ConvectiveKernel>;

    /**
     * @brief Constructor.
     *
     * @param flow_scratch_data Reference to the flow scratch data object (usually owned by the
     * corresponding operation class).
     */
    explicit DGOperatorExplicit(OperationScratchData<dim, number> &flow_scratch_data);

    /**
     * @brief Advances solver by a single time step.
     *
     * This function performs a single explicit time step of size @p time_step starting from the
     * solution at time @p time.
     *
     * @note The function does not take care about updating the solution history object or similar
     * operations which are not directly related to the integration. It **only** advances the
     * solution by a single time step starting from the current solution in the solution history
     * object of the @ref flow_scratch_data object.
     */
    void
    advance_time_step(number time, number time_step) override;

    /**
     * @brief Reinitialize the internal data structures.
     *
     * The reinitialization includes setting a new required size for the solution history object
     * according to the demands of the used time integrator.
     */
    void
    reinit() override;

    /**
     * @brief Computes the value of the function f(y) for the compressible Navier-Stokes equations of the
     * form y' = f(y).
     *
     * From a discretization perspective, f(y) is given by f(y) = M^(-1) * F(y),
     * where M is the mass matrix and F(y) is the sum of all flux contributions: F_v + F_c + F_rhs.
     *
     * @param time The current time at which the function is evaluated.
     * @param dst Vector where the computed value of f(y) is stored.
     * @param src The solution vector, y, at the current time.
     * @param func A function to be executed after f(y) has been computed. This function is applied
     * to the resulting vector in @p dst.
     */
    void
    apply_operator(number                                                 time,
                   number                                                 time_step,
                   VectorType                                            &dst,
                   const VectorType                                      &src,
                   const std::function<void(unsigned int, unsigned int)> &func) const;

    /**
     * @brief Add external fluid forces (e.g. gravity, ...).
     *
     * @param external_force A provided shared pointer to an external force definition.
     */
    void
    add_external_force(std::shared_ptr<ExternalFlowForce<dim, number, n_species>> external_force,
                       std::shared_ptr<ExternalFlowForceJacobian<dim, number, n_species>>) override;

  private:
    /// Scratch data for compressible flows
    OperationScratchData<dim, number> &flow_scratch_data;

    /// Time integrator class used for the time integration.
    TimeIntegration::LowStorageExplicitRungeKuttaIntegrator<number> time_integrator;

    /// This pointer may hold an instance of an external fluid force contribution
    /// (e.g., gravity, body forces, or user - defined source terms)
    std::vector<std::shared_ptr<ExternalFlowForce<dim, number, n_species>>> external_forces;

    /// Current time step size
    mutable number current_time_step;

    /**
     * Return true if the flow is viscous, false otherwise. In single component flows, this is
     * determined by the value of the dynamic viscosity in the material data. In multi-component
     * flows, a flow is considered viscous if the dynamic viscosity is greater than zero for at
     * least one of the components.
     */
    bool
    is_viscous() const;

    /**
     * @brief Local cell applier.
     *
     * Computes the cell contribution to the rhs if the compressible Navier-Stokes equations are
     * written in the form y'=rhs(y).
     *
     * @param matrix_free Matrix free object on which the applier works on.
     * @param dst Destination vector to which the result is added.
     * @param src Current solution.
     * @param cell_range Cell range which is considered in the applier.
     */
    void
    local_apply_cell(const dealii::MatrixFree<dim, number>       &matrix_free,
                     VectorType                                  &dst,
                     const VectorType                            &src,
                     const std::pair<unsigned int, unsigned int> &cell_range) const;

    /**
     * @brief Local face applier.
     *
     * Computes the face contribution to the rhs if the compressible Navier-Stokes equations are
     * written in the form y'=rhs(y).
     *
     * @param matrix_free Matrix free object on which the applier works on.
     * @param dst Destination vector to which the result is added.
     * @param src Current solution.
     * @param face_range Face range which is considered in the applier.
     */
    void
    local_apply_face(const dealii::MatrixFree<dim, number>       &matrix_free,
                     VectorType                                  &dst,
                     const VectorType                            &src,
                     const std::pair<unsigned int, unsigned int> &face_range) const;

    /**
     * @brief Local boundary face applier.
     *
     * Computes the boundary face contribution to the rhs if the compressible Navier-Stokes
     * equations are written in the form y'=rhs(y).
     *
     * @param matrix_free Matrix free object on which the applier works on.
     * @param dst Destination vector to which the result is added.
     * @param src Current solution.
     * @param face_range Boundary face range which is considered in the applier.
     */
    void
    local_apply_boundary_face(const dealii::MatrixFree<dim, number>       &matrix_free,
                              VectorType                                  &dst,
                              const VectorType                            &src,
                              const std::pair<unsigned int, unsigned int> &face_range) const;
  };
} // namespace MeltPoolDG::CompressibleFlow
