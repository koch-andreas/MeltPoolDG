#pragma once

#include <deal.II/base/config.h>

#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/flow/compressible_flow_data.hpp>
#include <meltpooldg/flow/compressible_flow_eos_utils_base.hpp>
#include <meltpooldg/flow/compressible_flow_ideal_gas_utils.hpp>
#include <meltpooldg/flow/compressible_flow_noble_abel_stiffened_gas_utils.hpp>
#include <meltpooldg/flow/compressible_flow_stiffened_gas_utils.hpp>
#include <meltpooldg/flow/compressible_flow_types.hpp>
#include <meltpooldg/flow/compressible_flow_utils.hpp>
#include <meltpooldg/utilities/dealii_tensor.hpp>

#include <utility>


namespace MeltPoolDG::Flow
{
  /**
   * @brief Viscous kernel operations for compressible flow solvers.
   *
   * This struct implements the evaluation of viscous fluxes and their linearizations
   * for compressible flow governed by the Euler or Navier–Stokes equations. It supports
   * flux evaluation on both volume cells and faces.
   *
   * It also provides functionality to compute the Jacobian (linearized form) of these fluxes,
   * which is required for implicit schemes.
   */
  template <int dim, typename number>
  struct CompressibleFlowViscousKernels
  {
    using ConservedVariablesType = CompressibleFlow::ConservedVariablesType<dim, number>;
    using ConservedVariablesGradType =
      CompressibleFlow::ConservedVariablesGradientType<dim, number>;

    explicit CompressibleFlowViscousKernels(
      const CompressibleFlowMaterial<dim, number> &material_in);
    /**
     * @brief Calculate the viscous stress tensor.
     *
     * Calculate the viscous stress tensor τ given by τ = μ*(grad(u)+grad(u)^T-2/3*(grad*u)*I),
     * where μ is the dynamic viscosity and I representing the identity matrix.
     *
     * @param grad_u Current gradient of the velocity field.
     *
     * @return Viscous stress tensor τ.
     */
    inline DEAL_II_ALWAYS_INLINE //
      dealii::Tensor<2, dim, dealii::VectorizedArray<number>>
      calculate_viscous_stress_tensor(
        const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> &grad_u) const;

    inline DEAL_II_ALWAYS_INLINE //
      dealii::Tensor<2, dim, dealii::VectorizedArray<number>>
      calculate_viscous_stress_tensor_damped(
        const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> &grad_u,
        const ConservedVariablesType     &conserved_variables,
      const ConservedVariablesGradType &grad_conserved_variables) const;

    /**
     * @brief Calculate the viscous flux F_v, i.e. F_v(u, grad(u)).
     *
     * @param conserved_variables Current values of the conserved variables.
     * @param grad_conserved_variables Current gradient of the conserved variables.
     *
     * @return Viscous flux F_v(u, grad(u)).
     */
    inline DEAL_II_ALWAYS_INLINE //
      ConservedVariablesGradType
      calculate_viscous_flux(const ConservedVariablesType     &conserved_variables,
                             const ConservedVariablesGradType &grad_conserved_variables,
                             const dealii::VectorizedArray<number> &quad_points = dealii::make_vectorized_array(0.)) const;

    /**
     * @brief Calculate the viscous numerical flux F_v^* using the symmetric interior penalty
     * method.
     *
     * @param u_m Current values of the conserved variables on the inner face.
     * @param u_p Current values of the conserved variables on the outer type.
     * @param grad_u_m Current values of the gradient of the conserved variables on the inner face.
     * @param grad_u_p Current values of the gradient of the conserved variables on the outer face.
     * @param normal Outer facing normal vector.
     * @param penalty_parameter Symmetric interior penalty parameter.
     *
     * @return Visocus numerical flux.
     */
    inline DEAL_II_ALWAYS_INLINE //
      ConservedVariablesType
      calculate_viscous_numerical_flux(
        const ConservedVariablesType                                  &u_m,
        const ConservedVariablesType                                  &u_p,
        const ConservedVariablesGradType                              &grad_u_m,
        const ConservedVariablesGradType                              &grad_u_p,
        const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
        dealii::VectorizedArray<number>                                penalty_parameter) const;

    /**
     * @brief Calculate the visocus flux, where jump(u) instead of grad(u) is used resulting in the
     * F_v(u, jump(u)).
     *
     * @param u_m Current values of the conserved variables on the inner face.
     * @param u_p Current values of the conserved variables on the outer type.
     * @param normal Outer facing normal vector.
     *
     * @return Viscous flux F_v(u, jump(u)).
     */
    inline DEAL_II_ALWAYS_INLINE //
      std::pair<ConservedVariablesGradType, ConservedVariablesGradType>
      calculate_viscous_numerical_flux_gradient(
        const ConservedVariablesType                                  &u_m,
        const ConservedVariablesType                                  &u_p,
        const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal) const;

    /**
     * @brief Compute the linearization of the viscous numerical flux with respect to the primary
     * variables.
     *
     * @param w_q Primary variables on the inner (first) and outer (second) face.
     * @param grad_w_q Gradient of the primary variables on the inner (first) and outer
     * (second) face.
     * @param delta_w_q Change in the primary variables n the inner (first) and outer (second) face.
     * @param grad_delta_w_q Gradient of the change in the primary variables on inner (first) and
     * outer (second) face.
     * @param normal Outer facing normal vector.
     * @param penalty_parameter Value of the symmetric interior penalty parameter.
     *
     * @return Linearized viscous numerical flux.
     */
    ConservedVariablesGradType
    calculate_jacobian_viscous_numerical_flux(
      const std::pair<ConservedVariablesType, ConservedVariablesType>         &w_q,
      const std::pair<ConservedVariablesGradType, ConservedVariablesGradType> &grad_w_q,
      const std::pair<ConservedVariablesType, ConservedVariablesType>         &delta_w_q,
      const std::pair<ConservedVariablesGradType, ConservedVariablesGradType> &grad_delta_w_q,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>>           &normal,
      dealii::VectorizedArray<number> penalty_parameter) const;

    /**
     * @brief Compute the linearization of the viscous flux with respect to the primary variables.
     *
     * @param w_q Primary variables.
     * @param grad_w_q Gradient of the primary variables.
     * @param delta_w_q Change in the primary variables.
     * @param grad_delta_w_q Gradient of the change in the primary variables.
     *
     * @return Linearized viscous flux.
     */
    ConservedVariablesGradType
    calculate_jacobian_viscous_flux(const ConservedVariablesType     &w_q,
                                    const ConservedVariablesGradType &grad_w_q,
                                    const ConservedVariablesType     &delta_w_q,
                                    const ConservedVariablesGradType &grad_delta_w_q) const;

    /**
     * @brief Compute the linearization of the viscous numerical flux jump term.
     *
     * Compute the Jacobian of the jump term in the viscous numerical flux. For the used symmetric
     * interior penalty approach the jump therm is given by the penalty parameter multiplied with
     * the jump in the primary variables.
     *
     * @param delta_w_q Change in the primary variables n the inner (first) and outer (second) face.
     * @param normal Outer facing normal vector.
     * @param penalty_parameter Value of the symmetric interior penalty parameter.
     *
     * @return Linearized jump term of the viscous numerical flux jump term.
     */
    ConservedVariablesGradType
    calculate_jacobian_viscous_numerical_flux_jump_term(
      const std::pair<ConservedVariablesType, ConservedVariablesType> &delta_w_q,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>>   &normal,
      dealii::VectorizedArray<number>                                  penalty_parameter) const;

  private:
    /// Object which provides all relevant material properties for a specific phase
    const CompressibleFlowMaterial<dim, number> &material;

    /// precomputed constant
    number lambda_div_c;
  };

  /********************************************************************************************
   * Inlined function definitions
   * *************************************************************************************+****/
  template <int dim, typename number>
  CompressibleFlowViscousKernels<dim, number>::CompressibleFlowViscousKernels(
    const CompressibleFlowMaterial<dim, number> &material_in)
    : material(material_in)
  {
    lambda_div_c = material.data.thermal_conductivity / material.data.specific_gas_constant *
                   (material.data.gamma - 1.0);
  }

  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<2, dim, dealii::VectorizedArray<number>>
    CompressibleFlowViscousKernels<dim, number>::calculate_viscous_stress_tensor(
      const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> &grad_u) const
  {
    const dealii::VectorizedArray<number> div_u = dealii::trace(grad_u);

    dealii::Tensor<2, dim, dealii::VectorizedArray<number>> out;
    for (unsigned int d = 0; d < dim; ++d)
      {
        for (unsigned int e = 0; e < dim; ++e)
          out[d][e] = material.data.dynamic_viscosity * (grad_u[d][e] + grad_u[e][d]);
        out[d][d] -= 2. / 3. * material.data.dynamic_viscosity * div_u;
      }

    // --- compression sensor: only activate in compression
    dealii::VectorizedArray<number> chi;

    for (unsigned int v = 0; v < chi.size(); ++v)
      {
        const number div_val = div_u[v];
        chi[v] = (div_val < 0.0) ? -div_val : 0.0;
      }

    // normalize sensor (optional but strongly recommended)
    chi /= (chi + 1e-12);

    // --- artificial viscosity coefficient
    const dealii::VectorizedArray<number> mu_art =
      0. * chi;

    // --- apply ONLY isotropic damping (bulk viscosity)
    for (unsigned int d = 0; d < dim; ++d)
      {
        out[d][d] -= mu_art * div_u;
      }

    return out;
  }

  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<2, dim, dealii::VectorizedArray<number>>
    CompressibleFlowViscousKernels<dim, number>::calculate_viscous_stress_tensor_damped(
      const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> &grad_u,
      const ConservedVariablesType     &conserved_variables,
      const ConservedVariablesGradType &grad_conserved_variables) const
  {
    const dealii::VectorizedArray<number> div_u = 2. / 3. * dealii::trace(grad_u);

    dealii::Tensor<2, dim, dealii::VectorizedArray<number>> out;
    for (unsigned int d = 0; d < dim; ++d)
      {
        for (unsigned int e = 0; e < dim; ++e)
          out[d][e] = material.data.dynamic_viscosity * (grad_u[d][e] + grad_u[e][d]);
        out[d][d] -= (material.data.dynamic_viscosity + 0. /** std::abs(material.eos_utils->calculate_grad_T(conserved_variables, grad_conserved_variables)[0])*/) * div_u;
      }

    return out;
  }

  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    auto
    CompressibleFlowViscousKernels<dim, number>::calculate_viscous_flux(
      const ConservedVariablesType     &conserved_variables,
      const ConservedVariablesGradType &grad_conserved_variables,
      const dealii::VectorizedArray<number> &quad_points) const
    -> ConservedVariablesGradType
  {
    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> velocity =
      calculate_velocity<dim, number>(conserved_variables);

    const auto grad_u = calculate_grad_velocity(conserved_variables, grad_conserved_variables);

    const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> viscous_stress =
      calculate_viscous_stress_tensor(grad_u);

    const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> viscous_stress_damped =
      calculate_viscous_stress_tensor_damped(grad_u, conserved_variables, grad_conserved_variables);

    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> neg_heat_flux =
      material.data.thermal_conductivity *
      material.eos_utils->calculate_grad_T(conserved_variables, grad_conserved_variables);

    ConservedVariablesGradType flux;
    for (unsigned int d = 0; d < dim; ++d)
      {
        // density
        flux[0][d] = 0.0;

        // momentum
        for (unsigned int e = 0; e < dim; ++e)
          flux[e + 1][d] = viscous_stress_damped[e][d];

        // energy
        flux[dim + 1][d] = neg_heat_flux[d];

        for (unsigned int e = 0; e < dim; ++e)
          flux[dim + 1][d] += velocity[e] * viscous_stress_damped[d][e];

        /*const number epsilon = 0.0000000000001;
        const dealii::VectorizedArray<number> grad_p = (material.data.gamma - 1.) / (1./conserved_variables[0] - material.data.eos_data.b)
              * material.data.specific_isobaric_heat / material.data.gamma * material.eos_utils->calculate_grad_T(conserved_variables,grad_conserved_variables)[0]
              + (material.data.gamma -1.) * material.eos_utils->calculate_temperature(conserved_variables) * material.data.specific_isobaric_heat
              / material.data.gamma * 1./((1./conserved_variables[0]-material.data.eos_data.b) * (1./conserved_variables[0]-material.data.eos_data.b))
                * 1. / conserved_variables[0] / conserved_variables[0] * grad_conserved_variables[0][0];*/

        /*const dealii::VectorizedArray<number> div_u = dealii::trace(grad_u);
        const dealii::VectorizedArray<number> div_mask = dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
                      -div_u,
                      dealii::make_vectorized_array(0.),
                      -div_u,
                      dealii::make_vectorized_array(0.));*/

        //flux[0][d] += epsilon * grad_conserved_variables[0][0] * std::abs(grad_p);
        //flux[1][d] += epsilon * grad_conserved_variables[1][0] * std::abs(grad_p);
        //flux[2][d] += epsilon * grad_conserved_variables[2][0] * std::abs(grad_p);
      }

    return flux;
  }

  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    auto
    CompressibleFlowViscousKernels<dim, number>::calculate_viscous_numerical_flux(
      const ConservedVariablesType                                  &u_m,
      const ConservedVariablesType                                  &u_p,
      const ConservedVariablesGradType                              &grad_u_m,
      const ConservedVariablesGradType                              &grad_u_p,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
      const dealii::VectorizedArray<number> penalty_parameter) const -> ConservedVariablesType
  {
    const auto flux_m = calculate_viscous_flux(u_m, grad_u_m);

    const auto flux_p = calculate_viscous_flux(u_p, grad_u_p);

    const auto average_density = (u_m[0] + u_p[0]) / 2.;

    return contract_average_tensor_with_vector<CompressibleFlow::n_conserved_variables<dim>,
                                               dim,
                                               dealii::VectorizedArray<number>>(flux_m,
                                                                                flux_p,
                                                                                normal) -
           penalty_parameter * material.data.dynamic_viscosity / average_density *
             (u_m - u_p);
  }

  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    auto
    CompressibleFlowViscousKernels<dim, number>::calculate_viscous_numerical_flux_gradient(
      const ConservedVariablesType                                  &u_m,
      const ConservedVariablesType                                  &u_p,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal) const
    -> std::pair<ConservedVariablesGradType, ConservedVariablesGradType>
  {
    ConservedVariablesGradType jump_u;
    for (unsigned int e = 0; e < CompressibleFlow::n_conserved_variables<dim>; ++e)
      for (unsigned int d = 0; d < dim; ++d)
        jump_u[e][d] = (u_m[e] - u_p[e]) * normal[d];

    // use jumps instead of gradients for evaluating the viscous flux
    const ConservedVariablesGradType flux_m = 0.5 * calculate_viscous_flux(u_m, jump_u);
    const ConservedVariablesGradType flux_p = 0.5 * calculate_viscous_flux(u_p, jump_u);

    return std::make_pair(flux_m, flux_p);
  }

  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    auto
    CompressibleFlowViscousKernels<dim, number>::calculate_jacobian_viscous_numerical_flux(
      const std::pair<ConservedVariablesType, ConservedVariablesType>         &w_q,
      const std::pair<ConservedVariablesGradType, ConservedVariablesGradType> &grad_w_q,
      const std::pair<ConservedVariablesType, ConservedVariablesType>         &delta_w_q,
      const std::pair<ConservedVariablesGradType, ConservedVariablesGradType> &grad_delta_w_q,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>>           &normal,
      dealii::VectorizedArray<number> penalty_parameter) const -> ConservedVariablesGradType
  {
    ConservedVariablesGradType flux_p =
      0.5 * calculate_jacobian_viscous_flux(w_q.second,
                                            grad_w_q.second,
                                            delta_w_q.second,
                                            grad_delta_w_q.second);
    ConservedVariablesGradType flux_m = 0.5 * calculate_jacobian_viscous_flux(w_q.first,
                                                                              grad_w_q.first,
                                                                              delta_w_q.first,
                                                                              grad_delta_w_q.first);

    ConservedVariablesGradType flux;
    for (unsigned int i = 0; i < CompressibleFlow::n_conserved_variables<dim>; ++i)
      flux[i] = (flux_p[i] + flux_m[i]);

    flux -=
      calculate_jacobian_viscous_numerical_flux_jump_term(delta_w_q, normal, penalty_parameter);
    return flux;
  }

  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    auto
    CompressibleFlowViscousKernels<dim, number>::calculate_jacobian_viscous_flux(
      const ConservedVariablesType     &w_q,
      const ConservedVariablesGradType &grad_w_q,
      const ConservedVariablesType     &delta_w_q,
      const ConservedVariablesGradType &grad_delta_w_q) const -> ConservedVariablesGradType
  {
    ConservedVariablesGradType viscous_differential_change;

    // precompute values
    dealii::VectorizedArray<number>                         rho_inv = 1.0 / w_q[0];
    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> m_q;
    for (unsigned int i = 0; i < dim; ++i)
      m_q[i] = w_q[i + 1];
    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> delta_m_q;
    for (unsigned int i = 0; i < dim; ++i)
      delta_m_q[i] = delta_w_q[i + 1];

    /* change in density flux */
    viscous_differential_change[0] = dealii::Tensor<1, dim, dealii::VectorizedArray<number>>();

    /* change in momentum flux */
    // density part
    dealii::Tensor<1, dim, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> grad_m_q;
    for (unsigned int i = 0; i < dim; ++i)
      grad_m_q[i] = grad_w_q[i + 1];
    dealii::Tensor<1, dim, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> grad_delta_m_q;
    for (unsigned int i = 0; i < dim; ++i)
      grad_delta_m_q[i] = grad_delta_w_q[i + 1];

    auto       v_q           = calculate_velocity<dim, number>(w_q);
    const auto grad_v_q_temp = calculate_grad_velocity(w_q, grad_w_q);
    dealii::Tensor<1, dim, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> grad_v_q;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        grad_v_q[i][j] = grad_v_q_temp[i][j];

    dealii::Tensor<1, dim, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> param_a =
      rho_inv * (-delta_w_q[0] * grad_v_q);
    param_a += rho_inv * rho_inv * delta_w_q[0] * dyadic_product(v_q, grad_w_q[0]);
    param_a -= rho_inv * dyadic_product(v_q, grad_delta_w_q[0]);

    dealii::Tensor<1, dim, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> param_c =
      param_a;
    param_c += transpose(param_a);
    param_c -= 2. / 3. * trace(param_a) * identity<dim, dealii::VectorizedArray<number>>();
    param_c *= material.data.dynamic_viscosity;

    dealii::Tensor<1, dim, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> param_b =
      rho_inv * grad_delta_m_q;
    param_b -= rho_inv * rho_inv * dyadic_product(delta_m_q, grad_w_q[0]);

    dealii::Tensor<1, dim, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> param_d =
      param_b;
    param_d += transpose(param_b);
    param_d -= 2. / 3. * trace(param_b) * identity<dim, dealii::VectorizedArray<number>>();
    param_d *= material.data.dynamic_viscosity;


    for (unsigned int i = 0; i < dim; ++i)
      {
        viscous_differential_change[i + 1] = param_c[i] + param_d[i];
      }

    /* change in energy density flux */
    const auto tau_temp = calculate_viscous_stress_tensor(grad_v_q_temp);

    dealii::Tensor<1, dim, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>> tau;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        tau[i][j] = tau_temp[i][j];
    // density part
    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> rho_energy_density;
    rho_energy_density += matrix_vector_product(param_c, m_q) * rho_inv;
    rho_energy_density -= rho_inv * rho_inv * matrix_vector_product(tau, m_q) * delta_w_q[0];


    rho_energy_density +=
      lambda_div_c * (-rho_inv * rho_inv * grad_w_q[dim + 1] * delta_w_q[0] +
                      rho_inv * rho_inv * rho_inv * w_q[dim + 1] * delta_w_q[0] * grad_w_q[0] +
                      grad_w_q[0] * w_q[dim + 1] * rho_inv * rho_inv * rho_inv * delta_w_q[0] -
                      w_q[dim + 1] * rho_inv * rho_inv * grad_delta_w_q[0] +
                      matrix_vector_product(grad_v_q, m_q) * rho_inv * rho_inv * delta_w_q[0]);

    rho_energy_density -=
      lambda_div_c * rho_inv * rho_inv *
      (rho_inv * delta_w_q[0] * matrix_vector_product(dyadic_product(v_q, grad_w_q[0]), m_q) -
       matrix_vector_product(dyadic_product(v_q, grad_delta_w_q[0]), m_q));


    rho_energy_density +=
      lambda_div_c * rho_inv * rho_inv * delta_w_q[0] * matrix_vector_product(grad_v_q, m_q);


    // momentum part
    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> momentum_energy_density;
    momentum_energy_density += rho_inv * matrix_vector_product(param_d, m_q);
    momentum_energy_density += rho_inv * matrix_vector_product(tau, delta_m_q);
    momentum_energy_density -= lambda_div_c * rho_inv * matrix_vector_product(grad_v_q, delta_m_q);
    momentum_energy_density -=
      lambda_div_c * rho_inv * rho_inv * matrix_vector_product(grad_delta_m_q, m_q);
    momentum_energy_density += lambda_div_c * rho_inv * rho_inv * rho_inv *
                               matrix_vector_product(dyadic_product(delta_m_q, grad_w_q[0]), m_q);


    // energy density part
    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> energy_energy_density;
    energy_energy_density += rho_inv * grad_delta_w_q[dim + 1];
    energy_energy_density -= grad_w_q[0] * rho_inv * rho_inv * delta_w_q[dim + 1];
    energy_energy_density *= lambda_div_c;

    // build the sum
    viscous_differential_change[dim + 1] =
      rho_energy_density + momentum_energy_density + energy_energy_density;
    return viscous_differential_change;
  }

  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    auto
    CompressibleFlowViscousKernels<dim, number>::
      calculate_jacobian_viscous_numerical_flux_jump_term(
        const std::pair<ConservedVariablesType, ConservedVariablesType> &delta_w_q,
        const dealii::Tensor<1, dim, dealii::VectorizedArray<number>>   &normal,
        dealii::VectorizedArray<number> penalty_parameter) const -> ConservedVariablesGradType
  {
    return matrix_matrix_product(
      penalty_parameter * material.data.dynamic_viscosity / material.data.reference_density *
        identity<CompressibleFlow::n_conserved_variables<dim>, dealii::VectorizedArray<number>>(),
      dyadic_product(delta_w_q.first - delta_w_q.second, normal));
  }
} // namespace MeltPoolDG::Flow
