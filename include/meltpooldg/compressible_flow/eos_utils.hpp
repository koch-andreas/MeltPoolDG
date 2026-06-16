#pragma once

#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/material.hpp>
#include <meltpooldg/compressible_flow/utils.hpp>

namespace MeltPoolDG::CompressibleFlow::EOS
{
  /**
   * @brief A base class for a collection of thermodynamic helper functions, which depend on the
   * equation of state.
   */
  template <int dim, typename number>
  class EquationOfStateUtils
  {
  public:
    virtual ~EquationOfStateUtils() = default;

    /**
     * @brief Calculate the pressure from the conserved variables for a specific equation of state.
     *
     * @param conserved_variables Current values of the conserved variables.
     *
     * @return Pressure resulting from the values of the conserved variables.
     */
    inline DEAL_II_ALWAYS_INLINE //
      virtual dealii::VectorizedArray<number>
      calculate_thermodynamic_pressure(
        const ConservedVariablesType<dim, number> &conserved_variables) const = 0;

    /**
     * @brief Calculate the gradient of the temperature from the conserved variables and their gradients
     * for a specific equation of state.
     *
     * @param conserved_variables Current values of the conserved variables.
     * @param grad_conserved_variables Current gradient of the conserved variables.
     *
     * @return Current gradient of the temperature field.
     */
    inline DEAL_II_ALWAYS_INLINE //
      virtual dealii::Tensor<1, dim, dealii::VectorizedArray<number>>
      calculate_grad_T(
        const ConservedVariablesType<dim, number>         &conserved_variables,
        const ConservedVariablesGradientType<dim, number> &grad_conserved_variables) const = 0;

    /**
     * @brief Calculate the speed of sound for a specific equation of state.
     *
     * @param conserved_variables Current values of the conserved variables.
     *
     * @return Speed of sound resulting from the values of the conserved variables.
     */
    inline DEAL_II_ALWAYS_INLINE //
      virtual dealii::VectorizedArray<number>
      calculate_speed_of_sound(
        const ConservedVariablesType<dim, number> &conserved_variables) const = 0;

    /**
     * @brief Calculate the temperature for a specific equation of state.
     *
     * @param conserved_variables Current values of the conserved variables.
     *
     * @return Temperature resulting from the values of the conserved variables.
     */
    inline DEAL_II_ALWAYS_INLINE //
      virtual dealii::VectorizedArray<number>
      calculate_temperature(
        const ConservedVariablesType<dim, number> &conserved_variables) const = 0;

    /**
     * @brief Calculate the total stress tensor from pressure contribution and viscous stress
     * contribution sigma_ij = tau_ij - p * delta_ij.
     *
     * @param conserved_variables Current values of the conserved variables.
     * @param viscous_stress_tensor Given viscous tress tensor.
     *
     * @return Stress tensor resulting from the values and gradients of the conserved variables.
     */
    inline DEAL_II_ALWAYS_INLINE //
      dealii::Tensor<2, dim, dealii::VectorizedArray<number>>
      calculate_stress_tensor(
        const ConservedVariablesType<dim, number>                     &conserved_variables,
        const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> &viscous_stress_tensor) const
    {
      const auto pressure_tensor =
        calculate_thermodynamic_pressure(conserved_variables) *
        dealii::unit_symmetric_tensor<dim, dealii::VectorizedArray<number>>();

      return viscous_stress_tensor - pressure_tensor;
    }

    /**
     * @brief Convert the given conservative variables (rho, momentum, total energy) to primitive
     * variables (pressure, velocity, temperature).
     *
     * @param u_cons Current values in conservative variables formulation.
     *
     * @return Current values in primitive variables formulation.
     */
    inline DEAL_II_ALWAYS_INLINE //
      ConservedVariablesType<dim, number>
      convert_conservative_into_primitive_variables(
        const ConservedVariablesType<dim, number> &u_cons) const
    {
      ConservedVariablesType<dim, number> u_prim;

      // pressure
      u_prim[0] = calculate_thermodynamic_pressure(u_cons);

      // velocity
      for (unsigned int i = 1; i < dim + 1; i++)
        u_prim[i] = u_cons[i] / u_cons[0];

      // temperature
      u_prim[dim + 1] = calculate_temperature(u_cons);

      return u_prim;
    }

    /**
     * @brief Convert the given primitive variables (pressure, velocity, temperature) to conservative
     * variables (rho, momentum, total energy).
     *
     * @param u_prim Current values in primitive variables formulation.
     *
     * @return Current values in conservative variables formulation.
     */
    inline DEAL_II_ALWAYS_INLINE //
      virtual ConservedVariablesType<dim, number>
      convert_primitive_into_conservative_variables(
        const ConservedVariablesType<dim, number> &u_prim) const = 0;

    /**
     * @brief Calculate the inner energy from a given pressure.
     *
     * @param pressure Given pressure value.
     * @param density Given density value.
     *
     * @return Inner energy resulting from the given pressure and density values.
     */
    inline DEAL_II_ALWAYS_INLINE //
      virtual dealii::VectorizedArray<number>
      compute_inner_energy_from_pressure(const dealii::VectorizedArray<number> &pressure,
                                         const dealii::VectorizedArray<number> &density) const = 0;
  };

  /**
   * @brief A class for a collection of thermodynamic helper functions for the ideal gas
   * equation of state.
   */
  template <int dim, typename number>
  class IdealGas : public EquationOfStateUtils<dim, number>
  {
  public:
    /**
     * @brief Constructor.
     *
     * @param material_data_in Reference to a material data object providing all relevant material
     * parameters.
     */
    explicit IdealGas(const MaterialPhaseData<number> &material_data_in)
      : material_data(material_data_in)
    {}

    /**
     * @brief Calculate the pressure from the conserved variables for an ideal gas.
     *
     * @param conserved_variables Current values of the conserved variables.
     *
     * @return Pressure resulting from the values of the conserved variables.
     */
    inline DEAL_II_ALWAYS_INLINE //
      dealii::VectorizedArray<number>
      calculate_thermodynamic_pressure(
        const ConservedVariablesType<dim, number> &conserved_variables) const override
    {
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> velocity =
        calculate_velocity<dim, number>(conserved_variables);

      return (material_data.gamma - 1.) *
             (conserved_variables[dim + 1] -
              conserved_variables[0] * 0.5 * scalar_product(velocity, velocity));
    }

    /**
     * @brief Calculate the gradient of the temperature from the conserved variables and their
     * gradients for an ideal gas.
     *
     * @param conserved_variables Current values of the conserved variables.
     * @param grad_conserved_variables Current gradient of the conserved variables.
     *
     * @return Current gradient of the temperature field.
     */
    inline DEAL_II_ALWAYS_INLINE //
      dealii::Tensor<1, dim, dealii::VectorizedArray<number>>
      calculate_grad_T(
        const ConservedVariablesType<dim, number>         &conserved_variables,
        const ConservedVariablesGradientType<dim, number> &grad_conserved_variables) const override
    {
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> u =
        calculate_velocity<dim, number>(conserved_variables);
      const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> grad_u =
        calculate_grad_velocity<dim, number>(conserved_variables, grad_conserved_variables);
      const dealii::VectorizedArray<number> rho     = conserved_variables[0];
      const dealii::VectorizedArray<number> inv_rho = dealii::VectorizedArray<number>(1.) / rho;
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> grad_rho =
        grad_conserved_variables[0];
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> grad_E =
        inv_rho *
        (grad_conserved_variables[dim + 1] - inv_rho * conserved_variables[dim + 1] * grad_rho);

      return (material_data.gamma - 1.0) / material_data.specific_gas_constant *
             (grad_E - grad_u * u);
    }

    /**
     * @brief Calculate the speed of sound for an ideal gas.
     *
     * @param conserved_variables Current values of the conserved variables.
     *
     * @return Speed of sound resulting from the values of the conserved variables.
     */
    inline DEAL_II_ALWAYS_INLINE //
      dealii::VectorizedArray<number>
      calculate_speed_of_sound(
        const ConservedVariablesType<dim, number> &conserved_variables) const override
    {
      const auto pressure = calculate_thermodynamic_pressure(conserved_variables);
      const auto density  = conserved_variables[0];

      return std::sqrt(material_data.gamma * pressure / density);
    }

    /**
     * @brief Calculate the temperature for an ideal gas.
     *
     * @param conserved_variables Current values of the conserved variables.
     *
     * @return Temperature resulting from the values of the conserved variables.
     */
    inline DEAL_II_ALWAYS_INLINE //
      dealii::VectorizedArray<number>
      calculate_temperature(
        const ConservedVariablesType<dim, number> &conserved_variables) const override
    {
      const auto pressure = calculate_thermodynamic_pressure(conserved_variables);
      const auto density  = conserved_variables[0];

      return pressure / (material_data.specific_gas_constant * density);
    }

    /**
     * @brief Convert the given primitive variables (pressure, velocity, temperature) to
     * conservative variables (rho, momentum, total energy).
     *
     * @param u_prim Current values in primitive variables formulation.
     *
     * @return Current values in conservative variables formulation.
     */
    inline DEAL_II_ALWAYS_INLINE //
      ConservedVariablesType<dim, number>
      convert_primitive_into_conservative_variables(
        const ConservedVariablesType<dim, number> &u_prim) const override
    {
      ConservedVariablesType<dim, number> u_cons;

      // density
      u_cons[0] = u_prim[0] / (material_data.specific_gas_constant * u_prim[dim + 1]);
      // momentum
      for (unsigned int i = 1; i < dim + 1; i++)
        u_cons[i] = u_prim[i] * u_cons[0];
      // total energy
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> velocity =
        calculate_velocity<dim, number>(u_cons);
      u_cons[dim + 1] = u_cons[0] * (material_data.specific_gas_constant /
                                       (material_data.gamma - 1.) * u_prim[dim + 1] +
                                     0.5 * scalar_product(velocity, velocity));
      return u_cons;
    }

    /**
     * @brief Calculate the inner energy from a given pressure.
     *
     * @param pressure Given pressure value.
     *
     * @return Inner energy resulting from the given pressure and density values.
     */
    inline DEAL_II_ALWAYS_INLINE //
      dealii::VectorizedArray<number>
      compute_inner_energy_from_pressure(const dealii::VectorizedArray<number> &pressure,
                                         const dealii::VectorizedArray<number> &) const override
    {
      return pressure / (material_data.gamma - 1.);
    }

  private:
    /// Material data object providing all relevant material parameters
    const MaterialPhaseData<number> &material_data;
  };

  /**
   * @brief A class for a collection of thermodynamic helper functions for the Noble-Abel stiffened
   * gas equation of state.
   */
  template <int dim, typename number>
  class NobleAbelStiffenedGas : public EquationOfStateUtils<dim, number>
  {
  public:
    /**
     * @brief Constructor.
     *
     * @param material_data_in Reference to a material data object providing all relevant material
     * parameters.
     */
    explicit NobleAbelStiffenedGas(const MaterialPhaseData<number> &material_data_in)
      : material_data(material_data_in)
    {}

    /**
     * @brief Calculate the pressure from the conserved variables for a Noble-Abel stiffened ideal
     * gas.
     *
     * @param conserved_variables Current values of the conserved variables.
     *
     * @return Pressure resulting from the values of the conserved variables.
     */
    inline DEAL_II_ALWAYS_INLINE //
      dealii::VectorizedArray<number>
      calculate_thermodynamic_pressure(
        const ConservedVariablesType<dim, number> &conserved_variables) const override
    {
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> velocity =
        calculate_velocity<dim, number>(conserved_variables);

      return ((material_data.gamma - 1.) *
                (conserved_variables[dim + 1] -
                 0.5 * conserved_variables[0] * scalar_product(velocity, velocity) -
                 conserved_variables[0] * material_data.eos_data.heat_bound) -
              material_data.gamma * material_data.eos_data.stiffening_pressure *
                (1. - conserved_variables[0] * material_data.eos_data.covolume)) /
             (1. - conserved_variables[0] * material_data.eos_data.covolume);
    }

    /**
     * @brief Calculate the gradient of the temperature from the conserved variables and their
     * gradients for a Noble-Abel stiffened ideal gas.
     *
     * @param conserved_variables Current values of the conserved variables.
     * @param grad_conserved_variables Current gradient of the conserved variables.
     *
     * @return Current gradient of the temperature field.
     */
    inline DEAL_II_ALWAYS_INLINE //
      dealii::Tensor<1, dim, dealii::VectorizedArray<number>>
      calculate_grad_T(
        const ConservedVariablesType<dim, number>         &conserved_variables,
        const ConservedVariablesGradientType<dim, number> &grad_conserved_variables) const override
    {
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> u =
        calculate_velocity<dim, number>(conserved_variables);
      const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> grad_u =
        calculate_grad_velocity<dim, number>(conserved_variables, grad_conserved_variables);
      const dealii::VectorizedArray<number> rho     = conserved_variables[0];
      const dealii::VectorizedArray<number> inv_rho = dealii::VectorizedArray<number>(1.) / rho;
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> grad_rho =
        grad_conserved_variables[0];
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> grad_E =
        inv_rho *
        (grad_conserved_variables[dim + 1] - inv_rho * conserved_variables[dim + 1] * grad_rho);

      return material_data.gamma / material_data.specific_isobaric_heat *
             (grad_E - grad_u * u +
              material_data.eos_data.stiffening_pressure * inv_rho * inv_rho * grad_rho);
    }

    /**
     * @brief Calculate the speed of sound for a Noble-Abel stiffened ideal gas.
     *
     * @param conserved_variables Current values of the conserved variables.
     *
     * @return Speed of sound resulting from the values of the conserved variables.
     */
    inline DEAL_II_ALWAYS_INLINE //
      dealii::VectorizedArray<number>
      calculate_speed_of_sound(
        const ConservedVariablesType<dim, number> &conserved_variables) const override
    {
      const auto pressure = calculate_thermodynamic_pressure(conserved_variables);
      const auto density  = conserved_variables[0];

      return std::sqrt(material_data.gamma *
                       (pressure + material_data.eos_data.stiffening_pressure) /
                       (density * (1. - density * material_data.eos_data.covolume)));
    }

    /**
     * @brief Calculate the temperature for a Noble-Abel stiffened ideal gas.
     *
     * @param conserved_variables Current values of the conserved variables.
     *
     * @return Temperature resulting from the values of the conserved variables.
     */
    inline DEAL_II_ALWAYS_INLINE //
      dealii::VectorizedArray<number>
      calculate_temperature(
        const ConservedVariablesType<dim, number> &conserved_variables) const override
    {
      const auto pressure = calculate_thermodynamic_pressure(conserved_variables);
      const auto density  = conserved_variables[0];

      return (pressure + material_data.eos_data.stiffening_pressure) *
             (1. / density - material_data.eos_data.covolume) * material_data.gamma /
             ((material_data.gamma - 1.) * material_data.specific_isobaric_heat);
    }

    /**
     * @brief Convert the given primitive variables (pressure, velocity, temperature) to conservative
     * variables (rho, momentum, total energy).
     *
     * @param u_prim Current values in primitive variables formulation.
     *
     * @return Current values in conservative variables formulation.
     */
    inline DEAL_II_ALWAYS_INLINE //
      ConservedVariablesType<dim, number>
      convert_primitive_into_conservative_variables(
        const ConservedVariablesType<dim, number> &u_prim) const override
    {
      ConservedVariablesType<dim, number> u_cons;

      // density
      u_cons[0] = (u_prim[0] + material_data.eos_data.stiffening_pressure) /
                  (material_data.specific_isobaric_heat * u_prim[dim + 1] *
                     (material_data.gamma - 1.) / material_data.gamma +
                   material_data.eos_data.covolume *
                     (u_prim[0] + material_data.eos_data.stiffening_pressure));
      // momentum
      for (unsigned int i = 1; i < dim + 1; i++)
        u_cons[i] = u_prim[i] * u_cons[0];
      // total energy
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> velocity =
        calculate_velocity<dim, number>(u_cons);
      u_cons[dim + 1] =
        u_cons[0] * (material_data.specific_isobaric_heat / material_data.gamma * u_prim[dim + 1] +
                     material_data.eos_data.stiffening_pressure *
                       (1. / u_cons[0] - material_data.eos_data.covolume) +
                     material_data.eos_data.heat_bound + 0.5 * scalar_product(velocity, velocity));

      return u_cons;
    }

    /**
     * @brief Calculate the inner energy from a given pressure.
     *
     * @param pressure Given pressure value.
     * @param density Given density value.
     *
     * @return Inner energy resulting from the given pressure and density values.
     */
    inline DEAL_II_ALWAYS_INLINE //
      dealii::VectorizedArray<number>
      compute_inner_energy_from_pressure(
        const dealii::VectorizedArray<number> &pressure,
        const dealii::VectorizedArray<number> &density) const override
    {
      return (pressure + material_data.gamma * material_data.eos_data.stiffening_pressure) /
               (material_data.gamma - 1.) * (1. - density * material_data.eos_data.covolume) +
             density * material_data.eos_data.heat_bound;
    }

  private:
    /// Material data object providing all relevant material parameters
    const MaterialPhaseData<number> &material_data;
  };

  /**
   * @brief A class for a collection of thermodynamic helper functions for the stiffened gas
   * equation of state.
   */
  template <int dim, typename number>
  class StiffenedGas : public EquationOfStateUtils<dim, number>
  {
  public:
    /**
     * @brief Constructor.
     *
     * @param material_data_in Reference to a material data object providing all relevant material
     * parameters.
     */
    explicit StiffenedGas(const MaterialPhaseData<number> &material_data_in)
      : material_data(material_data_in)
    {}

    /**
     * @brief Calculate the pressure from the conserved variables for a stiffened gas.
     *
     * @param conserved_variables Current values of the conserved variables.
     *
     * @return Pressure resulting from the values of the conserved variables.
     */
    inline DEAL_II_ALWAYS_INLINE //
      dealii::VectorizedArray<number>
      calculate_thermodynamic_pressure(
        const ConservedVariablesType<dim, number> &conserved_variables) const override
    {
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> velocity =
        calculate_velocity<dim, number>(conserved_variables);

      return (material_data.gamma - 1.) *
               (conserved_variables[dim + 1] -
                conserved_variables[0] * 0.5 * scalar_product(velocity, velocity)) -
             material_data.gamma * material_data.eos_data.stiffening_pressure;
    }

    /**
     * @brief Calculate the gradient of the temperature from the conserved variables and their
     * gradients for a stiffened gas.
     *
     * @param conserved_variables Current values of the conserved variables.
     * @param grad_conserved_variables Current gradient of the conserved variables.
     *
     * @return Current gradient of the temperature field.
     */
    inline DEAL_II_ALWAYS_INLINE //
      dealii::Tensor<1, dim, dealii::VectorizedArray<number>>
      calculate_grad_T(
        const ConservedVariablesType<dim, number>         &conserved_variables,
        const ConservedVariablesGradientType<dim, number> &grad_conserved_variables) const override
    {
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> u =
        calculate_velocity<dim, number>(conserved_variables);
      const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> grad_u =
        calculate_grad_velocity<dim, number>(conserved_variables, grad_conserved_variables);
      const dealii::VectorizedArray<number> rho     = conserved_variables[0];
      const dealii::VectorizedArray<number> inv_rho = dealii::VectorizedArray<number>(1.) / rho;
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> grad_rho =
        grad_conserved_variables[0];
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> grad_E =
        inv_rho *
        (grad_conserved_variables[dim + 1] - inv_rho * conserved_variables[dim + 1] * grad_rho);

      return material_data.gamma / material_data.specific_isobaric_heat *
             (grad_E - grad_u * u +
              material_data.eos_data.stiffening_pressure * inv_rho * inv_rho * grad_rho);
    }

    /**
     * @brief Calculate the speed of sound for a stiffened gas.
     *
     * @param conserved_variables Current values of the conserved variables.
     *
     * @return Speed of sound resulting from the values of the conserved variables.
     */
    inline DEAL_II_ALWAYS_INLINE //
      dealii::VectorizedArray<number>
      calculate_speed_of_sound(
        const ConservedVariablesType<dim, number> &conserved_variables) const override
    {
      const auto pressure = calculate_thermodynamic_pressure(conserved_variables);
      const auto density  = conserved_variables[0];

      return std::sqrt(material_data.gamma *
                       (pressure + material_data.eos_data.stiffening_pressure) / density);
    }

    /**
     * @brief Calculate the temperature for a stiffened gas.
     *
     * @param conserved_variables Current values of the conserved variables.
     *
     * @return Temperature resulting from the values of the conserved variables.
     */
    inline DEAL_II_ALWAYS_INLINE //
      dealii::VectorizedArray<number>
      calculate_temperature(
        const ConservedVariablesType<dim, number> &conserved_variables) const override
    {
      const auto pressure = calculate_thermodynamic_pressure(conserved_variables);
      const auto density  = conserved_variables[0];

      return (pressure + material_data.eos_data.stiffening_pressure) * material_data.gamma /
             (material_data.specific_isobaric_heat * density * (material_data.gamma - 1.));
    }

    /**
     * @brief Convert the given primitive variables (pressure, velocity, temperature) to conservative
     * variables (rho, momentum, total energy).
     *
     * @param u_prim Current values in primitive variables formulation.
     *
     * @return Current values in conservative variables formulation.
     */
    inline DEAL_II_ALWAYS_INLINE //
      ConservedVariablesType<dim, number>
      convert_primitive_into_conservative_variables(
        const ConservedVariablesType<dim, number> &u_prim) const override
    {
      ConservedVariablesType<dim, number> u_cons;

      // density
      u_cons[0] =
        (u_prim[0] + material_data.eos_data.stiffening_pressure) * material_data.gamma /
        (material_data.specific_isobaric_heat * u_prim[dim + 1] * (material_data.gamma - 1.));
      // momentum
      for (unsigned int i = 1; i < dim + 1; i++)
        u_cons[i] = u_prim[i] * u_cons[0];
      // total energy
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> velocity =
        calculate_velocity<dim, number>(u_cons);
      u_cons[dim + 1] =
        u_cons[0] * (material_data.specific_isobaric_heat / material_data.gamma * u_prim[dim + 1] +
                     material_data.eos_data.stiffening_pressure / u_cons[0] +
                     0.5 * scalar_product(velocity, velocity));

      return u_cons;
    }

    /**
     * @brief Calculate the inner energy from a given pressure.
     *
     * @param pressure Given pressure value.
     *
     * @return Inner energy resulting from the given pressure and density values.
     */
    inline DEAL_II_ALWAYS_INLINE //
      dealii::VectorizedArray<number>
      compute_inner_energy_from_pressure(const dealii::VectorizedArray<number> &pressure,
                                         const dealii::VectorizedArray<number> &) const override
    {
      return (pressure + material_data.gamma * material_data.eos_data.stiffening_pressure) /
             (material_data.gamma - 1.);
    }

  private:
    /// Material data object providing all relevant material parameters
    const MaterialPhaseData<number> &material_data;
  };
} // namespace MeltPoolDG::CompressibleFlow::EOS
