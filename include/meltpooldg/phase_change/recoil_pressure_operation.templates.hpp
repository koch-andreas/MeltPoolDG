#pragma once

#include <meltpooldg/phase_change/evaporation_tools.hpp>
#include <meltpooldg/phase_change/recoil_pressure_operation.hpp>
//
#include <deal.II/base/utilities.h>

#include <cmath>


namespace MeltPoolDG::Evaporation
{
  namespace internal
  {
    // In order to smoothly activate the recoil pressure, a scaling coefficient is computed and
    // multiplied with the recoil pressure coefficient
    //
    //                  -
    //                 |     0         if T <= T_ac
    //                 |
    //                 |  T - T_ac
    // scaling_coeff = |  --------     if T_ac < T < T_v
    //                 |  T_v - T_ac
    //                 |
    //                 |     1         if T >= T_v
    //                  -
    //
    // with @p T_ac the activation temperature of the recoil pressure and @p T_v the boiling temperature.

    template <typename number>
    inline dealii::VectorizedArray<number>
    compute_scaling_coeff(const dealii::VectorizedArray<number> &T,
                          const number                           T_ac,
                          const number                           T_v)
    {
      return compare_and_apply_mask<dealii::SIMDComparison::less_than>(
        T,
        T_ac,
        0.0,
        compare_and_apply_mask<dealii::SIMDComparison::greater_than_or_equal>(
          T, T_v, 1., (T - T_ac) / (T_v - T_ac)));
    }

    template <typename number>
    inline number
    compute_scaling_coeff(const number T, const number T_ac, const number T_v)
    {
      return (T < T_ac) ? 0.0 : (T >= T_v) ? 1. : (T - T_ac) / (T_v - T_ac);
    }
  } // namespace internal

  template <typename number>
  inline number
  RecoilPressureHybridModel<number>::compute_recoil_pressure_coefficient(
    const number T,
    const number m_dot,
    const number delta_coefficient) const
  {
    return recoil_phenomenological.compute_recoil_pressure_coefficient(T) -
           dealii::Utilities::fixed_power<2>(m_dot) * density_coeff * delta_coefficient;
  }

  template <typename number>
  inline dealii::VectorizedArray<number>
  RecoilPressureHybridModel<number>::compute_recoil_pressure_coefficient(
    const dealii::VectorizedArray<number> &T,
    const dealii::VectorizedArray<number> &m_dot,
    const dealii::VectorizedArray<number> &delta_coefficient) const
  {
    return recoil_phenomenological.compute_recoil_pressure_coefficient(T) -
           dealii::Utilities::fixed_power<2>(m_dot) * density_coeff * delta_coefficient;
  }

  template <typename number>
  inline dealii::VectorizedArray<number>
  RecoilPressurePhenomenologicalModel<number>::compute_recoil_pressure_coefficient(
    const dealii::VectorizedArray<number> &T) const
  {
    const number T_ac = recoil_data.activation_temperature;
    const number T_v  = boiling_temperature;

    dealii::VectorizedArray<number> recoil_pressure =
      recoil_data.pressure_coefficient *
      compute_saturated_gas_pressure(T,
                                     T_v,
                                     recoil_data.ambient_gas_pressure,
                                     recoil_data.temperature_constant);

    if (recoil_data.subtract_ambient_pressure)
      recoil_pressure -= recoil_data.ambient_gas_pressure;

    if (recoil_data.enable_linear_activation_ramp)
      {
        const dealii::VectorizedArray<number> scaling_coeff =
          internal::compute_scaling_coeff(T, T_ac, T_v);
        recoil_pressure *= scaling_coeff;
      }

    return compare_and_apply_mask<dealii::SIMDComparison::greater_than_or_equal>(
      recoil_pressure, 0, recoil_pressure, dealii::VectorizedArray<number>(0));
  }

  template <typename number>
  inline number
  RecoilPressurePhenomenologicalModel<number>::compute_recoil_pressure_coefficient(
    const number T) const
  {
    const number T_ac = recoil_data.activation_temperature;
    const number T_v  = boiling_temperature;


    number recoil_pressure = recoil_data.pressure_coefficient *
                             compute_saturated_gas_pressure(T,
                                                            T_v,
                                                            recoil_data.ambient_gas_pressure,
                                                            recoil_data.temperature_constant);

    if (recoil_data.subtract_ambient_pressure)
      recoil_pressure -= recoil_data.ambient_gas_pressure;

    if (recoil_data.enable_linear_activation_ramp)
      {
        const number scaling_coeff = internal::compute_scaling_coeff(T, T_ac, T_v);
        recoil_pressure *= scaling_coeff;
      }

    return (recoil_pressure >= 0.0) ? recoil_pressure : 0.0;
  }

} // namespace MeltPoolDG::Evaporation
