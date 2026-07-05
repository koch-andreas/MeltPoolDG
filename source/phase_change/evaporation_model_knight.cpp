#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>

#include <meltpooldg/phase_change/evaporation_model_knight.hpp>
#include <meltpooldg/phase_change/evaporation_tools.hpp>
#include <meltpooldg/utilities/numbers.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <cmath>
#include <numbers>

namespace MeltPoolDG::Evaporation
{
  template <typename number, typename number_2>
  EvaporationModelKnight<number, number_2>::EvaporationModelKnight(
    const number atmospheric_pressure,
    const number boiling_temperature_at_atmospheric_pressure,
    const number latent_heat_of_evaporation,
    const number specific_gas_constant,
    const number specific_heat_ratio_vapor)
    : atmospheric_pressure(atmospheric_pressure)
    , boiling_temperature_at_atmospheric_pressure(boiling_temperature_at_atmospheric_pressure)
    , latent_heat_of_evaporation(latent_heat_of_evaporation)
    , specific_gas_constant(specific_gas_constant)
    , specific_heat_ratio_vapor(specific_heat_ratio_vapor)
    , temperature_constant(latent_heat_of_evaporation / specific_gas_constant)
    , helper_mass_flux(std::sqrt(specific_gas_constant / (2. * std::numbers::pi)))
    , helper_temperature_ratio(0.5 * (specific_heat_ratio_vapor - 1.) /
                               (specific_heat_ratio_vapor + 1.))
  {
    AssertThrow(boiling_temperature_at_atmospheric_pressure > 1e-12,
                dealii::ExcMessage("The boiling temperature must not be zero."));
    AssertThrow(specific_gas_constant > 1e-12,
                dealii::ExcMessage("The specific gas constant must not be zero."));
    AssertThrow(specific_heat_ratio_vapor > 1e-12,
                dealii::ExcMessage("The specific heat ratio of the vapor phase must not be zero."));
  }

  template <typename number, typename number_2>
  void
  EvaporationModelKnight<number, number_2>::reinit(const number_2    &T_liquid,
                                                   const number_2    &Ma_gas,
                                                   const unsigned int n_active_lanes)
  {
    // Assert when Mach numbers are very high. The result is expected to be non-physical.
    bool valid = true;

    if constexpr (std::is_same_v<number_2, number>)
      {
        valid = Ma_gas < 10.;
      }
    else
      {
        for (unsigned int v = 0; v < n_active_lanes; ++v)
          if (Ma_gas[v] >= 10.)
            {
              valid = false;
              break;
            }
      }

    AssertThrow(valid,
                dealii::ExcMessage(
                  "The Mach number exceeds Ma = 10. The result is expected to be non-physical!"));

    // Choose a minimum finite Mach number to trigger evaporation initially
    const number_2 Ma_gas_corrected = std::max(Ma_gas, number_2(1.e-4));

    // Dimensionless gas velocity
    const number_2 m_g = Ma_gas_corrected * std::sqrt(0.5 * specific_heat_ratio_vapor);

    // Pre-compute expensive functions of m_g
    number_2 erfc_m_g{};
    if constexpr (std::is_same_v<number_2, number>)
      erfc_m_g = std::erfc(m_g);
    else
      erfc_m_g = UtilityFunctions::calculate_complementary_error_function_vec(m_g);

    const number_2 exp_m_g_2 = std::exp(m_g * m_g);

    // Saturated vapor temperature (assumption of thermal equilibrium with liquid surface)
    const number_2 T_sat = T_liquid;

    // Saturated vapor pressure according to Clausius-Clapeyron
    const number_2 p_sat =
      compute_saturated_gas_pressure<number, number_2>(T_sat,
                                                       boiling_temperature_at_atmospheric_pressure,
                                                       atmospheric_pressure,
                                                       temperature_constant);

    // Saturated vapor density from ideal gas law
    const number_2 rho_sat = p_sat / (specific_gas_constant * T_sat);

    // Temperature ratio: T_gas / T_sat
    const number_2 T_gas_over_T_sat = compute_temperature_ratio(m_g);

    // Density ratio: rho_gas / rho_sat
    const number_2 rho_gas_over_rho_sat =
      compute_density_ratio(T_gas_over_T_sat, m_g, exp_m_g_2, erfc_m_g);

    // Factor beta for back-scattered atoms
    const number_2 beta =
      compute_factor_beta(T_gas_over_T_sat, rho_gas_over_rho_sat, m_g, exp_m_g_2);

    const number_2 rho_gas = rho_gas_over_rho_sat * rho_sat;

    const number_2 T_gas = T_gas_over_T_sat * T_sat;

    // Temperature jump across interface for the current conditions
    temperature_jump = T_liquid - T_gas;

    // Mass flux computation

    // Evaporative mass flux
    const number_2 m_dot_plus = rho_sat * helper_mass_flux * std::sqrt(T_sat);

    // Condensation mass flux
    const number_2 m_dot_minus = beta * rho_gas * helper_mass_flux * std::sqrt(T_gas) *
                                 (numbers::sqrt_pi * m_g * erfc_m_g - number_2(1.) / exp_m_g_2);

    // Net mass flux
    evaporative_mass_flux = m_dot_plus + m_dot_minus;

    // Restrict evaporation to liquid temperatures above the boiling temperature
    apply_boiling_threshold(T_sat);
  }

  template <typename number, typename number_2>
  number_2
  EvaporationModelKnight<number, number_2>::compute_temperature_ratio(const number_2 &m_g) const
  {
    const number_2 tmp = helper_temperature_ratio * m_g;

    const number_2 sqrt_T_gas_over_T_sat =
      std::sqrt(1. + std::numbers::pi * tmp * tmp) - numbers::sqrt_pi * tmp;

    return sqrt_T_gas_over_T_sat * sqrt_T_gas_over_T_sat;
  }

  template <typename number, typename number_2>
  number_2
  EvaporationModelKnight<number, number_2>::compute_density_ratio(const number_2 &T_gas_over_T_sat,
                                                                  const number_2 &m_g,
                                                                  const number_2 &exp_m_g_2,
                                                                  const number_2 &erfc_m_g) const
  {
    return ((m_g * m_g + 0.5) * exp_m_g_2 * erfc_m_g - m_g / numbers::sqrt_pi) /
             std::sqrt(T_gas_over_T_sat) +
           0.5 / T_gas_over_T_sat * (1. - numbers::sqrt_pi * m_g * exp_m_g_2 * erfc_m_g);
  }

  template <typename number, typename number_2>
  number_2
  EvaporationModelKnight<number, number_2>::compute_factor_beta(
    const number_2 &T_gas_over_T_sat,
    const number_2 &rho_gas_over_rho_sat,
    const number_2 &m_g,
    const number_2 &exp_m_g_2) const
  {
    const number_2 sqrt_T_sat_over_T_gas = 1. / std::sqrt(T_gas_over_T_sat);

    return (2. * m_g * m_g + 1. - m_g * numbers::sqrt_pi * sqrt_T_sat_over_T_gas) * exp_m_g_2 /
           rho_gas_over_rho_sat * sqrt_T_sat_over_T_gas;
  }

  template <typename number, typename number_2>
  void
  EvaporationModelKnight<number, number_2>::apply_boiling_threshold(const number_2 &T_sat)
  {
    if constexpr (std::is_same_v<number_2, number>)
      {
        if (T_sat < boiling_temperature_at_atmospheric_pressure)
          {
            evaporative_mass_flux = 0.;
            temperature_jump      = 0.;
          }
      }
    else
      {
        const dealii::VectorizedArray<number> zero_vec = dealii::make_vectorized_array(0.);
        const dealii::VectorizedArray<number> boiling_temperature_at_atmospheric_pressure_vec =
          dealii::make_vectorized_array(boiling_temperature_at_atmospheric_pressure);
        evaporative_mass_flux = dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
          T_sat, boiling_temperature_at_atmospheric_pressure_vec, zero_vec, evaporative_mass_flux);
        temperature_jump = dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
          T_sat, boiling_temperature_at_atmospheric_pressure_vec, zero_vec, temperature_jump);
      }
  }

  template class EvaporationModelKnight<double>;
  template class EvaporationModelKnight<double, dealii::VectorizedArray<double>>;
} // namespace MeltPoolDG::Evaporation
