#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>

#include <meltpooldg/phase_change/evaporation_model_knight.hpp>
#include <meltpooldg/phase_change/evaporation_tools.hpp>

#include <cmath>
#include <numbers>

namespace MeltPoolDG::Evaporation
{
  template <typename number>
  EvaporationModelKnight<number>::EvaporationModelKnight(
    const number atmospheric_pressure,
    const number boiling_temperature_at_atmospheric_pressure,
    const number latent_heat_of_evaporation,
    const number specific_gas_constant,
    const number specific_heat_ratio_vapor)
    : atmospheric_pressure(atmospheric_pressure)
    , boiling_temperature_at_atmospheric_pressure(3133.0)
    , latent_heat_of_evaporation(latent_heat_of_evaporation)
    , specific_gas_constant(specific_gas_constant)
    , specific_heat_ratio_vapor(specific_heat_ratio_vapor)
    , temperature_constant(latent_heat_of_evaporation / specific_gas_constant)
  {
    AssertThrow(boiling_temperature_at_atmospheric_pressure > 1e-12,
                dealii::ExcMessage("The boiling temperature must not be zero."));
    AssertThrow(specific_gas_constant > 1e-12,
                dealii::ExcMessage("The specific gas constant must not be zero."));
    AssertThrow(specific_heat_ratio_vapor > 1e-12,
                dealii::ExcMessage("The specific heat ratio of the vapor phase must not be zero."));
  }

  template <typename number>
  void
  EvaporationModelKnight<number>::reinit(const number &T_liquid, const number &Ma_gas)
  {
    // Assert when Mach numbers are very high. The result is expected to be non-physical.
    AssertThrow(Ma_gas < 10.,
                dealii::ExcMessage("The Mach number exceeds Ma = 10. The result is expected to"
                                   " be non-physical!"));

    // Correct potentially slightly negative Ma numbers at interface position due to numerical
    // inaccuracies
    const number Ma_gas_corrected = std::max(Ma_gas, 1.e-4);

    // Dimensionless velocity
    const number m = Ma_gas_corrected * std::sqrt(0.5 * specific_heat_ratio_vapor);

    // Saturated vapor temperature (assumption of thermal equilibrium with liquid surface)
    const number T_sat = T_liquid;

    // Saturated vapor pressure according to Clausius-Clapeyron
    const number p_sat =
      compute_saturated_gas_pressure<number, number>(T_sat,
                                                     boiling_temperature_at_atmospheric_pressure,
                                                     atmospheric_pressure,
                                                     temperature_constant);

    // Saturated vapor density from ideal gas law
    const number rho_sat = p_sat / (specific_gas_constant * T_sat);

    // Temperature ratio: T_gas / T_sat
    const number T_gas_over_T_sat = compute_temperature_ratio(m);

    // Density ratio: rho_gas / rho_sat
    const number rho_gas_over_rho_sat = compute_density_ratio(T_gas_over_T_sat, m);

    // Factor beta for back-scattered atoms
    const number beta = compute_factor_beta(T_gas_over_T_sat, rho_gas_over_rho_sat, m);

    const number rho_gas = rho_gas_over_rho_sat * rho_sat;

    const number T_gas = T_gas_over_T_sat * T_sat;

    // Temperature jump across interface for the current conditions
    number T_jump = T_liquid - T_gas;

    // Mass flux computation

    const number helper = std::sqrt(specific_gas_constant / (2. * std::numbers::pi));

    // Evaporative mass flux
    const number m_dot_plus = rho_sat * helper * std::sqrt(T_sat);

    // Condensation mass flux
    constexpr number sqrt_pi = std::sqrt(std::numbers::pi);
    const number     m_dot_minus =
      beta * rho_gas * helper * std::sqrt(T_gas) * (sqrt_pi * m * std::erfc(m) - std::exp(-m * m));

    // Net mass flux
    number m_dot = m_dot_plus + m_dot_minus;

    // Exponential ramp correction (Knight's theory predicts mass flux for T_liquid < T_v).
    // The value for the damping factor was empirically determined and should not be changed.
    constexpr number damping_factor = 0.2;
    m_dot *=
      1 - std::exp(-damping_factor * (T_liquid - boiling_temperature_at_atmospheric_pressure));

    // No evaporation below boiling temperature
    if (T_sat < boiling_temperature_at_atmospheric_pressure)
      {
        m_dot  = 0.;
        T_jump = 0.;
      }

    // Set computed values
    evaporative_mass_flux = m_dot;
    temperature_jump      = T_jump;
  }

  template <typename number>
  number
  EvaporationModelKnight<number>::compute_temperature_ratio(const number &m) const
  {
    const number helper =
      0.5 * (specific_heat_ratio_vapor - 1.) / (specific_heat_ratio_vapor + 1.) * m;
    constexpr number sqrt_pi = std::sqrt(std::numbers::pi);

    const number sqrt_T_gas_over_T_sat =
      std::sqrt(1. + std::numbers::pi * helper * helper) - sqrt_pi * helper;

    return sqrt_T_gas_over_T_sat * sqrt_T_gas_over_T_sat;
  }

  template <typename number>
  number
  EvaporationModelKnight<number>::compute_density_ratio(const number &T_gas_over_T_sat,
                                                        const number &m) const
  {
    const number     exp_m_2 = std::exp(m * m);
    const number     erfc_m  = std::erfc(m);
    constexpr number sqrt_pi = std::sqrt(std::numbers::pi);

    const number rho_gas_over_rho_sat =
      ((m * m + 0.5) * exp_m_2 * erfc_m - m / sqrt_pi) / std::sqrt(T_gas_over_T_sat) +
      0.5 / T_gas_over_T_sat * (1. - sqrt_pi * m * exp_m_2 * erfc_m);

    return rho_gas_over_rho_sat;
  }

  template <typename number>
  number
  EvaporationModelKnight<number>::compute_factor_beta(const number &T_gas_over_T_sat,
                                                      const number &rho_gas_over_rho_sat,
                                                      const number &m) const
  {
    const number     sqrt_T_sat_over_T_gas = 1. / std::sqrt(T_gas_over_T_sat);
    const number     m_2                   = m * m;
    constexpr number sqrt_pi               = std::sqrt(std::numbers::pi);

    const number beta = (2. * m_2 + 1. - m * sqrt_pi * sqrt_T_sat_over_T_gas) * std::exp(m_2) /
                        rho_gas_over_rho_sat * sqrt_T_sat_over_T_gas;

    return beta;
  }

  template class EvaporationModelKnight<double>;
} // namespace MeltPoolDG::Evaporation
