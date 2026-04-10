#pragma once

#include <meltpooldg/phase_change/evaporative_cooling.hpp>
//
#include <deal.II/base/exceptions.h>

#include <meltpooldg/phase_change/evaporation_model_recoil_pressure.hpp>
#include <meltpooldg/phase_change/evaporation_model_saturated_vapor_pressure.hpp>
#include <meltpooldg/utilities/numbers.hpp>


namespace MeltPoolDG::Evaporation
{
  template <typename number>
  EvaporativeCooling<number>::EvaporativeCooling(const EvaporationData<number> &evapor_data,
                                                 const MaterialData<number>    &material_data,
                                                 const bool setup_internal_mass_flux_operator)
    : do_phenomenological_recoil_pressure(
        evapor_data.evaporative_cooling.consider_enthalpy_transport_vapor_mass_flux == "true")
    , latent_heat_of_evaporation(material_data.latent_heat_of_evaporation)
    , specific_heat_capacity(material_data.liquid.specific_heat_capacity)
    , specific_enthalpy_reference_temperature(material_data.specific_enthalpy_reference_temperature)
    , boiling_temperature(material_data.boiling_temperature)
    , ramp_enabled(evapor_data.evaporative_cooling.enable_linear_activation_ramp)
  {
    if (do_phenomenological_recoil_pressure)
      AssertThrow(not numbers::is_invalid(specific_enthalpy_reference_temperature),
                  dealii::ExcMessage(
                    "For the phenomenological recoil pressure model, the reference temperature "
                    "for computing the specific enthalpy must be specified. Abort..."));

    if (setup_internal_mass_flux_operator)
      {
        if (evapor_data.evaporative_mass_flux_model == EvaporationModelType::recoil_pressure)
          {
            mass_flux_operator = std::make_unique<EvaporationModelRecoilPressure<number>>(
              evapor_data.recoil,
              material_data.boiling_temperature,
              material_data.molar_mass,
              material_data.latent_heat_of_evaporation);
          }
        else if (evapor_data.evaporative_mass_flux_model ==
                 EvaporationModelType::saturated_vapor_pressure)
          {
            mass_flux_operator = std::make_unique<EvaporationModelSaturatedVaporPressure<number>>(
              evapor_data.recoil,
              material_data.boiling_temperature,
              material_data.molar_mass,
              material_data.latent_heat_of_evaporation);
          }
        else
          {
            AssertThrow(false, dealii::ExcNotImplemented());
          }


        if (numbers::is_invalid(evapor_data.evaporative_cooling.activation_temperature))
          {
            // Set the activation temperature so that the transition from the linear activation ramp
            // is kink-free.
            activation_temperature =
              material_data.boiling_temperature -
              compute_evaporative_cooling(material_data.boiling_temperature) /
                compute_evaporative_cooling_derivative_with_temperature_dependent_mass_flux(
                  material_data.boiling_temperature);
          }

        if (ramp_enabled)
          {
            activation_ramp_derivative =
              compute_evaporative_cooling(material_data.boiling_temperature) /
              (material_data.boiling_temperature - activation_temperature);
          }
      }
  }


  template <typename number>
  template <typename ValueType>
  inline ValueType
  EvaporativeCooling<number>::compute_evaporative_cooling(
    const ValueType                  &mass_flux,
    [[maybe_unused]] const ValueType &temperature) const
  {
    ValueType specific_enthalpy(0.0);
    if (do_phenomenological_recoil_pressure)
      specific_enthalpy = compute_phenomenological_specific_enthalpy(temperature);

    return -(latent_heat_of_evaporation + specific_enthalpy) * mass_flux;
  }


  template <typename number>
  inline number
  EvaporativeCooling<number>::compute_evaporative_cooling(const number temperature) const
  {
    Assert(mass_flux_operator,
           dealii::ExcMessage("To use this function, the class must be constructed with "
                              "setup_internal_mass_flux_operator = true."));

    if (temperature < activation_temperature)
      return 0.0;
    else if (temperature >= boiling_temperature || not ramp_enabled)
      return compute_evaporative_cooling(
        mass_flux_operator->local_compute_evaporative_mass_flux(temperature), temperature);
    else
      // linear activation ramp
      return activation_ramp_derivative * (temperature - activation_temperature);
  }


  template <typename number>
  dealii::VectorizedArray<number>
  EvaporativeCooling<number>::compute_evaporative_cooling(
    const dealii::VectorizedArray<number> &temperature) const
  {
    Assert(mass_flux_operator,
           dealii::ExcMessage("To use this function, the class must be constructed with "
                              "setup_internal_mass_flux_operator = true."));

    return dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
      temperature,
      activation_temperature,
      0,
      dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than_or_equal>(
        temperature,
        ramp_enabled ? boiling_temperature : activation_temperature,
        compute_evaporative_cooling(
          mass_flux_operator->local_compute_evaporative_mass_flux_vec(temperature), temperature),
        activation_ramp_derivative * (temperature - activation_temperature)));
  }


  template <typename number>
  template <typename ValueType>
  inline ValueType
  EvaporativeCooling<number>::compute_evaporative_cooling_derivative_constant_mass_flux(
    [[maybe_unused]] const ValueType &mass_flux) const
  {
    if (do_phenomenological_recoil_pressure)
      return -specific_heat_capacity * mass_flux;
    else
      return 0.0;
  }


  template <typename number>
  inline number
  EvaporativeCooling<number>::
    compute_evaporative_cooling_derivative_with_temperature_dependent_mass_flux(
      const number temperature) const
  {
    Assert(mass_flux_operator,
           dealii::ExcMessage("To use this function, the class must be constructed with "
                              "setup_internal_mass_flux_operator = true."));

    if (temperature < activation_temperature)
      return 0.0;
    else if (temperature >= boiling_temperature || not ramp_enabled)
      {
        const auto mass_flux_derivative =
          mass_flux_operator->local_compute_evaporative_mass_flux_derivative(temperature);
        if (do_phenomenological_recoil_pressure)
          return -specific_heat_capacity *
                   mass_flux_operator->local_compute_evaporative_mass_flux(temperature) -
                 (latent_heat_of_evaporation +
                  compute_phenomenological_specific_enthalpy(temperature)) *
                   mass_flux_derivative;
        else
          return -specific_heat_capacity * mass_flux_derivative;
      }
    else
      return activation_ramp_derivative;
  }


  template <typename number>
  dealii::VectorizedArray<number>
  EvaporativeCooling<number>::
    compute_evaporative_cooling_derivative_with_temperature_dependent_mass_flux(
      const dealii::VectorizedArray<number> &temperature) const
  {
    Assert(mass_flux_operator,
           dealii::ExcMessage("To use this function, the class must be constructed with "
                              "setup_internal_mass_flux_operator = true."));

    return dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
      temperature,
      activation_temperature,
      0,
      dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than_or_equal>(
        temperature,
        ramp_enabled ? boiling_temperature : activation_temperature,
        (do_phenomenological_recoil_pressure) ?
          -specific_heat_capacity *
              mass_flux_operator->local_compute_evaporative_mass_flux_vec(temperature) -
            (latent_heat_of_evaporation + compute_phenomenological_specific_enthalpy(temperature)) *
              mass_flux_operator->local_compute_evaporative_mass_flux_vec_derivative(temperature) :
          -specific_heat_capacity *
            mass_flux_operator->local_compute_evaporative_mass_flux_vec_derivative(temperature),
        activation_ramp_derivative));
  }

  template <typename number>
  template <typename ValueType>
  inline ValueType
  EvaporativeCooling<number>::compute_phenomenological_specific_enthalpy(
    const ValueType &temperature) const
  {
    return specific_heat_capacity * (temperature - specific_enthalpy_reference_temperature);
  }

} // namespace MeltPoolDG::Evaporation
