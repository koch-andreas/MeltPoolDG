#include <deal.II/base/patterns.h>

#include <meltpooldg/phase_change/recoil_pressure_data.hpp>
#include <meltpooldg/utilities/physical_constants.hpp>

namespace MeltPoolDG::Evaporation
{
  template <typename number>
  void
  RecoilPressureData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("recoil pressure");
    {
      prm.add_parameter(
        "enable",
        enable,
        "Set this parameter to true to prescribe the evaporation-induced jump in the pressure field "
        "(i.e. recoil pressure), considered as an interfacial force in the momentum balance equation."
        "If 'evaporative dilation rate' is enabled, this pressure jump will be added to the "
        "one resulting from the discontinuous normal velocity field.");
      prm.add_parameter(
        "enable linear activation ramp",
        enable_linear_activation_ramp,
        "Enable a linear activation ramp for recoil pressure between "
        "the activation temperature and the boiling temperature. "
        "If enabled, the recoil pressure increases smoothly and linearly within "
        "this temperature range. Otherwise, the recoil pressure is computed directly "
        "without applying a ramp.");
      prm.add_parameter("subtract ambient pressure",
                        subtract_ambient_pressure,
                        "Subtract ambient pressure from the recoil pressure.");
      prm.add_parameter("ambient gas pressure",
                        ambient_gas_pressure,
                        "Ambient gas pressure for the recoil pressure model.");
      prm.add_parameter("pressure coefficient",
                        pressure_coefficient,
                        "Pressure coefficient for the recoil pressure model.",
                        dealii::Patterns::Double(0.0, 1.0));
      prm.add_parameter("temperature constant",
                        temperature_constant,
                        "Temperature constant for the recoil pressure model. "
                        "If this parameter is not set, the value is computed by "
                        "latent_heat_evaporation * molar_mass / "
                        "universal_gas_constant;");
      prm.add_parameter("sticking constant",
                        sticking_constant,
                        "Sticking constant.",
                        dealii::Patterns::Double(0.0, 1.0));
      prm.add_parameter("interface distributed flux type",
                        interface_distributed_flux_type,
                        "Type that determines how the recoil pressure force is computed in the "
                        "interfacial zone.");
      prm.add_parameter(
        "activation temperature",
        activation_temperature,
        "Activation temperature for the recoil pressure. It must be smaller than or equal to the "
        "boiling temperature. As default value, the boiling temperature is chosen.");
      delta_approximation_phase_weighted.add_parameters(prm);
      prm.add_parameter(
        "type",
        type,
        "Choose the model to compute the recoil pressure coefficient: phenomenological "
        "or hybrid, in case there is also an evaporation-induced velocity jump.");
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  RecoilPressureData<number>::post(const MaterialData<number> &material)
  {
    // recoil pressure: set default value of activation temperature equal to the boiling
    // temperature
    if (numbers::is_invalid(activation_temperature))
      activation_temperature = material.boiling_temperature;

    // set automatic weights of asymmetric delta functions, if requested
    this->delta_approximation_phase_weighted.set_parameters(
      material, LevelSet::ParameterScaledInterpolationType::density);

    // set default value of temperature constant
    if (temperature_constant < 0)
      temperature_constant = material.latent_heat_of_evaporation * material.molar_mass /
                             PhysicalConstants::universal_gas_constant;
  }

  template struct RecoilPressureData<double>;
} // namespace MeltPoolDG::Evaporation
