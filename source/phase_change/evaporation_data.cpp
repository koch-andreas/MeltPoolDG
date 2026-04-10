#include <deal.II/base/exceptions.h>
#include <deal.II/base/patterns.h>

#include <meltpooldg/phase_change/evaporation_data.hpp>
#include <meltpooldg/phase_change/recoil_pressure_data.hpp>


namespace MeltPoolDG::Evaporation
{
  template <typename number>
  void
  EvaporationData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("evaporation");
    {
      prm.add_parameter("evaporative mass flux model",
                        evaporative_mass_flux_model,
                        "Choose the formulation how the evaporative mass flux mDot (kg/(m2s)) "
                        "will be calculated.");
      prm.add_parameter(
        "interface temperature evaluation type",
        interface_temperature_evaluation_type,
        "Choose the formulation how the (local) evaporative mass flux will be converted to a DoF vector."
        "will be calculated. "
        "When the CutFEM heat transfer operator is used, this input parameter is ignored and the temperature "
        "is evaluated at the sharp interface which is equivalent to \"sharp\".");

      prm.enter_subsection("analytical");
      {
        prm.add_parameter(
          "function",
          analytical.function,
          "For evapor evaporation model == analytical, prescribe a spatially constant "
          "mass flux due to evaporation (SI unit in kg/m²s), as a function over time t "
          ", e.g. min(2.*t,0.01).");
      }
      prm.leave_subsection();

      prm.enter_subsection("hardt wondra");
      {
        prm.add_parameter("coefficient",
                          hardt_wondra.coefficient,
                          "Evaporation coefficient for the model by Hardt and Wondra.");
      }
      prm.leave_subsection();

      prm.enter_subsection("evaporative dilation rate");
      {
        prm.add_parameter("enable",
                          evaporative_dilation_rate.enable,
                          "Set this parameter to true to consider the evaporative dilation "
                          "rate in the Navier-Stokes equation. This results in an "
                          "evaporation-induced jump in the normal velocity component. ");
        prm.add_parameter("model",
                          evaporative_dilation_rate.model,
                          "Select how the additional source term due to evaporation in the"
                          " continuity equation (=evaporative dilation rate) is computed.");
      }
      prm.leave_subsection();

      prm.enter_subsection("evaporative cooling");
      {
        prm.add_parameter("enable",
                          evaporative_cooling.enable,
                          "Set this parameter to true to consider evaporative cooling "
                          "in the heat equation");
        prm.add_parameter("enable linear activation ramp",
                          evaporative_cooling.enable_linear_activation_ramp,
                          "Enable a linear activation ramp for evaporative cooling between "
                          "the activation temperature and the boiling temperature. "
                          "If enabled, the mass flux increases smoothly and linearly within "
                          "this temperature range. Otherwise, the mass flux is computed directly "
                          "without applying a ramp.");
        prm.add_parameter("consider enthalpy transport vapor mass flux",
                          evaporative_cooling.consider_enthalpy_transport_vapor_mass_flux,
                          "Set this parameter to true to account for the enthalpy "
                          "transported by the vapor mass flux in the heat equation. "
                          "This is only recommended if the vapor mass flux is not "
                          "considered in the Navier-Stokes equations.",
                          dealii::Patterns::Selection("default|true|false"));
        prm.add_parameter(
          "activation temperature",
          evaporative_cooling.activation_temperature,
          "Activation temperature for the evaporative cooling. It must be smaller than or equal to the "
          "boiling temperature. By default, it will be chosen such that the transition from the linear "
          "activation ramp is kink-free.");
        prm.add_parameter("model",
                          evaporative_cooling.model,
                          "Select how the additional source term due to evaporation in the "
                          "heat equation (evaporative cooling) is computed.");
        // When the CutFEM heat transfer operator is used, this input parameter is ignored.
        evaporative_cooling.delta_approximation_phase_weighted.add_parameters(prm);
      }
      prm.leave_subsection();

      recoil.add_parameters(prm);

      prm.add_parameter("formulation source term level set",
                        formulation_source_term_level_set,
                        "Select the type how the evaporative mass flux should be considered "
                        "in the level set equation.");
      prm.add_parameter("do level set pressure gradient interpolation",
                        do_level_set_pressure_gradient_interpolation,
                        "Set if the level set gradient for computing the delta function within "
                        "the evaporative mass flux source terms should be computed based on an "
                        "interpolation to the pressure space. This is only implemented for "
                        "evapor_level_set_source_term_type = rhs.");
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  EvaporationData<number>::post(
    const MaterialData<number> &material,
    const bool                  use_volume_specific_thermal_capacity_for_phase_interpolation)
  {
    recoil.post(material);

    // set automatic weights of phase-weighted delta functions, if requested
    if (use_volume_specific_thermal_capacity_for_phase_interpolation)
      evaporative_cooling.delta_approximation_phase_weighted.set_parameters(
        material, LevelSet::ParameterScaledInterpolationType::volume_specific_heat_capacity);
    else
      evaporative_cooling.delta_approximation_phase_weighted.set_parameters(
        material, LevelSet::ParameterScaledInterpolationType::specific_heat_capacity_times_density);

    // Set increased enthalpy due to vapor mass flux, if it is not specified
    // by the user and the latter is not considered in the mass balance equation.
    if (evaporative_cooling.consider_enthalpy_transport_vapor_mass_flux == "default")
      {
        if (recoil.enable && recoil.type == RecoilPressureModelType::phenomenological)
          evaporative_cooling.consider_enthalpy_transport_vapor_mass_flux = "true";
        else
          evaporative_cooling.consider_enthalpy_transport_vapor_mass_flux = "false";
      }
  }

  template <typename number>
  void
  EvaporationData<number>::check_input_parameters(const MaterialData<number> &material) const
  {
    AssertThrow(!recoil.enable ||
                  (evaporative_dilation_rate.enable ||
                   recoil.type == Evaporation::RecoilPressureModelType::phenomenological),
                dealii::ExcMessage("For the phenomenological recoil pressure model, no velocity "
                                   "jump is allowed."));

    AssertThrow(!evaporative_cooling.enable || material.latent_heat_of_evaporation > 0.0,
                dealii::ExcMessage("To consider the evaporative heat flux the value for "
                                   ">>> latent heat of evaporation <<< "
                                   "must be larger than zero."));
  }

  template struct EvaporationData<double>;
} // namespace MeltPoolDG::Evaporation
