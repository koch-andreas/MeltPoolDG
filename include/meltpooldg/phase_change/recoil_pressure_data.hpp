#pragma once
#include <meltpooldg/core/material_data.hpp>
#include <meltpooldg/level_set/delta_approximation_phase_weighted_data.hpp>
#include <meltpooldg/utilities/enum.hpp>
#include <meltpooldg/utilities/numbers.hpp>

namespace MeltPoolDG::Evaporation
{

  BETTER_ENUM(RegularizedRecoilPressureTemperatureEvaluationType,
              char,
              // The flux distributed in the interfacial zone is computed based on local values
              // evaluated at the quadrature points.
              local_value,
              // The flux distributed in the interfacial zone is computed based on values evaluated
              // at the projected quadrature points to the level set = 0 isosurface.
              interface_value)

  BETTER_ENUM(RecoilPressureModelType,
              char,
              // default: compute phenomenological recoil pressure
              //    p_v(T) = p_recoil_phenomenological(T)
              phenomenological,
              // hybrid recoil pressure model, considering the evaporation-induced velocity jump;
              // The pressure jump is computed from
              //
              //    p_v(T) = p_recoil_phenomenological(T) - mDot^2*(1/rho_g-1/rho_l).
              //
              // This ensures that the evaporation-induced pressure jump is the same as
              // in the phenomenological recoil pressure model.
              hybrid)

  template <typename number>
  struct RecoilPressureData
  {
    // Enable or disable recoil-pressure related effects.
    bool enable = false;

    // TODO
    bool enable_linear_activation_ramp = true;

    // TODO
    bool subtract_ambient_pressure = false;

    // ambient gas pressure
    // default value for air in Pa
    number ambient_gas_pressure = 1.013e5;

    // recoil pressure constant
    // recommended as c_p = 0.55
    number pressure_coefficient = 0.55;

    // temperature constant
    // Only used if it is set >=0. In the default case it is computed from
    // c_T = h_v/R with the molar latent heat of evaporation h_v
    // and the universal gas constant R.
    number temperature_constant = -1;

    // sticking constant
    number sticking_constant = 1.0;

    // activation temperature of the recoil pressure; must be smaller than or equal to the boiling
    // temperature; this parameter enables a smooth activation of the recoil pressure
    number activation_temperature = numbers::invalid_double;

    // Choose how the recoil pressure flux across the interface should be computed:
    // * local_value: use the local temperature value
    // * interface_value: use the value at the interface (level set=0)
    RegularizedRecoilPressureTemperatureEvaluationType interface_distributed_flux_type =
      RegularizedRecoilPressureTemperatureEvaluationType::local_value;

    // Choose the delta-function for computing the continuum interface force.
    LevelSet::DeltaApproximationPhaseWeightedData<number> delta_approximation_phase_weighted;

    // Choose the model type to compute the recoil pressure:
    //   * phenomenological (default)
    //   * hybrid
    RecoilPressureModelType type = RecoilPressureModelType::phenomenological;

    /**
     * Add parameters to the ParameterHandler
     *
     * Attach the parsing recipe for recoil pressure related data to the
     * ParameterHandler.
     *
     * @param[in, out] prm ParameterHandler.
     */
    void
    add_parameters(dealii::ParameterHandler &prm);

    /**
     * Post operation
     *
     * Set default values of parameters, after parameters have been parsed.
     *
     * @param[in] material Material data.
     */
    void
    post(const MaterialData<number> &material);
  };
} // namespace MeltPoolDG::Evaporation
