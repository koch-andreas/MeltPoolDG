#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/level_set/delta_approximation_phase_weighted_data.hpp>
#include <meltpooldg/level_set/reinitialization_data.hpp>
#include <meltpooldg/phase_change/recoil_pressure_data.hpp>
#include <meltpooldg/utilities/enum.hpp>
#include <meltpooldg/utilities/numbers.hpp>

#include <string>


namespace MeltPoolDG::Evaporation
{
  // evaporation specific
  BETTER_ENUM(
    EvaporationModelType,
    char,
    // prescribe a (time-dependent) function for an evaporative mass flux being constant
    // over the domain
    analytical,
    // calculate the evaporative mass flux from the recoil pressure
    recoil_pressure,
    // calculate the evaporative mass flux from the saturated vapor pressure
    saturated_vapor_pressure,
    // calculate the evaporative mass flux according to the model proposed by Hardt & Wondra
    hardt_wondra)

  BETTER_ENUM(EvaporationLevelSetSourceTermType,
              char,
              // calculate the interface velocity from the velocity at the interface
              // TODO: this only makes sense without evaporation --> remove for
              // evaporation
              interface_velocity_sharp,
              // calculate the interface velocity by using the liquid velocity (H(phi)=1)
              interface_velocity_sharp_heavy,
              // calculate a divergence-free interface velocity and use it to advect the level set
              interface_velocity_local,
              // use the source term due to evaporation as right hand-side term
              rhs)

  BETTER_ENUM(InterfaceFluxType,
              char,
              // Regularized representation of the interface flux via a smoothed Dirac
              // delta function.
              regularized,
              // Sharp representation of the interface flux by performing a surface
              // integral over the interface.
              sharp)

  BETTER_ENUM(EvaporativeMassFluxTemperatureEvaluationType,
              char,
              // The evaporative mass flux distributed in the interfacial zone is computed based on
              // local values evaluated at the quadrature points.
              local_value,
              // The flux distributed in the interfacial zone is computed based on values evaluated
              // at the projected quadrature points to the level set = 0 isosurface.
              interface_value)

  BETTER_ENUM(EvaporCoolingInterfaceFluxType,
              char,
              none,
              // Regularized representation of the interface flux via a smoothed Dirac
              // delta function.
              regularized,
              // Sharp representation of the interface flux by performing a surface
              // integral over the interface.
              sharp,
              // Sharp representation of the interface flux by performing a surface
              // integral over the element edges that represent the interface. The usage
              // is only recommended for interfaces that are aligned with element
              // edges.
              sharp_conforming)

  template <typename number>
  struct EvaporationData
  {
    EvaporationModelType evaporative_mass_flux_model = EvaporationModelType::analytical;
    EvaporativeMassFluxTemperatureEvaluationType interface_temperature_evaluation_type =
      EvaporativeMassFluxTemperatureEvaluationType::local_value;

    // source terms
    struct EvaporativeDilationRate
    {
      bool              enable = false;
      InterfaceFluxType model  = InterfaceFluxType::regularized;
    } evaporative_dilation_rate;

    struct EvaporativeCooling
    {
      bool enable = false;

      std::string consider_enthalpy_transport_vapor_mass_flux = "default";

      number activation_temperature = numbers::invalid_double;

      bool enable_linear_activation_ramp = true;

      // only for the diffuse operator
      EvaporCoolingInterfaceFluxType model = EvaporCoolingInterfaceFluxType::regularized;
      LevelSet::DeltaApproximationPhaseWeightedData<number> delta_approximation_phase_weighted;

    } evaporative_cooling;

    RecoilPressureData<number> recoil;

    EvaporationLevelSetSourceTermType formulation_source_term_level_set =
      EvaporationLevelSetSourceTermType::interface_velocity_local;

    struct HardtWondraData
    {
      number coefficient = 0.0;
    } hardt_wondra;

    struct AnalyticalModelData
    {
      std::string function = "not_initialized";
    } analytical;

    bool do_level_set_pressure_gradient_interpolation = false;

    /**
     * Add parameters to the ParameterHandler
     *
     * Attach the parsing recipe for evaporation related data to the
     * ParameterHandler.
     *
     * @param[in, out] prm ParameterHandler.
     */
    void
    add_parameters(dealii::ParameterHandler &prm);

    void
    check_input_parameters(const MaterialData<number> &material) const;

    /**
     * Post operation
     *
     * Set default values of parameters, after parameters have been parsed.
     *
     * @param[in] material Material data.
     * @param[in] use_volume_specific_thermal_capacity_for_phase_interpolation
     * Set to true if the volume-specific heat capacity is interpolated
     * as a single quantity across the interfaces.
     */
    void
    post(const MaterialData<number> &material,
         const bool                  use_volume_specific_thermal_capacity_for_phase_interpolation);
  };
} // namespace MeltPoolDG::Evaporation
