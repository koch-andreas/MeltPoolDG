#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/utilities/enum.hpp>
#include <meltpooldg/utilities/numbers.hpp>

#include <string>

namespace MeltPoolDG
{
  BETTER_ENUM(
    MaterialTemplate,
    char,
    none,            // all material parameters must be specified
    stainless_steel, // melt and solid material parameters will be set to stainless steel values
    Ti64,            // melt and solid material parameters will be set to Ti-6Al-4V values
    Ti64Benchmark    // melt and solid material parameters will be set to Ti-6Al-4V values for the
                     // benchmark paper together with PolyMTL.
  )
  BETTER_ENUM(SolidLiquidPropertiesTransitionType,
              char,
              mushy_zone, // solid and liquid properties are interpolated between the solidus and
                          // liquidus temperature
              sharp // solid and liquid properties jump at the melting point, which is set via the
                    // solidus temperature
  )
  BETTER_ENUM(TwoPhaseFluidPropertiesTransitionType,
              char,
              sharp,  // properties jump at heaviside = 0.5
              smooth, // properties are smeared between the phases proportional to the heaviside
              consistent_with_evaporation //  same as "smooth", but the density is interpolated
                                          //  proportional by the harmonic mean.
                                          // consistent with the evaporation formulation
  )

  template <typename number>
  struct MaterialPhaseData
  {
    number thermal_conductivity   = 0.0;
    number specific_heat_capacity = 0.0;
    number density                = 0.0;
    number dynamic_viscosity      = 0.0;

    void
    add_parameters(dealii::ParameterHandler &prm, const std::string &phase_name);
  };

  /**
   * Parameters of the Material class.
   *
   * @warning If you want to use a predefined material as template (by specifying
   * <material template>) but also modify individual properties, you need to
   * specify it in the first place in the <material> section.
   */
  template <typename number>
  struct MaterialData
  {
  private:
    MaterialTemplate material_template = MaterialTemplate::none;

  public:
    /**
     * Material parameters of the gas phase; heaviside(level set) == 0
     */
    MaterialPhaseData<number> gas;

    /**
     * Material parameters of the liquid phase; heaviside(level set) == 1
     */
    MaterialPhaseData<number> liquid;

    /**
     * Solid material.
     */
    MaterialPhaseData<number> solid;

    number solidus_temperature        = 0.0;
    number liquidus_temperature       = 0.0;
    number boiling_temperature        = 0.0;
    number latent_heat_of_evaporation = 0.0;
    number molar_mass                 = 0.0;

    number specific_enthalpy_reference_temperature = numbers::invalid_double;

    SolidLiquidPropertiesTransitionType solid_liquid_properties_transition_type =
      SolidLiquidPropertiesTransitionType::mushy_zone;
    TwoPhaseFluidPropertiesTransitionType two_phase_fluid_properties_transition_type =
      TwoPhaseFluidPropertiesTransitionType::smooth;

    void
    add_parameters(dealii::ParameterHandler &prm);

    void
    check_parameters_heat_transfer(const bool do_two_phase, const bool do_solidification) const;

    /**
     * Creates MaterialData with all parameters set to the values of stainless steel:
     *
     * gas thermal conductivity      = 0.026  W / (m K)
     * gas specific heat capacity    = 10.0  J / (kg K)
     * gas density                   = 74.3  kg / m³
     * gas dynamic viscosity         = 6.0e-4  kg / (m s)
     * liquid thermal conductivity   = 35.95  W / (m K)
     * liquid specific heat capacity = 965  J / (kg K)
     * liquid density                = 7430  kg / m³
     * liquid dynamic viscosity      = 6.0e-3  kg / (m s)
     * solid thermal conductivity    = 35.95  W / (m K)
     * solid specific heat capacity  = 965  J / (kg K)
     * solid density                 = 7430  kg / m³
     * solid dynamic viscosity       = 0.6  kg / (m s)
     * solidus temperature           = 1700  K
     * liquidus temperature          = 2100  K
     * boiling temperature           = 3000  K
     * latent heat of evaporation    = 6.0e6  J / kg
     * molar mass                    = 5.22e-2  kg / mol
     * specific enthalpy reference temperature = 663.731  K
     */
    static MaterialData<number>
    create_stainless_steel_material_data();

    /**
     * Creates MaterialData with all parameters set to the values of Ti-6Al-4V:
     *
     * gas thermal conductivity      = 0.02863  W / (m K)
     * gas specific heat capacity    = 11.3  J / (kg K)
     * gas density                   = 44.1  kg / m³
     * gas dynamic viscosity         = 0.00035  kg / (m s)
     * liquid thermal conductivity   = 28.63  W / (m K)
     * liquid specific heat capacity = 1130  J / (kg K)
     * liquid density                = 4087  kg / m³
     * liquid dynamic viscosity      = 0.0035  kg / (m s)
     * solid thermal conductivity    = 28.63  W / (m K)
     * solid specific heat capacity  = 1130  J / (kg K)
     * solid density                 = 4087  kg / m³
     * solid dynamic viscosity       = 0.35  kg / (m s)
     * solidus temperature           = 1933  K
     * liquidus temperature          = 2200  K
     * boiling temperature           = 3133  K
     * latent heat of evaporation    = 8.84e6  J / kg
     * molar mass                    = 4.78e-2  kg / mol
     * specific enthalpy reference temperature = 538  K
     */
    static MaterialData<number>
    create_Ti64_material_data();

    /**
     * Creates MaterialData with all parameters set to the Ti-6Al-4V values
     * used in the benchmark paper with PolyMTL.
     */
    static MaterialData<number>
    create_Ti64_benchmark_material_data();
  };
} // namespace MeltPoolDG
