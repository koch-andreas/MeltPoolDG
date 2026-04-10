#pragma once

#include <deal.II/base/vectorization.h>

#include <meltpooldg/core/material_data.hpp>
#include <meltpooldg/phase_change/evaporation_data.hpp>
#include <meltpooldg/phase_change/evaporation_model_base.hpp>

#include <memory>


namespace MeltPoolDG::Evaporation
{
  /**
   * Compute the heat sink due to evaporative cooling
   *             .
   *    q_s =  - m · (h_v + h(T))
   *
   * with the latent heat of evaporation h_v, the specific enthalpy
   *
   *          T
   *         /
   *        |
   * h(T) = | c_p  .  (1)
   *        |
   *       /
   *     T_ref
   *
   * where T_ref denotes an artificial reference temperature for
   * the specific enthalpy.
   *
   * h_v + h(T) is the total enthalpy leaving the system with the
   * evaporative mass flux. h(T) can be typically found in tables.
   * However, we assume for the computation of the specific
   * enthalpy a linear temperature-dependence
   *
   *    h(T) = h_ref + c_p * T .  (2)
   *
   * see also
   * Meier, Christoph, et al. "A novel smoothed particle hydrodynamics
   * formulation for thermo-capillary phase change problems with focus
   * on metal additive manufacturing melt pool modeling." CMAME 381
   * (2021).
   *
   * By setting (1) equal to (2) and assuming the heat capacity to
   * be temperature-independent, we obtain
   *
   *    h_ref = - c_p * T_ref
   *
   * Inserting into (2) yields
   *
   *    h(T) = c_p * (T - T_ref).
   *
   *
   * @note For the computation of h(T), it is assumed that the
   *       specific heat capacity c_p corresponds to the value
   *       for the liquid and solid phase.
   *
   * @note Instead of T_ref we could have also introduced directly
   *       h_ref as an input parameter.
   */
  template <typename number>
  class EvaporativeCooling
  {
  public:
    EvaporativeCooling(const EvaporationData<number> &evapor_data,
                       const MaterialData<number>    &material_data,
                       const bool                     setup_internal_mass_flux_operator = false);


    /**
     * Compute the heat sink with given a constant @param mass_flux as:
     */
    template <typename ValueType>
    inline ValueType
    compute_evaporative_cooling(const ValueType                  &mass_flux,
                                [[maybe_unused]] const ValueType &temperature) const;


    /**
     * Compute the heat sink with the internal mass flux operator
     */
    inline number
    compute_evaporative_cooling(const number temperature) const;

    inline dealii::VectorizedArray<number>
    compute_evaporative_cooling(const dealii::VectorizedArray<number> &temperature) const;


    /**
     * Compute the heat sink derivative given a constant @param mass_flux .
     *
     * The derivative of specific enthalpy h(T) with respect to the temperature:
     *
     *  d h(T)
     * -------- = c_p^sl
     *    dT
     *
     * tangent of heat sink due to evaporation:
     *
     *  d q_s      .
     * ------- = - m * c_p^sl
     *    dT
     */
    template <typename ValueType>
    inline ValueType
    compute_evaporative_cooling_derivative_constant_mass_flux(
      [[maybe_unused]] const ValueType &mass_flux) const;


    /**
     * Compute the heat sink derivative with the internal mass flux operator
     *
     *                                            .
     *  d q_s               .                   d m
     * ------- = - c_p^sl * m - (h_v + h(T)) * -----
     *    dT                                     dT
     */
    inline number
    compute_evaporative_cooling_derivative_with_temperature_dependent_mass_flux(
      const number temperature) const;

    inline dealii::VectorizedArray<number>
    compute_evaporative_cooling_derivative_with_temperature_dependent_mass_flux(
      const dealii::VectorizedArray<number> &temperature) const;

  private:
    template <typename ValueType>
    inline ValueType
    compute_phenomenological_specific_enthalpy(const ValueType &temperature) const;

    const bool                                    do_phenomenological_recoil_pressure;
    const number                                  latent_heat_of_evaporation;
    const number                                  specific_heat_capacity;
    const number                                  specific_enthalpy_reference_temperature;
    const number                                  boiling_temperature;
    number                                        activation_temperature;
    number                                        activation_ramp_derivative = 0.0;
    std::unique_ptr<EvaporationModelBase<number>> mass_flux_operator;
    const bool                                    ramp_enabled;
  };
} // namespace MeltPoolDG::Evaporation
