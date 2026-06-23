#pragma once

#include <meltpooldg/compressible_flow/eos_utils.hpp>
#include <meltpooldg/compressible_flow/material.hpp>

#include <memory>

namespace MeltPoolDG::CompressibleFlow
{
  /**
   * @brief A class which provides all relevant material properties for a specific phase.
   *
   * A class that provides all relevant material parameters of the considered fluid phase, as
   * well as thermodynamic computations related to the specific equation of state.
   */
  template <int dim, typename number>
  class Material
  {
  public:
    /// Material data object providing all relevant material parameters
    const MaterialPhaseData<number> &data;

    /// Class specific to an equation of state for thermodynamics-related computations
    std::shared_ptr<EOS::EquationOfStateUtils<dim, number>> eos_utils;

    /**
     * @brief Constructor.
     *
     * @param material_data_in Reference to a material data object providing all relevant material
     * parameters.
     */
    explicit Material(const MaterialPhaseData<number> &material_data_in)
      : data(material_data_in)
    {
      eos_utils = make_eos_utils(data);
    }

  private:
    /**
     * @brief Set up helper class for thermodynamic relations.
     *
     * Sets the pointer to the correct helper class for thermodynamic-related computations according
     * to the defined type of equation of state.
     *
     * @param material_data Material data struct providing all relevant material parameters.
     *
     * @note Currently, three equations of state are implemented: ideal gas, stiffened gas,
     * Noble-Abel stiffened gas.
     */
    std::shared_ptr<EOS::EquationOfStateUtils<dim, number>>
    make_eos_utils(const MaterialPhaseData<number> &material_data)
    {
      switch (material_data.eos_type)
        {
          case EquationOfState::ideal_gas:
            return std::make_shared<EOS::IdealGas<dim, number>>(material_data);
          case EquationOfState::stiffened_gas:
            return std::make_shared<EOS::StiffenedGas<dim, number>>(material_data);
          case EquationOfState::noble_abel_stiffened_gas:
            return std::make_shared<EOS::NobleAbelStiffenedGas<dim, number>>(material_data);
          case EquationOfState::weakly_compressible:
            return std::make_shared<EOS::WeaklyCompressible<dim, number>>(material_data);
          default:
            DEAL_II_NOT_IMPLEMENTED();
        }
    }
  };
} // namespace MeltPoolDG::CompressibleFlow
