/**
 * @brief Simulation of an evaporating droplet.
 */

#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <meltpooldg/utilities/boundary_ids_colorized.hpp>

#include "../compressible_multiphase_case.hpp"

namespace MeltPoolDG::Simulation::CompressibleMultiphase
{
  template <int dim>
  class NegatedSphere : public dealii::Functions::SignedDistance::Sphere<dim>
    {
    public:
      NegatedSphere(const dealii::Point<dim> &center, double radius)
        : dealii::Functions::SignedDistance::Sphere<dim>(center, radius)
      {}

      double value(const dealii::Point<dim> &p,
                   const unsigned int component = 0) const override
      {
        return -dealii::Functions::SignedDistance::Sphere<dim>::value(p, component);
      }
    };

  /**
   * @brief Energy function for the fixed energy outflow boundary condition.
   */
  template <int dim, typename number>
  class FixedEnergyOutflowBoundaryFunction : public dealii::Function<dim, number>
  {
  public:
    /**
     * @brief Constructor.
     *
     * @param ic_gas_phase Function for the initial conditions of the gas phase.
     * @param ic_liquid_phase Function for the initial conditions of the liquid phase.
     * @param gas_phase_is_first Indicator whether the gas phase is the first phase in the two-phase
     * system.
     */
    explicit FixedEnergyOutflowBoundaryFunction(std::string ic_gas_phase,
                                                std::string ic_liquid_phase,
                                                const bool  gas_phase_is_first = true)
      : dealii::Function<dim, number>(2 /* liquid and gas phase*/)
      , gas_phase_is_first(gas_phase_is_first)
    {
      parsing_function_gas->initialize(dealii::FunctionParser<dim>::default_variable_names(),
                                       ic_gas_phase,
                                       typename dealii::FunctionParser<dim>::ConstMap(),
                                       false);
      parsing_function_liquid->initialize(dealii::FunctionParser<dim>::default_variable_names(),
                                          ic_liquid_phase,
                                          typename dealii::FunctionParser<dim>::ConstMap(),
                                          false);
    }

    /**
     * @brief Computes the current function value for a specific @p component at a given point @p p.
     *
     * @param p Point at which the function should be evaluated.
     * @param component Component for which the function value should be returned. 0 corresponds to
     * the liquid phase and 1 corresponds to the gas phase.
     */
    number
    value(const dealii::Point<dim, number> &p, const unsigned int component) const final
    {
      Assert(component < 2, dealii::ExcIndexRange(component, 0, 2));

      // gas phase
      if (component == not gas_phase_is_first)
        return parsing_function_gas->value(p, dim + 1);
      // liquid phase
      else
        return parsing_function_liquid->value(p, dim + 1);
    }

  private:
    /// Indicator whether the gas phase is the first phase in the two-phase system
    const bool gas_phase_is_first;

    /// Function parser for initial conditions in gas phase
    std::unique_ptr<dealii::FunctionParser<dim>> parsing_function_gas =
      std::make_unique<dealii::FunctionParser<dim>>(dim + 2);

    /// Function parser for initial conditions in liquid phase
    std::unique_ptr<dealii::FunctionParser<dim>> parsing_function_liquid =
      std::make_unique<dealii::FunctionParser<dim>>(dim + 2);
  };

  /**
   * @brief Pressure function for the fixed pressure outflow boundary condition.
   */
  template <int dim, typename number>
  class FixedPressureOutflowBoundaryFunction : public dealii::Function<dim, number>
  {
  public:
    /**
     * @brief Constructor.
     *
     * @param atmospheric_pressure_in Given atmospheric pressure for fixed-pressure outflow boundary
     * condition.
     * @param gas_phase_is_first Indicator whether the gas phase is the first phase in the two-phase
     * system.
     */
    explicit FixedPressureOutflowBoundaryFunction(const number &atmospheric_pressure_in,
                                                  const bool    gas_phase_is_first = true)
      : dealii::Function<dim, number>(2 /* liquid and gas phase*/)
      , atmospheric_pressure(atmospheric_pressure_in)
      , gas_phase_is_first(gas_phase_is_first)
    {}

    /**
     * @brief Computes the current function value for a specific @p component at a given point @p p.
     *
     * @param component Component for which the function value should be returned. 0 corresponds to
     * the liquid phase and 1 corresponds to the gas phase.
     */
    number
    value(const dealii::Point<dim, number> &, const unsigned int component) const final
    {
      Assert(component < 2, dealii::ExcIndexRange(component, 0, 2));

      return atmospheric_pressure;
    }

  private:
    /// Atmospheric pressure (SI: Pa)
    const number atmospheric_pressure;

    /// Indicator whether the gas phase is the first phase in the two-phase system
    const bool gas_phase_is_first;
  };

  /**
   * @brief Initial flow field function for the conserved variables.
   */
  template <int dim, typename number>
  class ConservedVariablesInitialField : public dealii::Function<dim, number>
  {
  public:
    /**
     * @brief Constructor.
     *
     * @param ic_gas_phase Function for the initial conditions of the gas phase.
     * @param ic_liquid_phase Function for the initial conditions of the liquid phase.
     * @param gas_phase_is_first Indicator whether the gas phase is the first phase in the two-phase
     * system.
     */
    explicit ConservedVariablesInitialField(std::string ic_gas_phase,
                                            std::string ic_liquid_phase,
                                            const bool  gas_phase_is_first = true)
      : dealii::Function<dim, number>(2 * (dim + 2))
      , gas_phase_is_first(gas_phase_is_first)
    {
      parsing_function_gas->initialize(dim == 3 ?
                                         std::string("x,y,z") :
                                         (dim == 2 ? std::string("x,y") : std::string("x")),
                                       ic_gas_phase,
                                       constants,
                                       false);
      parsing_function_liquid->initialize(dim == 3 ?
                                            std::string("x,y,z") :
                                            (dim == 2 ? std::string("x,y") : std::string("x")),
                                          ic_liquid_phase,
                                          constants,
                                          false);
    }

    /**
     * @brief Computes the current function value for a specific @p component at a given point @p p.
     *
     * @param p Point at which the function should be evaluated.
     * @param component Component for which the function value should be returned.
     */
    number
    value(const dealii::Point<dim, number> &p, const unsigned int component) const final
    {
      std::array<unsigned int, dim + 2> gas_components;
      std::array<unsigned int, dim + 2> liquid_components;

      for (unsigned int i = 0; i < dim + 2; ++i)
        {
          if (gas_phase_is_first)
            {
              gas_components[i]    = i;
              liquid_components[i] = i + dim + 2;
            }
          else
            {
              gas_components[i]    = i + dim + 2;
              liquid_components[i] = i;
            }
        }

      // gas phase
      if (component == gas_components[0])
        return parsing_function_gas->value(p, 0);
      else if (component == gas_components[1])
        return parsing_function_gas->value(p, 1);
      else if (component == gas_components[2])
        return parsing_function_gas->value(p, 2);
      else if (dim > 1 and component == gas_components[3])
        return parsing_function_gas->value(p, 3);
      else if (dim > 2 and component == gas_components[4])
        return parsing_function_gas->value(p, 4);

      // liquid phase
      else if (component == liquid_components[0])
        return parsing_function_liquid->value(p, 0);
      else if (component == liquid_components[1])
        return parsing_function_liquid->value(p, 1);
      else if (component == liquid_components[2])
        return parsing_function_liquid->value(p, 2);
      else if (dim > 1 and component == liquid_components[3])
        return parsing_function_liquid->value(p, 3);
      else if (dim > 2 and component == liquid_components[4])
        return parsing_function_liquid->value(p, 4);
      else
        return 0.;
    }

  private:
    /// Indicator whether the gas phase is the first phase in the two-phase system
    bool gas_phase_is_first;

    /// Constants for function parsing
    std::map<std::string, number> constants;

    /// Function parser for initial conditions in gas phase
    std::unique_ptr<dealii::FunctionParser<dim>> parsing_function_gas =
      std::make_unique<dealii::FunctionParser<dim>>(dim + 2);

    /// Function parser for initial conditions in liquid phase
    std::unique_ptr<dealii::FunctionParser<dim>> parsing_function_liquid =
      std::make_unique<dealii::FunctionParser<dim>>(dim + 2);
  };

  /**
   * @brief A specific compressible flow simulation setup for a one-dimensional two-phase case with
   * one (moving) interface.
   */
  template <int dim, typename number>
  class SimulationEvaporatingDroplet final : public Multiphase::CompressibleMultiphaseCase<dim, number>
  {
  public:
    /**
     * @brief Constructor.
     *
     * @param parameter_file Parameter file that contains simulation input settings.
     * @param mpi_communicator The MPI communicator used to run the simulation in parallel.
     */
    explicit SimulationEvaporatingDroplet(std::string parameter_file, const MPI_Comm mpi_communicator)
      : Multiphase::CompressibleMultiphaseCase<dim, number>(parameter_file, mpi_communicator)
    {}

    /**
     * @brief Creates the spatial discretization for the simulation setup.
     */
    void
    create_spatial_discretization() override
    {
      // TODO: no distributed triangulation possible for dim=1
      this->triangulation =
        std::make_shared<dealii::parallel::shared::Triangulation<dim>>(this->mpi_communicator);

      dealii::Point<dim, number> lower_left;
      for (unsigned int d = 0; d < dim; ++d)
        lower_left[d] = -5.e-4;

      dealii::Point<dim, number> upper_right;
      for (unsigned int d = 0; d < dim; ++d)
        upper_right[d] = 5.e-4;

      std::vector<unsigned int> subdivisions(dim, 1);
      for (unsigned int d = 0; d < dim; ++d)
        subdivisions[d] = 20;

      dealii::GridGenerator::subdivided_hyper_rectangle(
        *this->triangulation, subdivisions, lower_left, upper_right, true);

      this->triangulation->refine_global(this->parameters.base.global_refinements);
    }

    /**
     * @brief Sets the boundary conditions.
     */
    void
    set_boundary_conditions() override
    {
      // face numbering according to the deal.II colorize flag
      const auto [lower_bc, upper_bc, left_bc, right_bc, front_bc, back_bc] =
        get_colorized_rectangle_boundary_ids<dim>();

      attach_boundary(lower_bc, lower_boundary_condition);
      attach_boundary(upper_bc, upper_boundary_condition);
      if (dim==2)
        {
          attach_boundary(left_bc, left_boundary_condition);
          attach_boundary(right_bc, right_boundary_condition);
        }
      //attach_boundary(front_bc, front_boundary_condition);
      //attach_boundary(back_bc, back_boundary_condition);
    }

    /**
     * @brief Attach the boundary condition function to the current boundary id.
     *
     * @param id Boundary id.
     * @param type Boundary condition type.
     */
    void
    attach_boundary(const dealii::types::boundary_id id, const std::string &type)
    {
      auto function = create_boundary_function(type);

      this->attach_boundary_condition({id, function}, type, "compressible_multiphase_flow");
    }

    /**
     * @brief Set the boundary value function for a given boundary condition type.
     *
     * @param type Given boundary condition type.
     */
    std::shared_ptr<dealii::Function<dim, number>>
    create_boundary_function(const std::string &type)
    {
      if (type == "outflow_fixed_energy")
        {
          return std::make_shared<FixedEnergyOutflowBoundaryFunction<dim, number>>(
            initial_conditions.gas, initial_conditions.liquid);
        }
      else if (type == "outflow_fixed_pressure")
        {
          return std::make_shared<FixedPressureOutflowBoundaryFunction<dim, number>>(
            atmospheric_pressure);
        }
      else
        {
          return std::make_shared<ConservedVariablesInitialField<dim, number>>(
            initial_conditions.gas, initial_conditions.liquid);
        }
    }

    /**
     * @brief Sets the field functions for the simulation.
     */
    void
    set_field_conditions() override
    {
      // The solution vector is ordered, such that the liquid phase is the first phase and the gas
      // phase is the second phase.
      auto initial_condition =
        std::make_shared<ConservedVariablesInitialField<dim, number>>(initial_conditions.gas,
                                                                      initial_conditions.liquid,
                                                                      false /*gas_phase_is_first*/);
      this->attach_initial_condition(initial_condition, "compressible_multiphase_flow");

      // set level-set function
      dealii::Point<dim, number> p{};

      const number radius = 290.0e-6;

      const auto level_set = std::make_shared<NegatedSphere<dim>>(p, radius);
      this->attach_field_function(level_set, "level_set", "compressible_multiphase_flow");
    }

    /**
     * @brief Performs post-processing by evaluating and outputting error norms.
     *
     * @param generic_data_out A generic utility for managing simulation output data.
     */
    void
    do_postprocessing(const GenericDataOut<dim, number> &generic_data_out) const override
    {
      ConservedVariablesInitialField<dim, number> reference_values(initial_conditions.gas,
                                                                   initial_conditions.liquid);
      this->print_relative_norm(generic_data_out, reference_values, "error");
    }

    /**
     * @brief Add simulation specific parameters to the parameter handler.
     *
     * @param prm The parameter handler to which the parameters are added.
     */
    bool
    add_simulation_specific_parameters(dealii::ParameterHandler &prm) override
    {
      const std::string boundary_condition_types_doc =
        "The following boundary conditions are supported:\n"
        "\t 'no_slip_wall', 'slip_wall', 'inflow', 'outflow_fixed_energy', "
        "\t 'outflow_fixed_pressure'";

      prm.enter_subsection("simulation specific parameters");
      {
        prm.add_parameter("lower boundary condition",
                          lower_boundary_condition,
                          boundary_condition_types_doc);
        prm.add_parameter("upper boundary condition",
                          upper_boundary_condition,
                          boundary_condition_types_doc);
        prm.add_parameter("left boundary condition",
                          left_boundary_condition,
                          boundary_condition_types_doc);
        prm.add_parameter("right boundary condition",
                          right_boundary_condition,
                          boundary_condition_types_doc);
        prm.add_parameter("front boundary condition",
                          front_boundary_condition,
                          boundary_condition_types_doc);
        prm.add_parameter("back boundary condition",
                          back_boundary_condition,
                          boundary_condition_types_doc);
        prm.enter_subsection("initial conditions");
        {
          prm.add_parameter("gas",
                            initial_conditions.gas,
                            "Initial condition function for gas phase.");
          prm.add_parameter("liquid",
                            initial_conditions.liquid,
                            "Initial condition function for liquid phase.");
        }
        prm.leave_subsection();
        prm.add_parameter("atmospheric pressure",
                          atmospheric_pressure,
                          "Atmospheric pressure (Pa).",
                          dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
      }
      prm.leave_subsection();

      return this->parameters.base.do_print_parameters;
    }

  private:
    /// Boundary conditions, the options are:
    /// "no_slip_wall", "slip_wall", "inflow", "outflow_fixed_energy", "outflow_fixed_pressure"
    std::string lower_boundary_condition  = "no_slip_wall";
    std::string upper_boundary_condition = "no_slip_wall";
    std::string left_boundary_condition  = "no_slip_wall";
    std::string right_boundary_condition = "no_slip_wall";
    std::string front_boundary_condition  = "no_slip_wall";
    std::string back_boundary_condition = "no_slip_wall";

    /// Functions for the initial conditions
    struct InitialConditions
    {
      std::string gas{};
      std::string liquid{};
    } initial_conditions;

    /// Atmospheric pressure in the gas phase (SI: Pa)
    /// (Required for fixed pressure outflow boundary condition)
    number atmospheric_pressure{};
  };
} // namespace MeltPoolDG::Simulation::CompressibleMultiphase
