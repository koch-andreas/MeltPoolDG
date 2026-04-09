#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/manifold_lib.h>

#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/level_set/level_set_type.hpp>
#include <meltpooldg/utilities/characteristic_functions.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../../melt_pool_case.hpp"

namespace MeltPoolDG::Simulation::LaserMeltingSimonds
{
  using namespace dealii;

  inline static double width = 600e-6;

  // Note: For 1d we consider the coordinates along the y-axis.
  inline static double height_substrate = 430e-6; // m
  inline static double height_gas       = 170e-6;

  inline static double delta_h = 20e-6; // TODO: add to ParameterHandler

  // initial temperature values
  inline static double T_initial_top    = 298; // K
  inline static double T_initial_bottom = T_initial_top;

  // boundary conditions
  inline static double inflow_velocity    = 0.1;
  inline static double outlet_pressure    = 0.0;
  inline static double inflow_temperature = T_initial_top;

  // for heat application only
  inline static double eps_prefactor = 2.0;

  template <int dim>
  class InflowVelocity : public Function<dim>
  {
  public:
    InflowVelocity();

    double
    value(const Point<dim> &p, const unsigned int component) const override;
  };

  template <int dim>
  class InitialLevelSet : public dealii::Function<dim>
  {
  public:
    InitialLevelSet(const double                 eps,
                    const LevelSet::LevelSetType level_set_type = LevelSet::LevelSetType::tanh);

    double
    value(const dealii::Point<dim> &p, const unsigned int component) const override;

  private:
    const dealii::Functions::SignedDistance::Plane<dim> distance_plane;
    const double                                        eps;
    const LevelSet::LevelSetType                        level_set_type;
  };

  template <int dim>
  class InitialConditionTemperature : public Function<dim>
  {
  public:
    InitialConditionTemperature(const double T_initial_bottom,
                                const double T_initial_top,
                                const double y_min,
                                const double y_max);

    double
    value(const Point<dim> &p, const unsigned int component) const override;

    const double T_initial_bottom;
    const double T_initial_top;
    const double y_min;
    const double grad_T;
  };

  template <int dim, typename Number, typename CaseClass>
  class SimulationLaserMeltingSimonds : public CaseClass
  {
  private:
    std::vector<unsigned int> cell_repetitions;

    unsigned int n_local_refinement = 0;

    Point<dim> local_refinement_1_bottom_left;
    Point<dim> local_refinement_1_top_right;
    Point<dim> local_refinement_2_bottom_left;
    Point<dim> local_refinement_2_top_right;
    Point<dim> bottom_left;
    Point<dim> top_right;

    // Postprocessor
    mutable std::ofstream file_conservation_variables;
    mutable int           n_time_step = 0;

    mutable TableHandler output_table;

  public:
    SimulationLaserMeltingSimonds(std::string parameter_file, const MPI_Comm mpi_communicator);

    bool
    add_simulation_specific_parameters(dealii::ParameterHandler &prm) override;

    void
    create_spatial_discretization() override;

    void
    set_boundary_conditions() override;

    void
    set_field_conditions() override;

    void
    do_postprocessing(
      [[maybe_unused]] const GenericDataOut<dim, Number> &generic_data_out) const final;
  };

} // namespace MeltPoolDG::Simulation::LaserMeltingSimonds
