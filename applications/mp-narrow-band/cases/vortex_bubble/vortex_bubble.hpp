#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <meltpooldg/post_processing/generic_data_out.hpp>

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "../../narrow_band_case.hpp"
#include "../../narrow_band_postprocessing.hpp"

namespace MeltPoolDG::Simulation::NarrowBandVortexBubble
{
  /**
   * @brief Time-reversed single-vortex velocity field on the unit square (LeVeque 1996; Bell,
   * Colella and Glaz 1989):
   *
   * $\boldsymbol{u}(\boldsymbol{x}, t) = \cos(\pi t / T) \left(\sin(2\pi y) \sin^2(\pi x),
   * -\sin(2\pi x) \sin^2(\pi y)\right)$.
   *
   * The flow stretches a circle into a spiral up to t = T/2 and reverses afterwards such that the
   * circle returns to its initial shape at t = T.
   */
  template <int dim>
  class SingleVortex : public dealii::Function<dim>
  {
  public:
    explicit SingleVortex(const double period)
      : dealii::Function<dim>(dim)
      , period(period)
    {}

    double
    value(const dealii::Point<dim> &p, const unsigned int component) const override
    {
      constexpr double pi         = dealii::numbers::PI;
      const double     reversion  = std::cos(pi * this->get_time() / period);
      const double     sin_pi_x   = std::sin(pi * p[0]);
      const double     sin_pi_y   = std::sin(pi * p[1]);
      const double     sin_2_pi_x = std::sin(2. * pi * p[0]);
      const double     sin_2_pi_y = std::sin(2. * pi * p[1]);

      if (component == 0)
        return reversion * sin_2_pi_y * sin_pi_x * sin_pi_x;
      if (component == 1)
        return -reversion * sin_2_pi_x * sin_pi_y * sin_pi_y;
      return 0.;
    }

  private:
    const double period;
  };

  /**
   * @brief Signed distance function of a circle, positive inside.
   */
  template <int dim>
  class Circle : public dealii::Function<dim>
  {
  public:
    Circle(const dealii::Point<dim> &center, const double radius)
      : dealii::Function<dim>(1)
      , center(center)
      , radius(radius)
    {}

    double
    value(const dealii::Point<dim> &p, const unsigned int /*component*/) const override
    {
      return radius - p.distance(center);
    }

  private:
    const dealii::Point<dim> center;
    const double             radius;
  };

  /**
   * @brief Vortex bubble: a circle in the time-reversed single-vortex flow on the unit square.
   *
   * The initial level-set field is the signed distance function of the circle (positive inside).
   * If user-defined postprocessing is enabled, the size of the narrow band and the geometric
   * measures of the interface (enclosed area, interface length and circularity) are printed at
   * every output time. At the start time and at multiples of the period T, where the exact
   * solution is the initial signed distance function, the errors of the level-set field on the
   * cut cells are printed additionally.
   */
  template <int dim, typename number>
  class SimulationVortexBubble : public LevelSet::NarrowBandCase<dim, number>
  {
  public:
    SimulationVortexBubble(std::string parameter_file, const MPI_Comm mpi_communicator)
      : LevelSet::NarrowBandCase<dim, number>(parameter_file, mpi_communicator)
    {}

    void
    create_spatial_discretization() override
    {
      AssertThrow(dim == 2, dealii::ExcMessage("The case vortex_bubble requires dim = 2."));

      this->triangulation =
        std::make_shared<dealii::parallel::distributed::Triangulation<dim>>(this->mpi_communicator);

      dealii::GridGenerator::hyper_cube(*this->triangulation, 0.0, 1.0);
      this->triangulation->refine_global(this->parameters.base.global_refinements);
    }

    void
    set_boundary_conditions() override
    {
      // nothing to do: the velocity vanishes on the domain boundary
    }

    void
    set_field_conditions() override
    {
      AssertThrow(period > 0., dealii::ExcMessage("The period T must be positive."));

      this->attach_initial_condition(std::make_shared<Circle<dim>>(to_point(center_point), radius),
                                     "level_set");
      this->attach_field_function(std::make_shared<SingleVortex<dim>>(period),
                                  "prescribed_velocity",
                                  "level_set");
    }

    bool
    add_case_specific_parameters(dealii::ParameterHandler &prm) override
    {
      prm.add_parameter("center point",
                        center_point,
                        "Initial center point of the circle. Specify comma-separated values for "
                        "the x- and y-coordinate.");
      prm.add_parameter("radius",
                        radius,
                        "Radius of the circle.",
                        dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
      prm.add_parameter("T",
                        period,
                        "Period of the vortex flow. The flow reverses at T/2 and the circle "
                        "returns to its initial shape at T.");

      return this->parameters.base.do_print_parameters;
    }

    /**
     * @brief Print the size of the narrow band and the measures of the interface. At the start
     * time and at multiples of the period, print the errors with respect to the initial signed
     * distance function.
     */
    void
    do_postprocessing(const GenericDataOut<dim, number> &generic_data_out) const final
    {
      LevelSet::print_narrow_band_statistics<dim, number>(generic_data_out,
                                                          "level_set_band",
                                                          this->mpi_communicator,
                                                          "vortex_bubble");
      LevelSet::print_interface_measures<dim, number>(generic_data_out,
                                                      "level_set",
                                                      this->mpi_communicator,
                                                      "vortex_bubble");

      // the exact solution is known at multiples of the period
      const number n_periods =
        (generic_data_out.get_time() - this->parameters.time_stepping.start_time) / period;
      if (std::abs(n_periods - std::round(n_periods)) < 1e-10)
        {
          const Circle<dim> exact_solution(to_point(center_point), radius);
          LevelSet::print_narrow_band_errors<dim, number>(generic_data_out,
                                                          "level_set",
                                                          exact_solution,
                                                          2. * dealii::numbers::PI * radius,
                                                          this->mpi_communicator,
                                                          "vortex_bubble");
        }
    }

  private:
    dealii::Point<dim>
    to_point(const std::vector<number> &coordinates) const
    {
      AssertThrow(coordinates.size() >= dim,
                  dealii::ExcMessage("Specify at least dim coordinates for a point."));

      dealii::Point<dim> point;
      for (unsigned int d = 0; d < dim; ++d)
        point[d] = coordinates[d];
      return point;
    }

    /// Initial center point of the circle
    std::vector<number> center_point{0.5, 0.75};

    /// Radius of the circle
    number radius = 0.15;

    /// Period of the vortex flow
    number period = 2.0;
  };
} // namespace MeltPoolDG::Simulation::NarrowBandVortexBubble
