#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <meltpooldg/post_processing/generic_data_out.hpp>

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "../../narrow_band_case.hpp"
#include "../../narrow_band_postprocessing.hpp"

namespace MeltPoolDG::Simulation::NarrowBandZalesakDisk
{
  /**
   * @brief Rigid-body rotation about the point @p rotation_center with the angular velocity
   * @p angular_velocity (counterclockwise for positive values).
   */
  template <int dim>
  class RigidBodyRotation : public dealii::Function<dim>
  {
  public:
    RigidBodyRotation(const dealii::Point<dim> &rotation_center, const double angular_velocity)
      : dealii::Function<dim>(dim)
      , rotation_center(rotation_center)
      , angular_velocity(angular_velocity)
    {}

    double
    value(const dealii::Point<dim> &p, const unsigned int component) const override
    {
      const dealii::Tensor<1, dim> x = p - rotation_center;
      if (component == 0)
        return -angular_velocity * x[1];
      if (component == 1)
        return angular_velocity * x[0];
      return 0.;
    }

  private:
    const dealii::Point<dim> rotation_center;
    const double             angular_velocity;
  };

  /**
   * @brief Level set of Zalesak's slotted disk (dealii::Functions::SignedDistance::ZalesakDisk,
   * slot at the bottom of the disk), which rotates with a rigid-body rotation about
   * @p rotation_center. The level set is positive inside the disk.
   */
  template <int dim>
  class RotatingZalesakDisk : public dealii::Function<dim>
  {
  public:
    RotatingZalesakDisk(const dealii::Point<dim> &center,
                        const double              radius,
                        const double              notch_width,
                        const double              notch_height,
                        const dealii::Point<dim> &rotation_center,
                        const double              angular_velocity)
      : dealii::Function<dim>(1)
      , disk(center, radius, notch_width, notch_height)
      , rotation_center(rotation_center)
      , angular_velocity(angular_velocity)
    {}

    double
    value(const dealii::Point<dim> &p, const unsigned int /*component*/) const override
    {
      // rotate the point back to the initial configuration
      const double                 angle = -angular_velocity * this->get_time();
      const dealii::Tensor<1, dim> x     = p - rotation_center;
      dealii::Point<dim>           p0    = p;
      p0[0] = rotation_center[0] + std::cos(angle) * x[0] - std::sin(angle) * x[1];
      p0[1] = rotation_center[1] + std::sin(angle) * x[0] + std::cos(angle) * x[1];
      return -disk.value(p0);
    }

  private:
    const dealii::Functions::SignedDistance::ZalesakDisk<dim> disk;
    const dealii::Point<dim>                                  rotation_center;
    const double                                              angular_velocity;
  };

  /**
   * @brief Rigid-body rotation of Zalesak's slotted disk on the domain [-1,1]^2.
   *
   * Default setup (as the case zalesak_disk of mp-level-set): disk of radius 0.3 with the center
   * (0, 0.5), slot of width 0.1 and height 0.5 at the bottom of the disk, rotation about the origin
   * with the velocity u = (4y, -4x), i.e., one revolution in the time pi/2. The initial level set
   * is the signed distance function of the disk (positive inside, exact except near the corners of
   * the slot). If user-defined postprocessing is enabled, the size of the narrow band, the area and
   * the length of the interface and, after full revolutions, the errors of the level set with
   * respect to the initial disk are printed.
   */
  template <int dim, typename number>
  class SimulationZalesakDisk : public LevelSet::NarrowBandCase<dim, number>
  {
  public:
    SimulationZalesakDisk(std::string parameter_file, const MPI_Comm mpi_communicator)
      : LevelSet::NarrowBandCase<dim, number>(parameter_file, mpi_communicator)
    {}

    void
    create_spatial_discretization() override
    {
      AssertThrow(dim == 2, dealii::ExcMessage("The case zalesak_disk requires dim = 2."));

      this->triangulation =
        std::make_shared<dealii::parallel::distributed::Triangulation<dim>>(this->mpi_communicator);

      dealii::GridGenerator::hyper_cube(*this->triangulation, left_domain, right_domain);
      this->triangulation->refine_global(this->parameters.base.global_refinements);
    }

    void
    set_boundary_conditions() override
    {
      // nothing to do: phi_ext = phi_int on the domain boundary
    }

    void
    set_field_conditions() override
    {
      this->attach_initial_condition(create_exact_solution(), "level_set");
      this->attach_field_function(
        std::make_shared<RigidBodyRotation<dim>>(to_point(rotation_center), angular_velocity),
        "prescribed_velocity",
        "level_set");
    }

    bool
    add_case_specific_parameters(dealii::ParameterHandler &prm) override
    {
      prm.add_parameter("center point",
                        center_point,
                        "Initial center point of the disk. Specify comma-separated values for the "
                        "x- and y-coordinate.");
      prm.add_parameter("radius",
                        radius,
                        "Radius of the disk.",
                        dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
      prm.add_parameter("notch width",
                        notch_width,
                        "Width of the slot.",
                        dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
      prm.add_parameter("notch height",
                        notch_height,
                        "Height of the slot, measured from the bottom of the disk.",
                        dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
      prm.add_parameter("rotation center",
                        rotation_center,
                        "Center of the rigid-body rotation. Specify comma-separated values for the "
                        "x- and y-coordinate.");
      prm.add_parameter(
        "angular velocity",
        angular_velocity,
        "Angular velocity of the rigid-body rotation (counterclockwise for positive "
        "values).");

      return this->parameters.base.do_print_parameters;
    }

    void
    do_postprocessing(const GenericDataOut<dim, number> &generic_data_out) const final
    {
      LevelSet::print_narrow_band_statistics<dim, number>(generic_data_out,
                                                          "level_set_band",
                                                          this->mpi_communicator,
                                                          "zalesak_disk");
      LevelSet::print_interface_measures<dim, number>(generic_data_out,
                                                      "level_set",
                                                      this->mpi_communicator,
                                                      "zalesak_disk");
      if (this->parameters.narrow_band.continuous_level_set.enable)
        LevelSet::print_interface_measures<dim, number>(generic_data_out,
                                                        "level_set_cg",
                                                        this->mpi_communicator,
                                                        "level_set_cg");

      // the exact solution is the initial disk after full revolutions
      const number period = 2. * dealii::numbers::PI / std::abs(angular_velocity);
      const number n_periods =
        (generic_data_out.get_time() - this->parameters.time_stepping.start_time) / period;
      if (std::abs(n_periods - std::round(n_periods)) < 1e-8)
        {
          const auto exact_solution = create_exact_solution();
          exact_solution->set_time(0.);
          LevelSet::print_narrow_band_errors<dim, number>(generic_data_out,
                                                          "level_set",
                                                          *exact_solution,
                                                          0.,
                                                          this->mpi_communicator,
                                                          "zalesak_disk");
        }
    }

  private:
    std::shared_ptr<RotatingZalesakDisk<dim>>
    create_exact_solution() const
    {
      return std::make_shared<RotatingZalesakDisk<dim>>(to_point(center_point),
                                                        radius,
                                                        notch_width,
                                                        notch_height,
                                                        to_point(rotation_center),
                                                        angular_velocity);
    }

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

    /// Domain [left_domain, right_domain]^dim
    const number left_domain  = -1.0;
    const number right_domain = 1.0;

    /// Initial center point, radius and slot of the disk
    std::vector<number> center_point{0.0, 0.5};
    number              radius       = 0.3;
    number              notch_width  = 0.1;
    number              notch_height = 0.5;

    /// Rigid-body rotation (default: u = (4y, -4x))
    std::vector<number> rotation_center{0.0, 0.0};
    number              angular_velocity = -4.0;
  };
} // namespace MeltPoolDG::Simulation::NarrowBandZalesakDisk
