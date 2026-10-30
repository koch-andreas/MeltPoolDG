#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/vector.h>

#include <meltpooldg/post_processing/generic_data_out.hpp>

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "../../narrow_band_case.hpp"
#include "../../narrow_band_postprocessing.hpp"

namespace MeltPoolDG::Simulation::RotatingCircle
{
  /**
   * @brief Rigid-body rotation about the axis through @p rotation_center in the direction of the
   * z-axis with the angular velocity @p angular_velocity.
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
   * @brief Signed distance function of a hypersphere, which rotates with a rigid-body rotation
   * about the axis through @p rotation_center in the direction of the z-axis. The level set is
   * positive inside the hypersphere.
   */
  template <int dim>
  class RotatingSphere : public dealii::Function<dim>
  {
  public:
    RotatingSphere(const dealii::Point<dim> &initial_center,
                   const double              radius,
                   const dealii::Point<dim> &rotation_center,
                   const double              angular_velocity)
      : dealii::Function<dim>(1)
      , initial_center(initial_center)
      , radius(radius)
      , rotation_center(rotation_center)
      , angular_velocity(angular_velocity)
    {}

    /**
     * @brief Center of the hypersphere at the current time.
     */
    dealii::Point<dim>
    get_center() const
    {
      const double                 angle = angular_velocity * this->get_time();
      const dealii::Tensor<1, dim> x0    = initial_center - rotation_center;

      dealii::Point<dim> center = initial_center;
      center[0] = rotation_center[0] + std::cos(angle) * x0[0] - std::sin(angle) * x0[1];
      center[1] = rotation_center[1] + std::sin(angle) * x0[0] + std::cos(angle) * x0[1];
      return center;
    }

    double
    value(const dealii::Point<dim> &p, const unsigned int /*component*/) const override
    {
      return radius - p.distance(get_center());
    }

  private:
    const dealii::Point<dim> initial_center;
    const double             radius;
    const dealii::Point<dim> rotation_center;
    const double             angular_velocity;
  };

  /**
   * @brief Rigid-body rotation of a circle (sphere in 3D) on the domain [-1,1]^dim.
   *
   * The initial level-set field is the signed distance function of the circle (positive inside).
   * Since the rigid-body rotation preserves distances, the exact solution is the signed distance
   * function of the rotated circle at all times. If user-defined postprocessing is enabled, the
   * errors of the level-set field on the cut cells with respect to the exact solution are
   * printed.
   */
  template <int dim, typename number>
  class SimulationRotatingCircle : public LevelSet::NarrowBandCase<dim, number>
  {
  public:
    SimulationRotatingCircle(std::string parameter_file, const MPI_Comm mpi_communicator)
      : LevelSet::NarrowBandCase<dim, number>(parameter_file, mpi_communicator)
    {}

    void
    create_spatial_discretization() override
    {
      AssertThrow(dim >= 2, dealii::ExcMessage("The case rotating_circle requires dim >= 2."));

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
      this->attach_initial_condition(
        std::make_shared<RotatingSphere<dim>>(
          to_point(center_point), radius, to_point(rotation_center), angular_velocity),
        "level_set");
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
                        "Initial center point of the circle (sphere). Specify comma-separated "
                        "values for the x-, y- (and z-) coordinate.");
      prm.add_parameter("radius",
                        radius,
                        "Radius of the circle (sphere).",
                        dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
      prm.add_parameter("rotation center",
                        rotation_center,
                        "Point on the rotation axis. The rotation axis is parallel to the z-axis. "
                        "Specify comma-separated values for the x-, y- (and z-) coordinate.");
      prm.add_parameter("angular velocity",
                        angular_velocity,
                        "Angular velocity of the rigid-body rotation.");

      return this->parameters.base.do_print_parameters;
    }

    /**
     * @brief Print the size of the narrow band and the errors of the level-set field on the cut
     * cells with respect to the exact signed distance function.
     */
    void
    do_postprocessing(const GenericDataOut<dim, number> &generic_data_out) const final
    {
      RotatingSphere<dim> exact_solution(to_point(center_point),
                                         radius,
                                         to_point(rotation_center),
                                         angular_velocity);
      exact_solution.set_time(generic_data_out.get_time());

      const number exact_interface_measure =
        dim == 2 ? 2. * dealii::numbers::PI * radius :
                   (dim == 3 ? 4. * dealii::numbers::PI * radius * radius : 0.);

      LevelSet::print_narrow_band_statistics<dim, number>(generic_data_out,
                                                          "level_set_band",
                                                          this->mpi_communicator,
                                                          "rotating_circle");
      LevelSet::print_narrow_band_errors<dim, number>(generic_data_out,
                                                      "level_set",
                                                      exact_solution,
                                                      exact_interface_measure,
                                                      this->mpi_communicator,
                                                      "rotating_circle");
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

    /// Domain [left_domain, right_domain]^dim
    const number left_domain  = -1.0;
    const number right_domain = 1.0;

    /// Initial center point of the circle (sphere)
    std::vector<number> center_point{0.0, 0.5, 0.0};

    /// Radius of the circle (sphere)
    number radius = 0.25;

    /// Point on the rotation axis
    std::vector<number> rotation_center{0.0, 0.0, 0.0};

    /// Angular velocity of the rigid-body rotation
    number angular_velocity = 2. * dealii::numbers::PI;
  };
} // namespace MeltPoolDG::Simulation::RotatingCircle
