#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <meltpooldg/post_processing/generic_data_out.hpp>

#include <limits>
#include <memory>
#include <vector>

#include "../../narrow_band_case.hpp"
#include "../../narrow_band_postprocessing.hpp"

namespace MeltPoolDG::Simulation::StaticCircle
{
  /**
   * @brief Signed distance function of a hypersphere with the center @p center and the radius
   * @p radius (positive inside).
   */
  template <int dim>
  class SphereSignedDistance : public dealii::Function<dim>
  {
  public:
    SphereSignedDistance(const dealii::Point<dim> &center, const double radius)
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
   * @brief Scaled signed distance function of a hypersphere,
   * $\phi_0(x) = (|x - x_f|^2 + b) (R - |x - c|)$, with the distortion point $x_f$ and the
   * distortion offset $b > 0$. Its zero level set is the hypersphere, while its gradient
   * $|\nabla \phi_0| = |x - x_f|^2 + b$ on the interface varies along the interface.
   */
  template <int dim>
  class ScaledSphereSignedDistance : public dealii::Function<dim>
  {
  public:
    ScaledSphereSignedDistance(const dealii::Point<dim> &center,
                               const double              radius,
                               const dealii::Point<dim> &distortion_point,
                               const double              distortion_offset)
      : dealii::Function<dim>(1)
      , signed_distance(center, radius)
      , distortion_point(distortion_point)
      , distortion_offset(distortion_offset)
    {}

    double
    value(const dealii::Point<dim> &p, const unsigned int /*component*/) const override
    {
      return (p.distance_square(distortion_point) + distortion_offset) *
             signed_distance.value(p, 0);
    }

  private:
    const SphereSignedDistance<dim> signed_distance;
    const dealii::Point<dim>        distortion_point;
    const double                    distortion_offset;
  };

  /**
   * @brief Static circle (sphere in 3D) on the domain [-1,1]^dim for the convergence analysis of
   * the reinitialization.
   *
   * The initial level set is the scaled signed distance function
   * $\phi_0(x) = (|x - x_f|^2 + b) (R - |x - c|)$ (positive inside), i.e., its zero level set is
   * exactly the circle, but $|\nabla \phi_0| \neq 1$ varies along the interface (between
   * $(|c - x_f| - R)^2 + b$ and $(|c - x_f| + R)^2 + b$). By default, the initial level set is
   * reinitialized when it is set (parameter 'reinitialize initial condition'), such that the
   * output at the start time shows the errors of one reinitialization of the distorted field. The
   * velocity vanishes; with time steps, the semi-Lagrangian advection reproduces the level set,
   * and the level set is reinitialized according to the reinitialization frequency, i.e., the
   * time loop tests the drift of repeated reinitializations.
   *
   * If user-defined postprocessing is enabled, the size of the narrow band and the errors of the
   * level set of the cut cells with respect to the signed distance function of the circle are
   * printed at every output time (relative L2 error of the signed distance and L2 error of
   * |grad phi| - 1 on the cut cells, L2 and maximum distance of the zero level set to the circle,
   * relative error of its length/area). If the continuous level set of the narrow band is
   * enabled, the errors of the continuous level set and, with the curvature, of its curvature
   * (dim - 1) / R and of its normal vector are printed as well.
   */
  template <int dim, typename number>
  class SimulationStaticCircle : public LevelSet::NarrowBandCase<dim, number>
  {
  public:
    SimulationStaticCircle(std::string parameter_file, const MPI_Comm mpi_communicator)
      : LevelSet::NarrowBandCase<dim, number>(parameter_file, mpi_communicator)
    {}

    void
    create_spatial_discretization() override
    {
      AssertThrow(dim >= 2, dealii::ExcMessage("The case static_circle requires dim >= 2."));

      this->triangulation =
        std::make_shared<dealii::parallel::distributed::Triangulation<dim>>(this->mpi_communicator);

      dealii::GridGenerator::hyper_cube(*this->triangulation, left_domain, right_domain);
      this->triangulation->refine_global(this->parameters.base.global_refinements);
    }

    void
    set_boundary_conditions() override
    {
      // nothing to do: the interface is at rest
    }

    void
    set_field_conditions() override
    {
      AssertThrow(distortion_offset > 0.,
                  dealii::ExcMessage("The distortion offset must be positive."));

      this->attach_initial_condition(
        std::make_shared<ScaledSphereSignedDistance<dim>>(
          to_point(center_point), radius, to_point(distortion_point), distortion_offset),
        "level_set");
      this->attach_field_function(std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim),
                                  "prescribed_velocity",
                                  "level_set");
    }

    bool
    add_case_specific_parameters(dealii::ParameterHandler &prm) override
    {
      prm.add_parameter("center point",
                        center_point,
                        "Center point c of the circle (sphere). Specify comma-separated values "
                        "for the x-, y- (and z-) coordinate.");
      prm.add_parameter("radius",
                        radius,
                        "Radius R of the circle (sphere).",
                        dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
      prm.add_parameter("distortion point",
                        distortion_point,
                        "Distortion point x_f of the initial level set "
                        "phi_0(x) = (|x - x_f|^2 + b) (R - |x - c|). Specify comma-separated "
                        "values for the x-, y- (and z-) coordinate.");
      prm.add_parameter("distortion offset",
                        distortion_offset,
                        "Distortion offset b > 0 of the initial level set "
                        "phi_0(x) = (|x - x_f|^2 + b) (R - |x - c|).",
                        dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
      prm.add_parameter("reinitialize initial condition",
                        reinitialize_initial_condition_flag,
                        "Reinitialize the initial level set when it is set, i.e., the output at "
                        "the start time shows the errors after one reinitialization of the "
                        "distorted level set (otherwise, the errors of the interpolated distorted "
                        "level set).");

      return this->parameters.base.do_print_parameters;
    }

    bool
    reinitialize_initial_condition() const override
    {
      return reinitialize_initial_condition_flag;
    }

    /**
     * @brief Print the size of the narrow band and the errors of the level set with respect to
     * the signed distance function of the circle.
     */
    void
    do_postprocessing(const GenericDataOut<dim, number> &generic_data_out) const final
    {
      const SphereSignedDistance<dim> exact_solution(to_point(center_point), radius);

      const number exact_interface_measure =
        dim == 2 ? 2. * dealii::numbers::PI * radius :
                   (dim == 3 ? 4. * dealii::numbers::PI * radius * radius : 0.);

      LevelSet::print_narrow_band_statistics<dim, number>(generic_data_out,
                                                          "level_set_band",
                                                          this->mpi_communicator,
                                                          "static_circle");
      LevelSet::print_narrow_band_errors<dim, number>(generic_data_out,
                                                      "level_set",
                                                      exact_solution,
                                                      exact_interface_measure,
                                                      this->mpi_communicator,
                                                      "static_circle");

      // continuous level set: interface errors, curvature (dim - 1) / R (positive inside) and
      // inner normal vector of the circle
      if (this->parameters.narrow_band.continuous_level_set.enable)
        {
          LevelSet::print_narrow_band_errors<dim, number>(generic_data_out,
                                                          "level_set_cg",
                                                          exact_solution,
                                                          exact_interface_measure,
                                                          this->mpi_communicator,
                                                          "level_set_cg");

          if (this->parameters.curvature.enable)
            {
              const dealii::Point<dim> center = to_point(center_point);
              LevelSet::print_interface_geometry_errors<dim, number>(
                generic_data_out,
                (dim - 1.) / radius,
                [center](const dealii::Point<dim> &p) {
                  const dealii::Tensor<1, dim, number> r = center - p;
                  return r / r.norm();
                },
                this->mpi_communicator,
                "level_set_cg");
            }
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

    /// Domain [left_domain, right_domain]^dim
    const number left_domain  = -1.0;
    const number right_domain = 1.0;

    /// Center point of the circle (sphere): slightly off the mesh lines, so that the interface
    /// is not tangent to faces at its extreme points on all meshes
    std::vector<number> center_point{0.031, -0.027, 0.019};

    /// Radius of the circle (sphere)
    number radius = 0.5;

    /// Distortion point and offset of the initial level set
    std::vector<number> distortion_point{0.7, 0.7, 0.7};
    number              distortion_offset = 0.1;

    /// Reinitialize the initial level set when it is set
    bool reinitialize_initial_condition_flag = true;
  };
} // namespace MeltPoolDG::Simulation::StaticCircle
