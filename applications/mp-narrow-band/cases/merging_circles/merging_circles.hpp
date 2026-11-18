#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/vector.h>

#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/utilities/journal.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>

#include "../../narrow_band_case.hpp"
#include "../../narrow_band_postprocessing.hpp"

namespace MeltPoolDG::Simulation::MergingCircles
{
  /**
   * @brief Exact geometry of two circles (spheres in 3D) of the radius $r$, which move towards each
   * other along the x-axis with the velocity $U$ (see SimulationMergingCircles): the centers are
   * $\boldsymbol{c}_\pm(t) = \boldsymbol{m} \pm \frac{D(t)}{2} \boldsymbol{e}_x$ with the distance
   * $D(t) = D_0 - 2 U t \geq 0$, and the phase $\Omega^+$ is the union of the two disks (balls).
   *
   * value() is the signed distance to the boundary of the union (positive inside): outside the
   * union, $r - \min_\pm |\boldsymbol{x} - \boldsymbol{c}_\pm|$; inside, the distance to the
   * exposed arcs (spherical caps) or, after the contact ($D < 2r$), to the rim of the neck, i.e.,
   * the circle of the radius $w = \sqrt{r^2 - D^2/4}$ in the plane $x = m_x$ (two points in 2D).
   */
  template <int dim>
  class MergingSpheresSignedDistance : public dealii::Function<dim>
  {
  public:
    MergingSpheresSignedDistance(const dealii::Point<dim> &midpoint,
                                 const double              radius,
                                 const double              initial_distance,
                                 const double              velocity)
      : dealii::Function<dim>(1)
      , midpoint(midpoint)
      , radius(radius)
      , initial_distance(initial_distance)
      , velocity(velocity)
    {}

    /// Distance of the centers at the current time.
    double
    get_distance() const
    {
      return std::max(initial_distance - 2. * velocity * this->get_time(), 0.);
    }

    /// Half-width of the neck in the plane x = m_x (zero before the contact).
    double
    get_neck_radius() const
    {
      const double D = get_distance();
      return D < 2. * radius ? std::sqrt(radius * radius - 0.25 * D * D) : 0.;
    }

    /// Gap between the circles on the x-axis (zero after the contact).
    double
    get_gap() const
    {
      return std::max(get_distance() - 2. * radius, 0.);
    }

    /// Area (volume in 3D) of the union.
    double
    get_volume() const
    {
      const double D = get_distance();
      const double r = radius;
      const double h = D < 2. * r ? r - 0.5 * D : 0.; // height of the overlapping caps
      if constexpr (dim == 2)
        {
          // area of the lens of the two disks
          const double lens =
            h > 0. ? 2. * r * r * std::acos(0.5 * D / r) - 0.5 * D * std::sqrt(4. * r * r - D * D) :
                     0.;
          return 2. * dealii::numbers::PI * r * r - lens;
        }
      else
        return 8. / 3. * dealii::numbers::PI * r * r * r -
               2. * dealii::numbers::PI * h * h * (3. * r - h) / 3.;
    }

    /// Length (area in 3D) of the boundary of the union.
    double
    get_interface_measure() const
    {
      const double D = get_distance();
      const double r = radius;
      if constexpr (dim == 2)
        return 4. * dealii::numbers::PI * r - (D < 2. * r ? 4. * r * std::acos(0.5 * D / r) : 0.);
      else
        return 8. * dealii::numbers::PI * r * r -
               (D < 2. * r ? 4. * dealii::numbers::PI * r * (r - 0.5 * D) : 0.);
    }

    double
    value(const dealii::Point<dim> &p, const unsigned int /*component*/) const override
    {
      const double D = get_distance();
      const double r = radius;

      dealii::Point<dim> c_minus = midpoint, c_plus = midpoint;
      c_minus[0] -= 0.5 * D;
      c_plus[0] += 0.5 * D;

      const double d_minus = p.distance(c_minus);
      const double d_plus  = p.distance(c_plus);

      // outside of the union, or inside of one of two disjoint disks
      if (std::min(d_minus, d_plus) >= r or D >= 2. * r)
        return r - std::min(d_minus, d_plus);

      // inside of the union of overlapping disks: distance to the exposed arcs (the radial
      // projection onto circle -/+ lies on its exposed arc, if it is on the side x <= m_x /
      // x >= m_x) and to the rim of the neck
      double distance = std::numeric_limits<double>::max();
      if (d_minus > 0. and c_minus[0] + r * (p[0] - c_minus[0]) / d_minus <= midpoint[0])
        distance = std::min(distance, std::abs(r - d_minus));
      if (d_plus > 0. and c_plus[0] + r * (p[0] - c_plus[0]) / d_plus >= midpoint[0])
        distance = std::min(distance, std::abs(r - d_plus));

      double rho_square = 0.; // squared distance to the axis
      for (unsigned int d = 1; d < dim; ++d)
        rho_square += dealii::Utilities::fixed_power<2>(p[d] - midpoint[d]);
      const double w = std::sqrt(r * r - 0.25 * D * D);
      distance       = std::min(distance,
                          std::sqrt(dealii::Utilities::fixed_power<2>(p[0] - midpoint[0]) +
                                    dealii::Utilities::fixed_power<2>(std::sqrt(rho_square) - w)));
      return distance;
    }

  private:
    const dealii::Point<dim> midpoint;
    const double             radius;
    const double             initial_distance;
    const double             velocity;
  };

  /**
   * @brief Velocity of the two halves of the domain towards the plane $x = m_x$:
   * $\boldsymbol{u} = -U \operatorname{sign}(x - m_x) \boldsymbol{e}_x$ with
   * $\operatorname{sign}(0) = 1$, i.e., each half moves rigidly.
   */
  template <int dim>
  class ConvergingVelocity : public dealii::Function<dim>
  {
  public:
    ConvergingVelocity(const double plane, const double velocity)
      : dealii::Function<dim>(dim)
      , plane(plane)
      , velocity(velocity)
    {}

    double
    value(const dealii::Point<dim> &p, const unsigned int component) const override
    {
      if (component != 0)
        return 0.;
      return p[0] >= plane ? -velocity : velocity;
    }

  private:
    const double plane;
    const double velocity;
  };

  /**
   * @brief Two circles (spheres in 3D) on the domain [-1,1]^dim, which move towards each other and
   * merge (topology change).
   *
   * The circles of the radius $r$ have the initial distance $D_0 > 2r$ of their centers
   * $\boldsymbol{m} \pm \frac{D_0}{2} \boldsymbol{e}_x$. The velocity
   * $\boldsymbol{u} = -U \operatorname{sign}(x - m_x) \boldsymbol{e}_x$ moves both halves of the
   * domain rigidly towards the plane $x = m_x$. Since the characteristics converge at the plane,
   * the backward characteristics never cross it, and the exact solution is the union of the two
   * translated disks (balls) with the distance $D(t) = D_0 - 2Ut$ of the centers: the circles
   * touch at $t^* = (D_0 - 2r) / (2U)$ and then merge with a neck of the half-width
   * $w = \sqrt{r^2 - D^2/4}$. The overlapping parts of the disks vanish at the plane, i.e., the
   * area is not conserved, but known exactly. A smooth velocity field cannot merge the circles in
   * finite time (its flow map is a diffeomorphism); with this velocity, the merging is a property
   * of the exact solution and not an effect of the resolution.
   *
   * If user-defined postprocessing is enabled, the size of the narrow band, the area and the length
   * of the interface and their relative errors, the gap between the circles on the axis and the
   * half-width of the neck in the plane (zeros of the level set on the corresponding segments)
   * with their exact values, and the errors of the level set with respect to the signed distance
   * function of the union are printed at every output time (for the continuous level set of the
   * narrow band as well, if it is enabled). The end time must satisfy $D(t) \geq 0$.
   */
  template <int dim, typename number>
  class SimulationMergingCircles : public LevelSet::NarrowBandCase<dim, number>
  {
  public:
    SimulationMergingCircles(std::string parameter_file, const MPI_Comm mpi_communicator)
      : LevelSet::NarrowBandCase<dim, number>(parameter_file, mpi_communicator)
    {}

    void
    create_spatial_discretization() override
    {
      AssertThrow(dim >= 2, dealii::ExcMessage("The case merging_circles requires dim >= 2."));

      this->triangulation =
        std::make_shared<dealii::parallel::distributed::Triangulation<dim>>(this->mpi_communicator);

      dealii::GridGenerator::hyper_cube(*this->triangulation, left_domain, right_domain);
      this->triangulation->refine_global(this->parameters.base.global_refinements);
    }

    void
    set_boundary_conditions() override
    {
      // nothing to do: the interface does not reach the boundary of the domain
    }

    void
    set_field_conditions() override
    {
      AssertThrow(initial_distance > 2. * radius,
                  dealii::ExcMessage("The initial distance of the centers must exceed the "
                                     "diameter of the circles."));
      AssertThrow(initial_distance - 2. * velocity * this->parameters.time_stepping.end_time >=
                    -1e-12,
                  dealii::ExcMessage("The centers of the circles must not pass each other before "
                                     "the end time: D_0 - 2 U t_end >= 0."));

      this->attach_initial_condition(create_exact_solution(), "level_set");
      this->attach_field_function(std::make_shared<ConvergingVelocity<dim>>(
                                    to_point(center_point)[0], velocity),
                                  "prescribed_velocity",
                                  "level_set");
    }

    bool
    add_case_specific_parameters(dealii::ParameterHandler &prm) override
    {
      prm.add_parameter("center point",
                        center_point,
                        "Midpoint m between the centers of the circles (spheres); the circles move "
                        "towards the plane x = m_x. Specify comma-separated values for the x-, y- "
                        "(and z-) coordinate.");
      prm.add_parameter("radius",
                        radius,
                        "Radius r of the circles (spheres).",
                        dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
      prm.add_parameter("initial distance",
                        initial_distance,
                        "Initial distance D_0 > 2 r of the centers of the circles (spheres).",
                        dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
      prm.add_parameter("velocity",
                        velocity,
                        "Velocity U, with which each circle (sphere) moves towards the plane "
                        "x = m_x. The circles touch at t = (D_0 - 2 r) / (2 U).",
                        dealii::Patterns::Double(0., std::numeric_limits<number>::max()));

      return this->parameters.base.do_print_parameters;
    }

    /**
     * @brief Print the size of the narrow band, the measures of the interface, the gap and the
     * neck with their exact values, and the errors of the level set with respect to the signed
     * distance function of the union of the circles.
     */
    void
    do_postprocessing(const GenericDataOut<dim, number> &generic_data_out) const final
    {
      const auto exact_solution = create_exact_solution();
      exact_solution->set_time(generic_data_out.get_time());

      LevelSet::print_narrow_band_statistics<dim, number>(generic_data_out,
                                                          "level_set_band",
                                                          this->mpi_communicator,
                                                          "merging_circles");

      std::vector<std::pair<std::string, std::string>> fields{{"level_set", "merging_circles"}};
      if (this->parameters.narrow_band.continuous_level_set.enable)
        fields.emplace_back("level_set_cg", "level_set_cg");

      for (const auto &[field, label] : fields)
        {
          const auto [area, length] = LevelSet::print_interface_measures<dim, number>(
            generic_data_out, field, this->mpi_communicator, label, 10 /*area precision*/);
          print_measure_errors(area, length, *exact_solution, label);
          print_gap_and_neck(generic_data_out, field, *exact_solution, label);
          LevelSet::print_narrow_band_errors<dim, number>(generic_data_out,
                                                          field,
                                                          *exact_solution,
                                                          exact_solution->get_interface_measure(),
                                                          this->mpi_communicator,
                                                          label);
        }
    }

  private:
    std::shared_ptr<MergingSpheresSignedDistance<dim>>
    create_exact_solution() const
    {
      return std::make_shared<MergingSpheresSignedDistance<dim>>(to_point(center_point),
                                                                 radius,
                                                                 initial_distance,
                                                                 velocity);
    }

    void
    print_measure_errors(const number                             area,
                         const number                             length,
                         const MergingSpheresSignedDistance<dim> &exact_solution,
                         const std::string                       &label) const
    {
      dealii::ConditionalOStream pcout(
        std::cout, dealii::Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0);

      const double exact_area   = exact_solution.get_volume();
      const double exact_length = exact_solution.get_interface_measure();

      std::ostringstream str;
      str << std::scientific << std::setprecision(10) << "exact area: " << exact_area
          << std::setprecision(4) << "  rel. error area: " << (area - exact_area) / exact_area;
      Journal::print_line(pcout, str.str(), label);

      std::ostringstream str_length;
      str_length << std::scientific << std::setprecision(4) << "exact length: " << exact_length
                 << "  rel. error length: " << (length - exact_length) / exact_length;
      Journal::print_line(pcout, str_length.str(), label);
    }

    /**
     * @brief Print the gap between the circles (distance of the two inner zeros of the level set
     * on the segment of the axis between the centers at the start time, zero if there are less
     * than two zeros) and the half-width of the neck (half of the distance of the outermost zeros
     * on the segment of length 4r in the y-direction through the midpoint, zero if there are less
     * than two zeros) together with their exact values.
     */
    void
    print_gap_and_neck(const GenericDataOut<dim, number>       &generic_data_out,
                       const std::string                       &field,
                       const MergingSpheresSignedDistance<dim> &exact_solution,
                       const std::string                       &label) const
    {
      const dealii::Point<dim> m = to_point(center_point);

      // axis: [m_x - r, m_x + r] contains the inner zeros only (the outer zeros are at
      // m_x -+ (D/2 + r))
      dealii::Point<dim> axis_begin = m, axis_end = m;
      axis_begin[0] -= radius;
      axis_end[0] += radius;
      const auto axis_zeros = LevelSet::find_zeros_on_segment<dim, number>(
        generic_data_out, field, axis_begin, axis_end, this->mpi_communicator);
      const double gap =
        axis_zeros.size() >= 2 ? (axis_zeros.back() - axis_zeros.front()) * 2. * radius : 0.;

      // neck: segment of length 4r in the y-direction in the plane x = m_x
      dealii::Point<dim> neck_begin = m, neck_end = m;
      neck_begin[1] -= 2. * radius;
      neck_end[1] += 2. * radius;
      const auto neck_zeros = LevelSet::find_zeros_on_segment<dim, number>(
        generic_data_out, field, neck_begin, neck_end, this->mpi_communicator);
      const double neck =
        neck_zeros.size() >= 2 ? 0.5 * (neck_zeros.back() - neck_zeros.front()) * 4. * radius : 0.;

      dealii::ConditionalOStream pcout(
        std::cout, dealii::Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0);
      std::ostringstream str;
      str << std::scientific << std::setprecision(4) << "gap: " << gap
          << " (exact: " << exact_solution.get_gap() << ")  neck: " << neck
          << " (exact: " << exact_solution.get_neck_radius() << ")";
      Journal::print_line(pcout, str.str(), label);
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

    /// Midpoint between the centers: the plane x = m_x is a mesh plane, the axis is slightly off
    /// the mesh lines, so that the circles are not tangent to faces at their extreme points
    std::vector<number> center_point{0.0, 0.013, 0.007};

    /// Radius of the circles (spheres)
    number radius = 0.25;

    /// Initial distance of the centers (initial gap 0.1)
    number initial_distance = 0.6;

    /// Velocity of each circle towards the plane x = m_x (contact at t = 0.2)
    number velocity = 0.25;
  };
} // namespace MeltPoolDG::Simulation::MergingCircles
