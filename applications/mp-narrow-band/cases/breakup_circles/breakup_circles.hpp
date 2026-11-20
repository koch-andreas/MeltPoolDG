#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/utilities/journal.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>

#include "../../narrow_band_case.hpp"
#include "../../narrow_band_postprocessing.hpp"

namespace MeltPoolDG::Simulation::BreakupCircles
{
  /**
   * @brief Exact geometry of two overlapping circles of the radius $r$ with the centers
   * $\boldsymbol{c}_\pm = \boldsymbol{m} \pm \frac{D}{2} \boldsymbol{e}_x$ ($D < 2r$), which are
   * squeezed towards the axis $y = m_y$ with the velocity $U$ (see SimulationBreakupCircles).
   *
   * With the squeeze $s = U t$, the phase is
   * $\Omega^+(t) = \{\boldsymbol{x} : |y - m_y| + s < h_0(x)\}$, where $h_0(x) = \max_\pm
   * \sqrt{r^2 - (x - c_{\pm,x})^2}$ is the half-height of the union of the initial disks. Its part
   * $y \geq m_y$ is the union of the two disks shifted by $-s \boldsymbol{e}_y$, clipped by the
   * axis, and its part $y \leq m_y$ is the mirror image. The neck in the plane $x = m_x$ has the
   * half-width $w = w_0 - s$ with $w_0 = \sqrt{r^2 - D^2/4}$; it pinches off at
   * $t^* = w_0 / U$. Afterwards, the two parts are lenses (intersections of the disks shifted by
   * $\mp s \boldsymbol{e}_y$) with the gap $D - 2\sqrt{r^2 - s^2}$ between their tips on the axis.
   * The parts vanish at $t = r / U$.
   *
   * value() is the signed distance to the boundary of $\Omega^+$ (positive inside). By the
   * symmetry, the closest boundary point of a point $\boldsymbol{x}$ lies on the same side of the
   * axis; there, the boundary consists of the exposed arcs of the shifted circles (on the side of
   * $\boldsymbol{c}_-$ for $x \leq m_x$ and of $\boldsymbol{c}_+$ for $x \geq m_x$, above the
   * axis). The distance is the minimum of the distances to the radial projections onto the
   * circles, which lie on these arcs, and to the end points of the arcs: the corner of the neck
   * (before the pinch-off) and the points of the circles on the axis (tips).
   *
   * @note Only 2D is implemented.
   */
  template <int dim>
  class BreakupCirclesSignedDistance : public dealii::Function<dim>
  {
  public:
    BreakupCirclesSignedDistance(const dealii::Point<dim> &midpoint,
                                 const double              radius,
                                 const double              distance,
                                 const double              velocity)
      : dealii::Function<dim>(1)
      , midpoint(midpoint)
      , radius(radius)
      , distance(distance)
      , velocity(velocity)
    {
      AssertThrow(dim == 2, dealii::ExcNotImplemented());
    }

    /// Squeeze $s = U t$ of the circles towards the axis.
    double
    get_squeeze() const
    {
      return velocity * this->get_time();
    }

    /// Half-width of the initial neck $w_0 = \sqrt{r^2 - D^2/4}$.
    double
    get_initial_neck_radius() const
    {
      return std::sqrt(std::max(radius * radius - 0.25 * distance * distance, 0.));
    }

    /// Half-width of the neck in the plane x = m_x (zero after the pinch-off).
    double
    get_neck_radius() const
    {
      return std::max(get_initial_neck_radius() - get_squeeze(), 0.);
    }

    /// Gap between the tips of the two parts on the axis (zero before the pinch-off).
    double
    get_gap() const
    {
      const double s = get_squeeze();
      return std::max(distance - 2. * std::sqrt(std::max(radius * radius - s * s, 0.)), 0.);
    }

    /// Area of the phase.
    double
    get_volume() const
    {
      const double r  = radius;
      const double s  = std::min(get_squeeze(), r);
      const double w0 = get_initial_neck_radius();

      // area of a disk above a chord at the distance s from the center
      const double segment = r * r * std::acos(s / r) - s * std::sqrt(r * r - s * s);

      // area of the intersection of the two disks above the chord
      const auto F = [r](const double y) {
        return 0.5 * (y * std::sqrt(r * r - y * y) + r * r * std::asin(y / r));
      };
      const double intersection = s < w0 ? 2. * (F(w0) - F(s)) - distance * (w0 - s) : 0.;

      return 2. * (2. * segment - intersection);
    }

    /// Length of the boundary of the phase.
    double
    get_interface_measure() const
    {
      const double r      = radius;
      const double s      = std::min(get_squeeze(), r);
      const double alpha  = std::asin(s / r);
      const double theta0 = std::acos(std::min(0.5 * distance / r, 1.));
      return 4. * r * (dealii::numbers::PI - 2. * alpha - std::max(theta0 - alpha, 0.));
    }

    double
    value(const dealii::Point<dim> &p, const unsigned int /*component*/) const override
    {
      const double r  = radius;
      const double s  = get_squeeze();
      const double w0 = get_initial_neck_radius();

      // coordinates relative to the midpoint, mirrored into the upper half-plane
      const double x = p[0] - midpoint[0];
      const double y = std::abs(p[1] - midpoint[1]);

      // centers of the shifted circles (left, right)
      const std::array<double, 2> center_x{{-0.5 * distance, 0.5 * distance}};

      bool   inside   = false;
      double min_dist = std::numeric_limits<double>::max();
      for (unsigned int i = 0; i < 2; ++i)
        {
          const double dx = x - center_x[i];
          const double dy = y + s;
          const double d  = std::sqrt(dx * dx + dy * dy);
          inside          = inside or d < r;

          // radial projection: on the exposed arc above the axis
          if (d > 0.)
            {
              const double qx = center_x[i] + r * dx / d;
              const double qy = -s + r * dy / d;
              if (qy >= 0. and (i == 0 ? qx <= 0. : qx >= 0.))
                min_dist = std::min(min_dist, std::abs(r - d));
            }

          // tips: points of the circle on the axis on the exposed side
          if (s < r)
            {
              const double rho = std::sqrt(r * r - s * s);
              for (const double tip : {center_x[i] - rho, center_x[i] + rho})
                if (i == 0 ? tip <= 0. : tip >= 0.)
                  min_dist = std::min(min_dist, std::hypot(x - tip, y));
            }
        }

      // corner of the neck (before the pinch-off)
      if (w0 - s >= 0.)
        min_dist = std::min(min_dist, std::hypot(x, y - (w0 - s)));

      return inside ? min_dist : -min_dist;
    }

  private:
    const dealii::Point<dim> midpoint;
    const double             radius;
    const double             distance;
    const double             velocity;
  };

  /**
   * @brief Velocity of the two halves of the domain towards the axis $y = m_y$:
   * $\boldsymbol{u} = -U \operatorname{sign}(y - m_y) \boldsymbol{e}_y$ with
   * $\operatorname{sign}(0) = 1$, i.e., each half moves rigidly.
   */
  template <int dim>
  class SqueezingVelocity : public dealii::Function<dim>
  {
  public:
    SqueezingVelocity(const double axis, const double velocity)
      : dealii::Function<dim>(dim)
      , axis(axis)
      , velocity(velocity)
    {}

    double
    value(const dealii::Point<dim> &p, const unsigned int component) const override
    {
      if (component != 1)
        return 0.;
      return p[1] >= axis ? -velocity : velocity;
    }

  private:
    const double axis;
    const double velocity;
  };

  /**
   * @brief Break-up of two overlapping circles on the domain [-1,1]^2 (topology change).
   *
   * The circles of the radius $r$ have the centers $\boldsymbol{m} \pm \frac{D}{2}
   * \boldsymbol{e}_x$ with $D < 2r$, i.e., the initial phase is connected by a neck of the
   * half-width $w_0 = \sqrt{r^2 - D^2/4}$. The velocity $\boldsymbol{u} = -U
   * \operatorname{sign}(y - m_y) \boldsymbol{e}_y$ moves both halves of the domain rigidly towards
   * the axis $y = m_y$. Since the characteristics converge at the axis, the backward
   * characteristics never cross it, and the exact phase is $\{|y - m_y| + Ut < h_0(x)\}$, where
   * $h_0$ is the half-height of the initial union of the disks (see BreakupCirclesSignedDistance):
   * the neck thins with $w = w_0 - Ut$ and pinches off at $t^* = w_0 / U$; afterwards, the two
   * parts are lenses, whose tips on the axis move apart. The parts of the phase, which reach the
   * axis, vanish there, i.e., the area is not conserved, but known exactly. A smooth velocity field
   * cannot break up the phase in finite time (its flow map is a diffeomorphism); with this
   * velocity, the break-up is a property of the exact solution and not an effect of the
   * resolution. This case is the counterpart of merging_circles, where the circles move towards
   * the plane $x = m_x$ and merge.
   *
   * If user-defined postprocessing is enabled, the size of the narrow band, the area and the length
   * of the interface and their relative errors, the gap between the parts on the axis and the
   * half-width of the neck (zeros of the level set on the corresponding segments) with their exact
   * values, and the errors of the level set with respect to the exact signed distance function are
   * printed at every output time (for the continuous level set of the narrow band as well, if it
   * is enabled). The end time must satisfy $U t < r$. Only 2D is implemented.
   */
  template <int dim, typename number>
  class SimulationBreakupCircles : public LevelSet::NarrowBandCase<dim, number>
  {
  public:
    SimulationBreakupCircles(std::string parameter_file, const MPI_Comm mpi_communicator)
      : LevelSet::NarrowBandCase<dim, number>(parameter_file, mpi_communicator)
    {}

    void
    create_spatial_discretization() override
    {
      AssertThrow(dim == 2, dealii::ExcMessage("The case breakup_circles requires dim = 2."));

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
      AssertThrow(distance > 0. and distance < 2. * radius,
                  dealii::ExcMessage("The distance of the centers must be positive and smaller "
                                     "than the diameter of the circles (overlapping circles)."));
      AssertThrow(velocity * this->parameters.time_stepping.end_time < radius,
                  dealii::ExcMessage("The circles must not vanish before the end time: "
                                     "U t_end < r."));

      this->attach_initial_condition(create_exact_solution(), "level_set");
      this->attach_field_function(
        std::make_shared<SqueezingVelocity<dim>>(to_point(center_point)[1], velocity),
        "prescribed_velocity",
        "level_set");
    }

    bool
    add_case_specific_parameters(dealii::ParameterHandler &prm) override
    {
      prm.add_parameter("center point",
                        center_point,
                        "Midpoint m between the centers of the circles; the circles are squeezed "
                        "towards the axis y = m_y. Specify comma-separated values for the x- and "
                        "y-coordinate.");
      prm.add_parameter("radius",
                        radius,
                        "Radius r of the circles.",
                        dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
      prm.add_parameter("distance",
                        distance,
                        "Distance 0 < D < 2 r of the centers of the circles (overlapping circles "
                        "connected by a neck of the half-width sqrt(r^2 - D^2 / 4)).",
                        dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
      prm.add_parameter("velocity",
                        velocity,
                        "Velocity U, with which each half of the domain moves towards the axis "
                        "y = m_y. The neck pinches off at t = sqrt(r^2 - D^2 / 4) / U.",
                        dealii::Patterns::Double(0., std::numeric_limits<number>::max()));

      return this->parameters.base.do_print_parameters;
    }

    /**
     * @brief Print the size of the narrow band, the measures of the interface, the gap and the
     * neck with their exact values, and the errors of the level set with respect to the exact
     * signed distance function.
     */
    void
    do_postprocessing(const GenericDataOut<dim, number> &generic_data_out) const final
    {
      const auto exact_solution = create_exact_solution();
      exact_solution->set_time(generic_data_out.get_time());

      LevelSet::print_narrow_band_statistics<dim, number>(generic_data_out,
                                                          "level_set_band",
                                                          this->mpi_communicator,
                                                          "breakup_circles");

      std::vector<std::pair<std::string, std::string>> fields{{"level_set", "breakup_circles"}};
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
    std::shared_ptr<BreakupCirclesSignedDistance<dim>>
    create_exact_solution() const
    {
      return std::make_shared<BreakupCirclesSignedDistance<dim>>(to_point(center_point),
                                                                 radius,
                                                                 distance,
                                                                 velocity);
    }

    void
    print_measure_errors(const number                             area,
                         const number                             length,
                         const BreakupCirclesSignedDistance<dim> &exact_solution,
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
     * @brief Print the gap between the two parts (distance of the two inner zeros of the level set
     * on the segment [m_x - r, m_x + r] of the axis, zero if there are less than two zeros) and
     * the half-width of the neck (half of the distance of the outermost zeros on the segment of
     * length 4r in the y-direction through the midpoint, zero if there are less than two zeros)
     * together with their exact values.
     */
    void
    print_gap_and_neck(const GenericDataOut<dim, number>       &generic_data_out,
                       const std::string                       &field,
                       const BreakupCirclesSignedDistance<dim> &exact_solution,
                       const std::string                       &label) const
    {
      const dealii::Point<dim> m = to_point(center_point);

      // axis: [m_x - r, m_x + r] contains the inner tips only (the outer tips are at
      // m_x -+ (D/2 + sqrt(r^2 - s^2)) with D/2 + sqrt(r^2 - s^2) > r after the pinch-off)
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

    /// Midpoint between the centers: the axis y = m_y is a mesh line, the midpoint is slightly
    /// off the plane x = 0, so that the circles are not tangent to faces at their extreme points
    std::vector<number> center_point{0.013, 0.0};

    /// Radius of the circles
    number radius = 0.24;

    /// Distance of the centers (initial neck half-width sqrt(r^2 - D^2/4) = 0.0686)
    number distance = 0.46;

    /// Velocity of each half of the domain towards the axis (pinch-off at t = 0.274)
    number velocity = 0.25;
  };
} // namespace MeltPoolDG::Simulation::BreakupCircles
