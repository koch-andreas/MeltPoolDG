#include <deal.II/base/bounding_box.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/cell_id.h>

#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/non_matching/quadrature_generator.h>

#include <deal.II/numerics/rtree.h>

#include <meltpooldg/level_set/narrow_band_closest_point_tools.hpp>
#include <meltpooldg/level_set/narrow_band_interface_points.hpp>
#include <meltpooldg/utilities/journal.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <utility>
#include <vector>

namespace MeltPoolDG::LevelSet
{
  using NarrowBand::internal::AffineCellMap;

  namespace
  {
    /**
     * Signed distance of @p x to the flat patches of the points (see
     * NarrowBandInterfacePoints::signed_distance()), using the 8 (2D) or 24 (3D) nearest points of
     * @p tree (NarrowBand::internal::n_nearest_interface_points()). If
     * the closest point of the closest patch lies on its edge, i.e., @p x does not lie in front of
     * the patch (e.g., beyond the open end of a chain of points, whose cap has been removed), the
     * side of the patch is not defined, and the sign is @p fallback_sign (if not zero). If
     * @p orientation is given, only the points with a normal vector n satisfying
     * n . orientation > 0 are considered (distance to one branch of the interface).
     */
    template <int dim, typename number, typename TreeType>
    number
    signed_distance_to_patches(const dealii::Point<dim>                  &x,
                               const double                               max_radius,
                               const TreeType                            &tree,
                               const std::vector<dealii::Tensor<1, dim>> &normals,
                               const std::vector<number>                 &weights,
                               const double                               fallback_sign = 0.,
                               const dealii::Tensor<1, dim>              *orientation   = nullptr)
    {
      constexpr unsigned int n_nearest = NarrowBand::internal::n_nearest_interface_points<dim>();

      std::vector<std::pair<dealii::Point<dim>, unsigned int>> nearest;
      if (orientation != nullptr)
        tree.query(boost::geometry::index::nearest(x, n_nearest) and
                     boost::geometry::index::satisfies(
                       [&](const std::pair<dealii::Point<dim>, unsigned int> &value) {
                         return normals[value.second] * (*orientation) > 0.;
                       }),
                   std::back_inserter(nearest));
      else
        tree.query(boost::geometry::index::nearest(x, n_nearest), std::back_inserter(nearest));

      double min_distance = std::numeric_limits<double>::max();
      double sign         = 0.;
      for (const auto &[point, k] : nearest)
        {
          // radius of the patch: half the length (2D) or radius of the disk of the area (3D)
          double radius = 0.;
          if constexpr (dim == 2)
            radius = 0.5 * weights[k];
          else if constexpr (dim == 3)
            radius = std::sqrt(weights[k] / dealii::numbers::PI);
          radius = std::min(radius, max_radius);

          const dealii::Tensor<1, dim> d                = x - point;
          const double                 normal_distance  = normals[k] * d;
          const double                 tangent_distance = (d - normal_distance * normals[k]).norm();
          const double                 distance =
            std::sqrt(normal_distance * normal_distance +
                      dealii::Utilities::fixed_power<2>(std::max(0., tangent_distance - radius)));
          if (distance < min_distance)
            {
              min_distance = distance;
              sign         = normal_distance > 0. ? 1. : (normal_distance < 0. ? -1. : 0.);
              if (tangent_distance > radius and fallback_sign != 0.)
                sign = fallback_sign;
            }
        }

      return nearest.empty() ? 0. : sign * min_distance;
    }
  } // namespace

  template <int dim, typename number>
  NarrowBandInterfacePoints<dim, number>::NarrowBandInterfacePoints(
    const dealii::ConditionalOStream &pcout,
    const double                      unresolved_distance,
    const double                      support_tolerance,
    const double                      merge_distance,
    const bool                        conservative_insertion)
    : pcout(pcout)
    , unresolved_distance(unresolved_distance)
    , support_tolerance(support_tolerance)
    , merge_distance(merge_distance)
    , conservative_insertion(conservative_insertion)
  {
    AssertThrow(unresolved_distance >= 0. and support_tolerance > 0. and merge_distance >= 0.,
                dealii::ExcMessage("The unresolved distance and the merge distance must not be "
                                   "negative, and the support tolerance must be positive."));
  }

  template <int dim, typename number>
  template <typename CellIteratorType, typename QuadratureGeneratorType>
  void
  NarrowBandInterfacePoints<dim, number>::pack_surface_points(
    const CellIteratorType                                &cell,
    QuadratureGeneratorType                               &quadrature_generator,
    std::vector<double>                                   &data,
    const std::function<bool(const dealii::Point<dim> &)> &accept)
  {
    quadrature_generator.generate(cell);
    const auto &quadrature = quadrature_generator.get_surface_quadrature();
    if (quadrature.size() == 0)
      return;

    // JxW = w |det J| |J^{-T} n_ref|, n = J^{-T} n_ref / |J^{-T} n_ref|
    const AffineCellMap<dim> map(cell);
    const double             det_jacobian = std::abs(dealii::determinant(map.jacobian));
    for (unsigned int q = 0; q < quadrature.size(); ++q)
      {
        const dealii::Point<dim>     point = map.to_real(quadrature.point(q));
        const dealii::Tensor<1, dim> normal =
          map.inverse_jacobian_transposed * quadrature.normal_vector(q);
        const double norm = normal.norm();

        if (accept and not accept(point))
          continue;

        for (unsigned int d = 0; d < dim; ++d)
          data.push_back(point[d]);
        data.push_back(quadrature.weight(q) * det_jacobian * norm);
        for (unsigned int d = 0; d < dim; ++d)
          data.push_back(normal[d] / norm);
      }
  }

  template <int dim, typename number>
  void
  NarrowBandInterfacePoints<dim, number>::append_points(const std::vector<double> &local_data,
                                                        const MPI_Comm             mpi_comm)
  {
    for (const auto &data_of_process : dealii::Utilities::MPI::all_gather(mpi_comm, local_data))
      {
        AssertDimension(data_of_process.size() % (2 * dim + 1), 0);
        for (std::size_t k = 0; k < data_of_process.size();)
          {
            dealii::Point<dim>     point;
            dealii::Tensor<1, dim> normal;
            for (unsigned int d = 0; d < dim; ++d)
              point[d] = data_of_process[k++];
            const number weight = data_of_process[k++];
            for (unsigned int d = 0; d < dim; ++d)
              normal[d] = data_of_process[k++];

            points.push_back(point);
            weights.push_back(weight);
            normals.push_back(normal);
          }
      }
  }

  template <int dim, typename number>
  void
  NarrowBandInterfacePoints<dim, number>::conserve_cell_weights(const std::size_t first_new_point,
                                                                const MPI_Comm    mpi_comm)
  {
    // scaling factors of the points of the locally owned cells (zero: unchanged); every point is
    // assigned to at most one locally owned cell of one process
    std::vector<double> factors(points.size(), 0.);
    for (const auto &cell_points : points_of_cell)
      {
        double old_weight = 0., new_weight = 0.;
        for (const unsigned int k : cell_points)
          (k < first_new_point ? old_weight : new_weight) += weights[k];
        if (old_weight > 0. and new_weight > 0.)
          for (const unsigned int k : cell_points)
            factors[k] = old_weight / (old_weight + new_weight);
      }
    dealii::Utilities::MPI::sum(factors, mpi_comm, factors);

    for (std::size_t k = 0; k < points.size(); ++k)
      if (factors[k] > 0.)
        weights[k] *= factors[k];
  }

  template <int dim, typename number>
  void
  NarrowBandInterfacePoints<dim, number>::pack_gap_points(
    const dealii::DoFHandler<dim> &dof_handler,
    const VectorType              &level_set,
    const unsigned int             n_points_1d,
    const std::vector<bool>       *cells,
    std::vector<double>           &data) const
  {
    // Note: the quadrature generator keeps a pointer to the quadrature collection.
    const dealii::hp::QCollection<1> q_collection{dealii::QGauss<1>(n_points_1d)};
    dealii::NonMatching::DiscreteQuadratureGenerator<dim> quadrature_generator(q_collection,
                                                                               dof_handler,
                                                                               level_set);

    std::vector<std::pair<dealii::Point<dim>, unsigned int>> indexed_points;
    indexed_points.reserve(points.size());
    for (unsigned int k = 0; k < points.size(); ++k)
      indexed_points.emplace_back(points[k], k);
    const auto point_tree = dealii::pack_rtree(indexed_points);

    std::vector<std::pair<dealii::Point<dim>, unsigned int>> nearest;
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0 or
            (cells != nullptr and not(*cells)[cell->active_cell_index()]))
          continue;

        const double min_distance = cell->minimum_vertex_distance() / n_points_1d;
        pack_surface_points(cell, quadrature_generator, data, [&](const dealii::Point<dim> &x) {
          nearest.clear();
          point_tree.query(boost::geometry::index::nearest(x, 1), std::back_inserter(nearest));
          return nearest.empty() or nearest[0].first.distance(x) > min_distance;
        });
      }
  }

  template <int dim, typename number>
  void
  NarrowBandInterfacePoints<dim, number>::seed(const dealii::DoFHandler<dim> &dof_handler,
                                               const VectorType              &level_set,
                                               const unsigned int             n_points_1d)
  {
    // quadrature points of the zero level set of the locally owned cells
    std::vector<double> local_data;
    {
      const bool update_ghosts = not level_set.has_ghost_elements();
      if (update_ghosts)
        level_set.update_ghost_values();

      // Note: the quadrature generator keeps a pointer to the quadrature collection.
      const dealii::hp::QCollection<1> q_collection{dealii::QGauss<1>(n_points_1d)};
      dealii::NonMatching::DiscreteQuadratureGenerator<dim> quadrature_generator(q_collection,
                                                                                 dof_handler,
                                                                                 level_set);

      for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned() and cell->get_fe().n_dofs_per_cell() > 0)
          pack_surface_points(cell, quadrature_generator, local_data);

      if (update_ghosts)
        level_set.zero_out_ghost_values();
    }

    // points of all processes
    points.clear();
    weights.clear();
    normals.clear();
    append_points(local_data, dof_handler.get_mpi_communicator());

    // length scale for the finite differences of the velocity: minimum cell size
    double min_cell_size = std::numeric_limits<double>::max();
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        min_cell_size = std::min(min_cell_size, cell->minimum_vertex_distance());
    length_scale = dealii::Utilities::MPI::min(min_cell_size, dof_handler.get_mpi_communicator());

    approach.clear();
    convergence.clear();
    inner_approach.clear();
    inner_convergence.clear();
    crossed.clear();
    points_of_cell.clear();
    is_unresolved.clear();
    located = false;
  }

  template <int dim, typename number>
  void
  NarrowBandInterfacePoints<dim, number>::advect(dealii::Function<dim> &velocity,
                                                 const number           time,
                                                 const number           time_step)
  {
    AssertThrow(velocity.n_components == dim,
                dealii::ExcMessage("The velocity function needs dim components."));

    // forward integration of the characteristic dx/dt = u(x, t) and of the weighted normal
    // vector m = w n (Nanson's formula: dm/dt = (div u) m - (grad u)^T m, the weight is the area
    // element) from t^n to t^{n+1} with the classical fourth-order Runge-Kutta method; the
    // velocity gradient is approximated by central differences
    const number old_time = velocity.get_time();
    const number delta    = 1e-4 * length_scale;

    using StateType = std::pair<dealii::Tensor<1, dim>, dealii::Tensor<1, dim>>;
    const auto rhs  = [&](const dealii::Point<dim>     &x,
                         const dealii::Tensor<1, dim> &m,
                         const number                  t) {
      velocity.set_time(t);
      dealii::Tensor<1, dim> u;
      for (unsigned int d = 0; d < dim; ++d)
        u[d] = velocity.value(x, d);

      // grad_u[i][j] = du_i / dx_j
      dealii::Tensor<2, dim> grad_u;
      for (unsigned int j = 0; j < dim; ++j)
        {
          dealii::Point<dim> x_plus = x, x_minus = x;
          x_plus[j] += delta;
          x_minus[j] -= delta;
          for (unsigned int i = 0; i < dim; ++i)
            grad_u[i][j] = (velocity.value(x_plus, i) - velocity.value(x_minus, i)) / (2. * delta);
        }

      return StateType{u, dealii::trace(grad_u) * m - dealii::transpose(grad_u) * m};
    };

    const number                          h          = time_step;
    const std::vector<dealii::Point<dim>> old_points = points;
    for (unsigned int k = 0; k < points.size(); ++k)
      {
        const dealii::Point<dim>     x = points[k];
        const dealii::Tensor<1, dim> m = weights[k] * normals[k];

        const auto k1 = rhs(x, m, time);
        const auto k2 = rhs(x + 0.5 * h * k1.first, m + 0.5 * h * k1.second, time + 0.5 * h);
        const auto k3 = rhs(x + 0.5 * h * k2.first, m + 0.5 * h * k2.second, time + 0.5 * h);
        const auto k4 = rhs(x + h * k3.first, m + h * k3.second, time + h);

        points[k] = x + (h / 6.) * (k1.first + 2. * k2.first + 2. * k3.first + k4.first);
        const dealii::Tensor<1, dim> m_new =
          m + (h / 6.) * (k1.second + 2. * k2.second + 2. * k3.second + k4.second);
        weights[k] = m_new.norm();
        if (weights[k] > 0.)
          normals[k] = m_new / weights[k];
      }

    // approach of the flow on the outer side of each point within one time step (merge rule):
    // dt max(0, (u(x - l n) - u(x)) . n) at t^{n+1} with the length scale l, i.e., the
    // displacement, by which a part of the interface on the outer side approaches the point
    // within one time step (nonzero for converging flows, zero for rigid-body motions),
    // and the convergence ratio of this normal velocity difference and the magnitude of the
    // velocity difference (the cosine of the angle between the relative velocity of the flow on
    // this side and the direction towards the point); the same on the inner side (break-up
    // rule: the phase between two parts of the interface is squeezed)
    approach.assign(points.size(), 0.);
    convergence.assign(points.size(), 0.);
    inner_approach.assign(points.size(), 0.);
    inner_convergence.assign(points.size(), 0.);
    crossed.assign(points.size(), false);
    if (merge_distance > 0.)
      {
        // points, which have crossed a line of converging characteristics within the time step:
        // the velocity along the path of a point changes by Delta = u(x^n, t^n) -
        // u(x^{n+1}, t^{n+1}), and the velocity field already changes by more than |Delta| / 2
        // within the distance dt |Delta| from the new position in the direction -Delta, i.e., the
        // point has passed a jump of the velocity, which pushes it back (e.g., the points of the
        // two sides of a tip, which are squeezed together tangentially, cf. the pair test for the
        // normal collisions). For a Lipschitz-continuous velocity with dt |grad u| < 1/2, the
        // velocity changes by less than |Delta| / 2 over this distance, i.e., the test never
        // applies. The test only uses differences of velocities and is invariant under a
        // superposed uniform translation (in contrast to the condition u(x^n) . u(x^{n+1}) < 0).
        std::vector<dealii::Tensor<1, dim>> u_old(points.size());
        velocity.set_time(time);
        for (unsigned int k = 0; k < points.size(); ++k)
          for (unsigned int d = 0; d < dim; ++d)
            u_old[k][d] = velocity.value(old_points[k], d);

        velocity.set_time(time + h);
        for (unsigned int k = 0; k < points.size(); ++k)
          {
            dealii::Tensor<1, dim> u_new;
            for (unsigned int d = 0; d < dim; ++d)
              u_new[d] = velocity.value(points[k], d);
            const dealii::Tensor<1, dim> change = u_old[k] - u_new;
            const double                 jump   = change.norm();
            if (not(jump > 1e-12 * std::max(u_old[k].norm(), u_new.norm())))
              continue;

            const dealii::Point<dim> probe = points[k] - h * change;
            dealii::Tensor<1, dim>   u_probe;
            for (unsigned int d = 0; d < dim; ++d)
              u_probe[d] = velocity.value(probe, d);
            crossed[k] = (u_probe - u_new).norm() > 0.5 * jump;
          }

        velocity.set_time(time + h);
        for (unsigned int k = 0; k < points.size(); ++k)
          {
            const dealii::Point<dim> outer = points[k] - length_scale * normals[k];
            const dealii::Point<dim> inner = points[k] + length_scale * normals[k];
            dealii::Tensor<1, dim>   u_point, u_outer, u_inner;
            for (unsigned int d = 0; d < dim; ++d)
              {
                u_point[d] = velocity.value(points[k], d);
                u_outer[d] = velocity.value(outer, d);
                u_inner[d] = velocity.value(inner, d);
              }
            // the convergence ratio relates the normal velocity difference to the magnitude of
            // the velocity difference (not to the speed), so that it is invariant under a
            // superposed uniform translation; relative velocities below the round-off level of
            // the speed (e.g., a uniform translation) count as zero
            const auto ratio = [](const double rate, const dealii::Tensor<1, dim> &difference,
                                  const double speed) {
              const double magnitude = difference.norm();
              return magnitude > 1e-12 * speed ? std::max(0., rate) / magnitude : 0.;
            };

            const double rate = (u_outer - u_point) * normals[k];
            approach[k]       = h * std::max(0., rate);
            convergence[k] =
              ratio(rate, u_outer - u_point, std::max(u_point.norm(), u_outer.norm()));

            const double inner_rate = (u_point - u_inner) * normals[k];
            inner_approach[k]       = h * std::max(0., inner_rate);
            inner_convergence[k] =
              ratio(inner_rate, u_point - u_inner, std::max(u_point.norm(), u_inner.norm()));
          }
      }

    velocity.set_time(old_time);

    points_of_cell.clear();
    is_unresolved.clear();
    located = false;
  }

  template <int dim, typename number>
  std::vector<std::optional<typename NarrowBandInterfacePoints<dim, number>::Location>>
  NarrowBandInterfacePoints<dim, number>::find_cells(
    const dealii::DoFHandler<dim>         &dof_handler_band,
    const std::vector<dealii::Point<dim>> &query_points)
  {
    // cell of the band containing each point (locally owned and ghost cells): on the common
    // boundary of several cells, the cell with the smallest CellId (unique on all processes)
    std::vector<std::optional<Location>> locations(query_points.size());

    std::vector<std::pair<dealii::Point<dim>, unsigned int>> indexed_points;
    indexed_points.reserve(query_points.size());
    for (unsigned int k = 0; k < query_points.size(); ++k)
      indexed_points.emplace_back(query_points[k], k);
    const auto point_tree = dealii::pack_rtree(indexed_points);

    constexpr double tolerance = 1e-10;

    std::vector<std::pair<dealii::Point<dim>, unsigned int>> candidates;
    for (const auto &cell : dof_handler_band.active_cell_iterators())
      {
        if (cell->is_artificial() or cell->get_fe().n_dofs_per_cell() == 0)
          continue;

        const AffineCellMap<dim> map(cell);

        auto box = cell->bounding_box();
        box.extend(tolerance * map.cell_size);

        candidates.clear();
        point_tree.query(boost::geometry::index::intersects(box), std::back_inserter(candidates));

        for (const auto &[point, k] : candidates)
          {
            const dealii::Point<dim> unit_point = map.to_unit(point);
            bool                     is_inside  = true;
            for (unsigned int d = 0; d < dim; ++d)
              is_inside =
                is_inside and unit_point[d] >= -tolerance and unit_point[d] <= 1. + tolerance;
            if (not is_inside)
              continue;

            const dealii::CellId cell_id = cell->id();
            if (not locations[k] or cell_id < locations[k]->cell_id)
              locations[k] = Location{cell_id, cell->active_cell_index(), cell->is_locally_owned()};
          }
      }

    return locations;
  }

  template <int dim, typename number>
  void
  NarrowBandInterfacePoints<dim, number>::assign_points_to_cells(
    const dealii::DoFHandler<dim> &dof_handler_band)
  {
    const auto locations = find_cells(dof_handler_band, points);

    points_of_cell.assign(dof_handler_band.get_triangulation().n_active_cells(), {});
    for (unsigned int k = 0; k < points.size(); ++k)
      if (locations[k] and locations[k]->is_locally_owned)
        points_of_cell[locations[k]->active_cell_index].push_back(k);
  }

  template <int dim, typename number>
  void
  NarrowBandInterfacePoints<dim, number>::mark_collided_points(
    const dealii::DoFHandler<dim> &dof_handler_band,
    const VectorType              &level_set_band,
    const std::vector<double>     &values_at_points,
    const std::vector<double>     &cell_sizes,
    std::vector<unsigned int>     &status) const
  {
    const MPI_Comm mpi_comm = dof_handler_band.get_mpi_communicator();

    // Both rules consider a side of each point: the outer side (merge rule: the other phase
    // between two parts of the interface vanishes, e.g., merging droplets) and the inner side
    // (break-up rule: the phase between two parts of the interface vanishes, e.g., a neck, which
    // pinches off). Only points in a flow, which converges on this side, are candidates (the
    // positive part of the normal velocity difference over h on this side exceeds a quarter of
    // the magnitude of the velocity difference, i.e., relative to the point, the flow on this
    // side approaches it within an angle of about 75 degrees to the normal, e.g., two parts of
    // the interface pushed against each other; this is the condition "half of the speed" in the
    // frame of the mean velocity of the point and the probe and independent of a superposed
    // uniform translation), i.e., not the points of thin gaps or filaments in shear or rotating
    // flows, which the level set does not resolve (the points are the same on all processes,
    // i.e., all processes take the same decisions).
    struct Side
    {
      // +1: outer side (merge rule), -1: inner side (break-up rule)
      double                     orientation;
      const std::vector<double> *approach;
      const std::vector<double> *convergence;
    };
    const std::array<Side, 2> sides{
      {Side{1., &approach, &convergence}, Side{-1., &inner_approach, &inner_convergence}}};

    // points, which have crossed a line of converging characteristics (see advect())
    std::vector<bool> is_collided(points.size(), false);
    if (crossed.size() == points.size())
      for (unsigned int k = 0; k < points.size(); ++k)
        is_collided[k] = status[k] == 1 and crossed[k];

    for (const Side &side : sides)
      {
        const auto is_candidate = [&](const unsigned int k) {
          return status[k] == 1 and not is_collided[k] and
                 side.convergence->size() == points.size() and (*side.convergence)[k] > 0.25;
        };
        bool any_candidate = false;
        for (unsigned int k = 0; k < points.size(); ++k)
          any_candidate = any_candidate or is_candidate(k);
        if (not any_candidate)
          continue;

        const auto approach_of = [&](const unsigned int k) {
          return side.approach->size() == points.size() ? (*side.approach)[k] : 0.;
        };

        // direction from the point to the considered side
        const auto direction = [&](const unsigned int k) { return -side.orientation * normals[k]; };

        // distance of the rule: c_m h plus the approach of the flow on the side within one time
        // step (collisions within the time step)
        std::vector<double> distances(points.size());
        for (unsigned int k = 0; k < points.size(); ++k)
          distances[k] = merge_distance * cell_sizes[k] + approach_of(k);

        // (a) ray test: towards the side, i.e., at the probe x + delta d (d: direction to the
        // side), the advected level set changes by less than delta / 4 in the direction of the
        // phase on this side (for a point on the interface, it decreases by about delta on the
        // outer side and increases by about delta on the inner side), i.e., the point lies
        // inside of the other phase or at a gap (merge rule) or a part of the phase (break-up
        // rule) thinner than about delta
        std::vector<dealii::Point<dim>> probes(points.size());
        std::vector<bool>               candidates(points.size(), false);
        for (unsigned int k = 0; k < points.size(); ++k)
          {
            candidates[k] = is_candidate(k);
            probes[k]     = candidates[k] ? points[k] + distances[k] * direction(k) : points[k];
          }

        const auto locations = find_cells(dof_handler_band, probes);

        std::vector<std::vector<unsigned int>> probes_of_cell(
          dof_handler_band.get_triangulation().n_active_cells());
        for (unsigned int k = 0; k < points.size(); ++k)
          if (candidates[k] and locations[k] and locations[k]->is_locally_owned)
            probes_of_cell[locations[k]->active_cell_index].push_back(k);

        std::vector<double> probe_values(points.size(), 0.), probe_found(points.size(), 0.);
        {
          dealii::Vector<number> values;
          for (const auto &cell : dof_handler_band.active_cell_iterators())
            {
              if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
                continue;

              const auto &cell_probes = probes_of_cell[cell->active_cell_index()];
              if (cell_probes.empty())
                continue;

              const auto              &fe = cell->get_fe();
              const AffineCellMap<dim> map(cell);
              values.reinit(fe.n_dofs_per_cell());
              cell->get_dof_values(level_set_band, values);

              for (const unsigned int k : cell_probes)
                {
                  const dealii::Point<dim> unit_point = map.to_unit(probes[k]);
                  double                   value      = 0.;
                  for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
                    value += values[i] * fe.shape_value(i, unit_point);
                  probe_values[k] = value;
                  probe_found[k]  = 1.;
                }
            }
        }
        dealii::Utilities::MPI::sum(probe_values, mpi_comm, probe_values);
        dealii::Utilities::MPI::sum(probe_found, mpi_comm, probe_found);

        std::vector<bool> is_collided_side(points.size(), false);
        for (unsigned int k = 0; k < points.size(); ++k)
          if (candidates[k] and probe_found[k] > 0.5 and
              side.orientation * (probe_values[k] - values_at_points[k]) > 0.25 * distances[k])
            is_collided_side[k] = true;

        // (b) pair test: a point of a different orientation (n_i . n_j < 1/2) is closer than the
        // sum of the approaches of the flow of the two points within the time step, and each of
        // the two points lies on the considered side of the other one (merge rule: back to back,
        // i.e., the gap between the two parts of the interface has closed within the time step;
        // break-up rule: face to face, i.e., the part of the phase between them has vanished)
        std::vector<std::pair<dealii::Point<dim>, unsigned int>> indexed_points;
        for (unsigned int k = 0; k < points.size(); ++k)
          if (status[k] != 0)
            indexed_points.emplace_back(points[k], k);
        const auto point_tree = dealii::pack_rtree(indexed_points);

        double max_approach = 0.;
        for (unsigned int k = 0; k < points.size(); ++k)
          max_approach = std::max(max_approach, approach_of(k));

        std::vector<std::pair<dealii::Point<dim>, unsigned int>> neighbors;
        for (unsigned int k = 0; k < points.size(); ++k)
          {
            if (not candidates[k] or is_collided_side[k])
              continue;

            const double       radius = approach_of(k) + max_approach;
            dealii::Point<dim> lower = points[k], upper = points[k];
            for (unsigned int d = 0; d < dim; ++d)
              {
                lower[d] -= radius;
                upper[d] += radius;
              }
            neighbors.clear();
            point_tree.query(boost::geometry::index::intersects(
                               dealii::BoundingBox<dim>(std::make_pair(lower, upper))),
                             std::back_inserter(neighbors));
            for (const auto &[point, j] : neighbors)
              if (j != k and point.distance(points[k]) < approach_of(k) + approach_of(j) and
                  normals[k] * normals[j] < 0.5 and direction(k) * (points[j] - points[k]) >= 0. and
                  direction(j) * (points[k] - points[j]) >= 0.)
                {
                  is_collided_side[k] = true;
                  break;
                }
          }

        for (unsigned int k = 0; k < points.size(); ++k)
          if (is_collided_side[k])
            is_collided[k] = true;
      }

    for (unsigned int k = 0; k < points.size(); ++k)
      if (is_collided[k])
        status[k] = 3;
  }

  template <int dim, typename number>
  typename NarrowBandInterfacePoints<dim, number>::Statistics
  NarrowBandInterfacePoints<dim, number>::locate(const dealii::DoFHandler<dim> &dof_handler_band,
                                                 const VectorType              &level_set_band,
                                                 const unsigned int             n_points_1d,
                                                 const std::vector<bool>       &is_cut_cell,
                                                 const bool                     insert)
  {
    const auto &triangulation = dof_handler_band.get_triangulation();
    AssertDimension(is_cut_cell.size(), triangulation.n_active_cells());

    const MPI_Comm mpi_comm = dof_handler_band.get_mpi_communicator();
    Statistics     statistics;

    const bool update_ghosts = not level_set_band.has_ghost_elements();
    if (update_ghosts)
      level_set_band.update_ghost_values();

    // 1) assignment of the points to the locally owned cells of the band
    assign_points_to_cells(dof_handler_band);

    // 2) status of the points (on the process owning their cell): 0: outside of the band, 1:
    // supported by the advected level set, 2: not supported, i.e., |phi(x)| > c_s h or n . grad phi
    // <= 0 at the point x (e.g., parts of the interface, which are not resolved by the level set
    // and have been lost by it)
    std::vector<unsigned int> status_local(points.size(), 0), status(points.size(), 0);
    std::vector<double>       values_at_points(points.size(), 0.), cell_sizes(points.size(), 0.);
    {
      dealii::Vector<number> values;
      for (const auto &cell : dof_handler_band.active_cell_iterators())
        {
          if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
            continue;

          const auto &cell_points = points_of_cell[cell->active_cell_index()];
          if (cell_points.empty())
            continue;

          const auto              &fe = cell->get_fe();
          const AffineCellMap<dim> map(cell);
          values.reinit(fe.n_dofs_per_cell());
          cell->get_dof_values(level_set_band, values);

          for (const unsigned int k : cell_points)
            {
              const dealii::Point<dim> unit_point = map.to_unit(points[k]);
              number                   value      = 0.;
              dealii::Tensor<1, dim>   unit_gradient;
              for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
                {
                  value += values[i] * fe.shape_value(i, unit_point);
                  unit_gradient += values[i] * fe.shape_grad(i, unit_point);
                }
              const dealii::Tensor<1, dim> gradient =
                map.inverse_jacobian_transposed * unit_gradient;

              const bool is_supported =
                std::abs(value) <= support_tolerance * map.cell_size and normals[k] * gradient > 0.;
              status_local[k]     = is_supported ? 1 : 2;
              values_at_points[k] = value;
              cell_sizes[k]       = map.cell_size;
            }
        }
    }
    dealii::Utilities::MPI::sum(status_local, mpi_comm, status);

    // merge rule (status 3: collided points, removed)
    if (merge_distance > 0.)
      {
        dealii::Utilities::MPI::sum(values_at_points, mpi_comm, values_at_points);
        dealii::Utilities::MPI::sum(cell_sizes, mpi_comm, cell_sizes);
        mark_collided_points(
          dof_handler_band, level_set_band, values_at_points, cell_sizes, status);
      }

    {
      unsigned int n_kept = 0;
      for (unsigned int k = 0; k < points.size(); ++k)
        {
          if (status[k] == 0)
            ++statistics.n_points_outside;
          else if (status[k] == 2)
            ++statistics.n_unsupported_points;
          else if (status[k] == 3)
            ++statistics.n_collided_points;
          if (status[k] != 1)
            continue;

          points[n_kept]  = points[k];
          weights[n_kept] = weights[k];
          normals[n_kept] = normals[k];
          ++n_kept;
        }
      points.resize(n_kept);
      weights.resize(n_kept);
      normals.resize(n_kept);
    }

    assign_points_to_cells(dof_handler_band);

    // 3) locally owned cells: new points at the quadrature points of the zero level set of the
    // advected level set, which are farther than h / n_points_1d from all points (gaps of the
    // point set due to the stretching of the interface; only if insert is true, otherwise see
    // insert_points()), merging of the closest pairs of points in cells with too many points
    // (compression of the interface)
    const unsigned int max_n_points_per_cell = 2 * dealii::Utilities::pow(n_points_1d, dim - 1);

    std::vector<double> new_points_local;
    if (insert)
      pack_gap_points(
        dof_handler_band, level_set_band, n_points_1d, &is_cut_cell, new_points_local);

    std::vector<unsigned int> merged_pairs_local; // (removed point, kept point)
    {
      std::vector<unsigned int> active;
      for (const auto &cell : dof_handler_band.active_cell_iterators())
        {
          if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
            continue;

          const auto &cell_points = points_of_cell[cell->active_cell_index()];
          if (cell_points.size() > max_n_points_per_cell)
            {
              // greedy: remove one point of the closest pair of points with the same orientation
              // (n_i . n_j > 0) until the maximum number is reached
              active = cell_points;
              while (active.size() > max_n_points_per_cell)
                {
                  std::optional<std::pair<unsigned int, unsigned int>> closest_pair;
                  double min_distance = std::numeric_limits<double>::max();
                  for (unsigned int i = 0; i < active.size(); ++i)
                    for (unsigned int j = i + 1; j < active.size(); ++j)
                      if (const double distance = points[active[i]].distance(points[active[j]]);
                          distance < min_distance and normals[active[i]] * normals[active[j]] > 0.)
                        {
                          min_distance = distance;
                          closest_pair = std::make_pair(i, j);
                        }

                  if (not closest_pair)
                    break;

                  merged_pairs_local.push_back(active[closest_pair->second]);
                  merged_pairs_local.push_back(active[closest_pair->first]);
                  active.erase(active.begin() + closest_pair->second);
                }
            }
        }
    }

    if (update_ghosts)
      level_set_band.zero_out_ghost_values();

    // 4) update of the points on all processes: merging (in the order of the merges), removal of
    // the merged points, new points
    std::vector<bool> is_removed(points.size(), false);
    for (const auto &pairs_of_process :
         dealii::Utilities::MPI::all_gather(mpi_comm, merged_pairs_local))
      for (std::size_t i = 0; i < pairs_of_process.size(); i += 2)
        {
          const unsigned int removed = pairs_of_process[i], kept = pairs_of_process[i + 1];
          weights[kept] += weights[removed];
          is_removed[removed] = true;
          ++statistics.n_merged_points;
        }

    {
      unsigned int n_kept = 0;
      for (unsigned int k = 0; k < points.size(); ++k)
        {
          if (is_removed[k])
            continue;

          points[n_kept]  = points[k];
          weights[n_kept] = weights[k];
          normals[n_kept] = normals[k];
          ++n_kept;
        }
      points.resize(n_kept);
      weights.resize(n_kept);
      normals.resize(n_kept);
    }

    const std::size_t n_points_old = points.size();
    double            old_weight   = 0.;
    for (const auto weight : weights)
      old_weight += weight;
    append_points(new_points_local, mpi_comm);
    statistics.n_inserted_points = points.size() - n_points_old;
    for (std::size_t k = n_points_old; k < points.size(); ++k)
      statistics.inserted_weight += weights[k];

    // 5) assignment of the updated points, conservative insertion
    assign_points_to_cells(dof_handler_band);
    if (conservative_insertion and statistics.n_inserted_points > 0)
      conserve_cell_weights(n_points_old, mpi_comm);
    for (const auto weight : weights)
      statistics.added_weight += weight;
    statistics.added_weight -= old_weight;

    // 6), 7) unresolved points and statistics of the cells
    last_statistics = statistics;
    update_unresolved_points(triangulation, is_cut_cell, mpi_comm);
    if (insert)
      print_statistics();

    return last_statistics;
  }

  template <int dim, typename number>
  typename NarrowBandInterfacePoints<dim, number>::Statistics
  NarrowBandInterfacePoints<dim, number>::insert_points(
    const dealii::DoFHandler<dim> &dof_handler_cut,
    const VectorType              &level_set,
    const dealii::DoFHandler<dim> &dof_handler_band,
    const unsigned int             n_points_1d)
  {
    Assert(located, dealii::ExcMessage("The points are not assigned to the cells."));

    const auto    &triangulation = dof_handler_cut.get_triangulation();
    const MPI_Comm mpi_comm      = dof_handler_cut.get_mpi_communicator();

    // new points at the quadrature points of the zero level set of the locally owned cells with
    // DoFs, which are farther than h / n_points_1d from all points
    std::vector<double> new_points_local;
    {
      const bool update_ghosts = not level_set.has_ghost_elements();
      if (update_ghosts)
        level_set.update_ghost_values();

      pack_gap_points(dof_handler_cut, level_set, n_points_1d, nullptr, new_points_local);

      if (update_ghosts)
        level_set.zero_out_ghost_values();
    }

    const std::size_t n_points_old = points.size();
    double            old_weight   = 0.;
    for (const auto weight : weights)
      old_weight += weight;
    append_points(new_points_local, mpi_comm);
    last_statistics.n_inserted_points += points.size() - n_points_old;
    for (std::size_t k = n_points_old; k < points.size(); ++k)
      last_statistics.inserted_weight += weights[k];

    // assignment of all points to the cells of the band, conservative insertion, unresolved points
    // and statistics of the cells (cut cells: cells with DoFs of dof_handler_cut)
    assign_points_to_cells(dof_handler_band);
    if (conservative_insertion and points.size() > n_points_old)
      conserve_cell_weights(n_points_old, mpi_comm);
    for (const auto weight : weights)
      last_statistics.added_weight += weight;
    last_statistics.added_weight -= old_weight;

    std::vector<bool> is_cut_cell(triangulation.n_active_cells(), false);
    for (const auto &cell : dof_handler_cut.active_cell_iterators())
      if (cell->is_locally_owned() and cell->get_fe().n_dofs_per_cell() > 0)
        is_cut_cell[cell->active_cell_index()] = true;
    update_unresolved_points(triangulation, is_cut_cell, mpi_comm);
    print_statistics();

    return last_statistics;
  }

  template <int dim, typename number>
  void
  NarrowBandInterfacePoints<dim, number>::update_unresolved_points(
    const dealii::Triangulation<dim> &triangulation,
    const std::vector<bool>          &is_cut_cell,
    const MPI_Comm                    mpi_comm)
  {
    AssertDimension(is_cut_cell.size(), triangulation.n_active_cells());

    Statistics &statistics                = last_statistics;
    statistics.n_unresolved_points        = 0;
    statistics.n_points_in_non_cut_cells  = 0;
    statistics.n_cut_cells_without_points = 0;

    // 6) points of the locally owned cells with a point of opposite orientation (n_i . n_j < 0)
    // closer than the unresolved distance: the interface is not resolved (e.g., both sides of a
    // thin filament or a tip), and the points are not used in the surface quadrature
    is_unresolved.assign(points.size(), false);
    {
      std::vector<std::pair<dealii::Point<dim>, unsigned int>> indexed_points;
      indexed_points.reserve(points.size());
      for (unsigned int k = 0; k < points.size(); ++k)
        indexed_points.emplace_back(points[k], k);
      const auto point_tree = dealii::pack_rtree(indexed_points);

      std::vector<std::pair<dealii::Point<dim>, unsigned int>> neighbors;
      for (const auto &cell : triangulation.active_cell_iterators())
        {
          if (not cell->is_locally_owned())
            continue;

          const double h = unresolved_distance * cell->minimum_vertex_distance();
          for (const unsigned int k : points_of_cell[cell->active_cell_index()])
            {
              dealii::Point<dim> lower = points[k], upper = points[k];
              for (unsigned int d = 0; d < dim; ++d)
                {
                  lower[d] -= h;
                  upper[d] += h;
                }

              neighbors.clear();
              point_tree.query(boost::geometry::index::intersects(
                                 dealii::BoundingBox<dim>(std::make_pair(lower, upper))),
                               std::back_inserter(neighbors));
              for (const auto &[point, j] : neighbors)
                if (point.distance(points[k]) < h and normals[j] * normals[k] < 0.)
                  {
                    is_unresolved[k] = true;
                    ++statistics.n_unresolved_points;
                    break;
                  }
            }
        }
    }

    // 7) statistics of the cells: usable points in cells, which are not cut (not used), cut
    // cells without usable points (zero level set used)
    for (const auto &cell : triangulation.active_cell_iterators())
      {
        if (not cell->is_locally_owned())
          continue;

        const unsigned int index = cell->active_cell_index();
        const unsigned int n_usable_points =
          std::count_if(points_of_cell[index].begin(),
                        points_of_cell[index].end(),
                        [&](const unsigned int k) { return not is_unresolved[k]; });
        if (is_cut_cell[index] and n_usable_points == 0)
          ++statistics.n_cut_cells_without_points;
        if (not is_cut_cell[index])
          statistics.n_points_in_non_cut_cells += n_usable_points;
      }

    statistics.n_points = points.size();
    statistics.n_points_in_non_cut_cells =
      dealii::Utilities::MPI::sum(statistics.n_points_in_non_cut_cells, mpi_comm);
    statistics.n_cut_cells_without_points =
      dealii::Utilities::MPI::sum(statistics.n_cut_cells_without_points, mpi_comm);
    statistics.n_unresolved_points =
      dealii::Utilities::MPI::sum(statistics.n_unresolved_points, mpi_comm);

    // the points are stored on all processes
    statistics.total_weight = 0.;
    for (const auto weight : weights)
      statistics.total_weight += weight;

    located = true;
  }

  template <int dim, typename number>
  void
  NarrowBandInterfacePoints<dim, number>::print_statistics() const
  {
    const Statistics  &statistics = last_statistics;
    std::ostringstream str;
    str << "interface points: " << statistics.n_points << " (" << statistics.n_inserted_points
        << " new, " << statistics.n_merged_points << " merged, " << statistics.n_points_outside
        << " outside of the band, " << statistics.n_unsupported_points << " not supported";
    if (statistics.n_collided_points > 0)
      str << ", " << statistics.n_collided_points << " collided";
    str << "), " << statistics.n_points_in_non_cut_cells << " in cells, which are not cut, "
        << statistics.n_cut_cells_without_points << " cut cells without points, "
        << statistics.n_unresolved_points << " unresolved points";
    Journal::print_line(pcout, str.str(), "narrow_band");

    std::ostringstream str_weights;
    str_weights << std::scientific << std::setprecision(6) << "interface points: total weight "
                << statistics.total_weight;
    if (statistics.n_inserted_points > 0)
      str_weights << " (weight of the new points " << std::setprecision(3)
                  << statistics.inserted_weight << ", increase of the total weight "
                  << statistics.added_weight << ")";
    Journal::print_line(pcout, str_weights.str(), "narrow_band");
  }

  template <int dim, typename number>
  bool
  NarrowBandInterfacePoints<dim, number>::is_located() const
  {
    return located;
  }

  template <int dim, typename number>
  dealii::NonMatching::ImmersedSurfaceQuadrature<dim>
  NarrowBandInterfacePoints<dim, number>::get_surface_quadrature(
    const typename dealii::Triangulation<dim>::cell_iterator &cell) const
  {
    Assert(located, dealii::ExcMessage("The points are not assigned to the cells."));

    dealii::NonMatching::ImmersedSurfaceQuadrature<dim> quadrature;

    const unsigned int index = cell->active_cell_index();
    if (index >= points_of_cell.size() or points_of_cell[index].empty())
      return quadrature;

    // reference normal n_ref = J^T n / |J^T n| and reference weight w_ref = w |J^T n| / |det J|,
    // such that JxW = w_ref |det J| |J^{-T} n_ref| = w
    const AffineCellMap<dim> map(cell);
    const double             det_jacobian = std::abs(dealii::determinant(map.jacobian));
    for (const unsigned int k : points_of_cell[index])
      {
        if (is_unresolved[k])
          continue;

        dealii::Point<dim> unit_point = map.to_unit(points[k]);
        for (unsigned int d = 0; d < dim; ++d)
          unit_point[d] = std::clamp(unit_point[d], 0., 1.);

        const dealii::Tensor<1, dim> normal = dealii::transpose(map.jacobian) * normals[k];
        const double                 norm   = normal.norm();

        quadrature.push_back(unit_point, weights[k] * norm / det_jacobian, normal / norm);
      }

    return quadrature;
  }

  template <int dim, typename number>
  number
  NarrowBandInterfacePoints<dim, number>::signed_distance(const dealii::Point<dim> &x,
                                                          const double max_radius) const
  {
    std::vector<std::pair<dealii::Point<dim>, unsigned int>> indexed_points;
    indexed_points.reserve(points.size());
    for (unsigned int k = 0; k < points.size(); ++k)
      indexed_points.emplace_back(points[k], k);
    const auto point_tree = dealii::pack_rtree(indexed_points);

    return signed_distance_to_patches<dim, number>(x, max_radius, point_tree, normals, weights);
  }

  template <int dim, typename number>
  typename NarrowBandInterfacePoints<dim, number>::ProjectionStatistics
  NarrowBandInterfacePoints<dim, number>::for_each_unresolved_cell(
    const dealii::Mapping<dim>    &mapping,
    const dealii::DoFHandler<dim> &dof_handler,
    const VectorType              &level_set_advected,
    const double                   buffer,
    const unsigned int             n_subdivisions,
    const bool                     orientation_filter,
    const UnresolvedCellFunction  &function) const
  {
    Assert(located, dealii::ExcMessage("The points are not assigned to the cells."));

    // distance relative to the cell size, beyond which the signed distance must have the sign of
    // the advected level set
    constexpr double sign_check_distance = 0.25;

    ProjectionStatistics statistics;
    const MPI_Comm       mpi_comm = dof_handler.get_mpi_communicator();

    if (points.empty())
      return statistics;

    std::vector<std::pair<dealii::Point<dim>, unsigned int>> indexed_points;
    indexed_points.reserve(points.size());
    for (unsigned int k = 0; k < points.size(); ++k)
      indexed_points.emplace_back(points[k], k);
    const auto point_tree = dealii::pack_rtree(indexed_points);

    // unresolved points (a point of opposite orientation closer than the unresolved distance),
    // evaluated on demand with the cell size of the querying cell (-1: unknown)
    std::vector<signed char>                                 unresolved(points.size(), -1);
    std::vector<std::pair<dealii::Point<dim>, unsigned int>> candidates, neighbors;
    const auto is_unresolved_point = [&](const unsigned int k, const double h) {
      if (unresolved[k] < 0)
        {
          const double       distance = unresolved_distance * h;
          dealii::Point<dim> lower = points[k], upper = points[k];
          for (unsigned int d = 0; d < dim; ++d)
            {
              lower[d] -= distance;
              upper[d] += distance;
            }
          neighbors.clear();
          point_tree.query(boost::geometry::index::intersects(
                             dealii::BoundingBox<dim>(std::make_pair(lower, upper))),
                           std::back_inserter(neighbors));
          unresolved[k] = 0;
          for (const auto &[point, j] : neighbors)
            if (point.distance(points[k]) < distance and normals[j] * normals[k] < 0.)
              {
                unresolved[k] = 1;
                break;
              }
        }
      return unresolved[k] == 1;
    };

    std::unique_ptr<dealii::FEValues<dim>> fe_values;
    dealii::Vector<double>                 values_advected;
    std::vector<double>                    distances;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
          continue;

        // cells closer than buffer h to an unresolved point
        const double h          = cell->minimum_vertex_distance();
        auto         box        = cell->bounding_box();
        bool         unresolved = false;
        box.extend(buffer * h);
        candidates.clear();
        point_tree.query(boost::geometry::index::intersects(box), std::back_inserter(candidates));
        for (const auto &[point, k] : candidates)
          if (is_unresolved_point(k, h))
            {
              unresolved = true;
              break;
            }
        if (not unresolved)
          continue;

        const auto        &fe     = cell->get_fe();
        const unsigned int n_dofs = fe.n_dofs_per_cell();
        if (not fe_values or fe_values->get_fe().n_dofs_per_cell() != n_dofs)
          fe_values = std::make_unique<dealii::FEValues<dim>>(
            mapping,
            fe,
            dealii::QIterated<dim>(dealii::QGauss<1>(fe.degree + 1), n_subdivisions),
            dealii::update_values | dealii::update_quadrature_points | dealii::update_JxW_values);
        fe_values->reinit(cell);
        const unsigned int n_q_points = fe_values->n_quadrature_points;

        values_advected.reinit(n_dofs);
        cell->get_dof_values(level_set_advected, values_advected);

        // points of the cell with one orientation (pairwise n_i . n_j > 0), i.e., one branch of
        // the interface in the cell
        const auto &cell_points     = points_of_cell[cell->active_cell_index()];
        bool        one_orientation = not cell_points.empty();
        for (unsigned int i = 0; i < cell_points.size() and one_orientation; ++i)
          for (unsigned int j = i + 1; j < cell_points.size(); ++j)
            if (normals[cell_points[i]] * normals[cell_points[j]] <= 0.)
              {
                one_orientation = false;
                break;
              }

        // orientation filter: the distance is only computed to the points of this orientation,
        // i.e., to the branch of the interface in this cell, and not to a close branch of opposite
        // orientation in a neighbor cell
        std::optional<dealii::Tensor<1, dim>> orientation;
        if (orientation_filter and one_orientation)
          {
            dealii::Tensor<1, dim> sum;
            for (const unsigned int k : cell_points)
              sum += normals[k];
            if (sum.norm() > 0.)
              {
                orientation = sum / sum.norm();
                ++statistics.n_filtered_cells;
              }
          }

        // signed distance at x with the advected level set at x (sign beyond the edges of the
        // patches)
        const auto distance = [&](const dealii::Point<dim> &x, const double advected) {
          return signed_distance_to_patches<dim, number>(x,
                                                         h,
                                                         point_tree,
                                                         normals,
                                                         weights,
                                                         advected > 0. ? 1. :
                                                                         (advected < 0. ? -1. : 0.),
                                                         orientation ? &*orientation : nullptr);
        };

        // signed distance at the quadrature points; sign check against the advected level set
        // away from the points
        distances.resize(n_q_points);
        bool consistent = true;
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            double advected = 0.;
            for (unsigned int i = 0; i < n_dofs; ++i)
              advected += values_advected[i] * fe_values->shape_value(i, q);

            distances[q] = distance(fe_values->quadrature_point(q), advected);

            if (std::abs(distances[q]) > sign_check_distance * h and advected * distances[q] < 0.)
              consistent = false;
          }

        if (not consistent)
          {
            ++statistics.n_rejected_cells;
            continue;
          }

        // signed distance at other points of the cell (unit coordinates)
        const CellDistanceFunction distance_function = [&](const dealii::Point<dim> &unit_point) {
          double advected = 0.;
          for (unsigned int i = 0; i < n_dofs; ++i)
            advected += values_advected[i] * fe.shape_value(i, unit_point);
          return distance(mapping.transform_unit_to_real_cell(cell, unit_point), advected);
        };

        function(cell, *fe_values, distances, cell_points, one_orientation, distance_function);

        ++statistics.n_cells;
      }

    statistics.n_cells = dealii::Utilities::MPI::sum(statistics.n_cells, mpi_comm);
    statistics.n_rejected_cells =
      dealii::Utilities::MPI::sum(statistics.n_rejected_cells, mpi_comm);
    statistics.n_filtered_cells =
      dealii::Utilities::MPI::sum(statistics.n_filtered_cells, mpi_comm);
    return statistics;
  }

  template <int dim, typename number>
  typename NarrowBandInterfacePoints<dim, number>::ProjectionStatistics
  NarrowBandInterfacePoints<dim, number>::project_signed_distance(
    const dealii::Mapping<dim>    &mapping,
    const dealii::DoFHandler<dim> &dof_handler,
    const VectorType              &level_set_advected,
    VectorType                    &level_set,
    const double                   buffer,
    const unsigned int             n_subdivisions,
    VectorType                    *projected_dofs,
    const bool                     orientation_filter,
    const double                   pinning_penalty) const
  {
    dealii::FullMatrix<double> mass_matrix;
    dealii::Vector<double>     rhs, values_new;
    unsigned int               n_pinned_cells = 0;

    auto statistics = for_each_unresolved_cell(
      mapping,
      dof_handler,
      level_set_advected,
      buffer,
      n_subdivisions,
      orientation_filter,
      [&](const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
          const dealii::FEValues<dim>                                  &fe_values,
          const std::vector<double>                                    &distances,
          const std::vector<unsigned int>                              &cell_points,
          const bool                                                    one_orientation,
          const CellDistanceFunction &) {
        const auto        &fe         = cell->get_fe();
        const unsigned int n_dofs     = fe.n_dofs_per_cell();
        const unsigned int n_q_points = fe_values.n_quadrature_points;
        const double       h          = cell->minimum_vertex_distance();

        // L2-projection onto the polynomials of the cell
        mass_matrix.reinit(n_dofs, n_dofs);
        rhs.reinit(n_dofs);
        for (unsigned int q = 0; q < n_q_points; ++q)
          for (unsigned int i = 0; i < n_dofs; ++i)
            {
              const double value_i = fe_values.shape_value(i, q) * fe_values.JxW(q);
              rhs[i] += value_i * distances[q];
              for (unsigned int j = 0; j < n_dofs; ++j)
                mass_matrix(i, j) += value_i * fe_values.shape_value(j, q);
            }

        // pinned projection: penalty c h sum_k w_k phi(x_k)^2 of the zero level set at the points
        // of a cell with one branch of the interface
        if (pinning_penalty > 0. and one_orientation)
          {
            std::vector<double> shape_values(n_dofs);
            for (const unsigned int k : cell_points)
              {
                const auto   unit_point = mapping.transform_real_to_unit_cell(cell, points[k]);
                const double factor     = pinning_penalty * h * weights[k];
                for (unsigned int i = 0; i < n_dofs; ++i)
                  shape_values[i] = fe.shape_value(i, unit_point);
                for (unsigned int i = 0; i < n_dofs; ++i)
                  for (unsigned int j = 0; j < n_dofs; ++j)
                    mass_matrix(i, j) += factor * shape_values[i] * shape_values[j];
              }
            ++n_pinned_cells;
          }

        mass_matrix.gauss_jordan();
        values_new.reinit(n_dofs);
        mass_matrix.vmult(values_new, rhs);
        cell->set_dof_values(values_new, level_set);
        if (projected_dofs != nullptr)
          {
            values_new = 1.;
            cell->set_dof_values(values_new, *projected_dofs);
          }
      });

    statistics.n_pinned_cells =
      dealii::Utilities::MPI::sum(n_pinned_cells, dof_handler.get_mpi_communicator());
    return statistics;
  }

  template <int dim, typename number>
  const std::vector<dealii::Point<dim>> &
  NarrowBandInterfacePoints<dim, number>::get_points() const
  {
    return points;
  }

  template <int dim, typename number>
  const std::vector<number> &
  NarrowBandInterfacePoints<dim, number>::get_weights() const
  {
    return weights;
  }

  template <int dim, typename number>
  const std::vector<dealii::Tensor<1, dim>> &
  NarrowBandInterfacePoints<dim, number>::get_normals() const
  {
    return normals;
  }

  template class NarrowBandInterfacePoints<1, double>;
  template class NarrowBandInterfacePoints<2, double>;
  template class NarrowBandInterfacePoints<3, double>;
} // namespace MeltPoolDG::LevelSet
