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
#include <cmath>
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
     * NarrowBandInterfacePoints::signed_distance()), using the eight nearest points of @p tree. If
     * the closest point of the closest patch lies on its edge, i.e., @p x does not lie in front of
     * the patch (e.g., beyond the open end of a chain of points, whose cap has been removed), the
     * side of the patch is not defined, and the sign is @p fallback_sign (if not zero).
     */
    template <int dim, typename number, typename TreeType>
    number
    signed_distance_to_patches(const dealii::Point<dim>                  &x,
                               const double                               max_radius,
                               const TreeType                            &tree,
                               const std::vector<dealii::Tensor<1, dim>> &normals,
                               const std::vector<number>                 &weights,
                               const double                               fallback_sign = 0.)
    {
      std::vector<std::pair<dealii::Point<dim>, unsigned int>> nearest;
      tree.query(boost::geometry::index::nearest(x, 8), std::back_inserter(nearest));

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
    const double                      support_tolerance)
    : pcout(pcout)
    , unresolved_distance(unresolved_distance)
    , support_tolerance(support_tolerance)
  {
    AssertThrow(unresolved_distance >= 0. and support_tolerance > 0.,
                dealii::ExcMessage("The unresolved distance must not be negative, and the support "
                                   "tolerance must be positive."));
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

    const number h = time_step;
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

    velocity.set_time(old_time);

    points_of_cell.clear();
    is_unresolved.clear();
    located = false;
  }

  template <int dim, typename number>
  void
  NarrowBandInterfacePoints<dim, number>::assign_points_to_cells(
    const dealii::DoFHandler<dim> &dof_handler_band)
  {
    // cell of the band containing each point (locally owned and ghost cells): on the common
    // boundary of several cells, the cell with the smallest CellId (unique on all processes)
    struct Location
    {
      dealii::CellId cell_id;
      unsigned int   active_cell_index;
      bool           is_locally_owned;
    };
    std::vector<std::optional<Location>> locations(points.size());

    std::vector<std::pair<dealii::Point<dim>, unsigned int>> indexed_points;
    indexed_points.reserve(points.size());
    for (unsigned int k = 0; k < points.size(); ++k)
      indexed_points.emplace_back(points[k], k);
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

    points_of_cell.assign(dof_handler_band.get_triangulation().n_active_cells(), {});
    for (unsigned int k = 0; k < points.size(); ++k)
      if (locations[k] and locations[k]->is_locally_owned)
        points_of_cell[locations[k]->active_cell_index].push_back(k);
  }

  template <int dim, typename number>
  typename NarrowBandInterfacePoints<dim, number>::Statistics
  NarrowBandInterfacePoints<dim, number>::locate(const dealii::DoFHandler<dim> &dof_handler_band,
                                                 const VectorType              &level_set_band,
                                                 const unsigned int             n_points_1d,
                                                 const std::vector<bool>       &is_cut_cell)
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
              status_local[k] = is_supported ? 1 : 2;
            }
        }
    }
    dealii::Utilities::MPI::sum(status_local, mpi_comm, status);

    {
      unsigned int n_kept = 0;
      for (unsigned int k = 0; k < points.size(); ++k)
        {
          if (status[k] == 0)
            ++statistics.n_points_outside;
          else if (status[k] == 2)
            ++statistics.n_unsupported_points;
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
    // point set due to the stretching of the interface), merging of the closest pairs of points in
    // cells with too many points (compression of the interface)
    const unsigned int max_n_points_per_cell = 2 * dealii::Utilities::pow(n_points_1d, dim - 1);

    std::vector<double>       new_points_local;
    std::vector<unsigned int> merged_pairs_local; // (removed point, kept point)
    {
      // Note: the quadrature generator keeps a pointer to the quadrature collection.
      const dealii::hp::QCollection<1> q_collection{dealii::QGauss<1>(n_points_1d)};
      dealii::NonMatching::DiscreteQuadratureGenerator<dim> quadrature_generator(q_collection,
                                                                                 dof_handler_band,
                                                                                 level_set_band);

      std::vector<std::pair<dealii::Point<dim>, unsigned int>> indexed_points;
      indexed_points.reserve(points.size());
      for (unsigned int k = 0; k < points.size(); ++k)
        indexed_points.emplace_back(points[k], k);
      const auto point_tree = dealii::pack_rtree(indexed_points);

      std::vector<std::pair<dealii::Point<dim>, unsigned int>> nearest;
      std::vector<unsigned int>                                active;
      for (const auto &cell : dof_handler_band.active_cell_iterators())
        {
          if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
            continue;

          if (is_cut_cell[cell->active_cell_index()])
            {
              const double min_distance = cell->minimum_vertex_distance() / n_points_1d;
              pack_surface_points(
                cell, quadrature_generator, new_points_local, [&](const dealii::Point<dim> &x) {
                  nearest.clear();
                  point_tree.query(boost::geometry::index::nearest(x, 1),
                                   std::back_inserter(nearest));
                  return nearest.empty() or nearest[0].first.distance(x) > min_distance;
                });
            }

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
    append_points(new_points_local, mpi_comm);
    statistics.n_inserted_points = points.size() - n_points_old;

    // 5) assignment of the updated points
    assign_points_to_cells(dof_handler_band);

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

    {
      std::ostringstream str;
      str << "interface points: " << statistics.n_points << " (" << statistics.n_inserted_points
          << " new, " << statistics.n_merged_points << " merged, " << statistics.n_points_outside
          << " outside of the band, " << statistics.n_unsupported_points << " not supported), "
          << statistics.n_points_in_non_cut_cells << " in cells, which are not cut, "
          << statistics.n_cut_cells_without_points << " cut cells without points, "
          << statistics.n_unresolved_points << " unresolved points";
      Journal::print_line(pcout, str.str(), "narrow_band");
    }

    located = true;
    return statistics;
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
  NarrowBandInterfacePoints<dim, number>::project_signed_distance(
    const dealii::Mapping<dim>    &mapping,
    const dealii::DoFHandler<dim> &dof_handler,
    const VectorType              &level_set_advected,
    VectorType                    &level_set,
    const double                   buffer,
    const unsigned int             n_subdivisions) const
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
    dealii::FullMatrix<double>             mass_matrix;
    dealii::Vector<double>                 rhs, values_advected, values_new;
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

        // signed distance at the quadrature points (sign of the advected level set beyond the
        // edges of the patches); sign check against the advected level set away from the points
        distances.resize(n_q_points);
        bool consistent = true;
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            double advected = 0.;
            for (unsigned int i = 0; i < n_dofs; ++i)
              advected += values_advected[i] * fe_values->shape_value(i, q);

            distances[q] =
              signed_distance_to_patches<dim, number>(fe_values->quadrature_point(q),
                                                      h,
                                                      point_tree,
                                                      normals,
                                                      weights,
                                                      advected > 0. ? 1. :
                                                                      (advected < 0. ? -1. : 0.));

            if (std::abs(distances[q]) > sign_check_distance * h and advected * distances[q] < 0.)
              consistent = false;
          }

        if (not consistent)
          {
            ++statistics.n_rejected_cells;
            continue;
          }

        // L2-projection onto the polynomials of the cell
        mass_matrix.reinit(n_dofs, n_dofs);
        rhs.reinit(n_dofs);
        for (unsigned int q = 0; q < n_q_points; ++q)
          for (unsigned int i = 0; i < n_dofs; ++i)
            {
              const double value_i = fe_values->shape_value(i, q) * fe_values->JxW(q);
              rhs[i] += value_i * distances[q];
              for (unsigned int j = 0; j < n_dofs; ++j)
                mass_matrix(i, j) += value_i * fe_values->shape_value(j, q);
            }
        mass_matrix.gauss_jordan();
        values_new.reinit(n_dofs);
        mass_matrix.vmult(values_new, rhs);
        cell->set_dof_values(values_new, level_set);

        ++statistics.n_cells;
      }

    statistics.n_cells = dealii::Utilities::MPI::sum(statistics.n_cells, mpi_comm);
    statistics.n_rejected_cells =
      dealii::Utilities::MPI::sum(statistics.n_rejected_cells, mpi_comm);
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

  template class NarrowBandInterfacePoints<1, double>;
  template class NarrowBandInterfacePoints<2, double>;
  template class NarrowBandInterfacePoints<3, double>;
} // namespace MeltPoolDG::LevelSet
