#include <deal.II/base/bounding_box.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/vector.h>

#include <deal.II/non_matching/quadrature_generator.h>

#include <deal.II/numerics/rtree.h>

#include <meltpooldg/level_set/narrow_band_closest_point_tools.hpp>
#include <meltpooldg/level_set/narrow_band_semi_lagrangian_advection.hpp>
#include <meltpooldg/level_set/narrow_band_utilities.hpp>
#include <meltpooldg/utilities/journal.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iterator>
#include <limits>
#include <sstream>
#include <vector>

namespace MeltPoolDG::LevelSet
{
  template <int dim, typename number>
  NarrowBandSemiLagrangianAdvection<dim, number>::NarrowBandSemiLagrangianAdvection(
    const dealii::ConditionalOStream &pcout)
    : pcout(pcout)
  {}

  template <int dim, typename number>
  typename NarrowBandSemiLagrangianAdvection<dim, number>::Statistics
  NarrowBandSemiLagrangianAdvection<dim, number>::advect(
    const dealii::DoFHandler<dim> &dof_handler_cut,
    const VectorType              &level_set_cut,
    const dealii::DoFHandler<dim> &dof_handler_band,
    VectorType                    &level_set_band,
    const std::vector<int>        &cell_signs,
    dealii::Function<dim>         &velocity,
    const number                   time,
    const number                   time_step) const
  {
    using namespace NarrowBand;
    using namespace NarrowBand::internal;

    AssertThrow(velocity.n_components == dim,
                dealii::ExcMessage("The velocity function needs dim components."));
    AssertDimension(cell_signs.size(), dof_handler_band.get_triangulation().n_active_cells());
    AssertThrow(not dof_handler_cut.get_triangulation().has_hanging_nodes(),
                dealii::ExcMessage("The semi-Lagrangian advection does not support hanging "
                                   "nodes."));

    const auto &fe_cut = dof_handler_cut.get_fe_collection()[ActiveFEIndex::cut];
    AssertThrow(fe_cut.n_dofs_per_cell() > 0,
                dealii::ExcMessage("The semi-Lagrangian advection requires the cut DoFHandler."));

    const unsigned int            fe_degree = fe_cut.degree;
    const unsigned int            n_dofs    = fe_cut.n_dofs_per_cell();
    const TensorProductBasis<dim> basis(fe_cut);
    const MPI_Comm                mpi_comm = dof_handler_cut.get_mpi_communicator();

    // 1) Data of the locally owned cut cells (cell size, origin, Jacobian, DoF values and
    // quadrature points of the interface), packed into a vector.
    std::vector<double> local_data;
    {
      const bool update_ghosts = not level_set_cut.has_ghost_elements();
      if (update_ghosts)
        level_set_cut.update_ghost_values();

      const dealii::hp::QCollection<1> q_collection(dealii::QGauss<1>(fe_degree + 1));
      dealii::NonMatching::DiscreteQuadratureGenerator<dim> quadrature_generator(q_collection,
                                                                                 dof_handler_cut,
                                                                                 level_set_cut);
      dealii::Vector<number>                                values(n_dofs);

      for (const auto &cell : dof_handler_cut.active_cell_iterators())
        {
          if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
            continue;

          quadrature_generator.generate(cell);
          const auto &points = quadrature_generator.get_surface_quadrature().get_points();

          const AffineCellMap<dim> map(cell);
          cell->get_dof_values(level_set_cut, values);

          local_data.push_back(map.cell_size);
          for (unsigned int d = 0; d < dim; ++d)
            local_data.push_back(map.origin[d]);
          for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
              local_data.push_back(map.jacobian[i][j]);
          for (unsigned int i = 0; i < n_dofs; ++i)
            local_data.push_back(values[i]);
          local_data.push_back(points.size());
          for (const auto &unit_point : points)
            {
              const dealii::Point<dim> point = map.to_real(unit_point);
              for (unsigned int d = 0; d < dim; ++d)
                local_data.push_back(point[d]);
            }
        }

      if (update_ghosts)
        level_set_cut.zero_out_ghost_values();
    }

    // 2) Data of all cut cells of all processes, bounding boxes of the cut cells and quadrature
    // points of the interface.
    std::vector<CutCellData<dim, number>>                                  cut_cells;
    std::vector<std::pair<dealii::Point<dim>, unsigned int>>               interface_points;
    std::vector<std::pair<dealii::BoundingBox<dim, double>, unsigned int>> boxes;
    for (const auto &data_of_process : dealii::Utilities::MPI::all_gather(mpi_comm, local_data))
      for (std::size_t k = 0; k < data_of_process.size();)
        {
          const double       cell_size = data_of_process[k++];
          dealii::Point<dim> origin;
          for (unsigned int d = 0; d < dim; ++d)
            origin[d] = data_of_process[k++];
          dealii::Tensor<2, dim> jacobian;
          for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
              jacobian[i][j] = data_of_process[k++];

          CutCellData<dim, number> cut_cell{AffineCellMap<dim>(origin, jacobian, cell_size),
                                            dealii::Vector<number>(n_dofs),
                                            {}};
          for (unsigned int i = 0; i < n_dofs; ++i)
            cut_cell.values[i] = data_of_process[k++];

          const unsigned int n_points = static_cast<unsigned int>(data_of_process[k++]);
          for (unsigned int q = 0; q < n_points; ++q)
            {
              dealii::Point<dim> point;
              for (unsigned int d = 0; d < dim; ++d)
                point[d] = data_of_process[k++];
              cut_cell.interface_points.push_back(point);
              interface_points.emplace_back(point, cut_cells.size());
            }

          // bounding box of the parallelepiped (slightly enlarged)
          dealii::Point<dim> lower = origin, upper = origin;
          for (unsigned int v = 0; v < dealii::GeometryInfo<dim>::vertices_per_cell; ++v)
            {
              dealii::Point<dim> unit_vertex;
              for (unsigned int d = 0; d < dim; ++d)
                unit_vertex[d] = (v >> d) & 1U;
              const dealii::Point<dim> vertex = cut_cell.map.to_real(unit_vertex);
              for (unsigned int d = 0; d < dim; ++d)
                {
                  lower[d] = std::min(lower[d], vertex[d]);
                  upper[d] = std::max(upper[d], vertex[d]);
                }
            }
          dealii::BoundingBox<dim, double> box(std::make_pair(lower, upper));
          box.extend(1e-10 * cell_size);
          boxes.emplace_back(box, cut_cells.size());

          cut_cells.emplace_back(std::move(cut_cell));
        }

    Statistics statistics;

    if (interface_points.empty())
      return statistics;

    const auto box_tree   = dealii::pack_rtree(boxes);
    const auto point_tree = dealii::pack_rtree(interface_points);

    // number of nearest quadrature points of the interface, whose cut cells are candidates of the
    // closest-point search
    const unsigned int n_nearest_points = dim == 1 ? 2 : (dim == 2 ? 8 : 24);

    std::vector<std::pair<dealii::Point<dim>, unsigned int>>               nearest_points;
    std::vector<std::pair<dealii::BoundingBox<dim, double>, unsigned int>> containing_boxes;
    std::vector<unsigned int>                                              candidates;

    // closest point on the interface: first-order Taylor expansion of the level set from the
    // closest point y and the scaled distance |grad phi(y)| d, where d = |x - y| + |phi(y)| /
    // |grad phi(y)| is the estimate of the distance of the closest-point search. Both coincide for
    // a converged closest point; at an open end of the discrete interface, the Taylor expansion
    // vanishes on the continuation of the tangent, while the distance to the end point does not.
    const auto closest_point = [&](const dealii::Point<dim> &x) {
      nearest_points.clear();
      point_tree.query(boost::geometry::index::nearest(x, n_nearest_points),
                       std::back_inserter(nearest_points));

      candidates.clear();
      for (const auto &point : nearest_points)
        candidates.push_back(point.second);
      std::sort(candidates.begin(), candidates.end());
      candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

      number taylor_value = 0., scaled_distance = 0.;
      number min_distance = std::numeric_limits<number>::max();
      for (const unsigned int c : candidates)
        {
          const auto [y, distance, n_iterations] = search_closest_point(basis, cut_cells[c], x);
          (void)n_iterations;
          if (distance < min_distance)
            {
              min_distance                 = distance;
              const auto [value, gradient] = evaluate(basis, cut_cells[c], y);
              taylor_value                 = value + gradient * (x - y);
              scaled_distance              = gradient.norm() * distance;
            }
        }
      return std::make_pair(taylor_value, scaled_distance);
    };

    const auto is_inside = [](const dealii::Point<dim> &unit_point, const double tolerance) {
      for (unsigned int d = 0; d < dim; ++d)
        if (unit_point[d] < -tolerance or unit_point[d] > 1. + tolerance)
          return false;
      return true;
    };

    // cells sharing a vertex (for the location of the foot points in the cells, which are not
    // cut)
    const auto vertex_to_cells =
      dealii::GridTools::vertex_to_cell_map(dof_handler_band.get_triangulation());

    // backward integration of the characteristic dx/dt = u(x, t) from t^{n+1} to t^n with the
    // classical fourth-order Runge-Kutta method
    const number old_time = velocity.get_time();
    const auto   u        = [&](const dealii::Point<dim> &x, const number t) {
      velocity.set_time(t);
      dealii::Tensor<1, dim> result;
      for (unsigned int d = 0; d < dim; ++d)
        result[d] = velocity.value(x, d);
      return result;
    };
    const auto foot_point = [&](const dealii::Point<dim> &x) {
      const number h  = -time_step;
      const number t1 = time + time_step;
      const auto   k1 = u(x, t1);
      const auto   k2 = u(x + 0.5 * h * k1, t1 + 0.5 * h);
      const auto   k3 = u(x + 0.5 * h * k2, t1 + 0.5 * h);
      const auto   k4 = u(x + h * k3, t1 + h);
      return dealii::Point<dim>(x + (h / 6.) * (k1 + 2. * k2 + 2. * k3 + k4));
    };

    // 3) Evaluation of the back-traced level set at the nodes of the locally owned candidate cells
    for (const auto &cell : dof_handler_band.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
          continue;

        AssertThrow(cell->get_fe().has_support_points() and
                      cell->get_fe().n_dofs_per_cell() == n_dofs,
                    dealii::ExcMessage("The semi-Lagrangian advection requires the same nodal "
                                       "element on all candidate cells."));

        const auto              &unit_support_points = cell->get_fe().get_unit_support_points();
        const AffineCellMap<dim> map(cell);
        dealii::Vector<number>   values(n_dofs);

        // cells sharing a vertex with the candidate cell (including the cell itself)
        std::vector<typename dealii::Triangulation<dim>::active_cell_iterator> neighborhood;
        for (const unsigned int v : cell->vertex_indices())
          for (const auto &other : vertex_to_cells[cell->vertex_index(v)])
            if (std::find(neighborhood.begin(), neighborhood.end(), other) == neighborhood.end())
              neighborhood.push_back(other);

        for (unsigned int i = 0; i < n_dofs; ++i)
          {
            const dealii::Point<dim> x      = map.to_real(unit_support_points[i]);
            const dealii::Point<dim> x_foot = foot_point(x);

            ++statistics.n_nodes;
            statistics.max_displacement =
              std::max<number>(statistics.max_displacement, x.distance(x_foot) / map.cell_size);

            // (a) foot point in a cut cell: polynomial of the cut cell. On the common boundary of
            // several cut cells, the cell of the node is preferred: the level set may jump across
            // the faces, and a vanishing displacement reproduces the old polynomial.
            containing_boxes.clear();
            box_tree.query(boost::geometry::index::intersects(x_foot),
                           std::back_inserter(containing_boxes));

            int located_cell = -1;
            for (const auto &box : containing_boxes)
              {
                const auto &cut_cell = cut_cells[box.second];
                if (not is_inside(cut_cell.map.to_unit(x_foot), 1e-10))
                  continue;

                if (located_cell < 0)
                  located_cell = box.second;
                if (cut_cell.map.origin.distance(map.origin) < 1e-10 * map.cell_size)
                  {
                    located_cell = box.second;
                    break;
                  }
              }
            if (located_cell >= 0)
              {
                const auto            &cut_cell = cut_cells[located_cell];
                double                 value    = 0.;
                dealii::Tensor<1, dim> gradient;
                basis.evaluate(
                  cut_cell.values, cut_cell.map.to_unit(x_foot), value, gradient, nullptr);
                values[i] = value;
                continue;
              }

            // (b) foot point in a cell, which is not cut: distance to the interface with the sign
            // of the cell
            ++statistics.n_nodes_outside;
            const auto [taylor_value, scaled_distance] = closest_point(x_foot);

            int sign = 0;
            for (const auto &other : neighborhood)
              if (is_inside(AffineCellMap<dim>(other).to_unit(x_foot), 1e-10))
                {
                  sign = cell_signs[other->active_cell_index()];
                  break;
                }

            if (sign == 0)
              {
                // not located or unknown sign: sign of the Taylor expansion
                ++statistics.n_nodes_not_located;
                values[i] = (taylor_value >= 0. ? 1. : -1.) * scaled_distance;
              }
            else
              values[i] = sign * scaled_distance;
          }

        cell->set_dof_values(values, level_set_band);
      }

    velocity.set_time(old_time);

    statistics.n_nodes         = dealii::Utilities::MPI::sum(statistics.n_nodes, mpi_comm);
    statistics.n_nodes_outside = dealii::Utilities::MPI::sum(statistics.n_nodes_outside, mpi_comm);
    statistics.n_nodes_not_located =
      dealii::Utilities::MPI::sum(statistics.n_nodes_not_located, mpi_comm);
    statistics.max_displacement =
      dealii::Utilities::MPI::max(statistics.max_displacement, mpi_comm);

    {
      std::ostringstream str;
      str << "semi-Lagrangian advection: " << statistics.n_nodes << " nodes ("
          << statistics.n_nodes_outside << " foot points outside of the cut cells";
      if (statistics.n_nodes_not_located > 0)
        str << ", " << statistics.n_nodes_not_located << " not located";
      str << "), max. displacement / h: " << std::scientific << std::setprecision(2)
          << statistics.max_displacement;
      Journal::print_line(pcout, str.str(), "narrow_band");
    }

    return statistics;
  }

  template class NarrowBandSemiLagrangianAdvection<1, double>;
  template class NarrowBandSemiLagrangianAdvection<2, double>;
  template class NarrowBandSemiLagrangianAdvection<3, double>;
} // namespace MeltPoolDG::LevelSet
