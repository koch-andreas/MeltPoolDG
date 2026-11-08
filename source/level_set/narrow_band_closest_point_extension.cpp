#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/vector.h>

#include <deal.II/non_matching/quadrature_generator.h>

#include <deal.II/numerics/rtree.h>

#include <meltpooldg/level_set/narrow_band_closest_point_extension.hpp>
#include <meltpooldg/level_set/narrow_band_closest_point_tools.hpp>
#include <meltpooldg/level_set/narrow_band_utilities.hpp>
#include <meltpooldg/utilities/journal.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <limits>
#include <sstream>
#include <vector>

namespace MeltPoolDG::LevelSet
{
  using namespace NarrowBand::internal;

  template <int dim, typename number>
  NarrowBandClosestPointExtension<dim, number>::NarrowBandClosestPointExtension(
    const typename NarrowBandData<number>::Extension &data,
    const dealii::Mapping<dim>                       &mapping,
    const dealii::ConditionalOStream                 &pcout)
    : data(data)
    , mapping(mapping)
    , pcout(pcout)
  {}

  template <int dim, typename number>
  unsigned int
  NarrowBandClosestPointExtension<dim, number>::extend(const dealii::DoFHandler<dim> &dof_handler,
                                                       VectorType &level_set) const
  {
    using namespace NarrowBand;

    const auto &fe_collection = dof_handler.get_fe_collection();
    AssertThrow(dof_handler.has_hp_capabilities() and fe_collection.size() == 3 and
                  fe_collection[ActiveFEIndex::cut].n_dofs_per_cell() > 0 and
                  fe_collection[ActiveFEIndex::neighbor].n_dofs_per_cell() > 0,
                dealii::ExcMessage("The closest-point extension requires the band DoFHandler of "
                                   "the narrow-band level set."));
    AssertThrow(not dof_handler.get_triangulation().has_hanging_nodes(),
                dealii::ExcMessage("The narrow-band level set does not support hanging nodes."));

    const auto        &fe_cut      = fe_collection[ActiveFEIndex::cut];
    const auto        &fe_neighbor = fe_collection[ActiveFEIndex::neighbor];
    const unsigned int fe_degree   = fe_cut.degree;

    AssertThrow(fe_neighbor.has_support_points(),
                dealii::ExcMessage("The closest-point extension requires a nodal element."));

    const TensorProductBasis<dim> basis(fe_cut);

    const MPI_Comm mpi_comm = dof_handler.get_mpi_communicator();

    // The values of the cut cells of the ghost layer are required as well.
    VectorType level_set_relevant(dof_handler.locally_owned_dofs(),
                                  dealii::DoFTools::extract_locally_relevant_dofs(dof_handler),
                                  mpi_comm);
    level_set_relevant.copy_locally_owned_data_from(level_set);
    level_set_relevant.update_ghost_values();

    // 1) Collect the data of the locally relevant cut cells (locally owned and ghost cells): DoF
    // values, affine map and quadrature points of the discrete interface (starting points of the
    // closest-point search).
    std::vector<CutCellData<dim, number>>  cut_cells;
    std::vector<std::vector<unsigned int>> vertex_to_cut_cells(
      dof_handler.get_triangulation().n_vertices());
    {
      const dealii::hp::QCollection<1> q_collection(dealii::QGauss<1>(fe_degree + 1));
      dealii::NonMatching::DiscreteQuadratureGenerator<dim> quadrature_generator(
        q_collection, dof_handler, level_set_relevant);

      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (cell->is_artificial() or cell->active_fe_index() != ActiveFEIndex::cut)
            continue;

          CutCellData<dim, number> cut_cell{AffineCellMap<dim>(cell),
                                            dealii::Vector<number>(fe_cut.n_dofs_per_cell()),
                                            {}};
          cell->get_dof_values(level_set_relevant, cut_cell.values);

          quadrature_generator.generate(cell);
          for (const auto &unit_point : quadrature_generator.get_surface_quadrature().get_points())
            cut_cell.interface_points.push_back(cut_cell.map.to_real(unit_point));

          // volume fraction of the positive phase (reference cell with unit volume)
          if (data.minimum_source_volume_fraction > 0.)
            {
              number volume_fraction = 0.;
              for (const auto weight : quadrature_generator.get_outside_quadrature().get_weights())
                volume_fraction += weight;
              cut_cell.is_small = volume_fraction < data.minimum_source_volume_fraction;
            }

          // Note: a cut cell might not contain quadrature points of the interface after the
          // reinitialization. Such cells are still considered, the search starts from the point
          // of the cell nearest to the node.
          for (const unsigned int v : cell->vertex_indices())
            vertex_to_cut_cells[cell->vertex_index(v)].push_back(cut_cells.size());

          cut_cells.emplace_back(std::move(cut_cell));
        }
    }

    // 2) Closest-point search for the nodes of the locally owned neighbor cells.
    const auto &unit_support_points = fe_neighbor.get_unit_support_points();

    unsigned int max_n_iterations = 0;

    dealii::Vector<number>    values(fe_neighbor.n_dofs_per_cell());
    std::vector<unsigned int> candidates;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or cell->active_fe_index() != ActiveFEIndex::neighbor)
          continue;

        // candidates: cut cells sharing a vertex with the neighbor cell
        candidates.clear();
        for (const unsigned int v : cell->vertex_indices())
          for (const unsigned int c : vertex_to_cut_cells[cell->vertex_index(v)])
            candidates.push_back(c);
        std::sort(candidates.begin(), candidates.end());
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

        AssertThrow(not candidates.empty(),
                    dealii::ExcMessage("A neighbor cell of the narrow band does not share a vertex "
                                       "with a cut cell."));

        // small cut cells are only considered, if no other cut cells are available
        if (std::any_of(candidates.begin(), candidates.end(), [&](const unsigned int c) {
              return not cut_cells[c].is_small;
            }))
          candidates.erase(std::remove_if(candidates.begin(),
                                          candidates.end(),
                                          [&](const unsigned int c) {
                                            return cut_cells[c].is_small;
                                          }),
                           candidates.end());

        const AffineCellMap<dim> map(cell);

        for (unsigned int i = 0; i < fe_neighbor.n_dofs_per_cell(); ++i)
          {
            const dealii::Point<dim> x = map.to_real(unit_support_points[i]);

            number min_distance = std::numeric_limits<number>::max();
            for (const unsigned int c : candidates)
              {
                const auto [y, distance, n_iterations] =
                  search_closest_point(basis, cut_cells[c], x);
                max_n_iterations = std::max(max_n_iterations, n_iterations);

                if (distance < min_distance)
                  {
                    min_distance = distance;

                    // first-order Taylor expansion from the closest point
                    const auto [value, gradient] = evaluate(basis, cut_cells[c], y);
                    values[i]                    = value + gradient * (x - y);
                  }
              }
          }

        cell->set_dof_values(values, level_set);
      }

    max_n_iterations = dealii::Utilities::MPI::max(max_n_iterations, mpi_comm);

    const bool         use_sign_limiter = data.sign_limiter_tolerance >= 0.;
    const unsigned int n_limited_cells =
      use_sign_limiter ?
        limit_sign_changes(mapping, dof_handler, level_set, data.sign_limiter_tolerance) :
        0;

    // sign clamp: the extended level set does not change its sign within a neighbor cell
    const bool   use_sign_clamp   = data.sign_clamp_epsilon > 0.;
    unsigned int n_clamped_values = 0;
    if (use_sign_clamp)
      {
        std::vector<bool> is_neighbor_cell(dof_handler.get_triangulation().n_active_cells(), false);
        for (const auto &cell : dof_handler.active_cell_iterators())
          if (cell->is_locally_owned() and cell->active_fe_index() == ActiveFEIndex::neighbor)
            is_neighbor_cell[cell->active_cell_index()] = true;

        // Note: level_set_relevant holds the given values of the cut cells (including the ghost
        // cells).
        n_clamped_values = clamp_signs(
          dof_handler,
          level_set,
          compute_sides_from_cut_cells(mapping, dof_handler, level_set_relevant, is_neighbor_cell),
          data.sign_clamp_epsilon);
      }

    {
      std::ostringstream str;
      str << "closest-point extension to "
          << n_cells_with_active_fe_index(dof_handler, ActiveFEIndex::neighbor)
          << " neighbor cells: max. " << max_n_iterations << " iterations";
      if (use_sign_limiter)
        str << ", " << n_limited_cells << (use_sign_clamp ? " limited" : " limited cells");
      if (use_sign_clamp)
        str << ", " << n_clamped_values << " clamped";
      Journal::print_line(pcout, str.str(), "narrow_band");
    }

    return max_n_iterations;
  }

  template <int dim, typename number>
  unsigned int
  NarrowBandClosestPointExtension<dim, number>::redistance(
    const dealii::DoFHandler<dim> &dof_handler,
    VectorType                    &level_set,
    const std::vector<bool>       &cell_flags,
    const std::string             &label) const
  {
    using namespace NarrowBand;

    const auto &fe_collection = dof_handler.get_fe_collection();
    AssertThrow(dof_handler.has_hp_capabilities() and fe_collection.size() > ActiveFEIndex::cut and
                  fe_collection[ActiveFEIndex::cut].n_dofs_per_cell() > 0,
                dealii::ExcMessage("The redistancing requires the band DoFHandler of the "
                                   "narrow-band level set."));
    AssertDimension(cell_flags.size(), dof_handler.get_triangulation().n_active_cells());

    const auto        &fe_cut    = fe_collection[ActiveFEIndex::cut];
    const unsigned int fe_degree = fe_cut.degree;
    const unsigned int n_dofs    = fe_cut.n_dofs_per_cell();

    const TensorProductBasis<dim> basis(fe_cut);

    const MPI_Comm mpi_comm = dof_handler.get_mpi_communicator();

    // 1) Data of the locally owned cut cells (cell size, origin, Jacobian, DoF values and
    // quadrature points of the interface), packed into a vector.
    std::vector<double> local_data;
    {
      const bool update_ghosts = not level_set.has_ghost_elements();
      if (update_ghosts)
        level_set.update_ghost_values();

      const dealii::hp::QCollection<1> q_collection(dealii::QGauss<1>(fe_degree + 1));
      dealii::NonMatching::DiscreteQuadratureGenerator<dim> quadrature_generator(q_collection,
                                                                                 dof_handler,
                                                                                 level_set);
      dealii::Vector<number>                                values(n_dofs);

      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (not cell->is_locally_owned() or cell->active_fe_index() != ActiveFEIndex::cut)
            continue;

          quadrature_generator.generate(cell);
          const auto &points = quadrature_generator.get_surface_quadrature().get_points();
          if (points.empty())
            continue;

          const AffineCellMap<dim> map(cell);
          cell->get_dof_values(level_set, values);

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
        level_set.zero_out_ghost_values();
    }

    // 2) Data of all cut cells of all processes. The interface is small compared to the domain.
    std::vector<CutCellData<dim, number>>                    cut_cells;
    std::vector<std::pair<dealii::Point<dim>, unsigned int>> interface_points;
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

          cut_cells.emplace_back(std::move(cut_cell));
        }

    unsigned int n_redistanced_cells = 0;
    unsigned int max_n_iterations    = 0;

    const bool   use_sign_clamp   = data.sign_clamp_epsilon > 0.;
    unsigned int n_clamped_values = 0;

    if (not interface_points.empty())
      {
        const auto tree = dealii::pack_rtree(interface_points);

        // number of nearest quadrature points of the interface, whose cut cells are candidates of
        // the closest-point search
        const unsigned int n_nearest_points = dim == 1 ? 2 : (dim == 2 ? 8 : 24);

        std::vector<std::pair<dealii::Point<dim>, unsigned int>> nearest_points;
        std::vector<unsigned int>                                candidates;

        // 3) Closest-point search for the nodes of the flagged locally owned cells.
        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            if (not cell->is_locally_owned() or not cell_flags[cell->active_cell_index()] or
                cell->active_fe_index() == ActiveFEIndex::cut or
                cell->get_fe().n_dofs_per_cell() == 0)
              continue;

            AssertThrow(cell->get_fe().has_support_points(),
                        dealii::ExcMessage("The redistancing requires a nodal element."));

            const auto              &unit_support_points = cell->get_fe().get_unit_support_points();
            const AffineCellMap<dim> map(cell);
            dealii::Vector<number>   values(cell->get_fe().n_dofs_per_cell());

            // first-order Taylor expansion from the closest point on the interface
            const auto closest_point_value = [&](const dealii::Point<dim> &x) {
              nearest_points.clear();
              tree.query(boost::geometry::index::nearest(x, n_nearest_points),
                         std::back_inserter(nearest_points));

              candidates.clear();
              for (const auto &point : nearest_points)
                candidates.push_back(point.second);
              std::sort(candidates.begin(), candidates.end());
              candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

              number result       = 0.;
              number min_distance = std::numeric_limits<number>::max();
              for (const unsigned int c : candidates)
                {
                  const auto [y, distance, n_iterations] =
                    search_closest_point(basis, cut_cells[c], x);
                  max_n_iterations = std::max(max_n_iterations, n_iterations);

                  if (distance < min_distance)
                    {
                      min_distance                 = distance;
                      const auto [value, gradient] = evaluate(basis, cut_cells[c], y);
                      result                       = value + gradient * (x - y);
                    }
                }
              return result;
            };

            for (unsigned int i = 0; i < values.size(); ++i)
              values[i] = closest_point_value(map.to_real(unit_support_points[i]));

            // sign clamp: the side of the cell is given by the sign at the cell center
            if (use_sign_clamp)
              {
                dealii::Point<dim> unit_center;
                for (unsigned int d = 0; d < dim; ++d)
                  unit_center[d] = 0.5;

                const number center_value = closest_point_value(map.to_real(unit_center));
                const int    side         = center_value > 0. ? 1 : (center_value < 0. ? -1 : 0);

                if (side != 0)
                  for (auto &value : values)
                    if (side * value <= 0.)
                      {
                        value = side * data.sign_clamp_epsilon * cell->minimum_vertex_distance();
                        ++n_clamped_values;
                      }
              }

            cell->set_dof_values(values, level_set);
            ++n_redistanced_cells;
          }
      }

    n_redistanced_cells = dealii::Utilities::MPI::sum(n_redistanced_cells, mpi_comm);
    max_n_iterations    = dealii::Utilities::MPI::max(max_n_iterations, mpi_comm);
    n_clamped_values    = dealii::Utilities::MPI::sum(n_clamped_values, mpi_comm);

    {
      std::ostringstream str;
      str << "redistancing (" << label << ") of " << n_redistanced_cells << " cells: max. "
          << max_n_iterations << " iterations";
      if (use_sign_clamp)
        str << ", " << n_clamped_values << " clamped";
      Journal::print_line(pcout, str.str(), "narrow_band");
    }

    return n_redistanced_cells;
  }

  template class NarrowBandClosestPointExtension<1, double>;
  template class NarrowBandClosestPointExtension<2, double>;
  template class NarrowBandClosestPointExtension<3, double>;
} // namespace MeltPoolDG::LevelSet
