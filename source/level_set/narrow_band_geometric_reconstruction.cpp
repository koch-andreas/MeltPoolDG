#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/non_matching/quadrature_generator.h>

#include <deal.II/numerics/rtree.h>

#include <meltpooldg/level_set/narrow_band_closest_point_tools.hpp>
#include <meltpooldg/level_set/narrow_band_geometric_reconstruction.hpp>
#include <meltpooldg/level_set/narrow_band_utilities.hpp>
#include <meltpooldg/utilities/journal.hpp>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

namespace MeltPoolDG::LevelSet
{
  using namespace NarrowBand::internal;

  namespace
  {
    /**
     * Discrete interface of a discontinuous level set: the zero level sets of the cells (polynomial
     * and quadrature points of the interface of each cell) and isolated points of the faces, at
     * which the traces of the two adjacent cells have opposite signs. The data of the locally
     * owned cells is collected, gathered from all processes and stored in an R-tree of the
     * points.
     */
    template <int dim, typename number>
    class DiscreteInterface
    {
    public:
      DiscreteInterface(const unsigned int n_dofs, const unsigned int n_nearest_points)
        : n_dofs(n_dofs)
        , n_nearest_points(n_nearest_points)
      {}

      /**
       * Add the zero level set of a locally owned cell.
       */
      void
      add_cell(const AffineCellMap<dim>              &map,
               const dealii::Vector<number>          &values,
               const std::vector<dealii::Point<dim>> &points)
      {
        AssertDimension(values.size(), n_dofs);
        local_cell_data.push_back(map.cell_size);
        for (unsigned int d = 0; d < dim; ++d)
          local_cell_data.push_back(map.origin[d]);
        for (unsigned int i = 0; i < dim; ++i)
          for (unsigned int j = 0; j < dim; ++j)
            local_cell_data.push_back(map.jacobian[i][j]);
        for (unsigned int i = 0; i < n_dofs; ++i)
          local_cell_data.push_back(values[i]);
        local_cell_data.push_back(points.size());
        for (const auto &point : points)
          for (unsigned int d = 0; d < dim; ++d)
            local_cell_data.push_back(point[d]);
        ++n_local_cells;
      }

      /**
       * Add an isolated point of the interface (point of a face with a sign jump).
       */
      void
      add_point(const dealii::Point<dim> &point)
      {
        for (unsigned int d = 0; d < dim; ++d)
          local_point_data.push_back(point[d]);
      }

      /**
       * Gather the data of all processes and build the R-tree. Returns the global number of cells
       * and isolated points.
       */
      std::pair<unsigned int, unsigned int>
      gather(const MPI_Comm mpi_comm)
      {
        std::vector<double> local_data;
        local_data.reserve(local_cell_data.size() + local_point_data.size() + 2);
        local_data.push_back(n_local_cells);
        local_data.insert(local_data.end(), local_cell_data.begin(), local_cell_data.end());
        local_data.push_back(local_point_data.size() / dim);
        local_data.insert(local_data.end(), local_point_data.begin(), local_point_data.end());

        std::vector<std::pair<dealii::Point<dim>, unsigned int>> entries;
        std::vector<dealii::Point<dim>>                          isolated_points;

        for (const auto &data_of_process : dealii::Utilities::MPI::all_gather(mpi_comm, local_data))
          {
            std::size_t        k       = 0;
            const unsigned int n_cells = static_cast<unsigned int>(data_of_process[k++]);
            for (unsigned int c = 0; c < n_cells; ++c)
              {
                const double       cell_size = data_of_process[k++];
                dealii::Point<dim> origin;
                for (unsigned int d = 0; d < dim; ++d)
                  origin[d] = data_of_process[k++];
                dealii::Tensor<2, dim> jacobian;
                for (unsigned int i = 0; i < dim; ++i)
                  for (unsigned int j = 0; j < dim; ++j)
                    jacobian[i][j] = data_of_process[k++];

                CutCellData<dim, number> cell{AffineCellMap<dim>(origin, jacobian, cell_size),
                                              dealii::Vector<number>(n_dofs),
                                              {}};
                for (unsigned int i = 0; i < n_dofs; ++i)
                  cell.values[i] = data_of_process[k++];

                const unsigned int n_points = static_cast<unsigned int>(data_of_process[k++]);
                for (unsigned int q = 0; q < n_points; ++q)
                  {
                    dealii::Point<dim> point;
                    for (unsigned int d = 0; d < dim; ++d)
                      point[d] = data_of_process[k++];
                    cell.interface_points.push_back(point);
                    entries.emplace_back(point, cells.size());
                  }
                // bounding sphere of the cell
                dealii::Point<dim> unit_center;
                for (unsigned int d = 0; d < dim; ++d)
                  unit_center[d] = 0.5;
                const dealii::Point<dim> center = cell.map.to_real(unit_center);
                number                   radius = 0.;
                for (unsigned int v = 0; v < (1U << dim); ++v)
                  {
                    dealii::Point<dim> unit_vertex;
                    for (unsigned int d = 0; d < dim; ++d)
                      unit_vertex[d] = (v >> d) & 1U;
                    radius =
                      std::max<number>(radius, center.distance(cell.map.to_real(unit_vertex)));
                  }
                bounding_spheres.emplace_back(center, radius);

                cells.emplace_back(std::move(cell));
              }

            const unsigned int n_points = static_cast<unsigned int>(data_of_process[k++]);
            for (unsigned int q = 0; q < n_points; ++q)
              {
                dealii::Point<dim> point;
                for (unsigned int d = 0; d < dim; ++d)
                  point[d] = data_of_process[k++];
                isolated_points.push_back(point);
              }
          }

        // isolated points: indices after the cells
        for (const auto &point : isolated_points)
          entries.emplace_back(point, cells.size() + n_isolated_points++);

        n_entries = entries.size();
        if (not entries.empty())
          tree = dealii::pack_rtree(entries);

        return {cells.size(), n_isolated_points};
      }

      bool
      empty() const
      {
        return n_entries == 0;
      }

      /**
       * Distance of @p x to the discrete interface: minimum of the distances to the zero level
       * sets of the cells of the nearest points of the interface (closest-point search on the
       * polynomials) and to the nearest isolated points. Infinite, if the interface is empty.
       */
      number
      distance(const TensorProductBasis<dim> &basis, const dealii::Point<dim> &x) const
      {
        if (empty())
          return std::numeric_limits<number>::infinity();

        nearest_points.clear();
        tree.query(boost::geometry::index::nearest(x, n_nearest_points),
                   std::back_inserter(nearest_points));

        number min_distance = std::numeric_limits<number>::infinity();
        candidates.clear();
        for (const auto &[point, index] : nearest_points)
          {
            // every point of the interface is an upper bound of the distance
            min_distance = std::min<number>(min_distance, x.distance(point));
            if (index < cells.size())
              candidates.emplace_back(x.distance(bounding_spheres[index].first) -
                                        bounding_spheres[index].second,
                                      index);
          }
        std::sort(candidates.begin(), candidates.end());
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

        // closest-point search in the candidate cells, ordered by the lower bound of the distance
        // to the cell (cells farther away than the current distance are skipped)
        for (const auto &[lower_bound, c] : candidates)
          {
            if (lower_bound >= min_distance)
              break;
            const auto [y, distance, n_iterations] = search_closest_point(basis, cells[c], x);
            (void)y;
            max_n_iterations = std::max(max_n_iterations, n_iterations);
            min_distance     = std::min(min_distance, distance);
          }

        return min_distance;
      }

      unsigned int
      get_max_n_iterations() const
      {
        return max_n_iterations;
      }

    private:
      const unsigned int n_dofs;
      /// number of nearest points of the interface, whose cells are candidates of the search
      const unsigned int n_nearest_points;

      unsigned int        n_local_cells = 0;
      std::vector<double> local_cell_data;
      std::vector<double> local_point_data;

      std::vector<CutCellData<dim, number>> cells;
      /// center and radius of the bounding sphere of each cell
      std::vector<std::pair<dealii::Point<dim>, number>> bounding_spheres;
      unsigned int                                       n_isolated_points = 0;
      std::size_t                                        n_entries         = 0;

      dealii::RTree<std::pair<dealii::Point<dim>, unsigned int>> tree;

      mutable std::vector<std::pair<dealii::Point<dim>, unsigned int>> nearest_points;
      mutable std::vector<std::pair<number, unsigned int>>             candidates;
      mutable unsigned int                                             max_n_iterations = 0;
    };

    /**
     * Data of a locally owned cell of the band: affine map, size, evaluation points (quadrature
     * points of the projection, followed by the quadrature points of the discrete interface of
     * the level set before the reconstruction) and the reconstructed distance at these points.
     */
    template <int dim, typename number>
    struct LocalCell
    {
      typename dealii::DoFHandler<dim>::active_cell_iterator cell;
      AffineCellMap<dim>                                     map;
      number                                                 h;
      std::vector<dealii::Point<dim>>                        unit_points;
      std::vector<number>                                    surface_JxW;
      std::vector<number>                                    distance;
      std::vector<number>                                    signed_distance;
    };

    /**
     * Unit points on the faces of the reference cell for the detection of sign jumps between
     * the traces of two cells: tensor product of n Gauss points in each direction of the face.
     */
    template <int dim>
    std::vector<std::vector<dealii::Point<dim>>>
    create_face_points(const unsigned int n)
    {
      const dealii::QGauss<1> quadrature(n);

      unsigned int n_face_points = 1;
      for (unsigned int d = 0; d + 1 < dim; ++d)
        n_face_points *= n;

      std::vector<std::vector<dealii::Point<dim>>> face_points(2 * dim);
      for (unsigned int f = 0; f < 2 * dim; ++f)
        {
          const unsigned int direction = f / 2;
          for (unsigned int q = 0; q < n_face_points; ++q)
            {
              dealii::Point<dim> point;
              point[direction] = f % 2;
              for (unsigned int d = 0, remainder = q; d < dim; ++d)
                if (d != direction)
                  {
                    point[d] = quadrature.point(remainder % n)[0];
                    remainder /= n;
                  }
              face_points[f].push_back(point);
            }
        }
      return face_points;
    }

  } // namespace

  template <int dim, typename number>
  NarrowBandGeometricReconstruction<dim, number>::NarrowBandGeometricReconstruction(
    const typename NarrowBandData<number>::GeometricReconstruction &data,
    const dealii::ConditionalOStream                               &pcout)
    : data(data)
    , pcout(pcout)
  {}

  template <int dim, typename number>
  void
  NarrowBandGeometricReconstruction<dim, number>::reconstruct(
    const dealii::DoFHandler<dim> &dof_handler,
    VectorType                    &level_set) const
  {
    using namespace NarrowBand;

    const auto &fe_collection = dof_handler.get_fe_collection();
    AssertThrow(dof_handler.has_hp_capabilities() and fe_collection.size() > ActiveFEIndex::cut and
                  fe_collection[ActiveFEIndex::cut].n_dofs_per_cell() > 0,
                dealii::ExcMessage("The geometric reconstruction requires the band DoFHandler of "
                                   "the narrow-band level set."));
    AssertThrow(not dof_handler.get_triangulation().has_hanging_nodes(),
                dealii::ExcMessage("The narrow-band level set does not support hanging nodes."));

    const auto        &fe        = fe_collection[ActiveFEIndex::cut];
    const unsigned int fe_degree = fe.degree;
    const unsigned int n_dofs    = fe.n_dofs_per_cell();
    for (unsigned int i = 0; i < fe_collection.size(); ++i)
      AssertThrow(fe_collection[i].n_dofs_per_cell() == 0 or
                    fe_collection[i].n_dofs_per_cell() == n_dofs,
                  dealii::ExcMessage("The geometric reconstruction requires the same element on "
                                     "all cells of the band."));

    const TensorProductBasis<dim> basis(fe);
    const MPI_Comm                mpi_comm = dof_handler.get_mpi_communicator();

    // value of the polynomial of a cell at a unit point
    const auto evaluate_unit = [&](const dealii::Vector<number> &values,
                                   const dealii::Point<dim>     &unit_point) {
      number                         value = 0.;
      dealii::Tensor<1, dim, number> gradient;
      basis.evaluate(values, unit_point, value, gradient, nullptr);
      return value;
    };

    const auto sign = [](const number value) -> number {
      return value > 0. ? 1. : (value < 0. ? -1. : 0.);
    };

    // projection: Gauss-Lobatto quadrature with p+2 points (exact for the mass matrix; the points
    // include the vertices and points of the faces, at which the sign of the projection is
    // checked), mass matrix of the reference cell
    const dealii::QGaussLobatto<dim> quadrature(fe_degree + 2);
    const unsigned int               n_q = quadrature.size();
    dealii::FullMatrix<double>       shape_values(n_q, n_dofs);
    for (unsigned int q = 0; q < n_q; ++q)
      for (unsigned int i = 0; i < n_dofs; ++i)
        shape_values(q, i) = fe.shape_value(i, quadrature.point(q));

    dealii::FullMatrix<double> mass_matrix(n_dofs, n_dofs);
    for (unsigned int q = 0; q < n_q; ++q)
      for (unsigned int i = 0; i < n_dofs; ++i)
        for (unsigned int j = 0; j < n_dofs; ++j)
          mass_matrix(i, j) += quadrature.weight(q) * shape_values(q, i) * shape_values(q, j);
    dealii::FullMatrix<double> inverse_mass_matrix(n_dofs, n_dofs);
    inverse_mass_matrix.invert(mass_matrix);

    // cell-wise L2 projection of the distance at the quadrature points
    const auto project = [&](const std::vector<number> &distance, dealii::Vector<number> &values) {
      dealii::Vector<number> rhs(n_dofs);
      for (unsigned int q = 0; q < n_q; ++q)
        for (unsigned int i = 0; i < n_dofs; ++i)
          rhs[i] += quadrature.weight(q) * distance[q] * shape_values(q, i);
      inverse_mass_matrix.vmult(values, rhs);
    };

    const auto face_points = create_face_points<dim>(2 * (fe_degree + 1));

    // number of points per direction of the quadrature of the discrete interface of the level set
    // (points of the distance computation and of the penalty term)
    const unsigned int n_pinned_points_1d = 2 * (fe_degree + 1);

    const dealii::IndexSet locally_relevant_dofs =
      dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);

    // local cells (locally owned cells of the band): quadrature points of the projection
    std::vector<LocalCell<dim, number>> local_cells;
    std::vector<int> local_index(dof_handler.get_triangulation().n_active_cells(), -1);
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned() and cell->get_fe().n_dofs_per_cell() > 0)
        {
          local_index[cell->active_cell_index()] = local_cells.size();
          LocalCell<dim, number> local_cell{
            cell, AffineCellMap<dim>(cell), cell->minimum_vertex_distance(), {}, {}, {}, {}};
          local_cell.unit_points = quadrature.get_points();
          local_cells.emplace_back(std::move(local_cell));
        }

    // Discrete interface of the level set phi of the band (with ghost values): zero level sets of
    // the locally owned cells and points of the faces with sign jumps. If @p local_cells is given,
    // the quadrature points of the interface are stored as evaluation points of the cells.
    const auto collect_interface = [&](const VectorType                    &phi,
                                       DiscreteInterface<dim, number>      &interface,
                                       std::vector<LocalCell<dim, number>> *local_cells) {
      const dealii::hp::QCollection<1> q_collection(
        dealii::QGauss<1>(local_cells != nullptr ? n_pinned_points_1d : fe_degree + 1));
      dealii::NonMatching::DiscreteQuadratureGenerator<dim> quadrature_generator(q_collection,
                                                                                 dof_handler,
                                                                                 phi);
      dealii::Vector<number>                                values(n_dofs);
      dealii::Vector<number>                                neighbor_values(n_dofs);
      std::vector<dealii::Point<dim>>                       points;

      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
            continue;

          const AffineCellMap<dim> map(cell);
          cell->get_dof_values(phi, values);

          quadrature_generator.generate(cell);
          const auto &surface_quadrature = quadrature_generator.get_surface_quadrature();
          points.clear();
          for (const auto &unit_point : surface_quadrature.get_points())
            points.push_back(map.to_real(unit_point));
          if (not points.empty())
            interface.add_cell(map, values, points);

          if (local_cells != nullptr and local_index[cell->active_cell_index()] >= 0)
            {
              auto &local_cell = (*local_cells)[local_index[cell->active_cell_index()]];
              Assert(local_cell.cell == cell, dealii::ExcInternalError());
              const number det = std::abs(dealii::determinant(map.jacobian));
              for (unsigned int q = 0; q < surface_quadrature.size(); ++q)
                {
                  local_cell.unit_points.push_back(surface_quadrature.point(q));
                  local_cell.surface_JxW.push_back(
                    surface_quadrature.weight(q) * det *
                    (map.inverse_jacobian_transposed * surface_quadrature.normal_vector(q)).norm());
                }
            }

          // points of the faces, at which the traces of the cell and its neighbor have opposite
          // signs (each face once)
          for (const unsigned int f : cell->face_indices())
            {
              if (cell->at_boundary(f))
                continue;
              const auto neighbor = cell->neighbor(f);
              if (neighbor->is_artificial() or neighbor->get_fe().n_dofs_per_cell() == 0)
                continue;
              if (neighbor->is_locally_owned() ?
                    neighbor->active_cell_index() < cell->active_cell_index() :
                    neighbor->subdomain_id() < cell->subdomain_id())
                continue;

              const AffineCellMap<dim> neighbor_map(neighbor);
              neighbor->get_dof_values(phi, neighbor_values);
              for (const auto &unit_point : face_points[f])
                {
                  const dealii::Point<dim> point = map.to_real(unit_point);
                  const number             value = evaluate_unit(values, unit_point);
                  const number             neighbor_value =
                    evaluate_unit(neighbor_values, neighbor_map.to_unit(point));
                  if (value * neighbor_value < 0.)
                    interface.add_point(point);
                }
            }
        }
    };

    VectorType phi(dof_handler.locally_owned_dofs(), locally_relevant_dofs, mpi_comm);
    phi.copy_locally_owned_data_from(level_set);
    phi.update_ghost_values();

    // distance used if the interface is empty (or a phase has disappeared)
    const auto cap = [](const number distance, const number h) {
      return std::min<number>(distance, 1e3 * h);
    };

    unsigned int max_n_iterations = 0;

    // 1) signed distance to the discrete interface of the level set
    unsigned int n_interface_cells = 0, n_face_points = 0;
    {
      DiscreteInterface<dim, number> interface(n_dofs,
                                               n_nearest_interface_points<dim>() *
                                                 n_pinned_points_1d / (fe_degree + 1));
      collect_interface(phi, interface, &local_cells);
      std::tie(n_interface_cells, n_face_points) = interface.gather(mpi_comm);

      dealii::Vector<number> values(n_dofs);
      for (auto &local_cell : local_cells)
        {
          local_cell.cell->get_dof_values(phi, values);
          local_cell.distance.resize(local_cell.unit_points.size());
          for (unsigned int k = 0; k < local_cell.unit_points.size(); ++k)
            local_cell.distance[k] =
              k < n_q ?
                sign(evaluate_unit(values, local_cell.unit_points[k])) *
                  cap(interface.distance(basis, local_cell.map.to_real(local_cell.unit_points[k])),
                      local_cell.h) :
                0.;
        }
      for (auto &local_cell : local_cells)
        local_cell.signed_distance = local_cell.distance;
      max_n_iterations = std::max(max_n_iterations, interface.get_max_n_iterations());
    }

    // 2) morphological opening of the positive phase (s = 1) and of the negative phase (s = -1)
    // with the radius r = thickness * h / 2: erosion E = {s d > r} (zero level set of the
    // cell-wise projection of s d - r) and dilation s d = dist_s(x, boundary of E) + r
    if (data.thickness > 0.)
      for (const int s : {1, -1})
        {
          VectorType eroded(dof_handler.locally_owned_dofs(), locally_relevant_dofs, mpi_comm);
          dealii::Vector<number> values(n_dofs);
          for (const auto &local_cell : local_cells)
            {
              std::vector<number> distance(n_q);
              for (unsigned int q = 0; q < n_q; ++q)
                distance[q] = s * local_cell.distance[q];
              project(distance, values);
              const number radius = 0.5 * data.thickness * local_cell.h;
              for (auto &value : values)
                value -= radius;
              local_cell.cell->set_dof_values(values, eroded);
            }
          eroded.update_ghost_values();

          DiscreteInterface<dim, number> interface(n_dofs, n_nearest_interface_points<dim>());
          collect_interface(eroded, interface, nullptr);
          interface.gather(mpi_comm);

          for (auto &local_cell : local_cells)
            {
              local_cell.cell->get_dof_values(eroded, values);
              const number radius = 0.5 * data.thickness * local_cell.h;
              for (unsigned int k = 0; k < local_cell.unit_points.size(); ++k)
                {
                  const dealii::Point<dim> &unit_point = local_cell.unit_points[k];
                  const number              value      = evaluate_unit(values, unit_point);
                  // empty eroded set: the phase disappears
                  const number distance =
                    interface.empty() ?
                      -cap(std::numeric_limits<number>::infinity(), local_cell.h) :
                      sign(value) *
                        cap(interface.distance(basis, local_cell.map.to_real(unit_point)),
                            local_cell.h);
                  local_cell.distance[k] = s * (distance + radius);
                }
            }
          max_n_iterations = std::max(max_n_iterations, interface.get_max_n_iterations());
        }

    // 3) cell-wise projection with the penalty term on the points of the discrete interface,
    // which survive the opening (|d| < r/2 after the opening). The opened distance is only used
    // at the points, at which the opening has changed the signed distance by more than r/2 (thin
    // features, which are removed); elsewhere, the signed distance is projected. Smaller changes
    // (corners rounded by the opening, the undershoot of the projection in the erosion at ridges
    // of the distance) are ignored, such that repeated reconstructions do not erode the interface
    // step by step.
    constexpr number relative_tolerance = 0.5;
    const bool       had_ghost_values   = level_set.has_ghost_elements();
    if (had_ghost_values)
      level_set.zero_out_ghost_values();

    unsigned int n_surface_points = 0, n_pinned_points = 0, n_shifted_cells = 0;
    {
      const SignSampler<dim> sign_sampler(fe);
      // margin of the shifted polynomial relative to its undershoot
      constexpr number relative_margin = 0.1;

      dealii::FullMatrix<double> matrix(n_dofs, n_dofs);
      dealii::Vector<number>     rhs(n_dofs), values(n_dofs);
      std::vector<number>        surface_shape_values(n_dofs);
      std::vector<number>        target(n_q);

      for (const auto &local_cell : local_cells)
        {
          const number det    = std::abs(dealii::determinant(local_cell.map.jacobian));
          const number radius = 0.5 * data.thickness * local_cell.h;

          for (unsigned int i = 0; i < n_dofs; ++i)
            for (unsigned int j = 0; j < n_dofs; ++j)
              matrix(i, j) = det * mass_matrix(i, j);
          for (unsigned int q = 0; q < n_q; ++q)
            target[q] = std::abs(local_cell.distance[q] - local_cell.signed_distance[q]) <=
                            relative_tolerance * radius ?
                          local_cell.signed_distance[q] :
                          local_cell.distance[q];

          rhs = 0.;
          for (unsigned int q = 0; q < n_q; ++q)
            for (unsigned int i = 0; i < n_dofs; ++i)
              rhs[i] += det * quadrature.weight(q) * target[q] * shape_values(q, i);

          bool is_pinned = false;
          for (unsigned int k = n_q; k < local_cell.unit_points.size(); ++k)
            {
              ++n_surface_points;
              if (data.thickness > 0. and
                  std::abs(local_cell.distance[k]) >= relative_tolerance * radius)
                continue;
              ++n_pinned_points;
              is_pinned = true;
              for (unsigned int i = 0; i < n_dofs; ++i)
                surface_shape_values[i] = fe.shape_value(i, local_cell.unit_points[k]);
              const number factor = data.penalty * local_cell.h * local_cell.surface_JxW[k - n_q];
              for (unsigned int i = 0; i < n_dofs; ++i)
                for (unsigned int j = 0; j < n_dofs; ++j)
                  matrix(i, j) += factor * surface_shape_values[i] * surface_shape_values[j];
            }

          if (is_pinned)
            {
              matrix.gauss_jordan();
              matrix.vmult(values, rhs);
            }
          else
            {
              inverse_mass_matrix.vmult(values, rhs);
              values /= det;

              // Cells without remaining interface, in which the reconstructed distance has the
              // same sign at all quadrature points: the projection must not change the sign. Its
              // undershoot next to an interface, which touches the cell at a face or a vertex,
              // would create a spurious interface. The polynomial is shifted such that its
              // minimum is a fraction of the undershoot.
              const int side         = target[0] > 0. ? 1 : (target[0] < 0. ? -1 : 0);
              bool      has_one_sign = side != 0;
              for (unsigned int q = 1; q < n_q and has_one_sign; ++q)
                has_one_sign = side * target[q] > 0.;
              if (has_one_sign)
                {
                  const number min_value = sign_sampler.min_signed_value(values, side);
                  if (min_value < 0.)
                    {
                      for (auto &value : values)
                        value -= side * (1. + relative_margin) * min_value;
                      ++n_shifted_cells;
                    }
                }
            }

          local_cell.cell->set_dof_values(values, level_set);
        }
    }

    if (had_ghost_values)
      level_set.update_ghost_values();

    n_surface_points = dealii::Utilities::MPI::sum(n_surface_points, mpi_comm);
    n_pinned_points  = dealii::Utilities::MPI::sum(n_pinned_points, mpi_comm);
    n_shifted_cells  = dealii::Utilities::MPI::sum(n_shifted_cells, mpi_comm);
    max_n_iterations = dealii::Utilities::MPI::max(max_n_iterations, mpi_comm);

    std::ostringstream str;
    str << "geometric reconstruction: " << n_interface_cells << " interface cells, "
        << n_face_points << " face points, " << n_pinned_points << " of " << n_surface_points
        << " interface points kept, " << n_shifted_cells << " cells shifted, max. "
        << max_n_iterations << " iterations";
    Journal::print_line(pcout, str.str(), "narrow_band");
  }

  template class NarrowBandGeometricReconstruction<1, double>;
  template class NarrowBandGeometricReconstruction<2, double>;
  template class NarrowBandGeometricReconstruction<3, double>;
} // namespace MeltPoolDG::LevelSet
