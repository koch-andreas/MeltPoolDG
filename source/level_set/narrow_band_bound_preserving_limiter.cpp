#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/vector.h>

#include <meltpooldg/level_set/narrow_band_bound_preserving_limiter.hpp>
#include <meltpooldg/level_set/narrow_band_utilities.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace MeltPoolDG::LevelSet
{
  template <int dim, typename number>
  NarrowBandBoundPreservingLimiter<dim, number>::NarrowBandBoundPreservingLimiter(
    const typename NarrowBandData<number>::AdvectionLimiter &data,
    const dealii::Mapping<dim>                              &mapping)
    : data(data)
    , mapping(mapping)
  {}

  template <int dim, typename number>
  void
  NarrowBandBoundPreservingLimiter<dim, number>::evaluate_at_sample_points(
    const dealii::Vector<number> &dof_values,
    std::vector<number>          &sample_values) const
  {
    sample_values.assign(sample_shape_values.m(), 0.);
    for (unsigned int q = 0; q < sample_shape_values.m(); ++q)
      for (unsigned int i = 0; i < sample_shape_values.n(); ++i)
        sample_values[q] += sample_shape_values(q, i) * dof_values[i];
  }

  template <int dim, typename number>
  unsigned int
  NarrowBandBoundPreservingLimiter<dim, number>::compute_bounds(
    const dealii::DoFHandler<dim> &dof_handler,
    const VectorType              &level_set,
    dealii::Function<dim>         &velocity,
    const number                   time,
    const number                   time_step)
  {
    using namespace NarrowBand;

    AssertThrow(velocity.n_components == dim,
                dealii::ExcMessage("The advection velocity function needs dim components."));

    const auto &fe = dof_handler.get_fe_collection()[ActiveFEIndex::cut];
    AssertThrow(fe.n_dofs_per_cell() > 0 and fe.has_support_points(),
                dealii::ExcMessage("The limiter requires nodal elements on the band."));

    // values of the shape functions at the sample points (independent of the mapping)
    const dealii::QGaussLobatto<dim> sample_points(fe.degree + 2);
    sample_shape_values.reinit(sample_points.size(), fe.n_dofs_per_cell());
    for (unsigned int q = 0; q < sample_points.size(); ++q)
      for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
        sample_shape_values(q, i) = fe.shape_value(i, sample_points.point(q));

    const auto        &triangulation = dof_handler.get_triangulation();
    const unsigned int n_cells       = triangulation.n_active_cells();

    const bool use_strips = data.bounds == NarrowBandLimiterBounds::upwind_strips;

    // 1) range [min, max] of the level set of the locally owned band cells
    constexpr number                   inf = std::numeric_limits<number>::max();
    std::vector<std::array<number, 2>> ranges(n_cells, {{inf, -inf}});

    dealii::Vector<number> dof_values(fe.n_dofs_per_cell());
    std::vector<number>    sample_values;
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
          continue;

        AssertDimension(cell->get_fe().n_dofs_per_cell(), fe.n_dofs_per_cell());
        cell->get_dof_values(level_set, dof_values);
        evaluate_at_sample_points(dof_values, sample_values);

        ranges[cell->active_cell_index()] = {
          {*std::min_element(sample_values.begin(), sample_values.end()),
           *std::max_element(sample_values.begin(), sample_values.end())}};
      }

    // communicate the ranges of the locally owned cells to the ghost cells
    if (not use_strips and
        dynamic_cast<const dealii::parallel::TriangulationBase<dim> *>(&triangulation) != nullptr)
      dealii::GridTools::exchange_cell_data_to_ghosts<std::array<number, 2>,
                                                      dealii::DoFHandler<dim>>(
        dof_handler,
        [&](const typename dealii::DoFHandler<dim>::active_cell_iterator &cell)
          -> std::optional<std::array<number, 2>> { return ranges[cell->active_cell_index()]; },
        [&](const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
            const std::array<number, 2> &range) { ranges[cell->active_cell_index()] = range; });

    // the upwind strips are evaluated with the DoF values of the (ghost) neighbors: the ghost set
    // of the level-set vector (e.g., from MatrixFree) does not necessarily contain all DoFs of the
    // ghost cells
    VectorType level_set_ghosted;
    if (use_strips)
      {
        level_set_ghosted.reinit(dof_handler.locally_owned_dofs(),
                                 dealii::DoFTools::extract_locally_relevant_dofs(dof_handler),
                                 dof_handler.get_mpi_communicator());
        level_set_ghosted.copy_locally_owned_data_from(level_set);
        level_set_ghosted.update_ghost_values();
      }

    // 1D Gauss-Lobatto points for the tensor-product points on the faces
    const dealii::QGaussLobatto<1> points_1d(fe.degree + 2);
    unsigned int                   n_face_points = 1;
    for (unsigned int d = 0; d + 1 < dim; ++d)
      n_face_points *= points_1d.size();

    // range of the polynomial with the DoF values @p values_neighbor in the strip of reference
    // width delta_normal along the face @p face_no of the reference cell, which is extended by
    // delta_tangential beyond the edges of the face (extrapolation of the polynomial towards the
    // diagonal neighbors, which are passed by the characteristics of oblique flows)
    dealii::Vector<number> values_neighbor(fe.n_dofs_per_cell());
    const auto             strip_range =
      [&](const unsigned int face_no, const number delta_normal, const number delta_tangential) {
        const unsigned int    normal_direction = face_no / 2;
        std::array<number, 2> range{{inf, -inf}};
        for (const number s : {number(0.), number(0.5) * delta_normal, delta_normal})
          for (unsigned int q = 0; q < n_face_points; ++q)
            {
              dealii::Point<dim> point;
              point[normal_direction] = (face_no % 2 == 0) ? s : 1. - s;
              for (unsigned int d = 0, index = q; d < dim; ++d)
                if (d != normal_direction)
                  {
                    point[d] = -delta_tangential + (1. + 2. * delta_tangential) *
                                                     points_1d.point(index % points_1d.size())[0];
                    index /= points_1d.size();
                  }

              number value = 0.;
              for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
                value += values_neighbor[i] * fe.shape_value(i, point);
              range[0] = std::min(range[0], value);
              range[1] = std::max(range[1], value);
            }
        return range;
      };

    // 2) limited cells and their bounds
    bounds.assign(n_cells, {{-inf, inf}});
    is_limited_cell.assign(n_cells, false);

    // normal vectors and velocities at the Gauss-Lobatto points of the faces
    dealii::FEFaceValues<dim> fe_face_values(mapping,
                                             fe,
                                             dealii::QGaussLobatto<dim - 1>(fe.degree + 2),
                                             dealii::update_normal_vectors |
                                               dealii::update_quadrature_points);

    velocity.set_time(time);

    unsigned int n_limited_cells = 0;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
          continue;

        if (data.cells == NarrowBandLimiterCells::none or
            (data.cells == NarrowBandLimiterCells::non_cut_cells and
             cell->active_fe_index() == ActiveFEIndex::cut))
          continue;

        auto bounds_cell                   = ranges[cell->active_cell_index()];
        bool is_known_domain_of_dependence = true;

        for (const unsigned int f : cell->face_indices())
          {
            // maximum inflow velocity -u.n and maximum speed |u| over the face
            fe_face_values.reinit(cell, f);
            number max_inflow_velocity = 0., max_speed = 0.;
            for (const unsigned int q : fe_face_values.quadrature_point_indices())
              {
                const dealii::Point<dim> &x               = fe_face_values.quadrature_point(q);
                number                    normal_velocity = 0., speed_square = 0.;
                for (unsigned int d = 0; d < dim; ++d)
                  {
                    const number u_d = velocity.value(x, d);
                    normal_velocity += u_d * fe_face_values.normal_vector(q)[d];
                    speed_square += u_d * u_d;
                  }
                const number speed = std::sqrt(speed_square);
                // inflow with a tolerance for round-off errors of the normal vectors
                if (-normal_velocity > 1e-12 * speed)
                  max_inflow_velocity = std::max(max_inflow_velocity, -normal_velocity);
                max_speed = std::max(max_speed, speed);
              }
            const bool is_inflow_face = max_inflow_velocity > 0.;

            if (cell->at_boundary(f) or cell->neighbor(f)->get_fe().n_dofs_per_cell() == 0)
              {
                // The domain of dependence is not covered by the band. The level set of the
                // inactive cells has the sign of the adjacent band cell, since the inactive cells
                // are not cut, i.e., the sign preservation remains valid without these values.
                if (is_inflow_face and data.type == NarrowBandLimiterType::maximum_principle)
                  is_known_domain_of_dependence = false;
                continue;
              }

            if (not is_inflow_face and data.bounds != NarrowBandLimiterBounds::face_neighbors)
              continue;

            const auto neighbor = cell->neighbor(f);

            std::array<number, 2> range_neighbor;
            if (not use_strips)
              range_neighbor = ranges[neighbor->active_cell_index()];
            else
              {
                AssertThrow(not cell->neighbor_is_coarser(f) and not neighbor->has_children(),
                            dealii::ExcMessage("The upwind strips of the limiter are not "
                                               "implemented for hanging nodes."));

                // reference widths of the strip (normal displacement of the characteristics) and of
                // its extension beyond the edges of the face (tangential displacement)
                const unsigned int face_no_neighbor = cell->neighbor_of_neighbor(f);
                const number       h_neighbor       = neighbor->minimum_vertex_distance();
                const number       delta_normal =
                  std::min<number>(1., max_inflow_velocity * time_step / h_neighbor);
                const number delta_tangential =
                  std::min<number>(1., max_speed * time_step / h_neighbor);

                neighbor->get_dof_values(level_set_ghosted, values_neighbor);
                range_neighbor = strip_range(face_no_neighbor, delta_normal, delta_tangential);
              }

            bounds_cell[0] = std::min(bounds_cell[0], range_neighbor[0]);
            bounds_cell[1] = std::max(bounds_cell[1], range_neighbor[1]);
          }

        if (not is_known_domain_of_dependence)
          continue;

        const number h = cell->minimum_vertex_distance();
        if (data.type == NarrowBandLimiterType::maximum_principle)
          {
            bounds_cell[0] -= data.relaxation * h;
            bounds_cell[1] += data.relaxation * h;
          }
        else if (bounds_cell[0] > 0.)
          bounds_cell = {{std::min<number>(bounds_cell[0], data.sign_epsilon * h), inf}};
        else if (bounds_cell[1] < 0.)
          bounds_cell = {{-inf, std::max<number>(bounds_cell[1], -data.sign_epsilon * h)}};
        else
          continue; // both signs in the domain of dependence

        bounds[cell->active_cell_index()]          = bounds_cell;
        is_limited_cell[cell->active_cell_index()] = true;
        ++n_limited_cells;
      }

    return dealii::Utilities::MPI::sum(n_limited_cells, dof_handler.get_mpi_communicator());
  }

  template <int dim, typename number>
  typename NarrowBandBoundPreservingLimiter<dim, number>::Statistics
  NarrowBandBoundPreservingLimiter<dim, number>::limit(const dealii::DoFHandler<dim> &dof_handler,
                                                       VectorType &level_set) const
  {
    using namespace NarrowBand;

    AssertDimension(is_limited_cell.size(), dof_handler.get_triangulation().n_active_cells());

    const auto            &fe = dof_handler.get_fe_collection()[ActiveFEIndex::cut];
    dealii::FEValues<dim>  fe_values(mapping,
                                    fe,
                                    dealii::QGauss<dim>(fe.degree + 1),
                                    dealii::update_values | dealii::update_JxW_values);
    dealii::Vector<number> dof_values(fe.n_dofs_per_cell());
    std::vector<number>    sample_values;

    Statistics statistics;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or not is_limited_cell[cell->active_cell_index()])
          continue;

        AssertDimension(cell->get_fe().n_dofs_per_cell(), fe.n_dofs_per_cell());
        cell->get_dof_values(level_set, dof_values);
        evaluate_at_sample_points(dof_values, sample_values);

        const auto  &bounds_cell = bounds[cell->active_cell_index()];
        const number min_value   = *std::min_element(sample_values.begin(), sample_values.end());
        const number max_value   = *std::max_element(sample_values.begin(), sample_values.end());

        // tolerance for round-off errors
        const number tolerance = 1e-12 * cell->minimum_vertex_distance();
        if (min_value >= bounds_cell[0] - tolerance and max_value <= bounds_cell[1] + tolerance)
          continue;

        // cell mean (the Lagrange basis is a partition of unity: the scaling of the nodal values
        // scales the deviation of the polynomial from its mean)
        fe_values.reinit(cell);
        number mean = 0., volume = 0.;
        for (const unsigned int q : fe_values.quadrature_point_indices())
          {
            number value = 0.;
            for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
              value += dof_values[i] * fe_values.shape_value(i, q);
            mean += value * fe_values.JxW(q);
            volume += fe_values.JxW(q);
          }
        mean /= volume;

        // the scaling around the cell mean cannot restore the bounds
        if (mean > bounds_cell[1] or mean < bounds_cell[0])
          {
            ++statistics.n_mean_violations;
            continue;
          }

        number theta = 1.;
        if (max_value > bounds_cell[1])
          theta = std::min(theta, (bounds_cell[1] - mean) / (max_value - mean));
        if (min_value < bounds_cell[0])
          theta = std::min(theta, (mean - bounds_cell[0]) / (mean - min_value));
        theta = std::max<number>(theta, 0.);

        for (auto &value : dof_values)
          value = mean + theta * (value - mean);

        cell->set_dof_values(dof_values, level_set);
        ++statistics.n_limited_cells;
      }

    const auto comm              = dof_handler.get_mpi_communicator();
    statistics.n_limited_cells   = dealii::Utilities::MPI::sum(statistics.n_limited_cells, comm);
    statistics.n_mean_violations = dealii::Utilities::MPI::sum(statistics.n_mean_violations, comm);
    return statistics;
  }

  template class NarrowBandBoundPreservingLimiter<1, double>;
  template class NarrowBandBoundPreservingLimiter<2, double>;
  template class NarrowBandBoundPreservingLimiter<3, double>;
} // namespace MeltPoolDG::LevelSet
