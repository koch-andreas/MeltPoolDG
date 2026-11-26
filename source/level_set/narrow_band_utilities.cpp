#include <deal.II/base/exceptions.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/full_matrix.h>

#include <deal.II/non_matching/quadrature_generator.h>

#include <meltpooldg/level_set/narrow_band_closest_point_tools.hpp>
#include <meltpooldg/level_set/narrow_band_utilities.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <optional>

namespace MeltPoolDG::LevelSet::NarrowBand
{
  template <int dim>
  dealii::hp::FECollection<dim>
  create_band_fe_collection(const unsigned int fe_degree)
  {
    dealii::hp::FECollection<dim> fe_collection;
    fe_collection.push_back(dealii::FE_Nothing<dim>());      // inactive
    fe_collection.push_back(dealii::FE_DGQ<dim>(fe_degree)); // cut
    fe_collection.push_back(dealii::FE_DGQ<dim>(fe_degree)); // neighbor
    return fe_collection;
  }

  template <int dim>
  dealii::hp::FECollection<dim>
  create_cut_fe_collection(const unsigned int fe_degree)
  {
    dealii::hp::FECollection<dim> fe_collection;
    fe_collection.push_back(dealii::FE_Nothing<dim>());      // inactive
    fe_collection.push_back(dealii::FE_DGQ<dim>(fe_degree)); // cut
    fe_collection.push_back(dealii::FE_Nothing<dim>());      // neighbor
    return fe_collection;
  }

  template <int dim>
  dealii::hp::FECollection<dim>
  create_layered_band_fe_collection(const unsigned int fe_degree)
  {
    dealii::hp::FECollection<dim> fe_collection;
    fe_collection.push_back(dealii::FE_Nothing<dim>());      // inactive
    fe_collection.push_back(dealii::FE_DGQ<dim>(fe_degree)); // cut
    fe_collection.push_back(dealii::FE_DGQ<dim>(fe_degree)); // first layer
    fe_collection.push_back(dealii::FE_DGQ<dim>(fe_degree)); // outer layers
    return fe_collection;
  }

  template <int dim>
  dealii::hp::FECollection<dim>
  create_layered_cut_fe_collection(const unsigned int fe_degree)
  {
    dealii::hp::FECollection<dim> fe_collection;
    fe_collection.push_back(dealii::FE_Nothing<dim>());      // inactive
    fe_collection.push_back(dealii::FE_DGQ<dim>(fe_degree)); // cut
    fe_collection.push_back(dealii::FE_Nothing<dim>());      // first layer
    fe_collection.push_back(dealii::FE_Nothing<dim>());      // outer layers
    return fe_collection;
  }

  template <int dim, typename number>
  std::vector<bool>
  compute_cut_cells(const dealii::DoFHandler<dim>                            &dof_handler,
                    const dealii::LinearAlgebra::distributed::Vector<number> &level_set,
                    const unsigned int                                        fe_degree,
                    std::vector<number>                                      *volume_fractions)
  {
    std::vector<bool> is_cut_cell(dof_handler.get_triangulation().n_active_cells(), false);

    if (volume_fractions != nullptr)
      volume_fractions->assign(dof_handler.get_triangulation().n_active_cells(), 0.);

    const bool update_ghosts = not level_set.has_ghost_elements();
    if (update_ghosts)
      level_set.update_ghost_values();

    const dealii::hp::QCollection<1> q_collection(dealii::QGauss<1>(fe_degree + 1));

    dealii::NonMatching::DiscreteQuadratureGenerator<dim> quadrature_generator(q_collection,
                                                                               dof_handler,
                                                                               level_set);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
          continue;

        quadrature_generator.generate(cell);
        is_cut_cell[cell->active_cell_index()] =
          quadrature_generator.get_surface_quadrature().size() > 0;

        // volume fraction of the positive phase: the weights of the quadrature of the region
        // phi > 0 on the reference cell (with unit volume) sum up to its volume fraction
        if (volume_fractions != nullptr and is_cut_cell[cell->active_cell_index()])
          {
            number volume_fraction = 0.;
            for (const auto weight : quadrature_generator.get_outside_quadrature().get_weights())
              volume_fraction += weight;
            (*volume_fractions)[cell->active_cell_index()] = volume_fraction;
          }
      }

    if (update_ghosts)
      level_set.zero_out_ghost_values();

    return is_cut_cell;
  }

  template <int dim, typename number>
  std::vector<number>
  compute_cut_cell_volume_fractions(
    const dealii::DoFHandler<dim>                            &dof_handler,
    const dealii::LinearAlgebra::distributed::Vector<number> &level_set,
    const unsigned int                                        fe_degree)
  {
    std::vector<number> volume_fractions(dof_handler.get_triangulation().n_active_cells(), 0.);

    const bool update_ghosts = not level_set.has_ghost_elements();
    if (update_ghosts)
      level_set.update_ghost_values();

    const dealii::hp::QCollection<1> q_collection(dealii::QGauss<1>(fe_degree + 1));

    dealii::NonMatching::DiscreteQuadratureGenerator<dim> quadrature_generator(q_collection,
                                                                               dof_handler,
                                                                               level_set);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_artificial() or cell->active_fe_index() != ActiveFEIndex::cut)
          continue;

        // the weights of the quadrature of the region phi > 0 on the reference cell (with unit
        // volume) sum up to its volume fraction
        quadrature_generator.generate(cell);
        number volume_fraction = 0.;
        for (const auto weight : quadrature_generator.get_outside_quadrature().get_weights())
          volume_fraction += weight;
        volume_fractions[cell->active_cell_index()] = volume_fraction;
      }

    if (update_ghosts)
      level_set.zero_out_ghost_values();

    return volume_fractions;
  }

  template <int dim>
  void
  distribute_dofs(
    const std::vector<bool>                       &is_cut_cell,
    dealii::DoFHandler<dim>                       &dof_handler_cut,
    const dealii::hp::FECollection<dim>           &fe_collection_cut,
    dealii::DoFHandler<dim>                       &dof_handler_band,
    const dealii::hp::FECollection<dim>           &fe_collection_band,
    const std::vector<bool>                       &is_small_cut_cell,
    const std::function<bool(const typename dealii::Triangulation<dim>::active_cell_iterator &,
                             const unsigned int)> &is_activating_face)
  {
    AssertDimension(is_cut_cell.size(), dof_handler_cut.get_triangulation().n_active_cells());
    Assert(is_small_cut_cell.empty() or is_small_cut_cell.size() == is_cut_cell.size(),
           dealii::ExcDimensionMismatch(is_small_cut_cell.size(), is_cut_cell.size()));
    AssertThrow(not dof_handler_cut.get_triangulation().has_hanging_nodes(),
                dealii::ExcMessage("The narrow-band level set does not support hanging nodes."));

    const auto is_small = [&](const unsigned int active_cell_index) {
      return not is_small_cut_cell.empty() and is_small_cut_cell[active_cell_index];
    };

    // 1) Set the FE index of the cut cells and distribute the DoFs of the cut DoFHandler
    // preliminarily: the active FE indices of the ghost cells are communicated during the
    // distribution of the DoFs, which allows to identify the neighbor cells of the cut cells of
    // other processes. Cut cells, which do not activate their face neighbors, temporarily get the
    // FE index of the neighbor cells (FE_Nothing in the cut DoFHandler).
    for (const auto &cell : dof_handler_cut.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const unsigned int index = cell->active_cell_index();
          cell->set_active_fe_index(not is_cut_cell[index] ? ActiveFEIndex::inactive :
                                    is_small(index)        ? ActiveFEIndex::neighbor :
                                                             ActiveFEIndex::cut);
        }

    dof_handler_cut.distribute_dofs(fe_collection_cut);

    // 2) Identify the neighbor cells.
    std::vector<unsigned int> active_fe_indices(is_cut_cell.size(), ActiveFEIndex::inactive);
    for (const auto &cell : dof_handler_cut.active_cell_iterators())
      {
        if (not cell->is_locally_owned())
          continue;

        if (is_cut_cell[cell->active_cell_index()])
          {
            active_fe_indices[cell->active_cell_index()] = ActiveFEIndex::cut;
            continue;
          }

        for (const unsigned int f : cell->face_indices())
          {
            if (cell->at_boundary(f))
              continue;

            const auto neighbor = cell->neighbor(f);
            Assert(neighbor->is_active(), dealii::ExcNotImplemented());

            // only cut cells with the (preliminary) FE index NarrowBand::cut activate their face
            // neighbors (optionally only across the activating faces)
            if (neighbor->active_fe_index() == ActiveFEIndex::cut and
                (not is_activating_face or
                 is_activating_face(typename dealii::Triangulation<dim>::active_cell_iterator(cell),
                                    f)))
              {
                active_fe_indices[cell->active_cell_index()] = ActiveFEIndex::neighbor;
                break;
              }
          }
      }

    // 3) Both DoFHandlers use the same active FE indices, as required by MatrixFree for several
    // hp-DoFHandlers. The neighbor cells hold FE_Nothing elements in the cut DoFHandler.
    for (auto *dof_handler : {&dof_handler_cut, &dof_handler_band})
      for (const auto &cell : dof_handler->active_cell_iterators())
        if (cell->is_locally_owned())
          cell->set_active_fe_index(active_fe_indices[cell->active_cell_index()]);

    dof_handler_cut.distribute_dofs(fe_collection_cut);
    dof_handler_band.distribute_dofs(fe_collection_band);
  }

  template <int dim>
  bool
  is_downstream_face(const dealii::Mapping<dim>                                      &mapping,
                     dealii::Function<dim>                                           &velocity,
                     const std::vector<double>                                       &times,
                     const typename dealii::Triangulation<dim>::active_cell_iterator &cell,
                     const unsigned int                                               face_no,
                     const unsigned int                                               n_points_1d)
  {
    AssertThrow(velocity.n_components == dim,
                dealii::ExcMessage("The velocity function needs dim components."));
    AssertThrow(n_points_1d >= 2, dealii::ExcMessage("At least two points are required."));

    // only the quadrature points and normal vectors are needed (any element)
    const dealii::FE_Nothing<dim>     fe;
    const dealii::Quadrature<dim - 1> face_quadrature = [&]() -> dealii::Quadrature<dim - 1> {
      if constexpr (dim == 1)
        return dealii::QGauss<0>(1);
      else
        return dealii::QGaussLobatto<dim - 1>(n_points_1d);
    }();
    dealii::FEFaceValues<dim> fe_face_values(mapping,
                                             fe,
                                             face_quadrature,
                                             dealii::update_quadrature_points |
                                               dealii::update_normal_vectors);
    fe_face_values.reinit(cell, face_no);

    const double old_time = velocity.get_time();

    bool is_downstream = false;
    for (const double time : times)
      {
        velocity.set_time(time);
        for (const unsigned int q : fe_face_values.quadrature_point_indices())
          {
            double normal_velocity = 0., speed_square = 0.;
            for (unsigned int d = 0; d < dim; ++d)
              {
                const double u_d = velocity.value(fe_face_values.quadrature_point(q), d);
                normal_velocity += u_d * fe_face_values.normal_vector(q)[d];
                speed_square += u_d * u_d;
              }
            if (normal_velocity < -1e-12 * std::sqrt(speed_square))
              is_downstream = true;
          }
      }

    velocity.set_time(old_time);

    return is_downstream;
  }

  template <int dim>
  std::vector<unsigned int>
  compute_layers(const dealii::Triangulation<dim> &triangulation,
                 const std::vector<bool>          &is_cut_cell,
                 const unsigned int                n_layers)
  {
    AssertDimension(is_cut_cell.size(), triangulation.n_active_cells());

    constexpr unsigned int    invalid = dealii::numbers::invalid_unsigned_int;
    std::vector<unsigned int> layers(triangulation.n_active_cells(), invalid);

    const bool is_parallel =
      dynamic_cast<const dealii::parallel::TriangulationBase<dim> *>(&triangulation) != nullptr;

    // communicate the layers of the locally owned cells to the ghost cells
    const auto exchange = [&]() {
      if (not is_parallel)
        return;

      dealii::GridTools::exchange_cell_data_to_ghosts<unsigned int, dealii::Triangulation<dim>>(
        triangulation,
        [&](const typename dealii::Triangulation<dim>::active_cell_iterator &cell)
          -> std::optional<unsigned int> { return layers[cell->active_cell_index()]; },
        [&](const typename dealii::Triangulation<dim>::active_cell_iterator &cell,
            const unsigned int &layer) { layers[cell->active_cell_index()] = layer; });
    };

    // cells sharing a vertex (locally owned and ghost cells)
    std::vector<std::vector<unsigned int>> vertex_to_cells(triangulation.n_vertices());
    for (const auto &cell : triangulation.active_cell_iterators())
      if (not cell->is_artificial())
        for (const unsigned int v : cell->vertex_indices())
          vertex_to_cells[cell->vertex_index(v)].push_back(cell->active_cell_index());

    for (const auto &cell : triangulation.active_cell_iterators())
      if (cell->is_locally_owned() and is_cut_cell[cell->active_cell_index()])
        layers[cell->active_cell_index()] = 0;
    exchange();

    for (unsigned int layer = 1; layer <= n_layers; ++layer)
      {
        for (const auto &cell : triangulation.active_cell_iterators())
          {
            if (not cell->is_locally_owned() or layers[cell->active_cell_index()] != invalid)
              continue;

            bool is_in_layer = false;
            for (const unsigned int v : cell->vertex_indices())
              for (const unsigned int other : vertex_to_cells[cell->vertex_index(v)])
                if (layers[other] == layer - 1)
                  is_in_layer = true;

            if (is_in_layer)
              layers[cell->active_cell_index()] = layer;
          }
        exchange();
      }

    return layers;
  }

  template <int dim>
  std::vector<unsigned int>
  distribute_dofs_layered(const std::vector<bool>             &is_cut_cell,
                          const unsigned int                   n_layers,
                          dealii::DoFHandler<dim>             &dof_handler_cut,
                          const dealii::hp::FECollection<dim> &fe_collection_cut,
                          dealii::DoFHandler<dim>             &dof_handler_band,
                          const dealii::hp::FECollection<dim> &fe_collection_band)
  {
    AssertThrow(n_layers >= 1, dealii::ExcMessage("The layered band requires at least one layer."));
    AssertThrow(fe_collection_cut.size() == 4 and fe_collection_band.size() == 4,
                dealii::ExcMessage("The advected band requires the layered FE collections."));
    AssertThrow(not dof_handler_cut.get_triangulation().has_hanging_nodes(),
                dealii::ExcMessage("The narrow-band level set does not support hanging nodes."));

    const std::vector<unsigned int> layers =
      compute_layers(dof_handler_cut.get_triangulation(), is_cut_cell, n_layers);

    for (auto *dof_handler : {&dof_handler_cut, &dof_handler_band})
      for (const auto &cell : dof_handler->active_cell_iterators())
        if (cell->is_locally_owned())
          {
            const unsigned int layer = layers[cell->active_cell_index()];
            cell->set_active_fe_index(layer == dealii::numbers::invalid_unsigned_int ?
                                        ActiveFEIndex::inactive :
                                      layer == 0 ? ActiveFEIndex::cut :
                                      layer == 1 ? ActiveFEIndex::neighbor :
                                                   ActiveFEIndex::outer);
          }

    dof_handler_cut.distribute_dofs(fe_collection_cut);
    dof_handler_band.distribute_dofs(fe_collection_band);

    return layers;
  }

  template <int dim, typename number>
  std::vector<dealii::Vector<number>>
  extract_cell_values(const dealii::DoFHandler<dim>                            &dof_handler,
                      const dealii::LinearAlgebra::distributed::Vector<number> &vector,
                      const std::vector<bool>                                  &cell_flags)
  {
    AssertDimension(cell_flags.size(), dof_handler.get_triangulation().n_active_cells());

    std::vector<dealii::Vector<number>> cell_values(cell_flags.size());

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or not cell_flags[cell->active_cell_index()])
          continue;

        auto &values = cell_values[cell->active_cell_index()];
        values.reinit(cell->get_fe().n_dofs_per_cell());
        cell->get_dof_values(vector, values);
      }

    return cell_values;
  }

  template <int dim, typename number>
  void
  set_cell_values(const dealii::DoFHandler<dim>                      &dof_handler,
                  const std::vector<dealii::Vector<number>>          &cell_values,
                  dealii::LinearAlgebra::distributed::Vector<number> &vector)
  {
    AssertDimension(cell_values.size(), dof_handler.get_triangulation().n_active_cells());

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
          continue;

        const auto &values = cell_values[cell->active_cell_index()];
        if (values.size() == 0)
          continue;

        AssertDimension(values.size(), cell->get_fe().n_dofs_per_cell());
        cell->set_dof_values(values, vector);
      }
  }

  template <int dim, typename number>
  void
  copy_cut_cell_values(const dealii::DoFHandler<dim>                            &dof_handler_src,
                       const dealii::LinearAlgebra::distributed::Vector<number> &src,
                       const dealii::DoFHandler<dim>                            &dof_handler_dst,
                       dealii::LinearAlgebra::distributed::Vector<number>       &dst)
  {
    dealii::Vector<number> values;

    for (const auto &cell : dof_handler_src.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or cell->active_fe_index() != ActiveFEIndex::cut)
          continue;

        const auto cell_dst = cell->as_dof_handler_iterator(dof_handler_dst);
        AssertThrow(cell_dst->active_fe_index() == ActiveFEIndex::cut,
                    dealii::ExcMessage("The cut cells of both DoFHandlers do not coincide."));
        AssertDimension(cell->get_fe().n_dofs_per_cell(), cell_dst->get_fe().n_dofs_per_cell());

        values.reinit(cell->get_fe().n_dofs_per_cell());
        cell->get_dof_values(src, values);
        cell_dst->set_dof_values(values, dst);
      }
  }

  template <int dim, typename number>
  void
  copy_common_cell_values(const dealii::DoFHandler<dim>                            &dof_handler_src,
                          const dealii::LinearAlgebra::distributed::Vector<number> &src,
                          const dealii::DoFHandler<dim>                            &dof_handler_dst,
                          dealii::LinearAlgebra::distributed::Vector<number>       &dst)
  {
    dealii::Vector<number> values;

    for (const auto &cell : dof_handler_src.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
          continue;

        const auto cell_dst = cell->as_dof_handler_iterator(dof_handler_dst);
        if (cell_dst->get_fe().n_dofs_per_cell() == 0)
          continue;

        AssertDimension(cell->get_fe().n_dofs_per_cell(), cell_dst->get_fe().n_dofs_per_cell());

        values.reinit(cell->get_fe().n_dofs_per_cell());
        cell->get_dof_values(src, values);
        cell_dst->set_dof_values(values, dst);
      }
  }

  template <int dim, typename number>
  number
  compute_eikonal_error(const dealii::Mapping<dim>                               &mapping,
                        const dealii::DoFHandler<dim>                            &dof_handler,
                        const dealii::LinearAlgebra::distributed::Vector<number> &level_set)
  {
    const bool update_ghosts = not level_set.has_ghost_elements();
    if (update_ghosts)
      level_set.update_ghost_values();

    const auto                  &fe_collection = dof_handler.get_fe_collection();
    dealii::hp::QCollection<dim> q_collection;
    for (unsigned int i = 0; i < fe_collection.size(); ++i)
      q_collection.push_back(dealii::QGauss<dim>(fe_collection[i].degree + 1));

    // Note: hp::FEValues keeps pointers to the collections; thus, they must outlive it.
    const dealii::hp::MappingCollection<dim> mapping_collection(mapping);
    dealii::hp::FEValues<dim>                hp_fe_values(mapping_collection,
                                           fe_collection,
                                           q_collection,
                                           dealii::update_gradients | dealii::update_JxW_values);

    std::vector<dealii::Tensor<1, dim, number>> gradients;

    number error = 0., volume = 0.;
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or cell->active_fe_index() != ActiveFEIndex::cut or
            cell->get_fe().n_dofs_per_cell() == 0)
          continue;

        hp_fe_values.reinit(cell);
        const auto &fe_values = hp_fe_values.get_present_fe_values();

        gradients.resize(fe_values.n_quadrature_points);
        fe_values.get_function_gradients(level_set, gradients);

        for (const unsigned int q : fe_values.quadrature_point_indices())
          {
            error +=
              dealii::Utilities::fixed_power<2>(gradients[q].norm() - 1.0) * fe_values.JxW(q);
            volume += fe_values.JxW(q);
          }
      }

    if (update_ghosts)
      level_set.zero_out_ghost_values();

    const auto mpi_comm = dof_handler.get_mpi_communicator();
    error               = dealii::Utilities::MPI::sum(error, mpi_comm);
    volume              = dealii::Utilities::MPI::sum(volume, mpi_comm);

    return volume > 0. ? std::sqrt(error / volume) : 0.;
  }

  template <int dim, typename number>
  unsigned int
  limit_sign_changes(const dealii::Mapping<dim>                         &mapping,
                     const dealii::DoFHandler<dim>                      &dof_handler,
                     dealii::LinearAlgebra::distributed::Vector<number> &level_set,
                     const number                                        tolerance)
  {
    // If the level set changes its sign within a neighbor cell (beyond the tolerance), the
    // deviation from the cell mean is scaled such that the minimum of sign(mean) * phi equals
    // margin * |mean|.
    constexpr number margin = 0.01;

    const auto        &fe        = dof_handler.get_fe_collection()[ActiveFEIndex::neighbor];
    const unsigned int n_dofs    = fe.n_dofs_per_cell();
    const unsigned int fe_degree = fe.degree;

    // shape values at the sample points (the sign of the level set does not depend on the mapping)
    const dealii::QGaussLobatto<dim> sample_points(fe_degree + 2);
    dealii::FullMatrix<number>       shape_values(sample_points.size(), n_dofs);
    for (unsigned int q = 0; q < sample_points.size(); ++q)
      for (unsigned int i = 0; i < n_dofs; ++i)
        shape_values(q, i) = fe.shape_value(i, sample_points.point(q));

    // cell mean
    dealii::FEValues<dim> fe_values(mapping,
                                    fe,
                                    dealii::QGauss<dim>(fe_degree + 1),
                                    dealii::update_values | dealii::update_JxW_values);

    dealii::Vector<number> values(n_dofs);
    unsigned int           n_limited_cells = 0;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or cell->active_fe_index() != ActiveFEIndex::neighbor)
          continue;

        cell->get_dof_values(level_set, values);

        fe_values.reinit(cell);
        number mean = 0., volume = 0.;
        for (const unsigned int q : fe_values.quadrature_point_indices())
          {
            number value = 0.;
            for (unsigned int i = 0; i < n_dofs; ++i)
              value += values[i] * fe_values.shape_value(i, q);
            mean += value * fe_values.JxW(q);
            volume += fe_values.JxW(q);
          }
        mean /= volume;

        if (mean == 0.)
          continue;

        const number sign = mean > 0. ? 1. : -1.;

        // sign * phi at the sample points
        number min_value = std::numeric_limits<number>::max();
        for (unsigned int q = 0; q < sample_points.size(); ++q)
          {
            number value = 0.;
            for (unsigned int i = 0; i < n_dofs; ++i)
              value += values[i] * shape_values(q, i);
            min_value = std::min(min_value, sign * value);
          }

        // Small sign changes below the tolerance are accepted. Otherwise, the scaling factor
        // theta in [0, 1) is chosen such that sign * (mean + theta * (phi - mean)) >= margin *
        // |mean|.
        number theta = 1.;
        if (min_value < -tolerance * cell->minimum_vertex_distance())
          theta = (1. - margin) * std::abs(mean) / (std::abs(mean) - min_value);

        if (theta < 1.)
          {
            // the Lagrange basis is a partition of unity: scaling the nodal values scales the
            // deviation of the polynomial from its mean
            for (unsigned int i = 0; i < n_dofs; ++i)
              values[i] = mean + theta * (values[i] - mean);
            cell->set_dof_values(values, level_set);
            ++n_limited_cells;
          }
      }

    return dealii::Utilities::MPI::sum(n_limited_cells, dof_handler.get_mpi_communicator());
  }

  template <int dim, typename number>
  std::vector<int>
  compute_sides_from_cut_cells(const dealii::Mapping<dim>    &mapping,
                               const dealii::DoFHandler<dim> &dof_handler,
                               const dealii::LinearAlgebra::distributed::Vector<number> &level_set,
                               const std::vector<bool>                                  &cell_flags)
  {
    AssertDimension(cell_flags.size(), dof_handler.get_triangulation().n_active_cells());

    std::vector<int> sides(cell_flags.size(), 0);

    const auto &fe_cut = dof_handler.get_fe_collection()[ActiveFEIndex::cut];

    const dealii::QGauss<dim - 1> face_quadrature(fe_cut.degree + 1);
    dealii::FEFaceValues<dim>     fe_face_values(mapping,
                                             fe_cut,
                                             face_quadrature,
                                             dealii::update_values | dealii::update_JxW_values);
    std::vector<number>           values(face_quadrature.size());

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or not cell_flags[cell->active_cell_index()] or
            cell->active_fe_index() == ActiveFEIndex::cut)
          continue;

        number integral = 0.;
        for (const unsigned int f : cell->face_indices())
          {
            if (cell->at_boundary(f))
              continue;

            const auto neighbor = cell->neighbor(f);
            if (neighbor->active_fe_index() != ActiveFEIndex::cut)
              continue;

            // trace of the cut cell on the shared face
            fe_face_values.reinit(neighbor, cell->neighbor_of_neighbor(f));
            fe_face_values.get_function_values(level_set, values);
            for (const unsigned int q : fe_face_values.quadrature_point_indices())
              integral += values[q] * fe_face_values.JxW(q);
          }

        sides[cell->active_cell_index()] = integral > 0. ? 1 : (integral < 0. ? -1 : 0);
      }

    return sides;
  }

  template <int dim, typename number>
  unsigned int
  clamp_signs(const dealii::DoFHandler<dim>                      &dof_handler,
              dealii::LinearAlgebra::distributed::Vector<number> &level_set,
              const std::vector<int>                             &cell_sides,
              const number                                        epsilon)
  {
    AssertDimension(cell_sides.size(), dof_handler.get_triangulation().n_active_cells());

    unsigned int           n_clamped_values = 0;
    dealii::Vector<number> values;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
          continue;

        const int side = cell_sides[cell->active_cell_index()];
        if (side == 0)
          continue;

        values.reinit(cell->get_fe().n_dofs_per_cell());
        cell->get_dof_values(level_set, values);

        const number clamped_value = side * epsilon * cell->minimum_vertex_distance();

        bool is_modified = false;
        for (auto &value : values)
          if (side * value <= 0.)
            {
              value       = clamped_value;
              is_modified = true;
              ++n_clamped_values;
            }

        if (is_modified)
          cell->set_dof_values(values, level_set);
      }

    return dealii::Utilities::MPI::sum(n_clamped_values, dof_handler.get_mpi_communicator());
  }

  template <int dim, typename number>
  unsigned int
  remove_thin_features(const dealii::DoFHandler<dim>                      &dof_handler,
                       dealii::LinearAlgebra::distributed::Vector<number> &level_set,
                       const unsigned int                                  fe_degree,
                       const number                                        thickness,
                       const number                                        epsilon,
                       std::vector<bool>                                  *cleared_cells_flags)
  {
    using namespace internal;
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    // maximum cosine of the angle between the normals of two facing branches
    constexpr number max_normal_product = -0.5;

    // The level set of the face neighbors (including the ghost cells) is required as well.
    VectorType level_set_relevant(dof_handler.locally_owned_dofs(),
                                  dealii::DoFTools::extract_locally_relevant_dofs(dof_handler),
                                  dof_handler.get_mpi_communicator());
    level_set_relevant.copy_locally_owned_data_from(level_set);
    level_set_relevant.update_ghost_values();

    const dealii::hp::QCollection<1> q_collection(dealii::QGauss<1>(fe_degree + 1));
    dealii::NonMatching::DiscreteQuadratureGenerator<dim> quadrature_generator(q_collection,
                                                                               dof_handler,
                                                                               level_set_relevant);

    // part of a face in the phase: face quadrature of the discrete level set of the cell (the
    // phase touches the face, if the quadrature is not empty, even between fixed sample points)
    dealii::NonMatching::DiscreteFaceQuadratureGenerator<dim> face_quadrature_generator(
      q_collection, dof_handler, level_set_relevant);

    // basis and sign check of the FE_DGQ elements (the same degree on all cells with DoFs)
    std::optional<TensorProductBasis<dim>> basis;
    std::optional<SignSampler<dim>>        sign_sampler;

    const auto evaluate_value = [&](const CutCellData<dim, number> &cell_data,
                                    const dealii::Point<dim>       &real_point) {
      return evaluate(*basis, cell_data, real_point).first;
    };

    // new DoF values of the cleared cells
    std::vector<
      std::pair<typename dealii::DoFHandler<dim>::active_cell_iterator, dealii::Vector<number>>>
      cleared_cells;

    // candidates for the clearing: remaining side and thickness of the removed phase (indexed by
    // the active cell index, including the ghost cells)
    const unsigned int  n_active_cells = dof_handler.get_triangulation().n_active_cells();
    std::vector<int>    remaining_sides(n_active_cells, 0);
    std::vector<number> removed_thicknesses(n_active_cells, std::numeric_limits<number>::max());

    std::vector<dealii::Point<dim>>             points;
    std::vector<dealii::Tensor<1, dim, number>> normals;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
          continue;

        quadrature_generator.generate(cell);
        const auto &surface_quadrature = quadrature_generator.get_surface_quadrature();
        if (surface_quadrature.size() == 0)
          continue;

        if (not basis)
          {
            basis.emplace(cell->get_fe());
            sign_sampler.emplace(cell->get_fe());
          }

        CutCellData<dim, number> cut_cell{AffineCellMap<dim>(cell),
                                          dealii::Vector<number>(cell->get_fe().n_dofs_per_cell()),
                                          {}};
        cell->get_dof_values(level_set_relevant, cut_cell.values);

        const number h      = cut_cell.map.cell_size;
        const number det    = std::abs(dealii::determinant(cut_cell.map.jacobian));
        const number length = thickness * h;

        // interface points, normals and measure of the interface in real space
        points.clear();
        normals.clear();
        number interface_measure = 0.;
        for (unsigned int q = 0; q < surface_quadrature.size(); ++q)
          {
            interface_measure +=
              surface_quadrature.weight(q) * det *
              (cut_cell.map.inverse_jacobian_transposed * surface_quadrature.normal_vector(q))
                .norm();

            const dealii::Point<dim> x   = cut_cell.map.to_real(surface_quadrature.point(q));
            const auto [value, gradient] = evaluate(*basis, cut_cell, x);
            (void)value;
            if (gradient.norm() > 0.)
              {
                points.push_back(x);
                normals.push_back(gradient / gradient.norm());
              }
          }

        if (interface_measure <= 0.)
          continue;

        // volumes of the phases phi > 0 (outside) and phi < 0 (inside) in real space
        number volume_positive = 0., volume_negative = 0.;
        for (const auto weight : quadrature_generator.get_outside_quadrature().get_weights())
          volume_positive += weight * det;
        for (const auto weight : quadrature_generator.get_inside_quadrature().get_weights())
          volume_negative += weight * det;

        // Closed faces of the phase: faces, at whose points in the phase (points of the face
        // quadrature of {s phi_K > 0}, at least two) the face neighbor has the other sign, i.e.,
        // the face acts as a boundary of the phase (e.g., a thin film of the phase along the
        // face). At least two points are required and all points of the face have to be closed,
        // such that small mismatches of the discontinuous level set at the crossings of a resolved
        // interface with the face are not considered. The points of the closed faces and the
        // directions into the cell are collected. Returns true, if the phase does not continue
        // into any face neighbor (faces at the boundary of the domain or to cells without DoFs
        // count as continuation). The face quadrature also detects a phase, which touches the face
        // only between fixed sample points (e.g., a flat cap of a resolved interface, which is
        // almost tangential to the face and continues into the face neighbor).
        std::vector<dealii::Point<dim>>             closed_points;
        std::vector<dealii::Tensor<1, dim, number>> closed_directions;
        std::vector<dealii::Point<dim>>             face_points;
        const auto collect_closed_face_points = [&](const int side) {
          closed_points.clear();
          closed_directions.clear();
          bool is_enclosed = true;
          for (const unsigned int f : cell->face_indices())
            {
              const bool is_open =
                cell->at_boundary(f) or cell->neighbor(f)->get_fe().n_dofs_per_cell() == 0;

              std::optional<CutCellData<dim, number>> neighbor_data;
              bool                                    is_closed_face = true;
              face_points.clear();

              face_quadrature_generator.generate(cell, f);
              const auto &face_quadrature = side > 0 ?
                                              face_quadrature_generator.get_outside_quadrature() :
                                              face_quadrature_generator.get_inside_quadrature();
              for (const auto &face_point : face_quadrature.get_points())
                {
                  // point of the unit face on the unit cell: the face coordinate k is the
                  // coordinate (n + 1 + k) mod dim of the cell, where n is the normal direction
                  // (cross section of the FaceQuadratureGenerator)
                  dealii::Point<dim> unit_point;
                  unit_point[f / 2] = f % 2;
                  for (unsigned int k = 0; k + 1 < dim; ++k)
                    unit_point[(f / 2 + 1 + k) % dim] = face_point[k];

                  const dealii::Point<dim> x = cut_cell.map.to_real(unit_point);
                  if (side * evaluate_value(cut_cell, x) <= 0.)
                    continue;

                  face_points.push_back(x);

                  if (is_open)
                    {
                      is_enclosed    = false;
                      is_closed_face = false;
                      continue;
                    }

                  if (not neighbor_data)
                    {
                      const auto neighbor = cell->neighbor(f);
                      neighbor_data.emplace(CutCellData<dim, number>{
                        AffineCellMap<dim>(neighbor),
                        dealii::Vector<number>(neighbor->get_fe().n_dofs_per_cell()),
                        {}});
                      neighbor->get_dof_values(level_set_relevant, neighbor_data->values);
                    }

                  if (side * evaluate_value(*neighbor_data, x) > 0.)
                    {
                      is_enclosed    = false;
                      is_closed_face = false;
                    }
                }

              if (is_closed_face and face_points.size() >= 2)
                {
                  // unit normal of the face pointing into the cell
                  dealii::Tensor<1, dim, number> direction;
                  direction[f / 2] = (f % 2 == 0) ? 1. : -1.;
                  direction        = cut_cell.map.inverse_jacobian_transposed * direction;
                  direction /= direction.norm();

                  for (const auto &x : face_points)
                    {
                      closed_points.push_back(x);
                      closed_directions.push_back(direction);
                    }
                }
            }
          return is_enclosed;
        };

        // (a) two boundaries of the phase of opposite orientation facing each other across the
        // phase: two branches of the interface or a branch of the interface and a closed face
        // (directions m pointing into the phase: s n for the interface, the inward normal for a
        // closed face)
        const auto has_facing_boundaries = [&](const int side) {
          const auto are_facing = [&](const dealii::Point<dim>             &x_i,
                                      const dealii::Tensor<1, dim, number> &m_i,
                                      const dealii::Point<dim>             &x_j,
                                      const dealii::Tensor<1, dim, number> &m_j) {
            const dealii::Tensor<1, dim, number> difference = x_j - x_i;
            return m_i * m_j < max_normal_product and difference.norm() < length and
                   difference * m_i > 0. and difference * m_j < 0.;
          };

          for (unsigned int i = 0; i < points.size(); ++i)
            {
              const dealii::Tensor<1, dim, number> m_i = side * normals[i];
              for (unsigned int j = i + 1; j < points.size(); ++j)
                if (are_facing(points[i], m_i, points[j], side * normals[j]))
                  return true;
              for (unsigned int k = 0; k < closed_points.size(); ++k)
                if (are_facing(points[i], m_i, closed_points[k], closed_directions[k]))
                  return true;
            }
          return false;
        };

        // thin phase (thickness 2 |K cap phase| / |Gamma cap K| below the threshold), which is
        // under-resolved by (a) or (b); the thinner phase is removed, if both phases qualify
        int    removed_side      = 0;
        number removed_thickness = std::numeric_limits<number>::max();
        for (const int side : {1, -1})
          {
            const number phase_thickness =
              2. * (side > 0 ? volume_positive : volume_negative) / interface_measure;
            if (phase_thickness >= length or phase_thickness >= removed_thickness)
              continue;

            // (b) the phase is enclosed by the interface and closed faces
            const bool is_enclosed = collect_closed_face_points(side);
            if (is_enclosed or has_facing_boundaries(side))
              {
                removed_side      = side;
                removed_thickness = phase_thickness;
              }
          }

        if (removed_side == 0)
          continue;

        // clear the cell from the removed phase: all nodal values on the side of the remaining
        // phase with a magnitude of at least epsilon * h
        const int              side    = -removed_side;
        const number           minimum = epsilon * h;
        dealii::Vector<number> values  = cut_cell.values;
        for (auto &value : values)
          value = side * std::max(side * value, minimum);

        // the polynomial might still change its sign between the nodes: constant value
        if (sign_sampler->min_signed_value(values, side) < 0.5 * minimum)
          {
            number mean = 0.;
            for (const auto value : cut_cell.values)
              mean += value;
            mean /= cut_cell.values.size();
            for (auto &value : values)
              value = side * std::max(side * mean, minimum);
          }

        cleared_cells.emplace_back(cell, std::move(values));
        remaining_sides[cell->active_cell_index()]     = side;
        removed_thicknesses[cell->active_cell_index()] = removed_thickness;
      }

    // communicate the candidates to the ghost cells
    using CandidateData = std::pair<int, number>;
    if (dynamic_cast<const dealii::parallel::TriangulationBase<dim> *>(
          &dof_handler.get_triangulation()) != nullptr)
      dealii::GridTools::exchange_cell_data_to_ghosts<CandidateData, dealii::Triangulation<dim>>(
        dof_handler.get_triangulation(),
        [&](const typename dealii::Triangulation<dim>::active_cell_iterator &cell)
          -> std::optional<CandidateData> {
          return CandidateData(remaining_sides[cell->active_cell_index()],
                               removed_thicknesses[cell->active_cell_index()]);
        },
        [&](const typename dealii::Triangulation<dim>::active_cell_iterator &cell,
            const CandidateData                                             &data) {
          remaining_sides[cell->active_cell_index()]     = data.first;
          removed_thicknesses[cell->active_cell_index()] = data.second;
        });

    // A cleared cell must not have a face neighbor, which has the other sign in the whole cell
    // after the clearing: the interface would lie on the shared face, where it is not represented
    // by any cut cell (e.g., a thin film of one phase and a thin filament of the other phase on
    // both sides of a face, which are cleared at the same time). Of two face neighbors, which
    // are cleared to different sides, only the one with the thinner removed phase is cleared
    // (ties: smaller cell id); a cell is not cleared next to a cell, which is not cut and has the
    // other sign.
    dealii::Vector<number> neighbor_values;
    const auto             is_accepted = [&](const auto &cell) {
      const unsigned int index = cell->active_cell_index();
      const int          side  = remaining_sides[index];
      for (const unsigned int f : cell->face_indices())
        {
          if (cell->at_boundary(f))
            continue;

          const auto neighbor = cell->neighbor(f);
          if (neighbor->has_children() or neighbor->get_fe().n_dofs_per_cell() == 0)
            continue;

          const unsigned int neighbor_index = neighbor->active_cell_index();
          if (remaining_sides[neighbor_index] == -side)
            {
              const number thickness          = removed_thicknesses[index];
              const number neighbor_thickness = removed_thicknesses[neighbor_index];
              if (neighbor_thickness < thickness or
                  (neighbor_thickness == thickness and neighbor->id() < cell->id()))
                return false;
            }
          else if (remaining_sides[neighbor_index] == 0)
            {
              neighbor_values.reinit(neighbor->get_fe().n_dofs_per_cell());
              neighbor->get_dof_values(level_set_relevant, neighbor_values);
              if (sign_sampler->min_signed_value(neighbor_values, -side) > 0.)
                return false;
            }
        }
      return true;
    };

    cleared_cells.erase(std::remove_if(cleared_cells.begin(),
                                       cleared_cells.end(),
                                       [&](const auto &cleared_cell) {
                                         return not is_accepted(cleared_cell.first);
                                       }),
                        cleared_cells.end());

    const bool has_ghosts = level_set.has_ghost_elements();
    if (has_ghosts)
      level_set.zero_out_ghost_values();

    for (const auto &[cell, values] : cleared_cells)
      {
        cell->set_dof_values(values, level_set);
        if (cleared_cells_flags != nullptr)
          (*cleared_cells_flags)[cell->active_cell_index()] = true;
      }

    if (has_ghosts)
      level_set.update_ghost_values();

    return dealii::Utilities::MPI::sum(static_cast<unsigned int>(cleared_cells.size()),
                                       dof_handler.get_mpi_communicator());
  }

  template <int dim, typename number>
  std::vector<bool>
  find_clipped_cells(const dealii::DoFHandler<dim>                            &dof_handler,
                     const dealii::LinearAlgebra::distributed::Vector<number> &level_set,
                     const std::vector<bool>                                  &is_cut_cell)
  {
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    const auto &triangulation = dof_handler.get_triangulation();
    AssertDimension(is_cut_cell.size(), triangulation.n_active_cells());

    std::vector<bool> is_clipped(triangulation.n_active_cells(), false);

    // cut flags of the locally owned and ghost cells
    std::vector<int> cut_flags(triangulation.n_active_cells(), 0);
    for (const auto &cell : triangulation.active_cell_iterators())
      if (cell->is_locally_owned())
        cut_flags[cell->active_cell_index()] = is_cut_cell[cell->active_cell_index()] ? 1 : 0;
    if (dynamic_cast<const dealii::parallel::TriangulationBase<dim> *>(&triangulation) != nullptr)
      dealii::GridTools::exchange_cell_data_to_ghosts<int, dealii::Triangulation<dim>>(
        triangulation,
        [&](const typename dealii::Triangulation<dim>::active_cell_iterator &cell)
          -> std::optional<int> { return cut_flags[cell->active_cell_index()]; },
        [&](const typename dealii::Triangulation<dim>::active_cell_iterator &cell,
            const int &flag) { cut_flags[cell->active_cell_index()] = flag; });

    // level set of the ghost cells (face and vertex neighbors)
    VectorType level_set_relevant(dof_handler.locally_owned_dofs(),
                                  dealii::DoFTools::extract_locally_relevant_dofs(dof_handler),
                                  dof_handler.get_mpi_communicator());
    level_set_relevant.copy_locally_owned_data_from(level_set);
    level_set_relevant.update_ghost_values();

    // cut cells (owned and ghost) at each vertex of the cut cells
    std::map<unsigned int, std::vector<typename dealii::DoFHandler<dim>::active_cell_iterator>>
      cut_cells_at_vertex;
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (not cell->is_artificial() and cut_flags[cell->active_cell_index()] == 1 and
          cell->get_fe().n_dofs_per_cell() > 0)
        for (const unsigned int v : cell->vertex_indices())
          cut_cells_at_vertex[cell->vertex_index(v)].push_back(cell);

    if (cut_cells_at_vertex.empty())
      return is_clipped;

    const auto &fe = dof_handler.get_fe_collection()[ActiveFEIndex::cut];
    AssertThrow(fe.has_support_points(),
                dealii::ExcMessage("The detection of clipped cells requires a nodal element."));

    // node of the FE_DGQ element at each vertex of the unit cell
    std::vector<unsigned int> vertex_node(dealii::GeometryInfo<dim>::vertices_per_cell,
                                          dealii::numbers::invalid_unsigned_int);
    for (unsigned int v = 0; v < dealii::GeometryInfo<dim>::vertices_per_cell; ++v)
      for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
        if (fe.get_unit_support_points()[i].distance(
              dealii::GeometryInfo<dim>::unit_cell_vertex(v)) < 1e-12)
          vertex_node[v] = i;
    for (const unsigned int node : vertex_node)
      AssertThrow(node != dealii::numbers::invalid_unsigned_int,
                  dealii::ExcMessage("The element has no support point at a vertex."));

    // traces of the cut cells on the faces (Gauss-Lobatto points)
    const dealii::Quadrature<dim - 1> face_quadrature = [&]() -> dealii::Quadrature<dim - 1> {
      if constexpr (dim == 1)
        return dealii::QGauss<0>(1);
      else
        return dealii::QGaussLobatto<dim - 1>(fe.degree + 2);
    }();
    dealii::FEFaceValues<dim> fe_face_values(fe, face_quadrature, dealii::update_values);
    std::vector<number>       face_values(face_quadrature.size());

    dealii::Vector<number> values(fe.n_dofs_per_cell());
    dealii::Vector<number> neighbor_values(fe.n_dofs_per_cell());

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or cut_flags[cell->active_cell_index()] == 1 or
            cell->get_fe().n_dofs_per_cell() == 0)
          continue;

        // sign of the cell (the polynomial of a cell, which is not cut, does not change its sign)
        values.reinit(cell->get_fe().n_dofs_per_cell());
        cell->get_dof_values(level_set_relevant, values);
        number mean = 0.;
        for (const auto value : values)
          mean += value;
        const int side = mean > 0. ? 1 : (mean < 0. ? -1 : 0);
        if (side == 0)
          continue;

        // tolerance for an interface through a vertex or along a face
        const number tolerance = 1e-10 * cell->minimum_vertex_distance();

        bool clipped = false;

        // (a) vertices shared with cut cells
        for (const unsigned int v : cell->vertex_indices())
          {
            if (clipped)
              break;
            const auto it = cut_cells_at_vertex.find(cell->vertex_index(v));
            if (it == cut_cells_at_vertex.end())
              continue;
            for (const auto &cut_cell : it->second)
              {
                unsigned int local_vertex = 0;
                while (cut_cell->vertex_index(local_vertex) != cell->vertex_index(v))
                  ++local_vertex;
                neighbor_values.reinit(cut_cell->get_fe().n_dofs_per_cell());
                cut_cell->get_dof_values(level_set_relevant, neighbor_values);
                if (side * neighbor_values[vertex_node[local_vertex]] < -tolerance)
                  {
                    clipped = true;
                    break;
                  }
              }
          }

        // (b) faces shared with cut cells
        for (const unsigned int f : cell->face_indices())
          {
            if (clipped)
              break;
            if (cell->at_boundary(f))
              continue;
            const auto neighbor = cell->neighbor(f);
            if (neighbor->has_children() or neighbor->is_artificial() or
                cut_flags[neighbor->active_cell_index()] != 1)
              continue;

            fe_face_values.reinit(typename dealii::Triangulation<dim>::cell_iterator(neighbor),
                                  cell->neighbor_of_neighbor(f));
            neighbor_values.reinit(neighbor->get_fe().n_dofs_per_cell());
            neighbor->get_dof_values(level_set_relevant, neighbor_values);
            for (const unsigned int q : fe_face_values.quadrature_point_indices())
              {
                number value = 0.;
                for (unsigned int i = 0; i < neighbor_values.size(); ++i)
                  value += neighbor_values[i] * fe_face_values.shape_value(i, q);
                if (side * value < -tolerance)
                  {
                    clipped = true;
                    break;
                  }
              }
          }

        is_clipped[cell->active_cell_index()] = clipped;
      }

    return is_clipped;
  }

  template <int dim, typename number>
  std::pair<unsigned int, unsigned int>
  count_new_cut_cells(const dealii::Mapping<dim>                               &mapping,
                      const dealii::DoFHandler<dim>                            &dof_handler,
                      const dealii::LinearAlgebra::distributed::Vector<number> &level_set,
                      const std::vector<bool>                                  &is_cut_cell)
  {
    AssertDimension(is_cut_cell.size(), dof_handler.get_triangulation().n_active_cells());

    const auto &fe = dof_handler.get_fe_collection()[ActiveFEIndex::cut];

    const dealii::Quadrature<dim - 1> face_quadrature = [&]() -> dealii::Quadrature<dim - 1> {
      if constexpr (dim == 1)
        return dealii::QGauss<0>(1);
      else
        return dealii::QGaussLobatto<dim - 1>(fe.degree + 2);
    }();
    dealii::FEFaceValues<dim> fe_face_values(mapping, fe, face_quadrature, dealii::update_values);
    std::vector<number>       values(face_quadrature.size());

    unsigned int n_new_cut_cells = 0, n_unconnected_cut_cells = 0;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or not is_cut_cell[cell->active_cell_index()] or
            cell->active_fe_index() == ActiveFEIndex::cut or cell->get_fe().n_dofs_per_cell() == 0)
          continue;

        ++n_new_cut_cells;

        bool is_connected = false;
        for (const unsigned int f : cell->face_indices())
          {
            if (cell->at_boundary(f) or cell->neighbor(f)->active_fe_index() != ActiveFEIndex::cut)
              continue;

            fe_face_values.reinit(cell, f);
            fe_face_values.get_function_values(level_set, values);
            if (*std::min_element(values.begin(), values.end()) <= 0. and
                *std::max_element(values.begin(), values.end()) >= 0.)
              is_connected = true;
          }

        if (not is_connected)
          ++n_unconnected_cut_cells;
      }

    const MPI_Comm comm = dof_handler.get_mpi_communicator();
    return {dealii::Utilities::MPI::sum(n_new_cut_cells, comm),
            dealii::Utilities::MPI::sum(n_unconnected_cut_cells, comm)};
  }

  template <int dim, typename number>
  unsigned int
  count_band_boundary_crossings(const dealii::Mapping<dim>    &mapping,
                                const dealii::DoFHandler<dim> &dof_handler,
                                const dealii::LinearAlgebra::distributed::Vector<number> &level_set,
                                const std::vector<int> *cell_signs)
  {
    if (cell_signs != nullptr)
      AssertDimension(cell_signs->size(), dof_handler.get_triangulation().n_active_cells());

    const auto &fe = dof_handler.get_fe_collection()[ActiveFEIndex::cut];

    const dealii::Quadrature<dim - 1> face_quadrature = [&]() -> dealii::Quadrature<dim - 1> {
      if constexpr (dim == 1)
        return dealii::QGauss<0>(1);
      else
        return dealii::QGaussLobatto<dim - 1>(fe.degree + 2);
    }();
    dealii::FEFaceValues<dim> fe_face_values(mapping, fe, face_quadrature, dealii::update_values);
    std::vector<number>       values(face_quadrature.size());

    unsigned int n_faces = 0;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
          continue;

        for (const unsigned int f : cell->face_indices())
          {
            if (cell->at_boundary(f))
              continue;

            const auto neighbor = cell->neighbor(f);
            if (neighbor->has_children() or neighbor->get_fe().n_dofs_per_cell() > 0)
              continue;

            fe_face_values.reinit(cell, f);
            fe_face_values.get_function_values(level_set, values);
            const number min_value = *std::min_element(values.begin(), values.end());
            const number max_value = *std::max_element(values.begin(), values.end());

            const int side = cell_signs ? (*cell_signs)[neighbor->active_cell_index()] : 0;
            if ((side > 0 and min_value <= 0.) or (side < 0 and max_value >= 0.) or
                (side == 0 and min_value <= 0. and max_value >= 0.))
              ++n_faces;
          }
      }

    return dealii::Utilities::MPI::sum(n_faces, dof_handler.get_mpi_communicator());
  }

  template <int dim>
  ModalSmoothnessIndicator<dim>::ModalSmoothnessIndicator(const dealii::FiniteElement<dim> &fe)
  {
    const unsigned int p = fe.degree;
    AssertThrow(p >= 2 and fe.n_dofs_per_cell() == dealii::Utilities::pow(p + 1, dim),
                dealii::ExcMessage("The modal smoothness indicator requires a tensor-product "
                                   "element of degree p >= 2."));

    const unsigned int n_modes = dealii::Utilities::pow(p + 1, dim);

    // Legendre polynomials on the unit interval
    std::vector<dealii::Polynomials::Legendre> legendre;
    for (unsigned int k = 0; k <= p; ++k)
      legendre.emplace_back(k);

    // Gauss quadrature, which integrates the products of the Legendre modes and the shape
    // functions exactly
    const dealii::QGauss<dim> quadrature(p + 1);

    modal_projection.reinit(n_modes, fe.n_dofs_per_cell());
    mode_norms.assign(n_modes, 0.);
    is_highest_mode.assign(n_modes, false);

    for (unsigned int a = 0; a < n_modes; ++a)
      {
        // tensor index of the mode (the constant mode a = 0 first)
        std::array<unsigned int, dim> alpha;
        for (unsigned int d = 0, index = a; d < dim; ++d)
          {
            alpha[d] = index % (p + 1);
            index /= (p + 1);
            if (alpha[d] == p)
              is_highest_mode[a] = true;
          }

        for (unsigned int q = 0; q < quadrature.size(); ++q)
          {
            double mode = 1.;
            for (unsigned int d = 0; d < dim; ++d)
              mode *= legendre[alpha[d]].value(quadrature.point(q)[d]);

            mode_norms[a] += mode * mode * quadrature.weight(q);
            for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
              modal_projection(a, i) +=
                mode * fe.shape_value(i, quadrature.point(q)) * quadrature.weight(q);
          }

        for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
          modal_projection(a, i) /= mode_norms[a];
      }
  }

  template <int dim>
  unsigned int
  n_cells_with_active_fe_index(const dealii::DoFHandler<dim> &dof_handler,
                               const unsigned int             active_fe_index)
  {
    unsigned int n_cells = 0;
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned() and cell->active_fe_index() == active_fe_index)
        ++n_cells;

    return dealii::Utilities::MPI::sum(n_cells, dof_handler.get_mpi_communicator());
  }

  // explicit instantiations
#define MPDG_INSTANTIATE_NARROW_BAND_UTILITIES(dim)                                             \
  template dealii::hp::FECollection<dim> create_band_fe_collection<dim>(const unsigned int);    \
  template dealii::hp::FECollection<dim> create_cut_fe_collection<dim>(const unsigned int);     \
  template dealii::hp::FECollection<dim> create_layered_band_fe_collection<dim>(                \
    const unsigned int);                                                                        \
  template dealii::hp::FECollection<dim> create_layered_cut_fe_collection<dim>(                 \
    const unsigned int);                                                                        \
  template std::vector<unsigned int> compute_layers<dim>(const dealii::Triangulation<dim> &,    \
                                                         const std::vector<bool> &,             \
                                                         const unsigned int);                   \
  template std::vector<unsigned int> distribute_dofs_layered<dim>(                              \
    const std::vector<bool> &,                                                                  \
    const unsigned int,                                                                         \
    dealii::DoFHandler<dim> &,                                                                  \
    const dealii::hp::FECollection<dim> &,                                                      \
    dealii::DoFHandler<dim> &,                                                                  \
    const dealii::hp::FECollection<dim> &);                                                     \
  template void copy_common_cell_values<dim, double>(                                           \
    const dealii::DoFHandler<dim> &,                                                            \
    const dealii::LinearAlgebra::distributed::Vector<double> &,                                 \
    const dealii::DoFHandler<dim> &,                                                            \
    dealii::LinearAlgebra::distributed::Vector<double> &);                                      \
  template std::vector<bool> compute_cut_cells<dim, double>(                                    \
    const dealii::DoFHandler<dim> &,                                                            \
    const dealii::LinearAlgebra::distributed::Vector<double> &,                                 \
    const unsigned int,                                                                         \
    std::vector<double> *);                                                                     \
  template std::vector<double> compute_cut_cell_volume_fractions<dim, double>(                  \
    const dealii::DoFHandler<dim> &,                                                            \
    const dealii::LinearAlgebra::distributed::Vector<double> &,                                 \
    const unsigned int);                                                                        \
  template void distribute_dofs<dim>(                                                           \
    const std::vector<bool> &,                                                                  \
    dealii::DoFHandler<dim> &,                                                                  \
    const dealii::hp::FECollection<dim> &,                                                      \
    dealii::DoFHandler<dim> &,                                                                  \
    const dealii::hp::FECollection<dim> &,                                                      \
    const std::vector<bool> &,                                                                  \
    const std::function<bool(const typename dealii::Triangulation<dim>::active_cell_iterator &, \
                             const unsigned int)> &);                                           \
  template bool is_downstream_face<dim>(                                                        \
    const dealii::Mapping<dim> &,                                                               \
    dealii::Function<dim> &,                                                                    \
    const std::vector<double> &,                                                                \
    const typename dealii::Triangulation<dim>::active_cell_iterator &,                          \
    const unsigned int,                                                                         \
    const unsigned int);                                                                        \
  template std::vector<dealii::Vector<double>> extract_cell_values<dim, double>(                \
    const dealii::DoFHandler<dim> &,                                                            \
    const dealii::LinearAlgebra::distributed::Vector<double> &,                                 \
    const std::vector<bool> &);                                                                 \
  template void set_cell_values<dim, double>(                                                   \
    const dealii::DoFHandler<dim> &,                                                            \
    const std::vector<dealii::Vector<double>> &,                                                \
    dealii::LinearAlgebra::distributed::Vector<double> &);                                      \
  template void copy_cut_cell_values<dim, double>(                                              \
    const dealii::DoFHandler<dim> &,                                                            \
    const dealii::LinearAlgebra::distributed::Vector<double> &,                                 \
    const dealii::DoFHandler<dim> &,                                                            \
    dealii::LinearAlgebra::distributed::Vector<double> &);                                      \
  template double compute_eikonal_error<dim, double>(                                           \
    const dealii::Mapping<dim> &,                                                               \
    const dealii::DoFHandler<dim> &,                                                            \
    const dealii::LinearAlgebra::distributed::Vector<double> &);                                \
  template unsigned int limit_sign_changes<dim, double>(                                        \
    const dealii::Mapping<dim> &,                                                               \
    const dealii::DoFHandler<dim> &,                                                            \
    dealii::LinearAlgebra::distributed::Vector<double> &,                                       \
    const double);                                                                              \
  template std::vector<int> compute_sides_from_cut_cells<dim, double>(                          \
    const dealii::Mapping<dim> &,                                                               \
    const dealii::DoFHandler<dim> &,                                                            \
    const dealii::LinearAlgebra::distributed::Vector<double> &,                                 \
    const std::vector<bool> &);                                                                 \
  template unsigned int clamp_signs<dim, double>(                                               \
    const dealii::DoFHandler<dim> &,                                                            \
    dealii::LinearAlgebra::distributed::Vector<double> &,                                       \
    const std::vector<int> &,                                                                   \
    const double);                                                                              \
  template unsigned int remove_thin_features<dim, double>(                                      \
    const dealii::DoFHandler<dim> &,                                                            \
    dealii::LinearAlgebra::distributed::Vector<double> &,                                       \
    const unsigned int,                                                                         \
    const double,                                                                               \
    const double,                                                                               \
    std::vector<bool> *);                                                                       \
  template std::vector<bool> find_clipped_cells<dim, double>(                                   \
    const dealii::DoFHandler<dim> &,                                                            \
    const dealii::LinearAlgebra::distributed::Vector<double> &,                                 \
    const std::vector<bool> &);                                                                 \
  template std::pair<unsigned int, unsigned int> count_new_cut_cells<dim, double>(              \
    const dealii::Mapping<dim> &,                                                               \
    const dealii::DoFHandler<dim> &,                                                            \
    const dealii::LinearAlgebra::distributed::Vector<double> &,                                 \
    const std::vector<bool> &);                                                                 \
  template unsigned int count_band_boundary_crossings<dim, double>(                             \
    const dealii::Mapping<dim> &,                                                               \
    const dealii::DoFHandler<dim> &,                                                            \
    const dealii::LinearAlgebra::distributed::Vector<double> &,                                 \
    const std::vector<int> *);                                                                  \
  template unsigned int n_cells_with_active_fe_index<dim>(const dealii::DoFHandler<dim> &,      \
                                                          const unsigned int);

  MPDG_INSTANTIATE_NARROW_BAND_UTILITIES(1)
  MPDG_INSTANTIATE_NARROW_BAND_UTILITIES(2)
  MPDG_INSTANTIATE_NARROW_BAND_UTILITIES(3)

#undef MPDG_INSTANTIATE_NARROW_BAND_UTILITIES

  template class ModalSmoothnessIndicator<1>;
  template class ModalSmoothnessIndicator<2>;
  template class ModalSmoothnessIndicator<3>;
} // namespace MeltPoolDG::LevelSet::NarrowBand
