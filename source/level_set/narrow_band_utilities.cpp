#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/full_matrix.h>

#include <deal.II/non_matching/quadrature_generator.h>

#include <meltpooldg/level_set/narrow_band_utilities.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

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
  distribute_dofs(const std::vector<bool>             &is_cut_cell,
                  dealii::DoFHandler<dim>             &dof_handler_cut,
                  const dealii::hp::FECollection<dim> &fe_collection_cut,
                  dealii::DoFHandler<dim>             &dof_handler_band,
                  const dealii::hp::FECollection<dim> &fe_collection_band,
                  const std::vector<bool>             &is_small_cut_cell)
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
            // neighbors
            if (neighbor->active_fe_index() == ActiveFEIndex::cut)
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
        if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
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
#define MPDG_INSTANTIATE_NARROW_BAND_UTILITIES(dim)                                                        \
  template dealii::hp::FECollection<dim> create_band_fe_collection<dim>(const unsigned int);               \
  template dealii::hp::FECollection<dim> create_cut_fe_collection<dim>(const unsigned int);                \
  template std::vector<bool>             compute_cut_cells<dim, double>(                                   \
    const dealii::DoFHandler<dim> &,                                                           \
    const dealii::LinearAlgebra::distributed::Vector<double> &,                                \
    const unsigned int,                                                                        \
    std::vector<double> *);                                                                    \
  template std::vector<double> compute_cut_cell_volume_fractions<dim, double>(                             \
    const dealii::DoFHandler<dim> &,                                                                       \
    const dealii::LinearAlgebra::distributed::Vector<double> &,                                            \
    const unsigned int);                                                                                   \
  template void                                distribute_dofs<dim>(const std::vector<bool> &,             \
                                     dealii::DoFHandler<dim> &,             \
                                     const dealii::hp::FECollection<dim> &, \
                                     dealii::DoFHandler<dim> &,             \
                                     const dealii::hp::FECollection<dim> &, \
                                     const std::vector<bool> &);            \
  template std::vector<dealii::Vector<double>> extract_cell_values<dim, double>(                           \
    const dealii::DoFHandler<dim> &,                                                                       \
    const dealii::LinearAlgebra::distributed::Vector<double> &,                                            \
    const std::vector<bool> &);                                                                            \
  template void set_cell_values<dim, double>(                                                              \
    const dealii::DoFHandler<dim> &,                                                                       \
    const std::vector<dealii::Vector<double>> &,                                                           \
    dealii::LinearAlgebra::distributed::Vector<double> &);                                                 \
  template void copy_cut_cell_values<dim, double>(                                                         \
    const dealii::DoFHandler<dim> &,                                                                       \
    const dealii::LinearAlgebra::distributed::Vector<double> &,                                            \
    const dealii::DoFHandler<dim> &,                                                                       \
    dealii::LinearAlgebra::distributed::Vector<double> &);                                                 \
  template double compute_eikonal_error<dim, double>(                                                      \
    const dealii::Mapping<dim> &,                                                                          \
    const dealii::DoFHandler<dim> &,                                                                       \
    const dealii::LinearAlgebra::distributed::Vector<double> &);                                           \
  template unsigned int limit_sign_changes<dim, double>(                                                   \
    const dealii::Mapping<dim> &,                                                                          \
    const dealii::DoFHandler<dim> &,                                                                       \
    dealii::LinearAlgebra::distributed::Vector<double> &,                                                  \
    const double);                                                                                         \
  template unsigned int n_cells_with_active_fe_index<dim>(const dealii::DoFHandler<dim> &,                 \
                                                          const unsigned int);

  MPDG_INSTANTIATE_NARROW_BAND_UTILITIES(1)
  MPDG_INSTANTIATE_NARROW_BAND_UTILITIES(2)
  MPDG_INSTANTIATE_NARROW_BAND_UTILITIES(3)

#undef MPDG_INSTANTIATE_NARROW_BAND_UTILITIES
} // namespace MeltPoolDG::LevelSet::NarrowBand
