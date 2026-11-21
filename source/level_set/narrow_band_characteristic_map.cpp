#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/vector.h>

#include <meltpooldg/level_set/narrow_band_characteristic_map.hpp>
#include <meltpooldg/level_set/narrow_band_closest_point_tools.hpp>
#include <meltpooldg/utilities/journal.hpp>

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>

namespace MeltPoolDG::LevelSet
{
  namespace
  {
    /**
     * FE collection of the continuous field: FE_Q on the candidate cells of the reference (index
     * 0) and a non-dominating FE_Nothing otherwise (index 1, no constraints at the faces to the
     * other cells).
     */
    template <int dim>
    dealii::hp::FECollection<dim>
    create_fe_collection(const unsigned int fe_degree)
    {
      dealii::hp::FECollection<dim> fe_collection;
      fe_collection.push_back(dealii::FE_Q<dim>(fe_degree));
      fe_collection.push_back(dealii::FE_Nothing<dim>());
      return fe_collection;
    }

    /**
     * Distance of a unit point from the reference cell in unit coordinates (zero inside).
     */
    template <int dim>
    double
    distance_from_unit_cell(const dealii::Point<dim> &unit_point)
    {
      double distance = 0.;
      for (unsigned int d = 0; d < dim; ++d)
        distance = std::max({distance, -unit_point[d], unit_point[d] - 1.});
      return distance;
    }
  } // namespace

  template <int dim, typename number>
  NarrowBandCharacteristicMap<dim, number>::NarrowBandCharacteristicMap(
    const dealii::Triangulation<dim> &triangulation,
    const unsigned int                fe_degree,
    const dealii::ConditionalOStream &pcout)
    : pcout(pcout)
    , fe_collection(create_fe_collection<dim>(fe_degree))
    , dof_handler(triangulation)
  {}

  template <int dim, typename number>
  void
  NarrowBandCharacteristicMap<dim, number>::reset(const dealii::DoFHandler<dim> &dof_handler_band)
  {
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        cell->set_active_fe_index(
          cell->as_dof_handler_iterator(dof_handler_band)->get_fe().n_dofs_per_cell() > 0 ? 0 : 1);

    dof_handler.distribute_dofs(fe_collection);

    const MPI_Comm         mpi_comm = dof_handler.get_mpi_communicator();
    const dealii::IndexSet locally_relevant_dofs =
      dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);
    for (auto &values : displacement_values)
      {
        values.reinit(dof_handler.locally_owned_dofs(), locally_relevant_dofs, mpi_comm);
        values.update_ghost_values();
      }

    vertex_to_cells = dealii::GridTools::vertex_to_cell_map(dof_handler.get_triangulation());
    is_initialized  = true;
  }

  template <int dim, typename number>
  typename NarrowBandCharacteristicMap<dim, number>::Statistics
  NarrowBandCharacteristicMap<dim, number>::advance(dealii::Function<dim> &velocity,
                                                    const number           time,
                                                    const number           time_step)
  {
    using NarrowBand::internal::AffineCellMap;

    AssertThrow(velocity.n_components == dim,
                dealii::ExcMessage("The velocity function needs dim components."));
    AssertThrow(is_initialized, dealii::ExcMessage("The characteristic map has not been reset."));

    // classical fourth-order Runge-Kutta step backward in time from t^{n+1} to t^n
    const number old_time = velocity.get_time();
    const auto   u        = [&](const dealii::Point<dim> &x, const number t) {
      velocity.set_time(t);
      dealii::Tensor<1, dim> result;
      for (unsigned int d = 0; d < dim; ++d)
        result[d] = velocity.value(x, d);
      return result;
    };
    const auto backward_step = [&](const dealii::Point<dim> &x) {
      const number       h  = -time_step;
      const number       t1 = time + time_step;
      dealii::Point<dim> y  = x;
      const auto         k1 = u(y, t1);
      const auto         k2 = u(y + 0.5 * h * k1, t1 + 0.5 * h);
      const auto         k3 = u(y + 0.5 * h * k2, t1 + 0.5 * h);
      const auto         k4 = u(y + h * k3, t1 + h);
      y += (h / 6.) * (k1 + 2. * k2 + 2. * k3 + k4);
      return y;
    };

    const auto &fe                  = fe_collection[0];
    const auto &unit_support_points = fe.get_unit_support_points();

    std::array<VectorType, dim> new_values;
    for (unsigned int d = 0; d < dim; ++d)
      new_values[d].reinit(displacement_values[d]);

    // each locally owned node is updated once
    const dealii::IndexSet &locally_owned_dofs = dof_handler.locally_owned_dofs();
    std::vector<bool>       is_updated(locally_owned_dofs.n_elements(), false);

    Statistics                                   statistics;
    std::vector<dealii::types::global_dof_index> dof_indices(fe.n_dofs_per_cell());
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
          continue;

        cell->get_dof_indices(dof_indices);
        const AffineCellMap<dim> map(cell);

        for (unsigned int i = 0; i < dof_indices.size(); ++i)
          {
            const auto dof = dof_indices[i];
            if (not locally_owned_dofs.is_element(dof))
              continue;
            const auto index = locally_owned_dofs.index_within_set(dof);
            if (is_updated[index])
              continue;
            is_updated[index] = true;

            const dealii::Point<dim> x = map.to_real(unit_support_points[i]);
            const dealii::Point<dim> y = backward_step(x);

            bool                         extrapolated = false;
            const dealii::Tensor<1, dim> old_displacement =
              evaluate_in_neighborhood(cell, y, extrapolated);

            dealii::Tensor<1, dim> displacement;
            for (unsigned int d = 0; d < dim; ++d)
              {
                displacement[d]    = (x[d] - y[d]) + old_displacement[d];
                new_values[d](dof) = displacement[d];
              }

            ++statistics.n_nodes;
            if (extrapolated)
              ++statistics.n_nodes_extrapolated;
            statistics.max_step_displacement =
              std::max<number>(statistics.max_step_displacement, x.distance(y) / map.cell_size);
            statistics.max_displacement =
              std::max<number>(statistics.max_displacement, displacement.norm() / map.cell_size);
          }
      }

    velocity.set_time(old_time);

    for (unsigned int d = 0; d < dim; ++d)
      {
        displacement_values[d].copy_locally_owned_data_from(new_values[d]);
        displacement_values[d].update_ghost_values();
      }

    const MPI_Comm mpi_comm = dof_handler.get_mpi_communicator();
    statistics.n_nodes      = dealii::Utilities::MPI::sum(statistics.n_nodes, mpi_comm);
    statistics.n_nodes_extrapolated =
      dealii::Utilities::MPI::sum(statistics.n_nodes_extrapolated, mpi_comm);
    statistics.max_step_displacement =
      dealii::Utilities::MPI::max(statistics.max_step_displacement, mpi_comm);
    statistics.max_displacement =
      dealii::Utilities::MPI::max(statistics.max_displacement, mpi_comm);

    std::ostringstream str;
    str << "characteristic map: " << statistics.n_nodes << " nodes";
    if (statistics.n_nodes_extrapolated > 0)
      str << " (" << statistics.n_nodes_extrapolated
          << " extrapolated at the boundary of the candidate cells)";
    str << ", max. displacement / h of the time step: " << std::scientific << std::setprecision(2)
        << statistics.max_step_displacement
        << ", since the reference: " << statistics.max_displacement;
    Journal::print_line(pcout, str.str(), "narrow_band");

    return statistics;
  }

  template <int dim, typename number>
  dealii::Tensor<1, dim>
  NarrowBandCharacteristicMap<dim, number>::evaluate_in_neighborhood(
    const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
    const dealii::Point<dim>                                     &point,
    bool                                                         &extrapolated) const
  {
    using NarrowBand::internal::AffineCellMap;

    const auto &fe       = fe_collection[0];
    const auto  evaluate = [&](const typename dealii::DoFHandler<dim>::cell_iterator &dof_cell,
                              const dealii::Point<dim>                              &unit_point) {
      dealii::Vector<number> values(fe.n_dofs_per_cell());
      std::vector<number>    shape_values(fe.n_dofs_per_cell());
      for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
        shape_values[j] = fe.shape_value(j, unit_point);

      dealii::Tensor<1, dim> result;
      for (unsigned int d = 0; d < dim; ++d)
        {
          dof_cell->get_dof_values(displacement_values[d], values);
          for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
            result[d] += values[j] * shape_values[j];
        }
      return result;
    };

    // usually, the point lies in the cell itself
    extrapolated = false;
    {
      const dealii::Point<dim> unit_point = AffineCellMap<dim>(cell).to_unit(point);
      if (distance_from_unit_cell(unit_point) <= 1e-10)
        return evaluate(cell, unit_point);
    }

    // the cell itself and its vertex neighbors with DoFs (locally owned or ghost cells)
    std::vector<typename dealii::DoFHandler<dim>::cell_iterator> candidates{cell};
    for (const unsigned int v : cell->vertex_indices())
      for (const auto &other : vertex_to_cells[cell->vertex_index(v)])
        {
          if (other->is_artificial())
            continue;
          const auto other_dof = other->as_dof_handler_iterator(dof_handler);
          if (other_dof->get_fe().n_dofs_per_cell() == 0 or
              std::find(candidates.begin(), candidates.end(), other_dof) != candidates.end())
            continue;
          candidates.push_back(other_dof);
        }

    // cell containing the point; otherwise the closest cell (extrapolation)
    unsigned int       best          = 0;
    double             best_distance = std::numeric_limits<double>::max();
    dealii::Point<dim> best_unit_point;
    for (unsigned int c = 0; c < candidates.size(); ++c)
      {
        const dealii::Point<dim> unit_point = AffineCellMap<dim>(candidates[c]).to_unit(point);
        const double             distance   = distance_from_unit_cell(unit_point);
        if (distance < best_distance)
          {
            best            = c;
            best_distance   = distance;
            best_unit_point = unit_point;
          }
        if (distance <= 1e-10)
          break;
      }
    extrapolated = best_distance > 1e-10;

    return evaluate(candidates[best], best_unit_point);
  }

  template <int dim, typename number>
  dealii::Tensor<1, dim>
  NarrowBandCharacteristicMap<dim, number>::displacement(
    const typename dealii::Triangulation<dim>::cell_iterator &cell,
    const dealii::Point<dim>                                 &unit_point) const
  {
    const auto  dof_cell = cell->as_dof_handler_iterator(dof_handler);
    const auto &fe       = dof_cell->get_fe();
    AssertThrow(fe.n_dofs_per_cell() > 0,
                dealii::ExcMessage("The characteristic map is not defined on this cell."));

    dealii::Vector<number> values(fe.n_dofs_per_cell());
    dealii::Tensor<1, dim> result;
    for (unsigned int d = 0; d < dim; ++d)
      {
        dof_cell->get_dof_values(displacement_values[d], values);
        for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
          result[d] += values[j] * fe.shape_value(j, unit_point);
      }
    return result;
  }

  template <int dim, typename number>
  dealii::Point<dim>
  NarrowBandCharacteristicMap<dim, number>::foot_point(
    const typename dealii::Triangulation<dim>::cell_iterator &cell,
    const dealii::Point<dim>                                 &unit_point) const
  {
    const NarrowBand::internal::AffineCellMap<dim> map(cell);
    return map.to_real(unit_point) - displacement(cell, unit_point);
  }

  template <int dim, typename number>
  std::vector<number>
  NarrowBandCharacteristicMap<dim, number>::get_cell_values(
    const typename dealii::Triangulation<dim>::cell_iterator &cell) const
  {
    const auto         dof_cell = cell->as_dof_handler_iterator(dof_handler);
    const unsigned int n_dofs   = dof_cell->get_fe().n_dofs_per_cell();

    std::vector<number>    result;
    dealii::Vector<number> values(n_dofs);
    for (unsigned int d = 0; d < dim and n_dofs > 0; ++d)
      {
        dof_cell->get_dof_values(displacement_values[d], values);
        result.insert(result.end(), values.begin(), values.end());
      }
    return result;
  }

  template <int dim, typename number>
  void
  NarrowBandCharacteristicMap<dim, number>::set_cell_values(
    const typename dealii::Triangulation<dim>::cell_iterator &cell,
    const std::vector<number>                                &values)
  {
    const auto         dof_cell = cell->as_dof_handler_iterator(dof_handler);
    const unsigned int n_dofs   = dof_cell->get_fe().n_dofs_per_cell();
    AssertThrow(values.size() == dim * n_dofs,
                dealii::ExcMessage("The restart data does not match the DoFs of the "
                                   "characteristic map."));
    if (n_dofs == 0)
      return;

    dealii::Vector<number> cell_values(n_dofs);
    for (unsigned int d = 0; d < dim; ++d)
      {
        std::copy(values.begin() + d * n_dofs,
                  values.begin() + (d + 1) * n_dofs,
                  cell_values.begin());
        dof_cell->set_dof_values(cell_values, displacement_values[d]);
      }
  }

  template <int dim, typename number>
  void
  NarrowBandCharacteristicMap<dim, number>::finish_set_cell_values()
  {
    for (auto &values : displacement_values)
      values.update_ghost_values();
  }

  template <int dim, typename number>
  const dealii::DoFHandler<dim> &
  NarrowBandCharacteristicMap<dim, number>::get_dof_handler() const
  {
    return dof_handler;
  }

  template <int dim, typename number>
  const std::array<typename NarrowBandCharacteristicMap<dim, number>::VectorType, dim> &
  NarrowBandCharacteristicMap<dim, number>::get_displacement() const
  {
    return displacement_values;
  }

  template class NarrowBandCharacteristicMap<1, double>;
  template class NarrowBandCharacteristicMap<2, double>;
  template class NarrowBandCharacteristicMap<3, double>;
} // namespace MeltPoolDG::LevelSet
