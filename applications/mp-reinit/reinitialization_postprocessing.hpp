#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>

#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/utilities/journal.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace MeltPoolDG::LevelSet
{
  /**
   * @brief Compute and print the errors of a level-set field with respect to an exact signed
   * distance function.
   *
   * The relative L2-error of the level-set field and the L2-norm of the eikonal residual
   * |grad phi| - 1 (normalized by the square root of the integration volume) are evaluated on all
   * cells, where the level-set field is defined, i.e., cells with FE_Nothing elements (e.g., for a
   * level-set field which is only defined on the intersected cells) are skipped.
   *
   * @param generic_data_out Output data containing the level-set vector and its DoFHandler.
   * @param vector_name Name of the level-set vector in @p generic_data_out.
   * @param exact_signed_distance Exact signed distance function.
   * @param mpi_communicator MPI communicator.
   * @param label Label of the printed line.
   */
  template <int dim, typename number>
  void
  print_signed_distance_errors(const GenericDataOut<dim, number>   &generic_data_out,
                               const std::string                   &vector_name,
                               const dealii::Function<dim, number> &exact_signed_distance,
                               const MPI_Comm                       mpi_communicator,
                               const std::string                   &label)
  {
    const auto &dof_handler = generic_data_out.get_dof_handler(vector_name);
    const auto &level_set   = generic_data_out.get_vector(vector_name);

    const bool has_ghost_elements = level_set.has_ghost_elements();
    if (not has_ghost_elements)
      level_set.update_ghost_values();

    const auto                  &fe_collection = dof_handler.get_fe_collection();
    dealii::hp::QCollection<dim> q_collection;
    for (unsigned int i = 0; i < fe_collection.size(); ++i)
      q_collection.push_back(dealii::QGauss<dim>(fe_collection[i].degree + 3));
    dealii::hp::MappingCollection<dim> mapping_collection(generic_data_out.get_mapping());

    dealii::hp::FEValues<dim> hp_fe_values(mapping_collection,
                                           fe_collection,
                                           q_collection,
                                           dealii::update_values | dealii::update_gradients |
                                             dealii::update_JxW_values |
                                             dealii::update_quadrature_points);

    std::vector<number>                         values;
    std::vector<dealii::Tensor<1, dim, number>> gradients;

    number error_value = 0, norm_exact = 0, error_eikonal = 0, volume = 0;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
          continue;

        hp_fe_values.reinit(cell);
        const auto &fe_values = hp_fe_values.get_present_fe_values();

        values.resize(fe_values.n_quadrature_points);
        gradients.resize(fe_values.n_quadrature_points);
        fe_values.get_function_values(level_set, values);
        fe_values.get_function_gradients(level_set, gradients);

        for (const unsigned int q : fe_values.quadrature_point_indices())
          {
            const number exact = exact_signed_distance.value(fe_values.quadrature_point(q), 0);
            error_value += dealii::Utilities::fixed_power<2>(values[q] - exact) * fe_values.JxW(q);
            norm_exact += exact * exact * fe_values.JxW(q);
            error_eikonal +=
              dealii::Utilities::fixed_power<2>(gradients[q].norm() - 1.0) * fe_values.JxW(q);
            volume += fe_values.JxW(q);
          }
      }

    if (not has_ghost_elements)
      level_set.zero_out_ghost_values();

    error_value   = dealii::Utilities::MPI::sum(error_value, mpi_communicator);
    norm_exact    = dealii::Utilities::MPI::sum(norm_exact, mpi_communicator);
    error_eikonal = dealii::Utilities::MPI::sum(error_eikonal, mpi_communicator);
    volume        = dealii::Utilities::MPI::sum(volume, mpi_communicator);

    dealii::ConditionalOStream pcout(std::cout,
                                     dealii::Utilities::MPI::this_mpi_process(mpi_communicator) ==
                                       0);

    std::ostringstream str;
    str << std::scientific << std::setprecision(4)
        << "rel. L2 error signed distance: " << std::sqrt(error_value / norm_exact)
        << "  L2 error |grad phi|-1: " << std::sqrt(error_eikonal / volume);
    Journal::print_line(pcout, str.str(), label);
  }
} // namespace MeltPoolDG::LevelSet
