#pragma once

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/base/tensor.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/linear_algebra/utilities_matrixfree.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>
#include <meltpooldg/utilities/vector_tools.templates.hpp>

namespace MeltPoolDG::Utilities
{
  /**
   * Minmod function for a container of values. This function computes the minmod of the given
   * values, which is defined as follows:
   * - If all values have the same sign, the minmod is the value with the smallest absolute value.
   * - If the values have different signs, the minmod is zero.
   *
   * @tparam Container The type of the container holding the values.
   *
   * @param values The container of values for which to compute the minmod.
   * @return The minmod of the given values.
   */
  template <int n_components, typename TensorType, typename Container>
  TensorType
  minmod(const Container &values)
  {
    if (values.begin() == values.end())
      return TensorType();

    TensorType result;
    for (unsigned int i = 0; i < n_components; ++i)
      {
        // Check sign consistency
        const bool all_positive =
          std::all_of(values.begin(), values.end(), [i](TensorType v) { return v[i] > 0; });
        const bool all_negative =
          std::all_of(values.begin(), values.end(), [i](TensorType v) { return v[i] < 0; });

        if (!(all_positive or all_negative))
          {
            result[i] = typename TensorType::value_type(0.);
          }
        else
          {
            // Find element with smallest absolute value
            auto it = std::min_element(values.begin(),
                                       values.end(),
                                       [i](const TensorType &a, const TensorType &b) {
                                         return std::abs(a[i]) < std::abs(b[i]);
                                       });

            result[i] = (*it)[i];
          }
      }

    return result;
  }

  template <int dim, int n_components, typename number>
  dealii::Tensor<1, n_components, number>
  apply_muscl_limiter_to_cell_dof(
    const dealii::CellDataStorage<typename dealii::Triangulation<dim>::active_cell_iterator,
                                  dealii::Tensor<1, n_components, number>> &cell_average_values,
    const typename dealii::Triangulation<dim>::active_cell_iterator        &cell,
    const dealii::Point<dim, number>                                       &dof_coordinates,
    const dealii::Tensor<1, n_components, number>                          &average_cell_gradient)
  {
    Assert(dim == 1,
           dealii::ExcMessage("MUSCL limiter is currently only implemented for 1D problems."));

    const dealii::Tensor<1, n_components, number> cell_average_value_lane =
      *cell_average_values.get_data(cell)[0];
    std::vector<dealii::Tensor<1, n_components, number>> minmod_input_values;

    minmod_input_values.push_back(average_cell_gradient);

    for (unsigned int face_no = 0; face_no < dealii::GeometryInfo<dim>::faces_per_cell; ++face_no)
      {
        const auto neighbor = cell->neighbor(face_no);
        if (!cell->at_boundary(face_no) and neighbor->is_active())
          {
            const dealii::Tensor<1, n_components, number> neighbor_average_value =
              *cell_average_values.get_data(neighbor)[0];

            auto   vector_to_neighbor   = cell->center() - neighbor->center();
            number distance_to_neighbor = vector_to_neighbor.norm();

            if (face_no == 0)
              minmod_input_values.push_back((cell_average_value_lane - neighbor_average_value) /
                                            distance_to_neighbor);
            else if (face_no == 1)
              minmod_input_values.push_back((neighbor_average_value - cell_average_value_lane) /
                                            distance_to_neighbor);
            else
              AssertThrow(false,
                          dealii::ExcMessage(
                            "MUSCL limiter is currently only implemented for 1D problems."));
          }
      }
    dealii::Tensor<1, n_components, number> minmod_values =
      minmod<n_components,
             dealii::Tensor<1, n_components, number>,
             std::vector<dealii::Tensor<1, n_components, number>>>(minmod_input_values);
    for (unsigned int c = 0; c < 1; ++c)
      {
        minmod_values[c] = 0.;
      }

    return cell_average_value_lane + (dof_coordinates[0] - cell->center()[0]) * minmod_values;
  }

  /**
   * Applies a MUSCL limiter to a given set of values.
   */
  template <int dim,
            int n_components,
            typename number,
            typename VectorizedArrayType = dealii::VectorizedArray<number>,
            typename VectorType          = dealii::LinearAlgebra::distributed::Vector<number>>
  void
  apply_muscl_limiter(const MatrixFreeContext<dim, number> &mf_context,
                      VectorType                           &dst,
                      const VectorType                     &src)
  {
    AssertThrow(mf_context.mf.get_dof_handler(mf_context.dof_idx).get_fe().degree == 1,
                dealii::ExcMessage(
                  "MUSCL limiter can only be applied for degree 1 finite elements."));

    AssertThrow(dim == 1,
                dealii::ExcMessage("MUSCL limiter is currently only implemented for 1D problems."));

    using active_cell_iterator = typename dealii::Triangulation<dim>::active_cell_iterator;
    using TensorType           = dealii::Tensor<1, n_components, number>;


    // Ensure that the cell average values of ghost cells are set correctly
    const auto exchange_cell_data_to_ghosts =
      [](const dealii::Triangulation<dim>                          &triangulation,
         dealii::CellDataStorage<active_cell_iterator, TensorType> &cell_average_values) {
        std::function<std::optional<TensorType>(const active_cell_iterator &)> pack =
          [&](const active_cell_iterator &cell) -> std::optional<TensorType> {
          return *cell_average_values.get_data(cell)[0];
        };
        std::function<void(const active_cell_iterator &, const TensorType &)> unpack =
          [&](const active_cell_iterator &cell, const TensorType &value) {
            if (cell->is_ghost())
              {
                cell_average_values.initialize(cell, 1);
                *cell_average_values.get_data(cell)[0] = value;
              }
          };
        dealii::GridTools::exchange_cell_data_to_ghosts(triangulation, pack, unpack);
      };


    dealii::CellDataStorage<active_cell_iterator, dealii::Tensor<1, n_components, number>>
      cell_average_values;

    // Step 1: Compute the cell average values for each cell in the mesh using the matrix-free
    // context
    const std::function<void(const dealii::MatrixFree<dim, number, VectorizedArrayType> &,
                             VectorType &,
                             const VectorType &,
                             const std::pair<unsigned int, unsigned int> &)>
      cell_loop = [&](const dealii::MatrixFree<dim, number, VectorizedArrayType> &matrix_free,
                      VectorType &,
                      const VectorType                            &src,
                      const std::pair<unsigned int, unsigned int> &cell_range) {
        FECellIntegrator<dim, n_components, number> fe_cell_integrator(matrix_free,
                                                                       mf_context.dof_idx,
                                                                       mf_context.quad_idx);

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            fe_cell_integrator.reinit(cell);
            fe_cell_integrator.gather_evaluate(src, dealii::EvaluationFlags::values);

            dealii::Tensor<1, n_components, VectorizedArrayType> cell_average_value;
            VectorizedArrayType                                  cell_volume = 0.;
            for (const unsigned int q : fe_cell_integrator.quadrature_point_indices())
              {
                cell_average_value += fe_cell_integrator.get_value(q) * fe_cell_integrator.JxW(q);
                cell_volume += fe_cell_integrator.JxW(q);
              }
            cell_average_value /= cell_volume;
            const auto &cells = cells_in_cell_batch(mf_context.mf, cell);

            for (unsigned int lane = 0; lane < mf_context.mf.n_active_entries_per_cell_batch(cell);
                 ++lane)
              {
                dealii::Tensor<1, n_components, number> cell_average_value_lane;
                for (unsigned int c = 0; c < n_components; ++c)
                  cell_average_value_lane[c] = cell_average_value[c][lane];

                cell_average_values.initialize(cells[lane], 1);
                *cell_average_values.get_data(cells[lane])[0] = cell_average_value_lane;
              }
          }
      };

    mf_context.mf.cell_loop(cell_loop, dst, src);

    exchange_cell_data_to_ghosts(
      mf_context.mf.get_dof_handler(mf_context.dof_idx).get_triangulation(), cell_average_values);

    // Step 2: Compute the limited dof values
    const std::function<void(const dealii::MatrixFree<dim, number, VectorizedArrayType> &,
                             VectorType &,
                             const VectorType &,
                             const std::pair<unsigned int, unsigned int> &)>
      limit_loop = [&](const dealii::MatrixFree<dim, number, VectorizedArrayType> &matrix_free,
                       VectorType                                                 &dst,
                       const VectorType                                           &src,
                       const std::pair<unsigned int, unsigned int>                &cell_range) {
        FECellIntegrator<dim, n_components, number> fe_cell_integrator(matrix_free,
                                                                       mf_context.dof_idx,
                                                                       mf_context.quad_idx);

        dealii::MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, n_components, number>
          inverse(fe_cell_integrator);

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            fe_cell_integrator.reinit(cell);
            fe_cell_integrator.gather_evaluate(src,
                                               dealii::EvaluationFlags::values |
                                                 dealii::EvaluationFlags::gradients);

            const auto &cells = cells_in_cell_batch(mf_context.mf, cell);

            for (const unsigned int q : fe_cell_integrator.quadrature_point_indices())
              {
                dealii::Tensor<1, n_components, VectorizedArrayType> limited_value;
                for (unsigned int lane = 0;
                     lane < mf_context.mf.n_active_entries_per_cell_batch(cell);
                     ++lane)
                  {
                    dealii::Tensor<1, n_components, number> gradient_value_lane;
                    for (unsigned int c = 0; c < n_components; ++c)
                      gradient_value_lane[c] = fe_cell_integrator.get_gradient(q)[c][0][lane];

                    dealii::Point<dim, number> dof_coordinates;
                    for (unsigned int d = 0; d < dim; ++d)
                      dof_coordinates[d] = fe_cell_integrator.quadrature_point(q)[d][lane];

                    dealii::Tensor<1, n_components, number> cell_limited_value_lane =
                      apply_muscl_limiter_to_cell_dof<dim, n_components, number>(
                        cell_average_values, cells[lane], dof_coordinates, gradient_value_lane);

                    for (unsigned int c = 0; c < n_components; ++c)
                      limited_value[c][lane] = cell_limited_value_lane[c];
                  }
                fe_cell_integrator.submit_dof_value(limited_value, q);
              }

            inverse.transform_from_q_points_to_basis(n_components,
                                                     fe_cell_integrator.begin_dof_values(),
                                                     fe_cell_integrator.begin_dof_values());
            fe_cell_integrator.set_dof_values(dst);
          }
      };

    mf_context.mf.cell_loop(limit_loop, dst, src, true);
  }
} // namespace MeltPoolDG::Utilities