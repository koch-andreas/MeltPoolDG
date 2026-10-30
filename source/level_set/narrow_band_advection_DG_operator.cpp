#include <deal.II/base/mpi.h>

#include <deal.II/matrix_free/operators.h>

#include <meltpooldg/level_set/narrow_band_advection_DG_operator.hpp>
#include <meltpooldg/level_set/narrow_band_utilities.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <algorithm>

namespace MeltPoolDG::LevelSet
{
  template <int dim, typename number>
  NarrowBandAdvectionDGOperator<dim, number>::NarrowBandAdvectionDGOperator(
    const ScratchData<dim, dim, number> &scratch_data,
    const unsigned int                   band_dof_idx,
    const unsigned int                   quad_idx)
    : scratch_data(scratch_data)
    , band_dof_idx(band_dof_idx)
    , quad_idx(quad_idx)
  {}

  template <int dim, typename number>
  void
  NarrowBandAdvectionDGOperator<dim, number>::set_velocity_function(
    const std::shared_ptr<dealii::Function<dim>> &velocity_function_in)
  {
    AssertThrow(velocity_function_in and velocity_function_in->n_components == dim,
                dealii::ExcMessage("The advection velocity function needs dim components."));
    velocity_function = velocity_function_in;
  }

  template <int dim, typename number>
  bool
  NarrowBandAdvectionDGOperator<dim, number>::is_band_fe_index(const unsigned int active_fe_index)
  {
    return active_fe_index == NarrowBand::ActiveFEIndex::cut or
           active_fe_index == NarrowBand::ActiveFEIndex::neighbor or
           active_fe_index == NarrowBand::ActiveFEIndex::outer;
  }

  template <int dim, typename number>
  void
  NarrowBandAdvectionDGOperator<dim, number>::local_apply_cell(
    const dealii::MatrixFree<dim, number>       &matrix_free,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    if (not is_band_fe_index(matrix_free.get_cell_range_category(cell_range, band_dof_idx)))
      return;

    FECellIntegrator<dim, 1, number> phi(matrix_free, cell_range, band_dof_idx, quad_idx);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(src, dealii::EvaluationFlags::gradients);

        for (const unsigned int q : phi.quadrature_point_indices())
          {
            const auto velocity =
              MeltPoolDG::VectorTools::evaluate_function_at_vectorized_points<dim, number>(
                *velocity_function, phi.quadrature_point(q));

            phi.submit_value(-(velocity * phi.get_gradient(q)), q);
          }

        phi.integrate_scatter(dealii::EvaluationFlags::values, dst);
      }
  }

  template <int dim, typename number>
  void
  NarrowBandAdvectionDGOperator<dim, number>::local_apply_face(
    const dealii::MatrixFree<dim, number>       &matrix_free,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    // Faces at the outer boundary of the band (adjacent to an inactive cell) do not contribute,
    // since phi_ext = phi_int is used there.
    const auto category = matrix_free.get_face_range_category(face_range, band_dof_idx);
    if (not is_band_fe_index(category.first) or not is_band_fe_index(category.second))
      return;

    FEFaceIntegrator<dim, 1, number> phi_m(matrix_free, face_range, true, band_dof_idx, quad_idx);
    FEFaceIntegrator<dim, 1, number> phi_p(matrix_free, face_range, false, band_dof_idx, quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_m.reinit(face);
        phi_p.reinit(face);
        phi_m.gather_evaluate(src, dealii::EvaluationFlags::values);
        phi_p.gather_evaluate(src, dealii::EvaluationFlags::values);

        for (const unsigned int q : phi_m.quadrature_point_indices())
          {
            const auto velocity =
              MeltPoolDG::VectorTools::evaluate_function_at_vectorized_points<dim, number>(
                *velocity_function, phi_m.quadrature_point(q));

            // normal velocity with respect to the outer normal of the interior (minus) cell
            const VectorizedArrayType normal_velocity = velocity * phi_m.normal_vector(q);
            const VectorizedArrayType jump            = phi_m.get_value(q) - phi_p.get_value(q);

            // upwind terms on the right-hand side: on the inflow part of the minus cell
            // (u.n_m < 0): (u.n_m)(phi_m - phi_p) v_m; on the inflow part of the plus cell
            // (u.n_p = -u.n_m < 0): (u.n_p)(phi_p - phi_m) v_p = (u.n_m)(phi_m - phi_p) v_p
            phi_m.submit_value(std::min(normal_velocity, VectorizedArrayType(0.)) * jump, q);
            phi_p.submit_value(std::max(normal_velocity, VectorizedArrayType(0.)) * jump, q);
          }

        phi_m.integrate_scatter(dealii::EvaluationFlags::values, dst);
        phi_p.integrate_scatter(dealii::EvaluationFlags::values, dst);
      }
  }

  template <int dim, typename number>
  void
  NarrowBandAdvectionDGOperator<dim, number>::local_apply_inverse_mass_matrix(
    const dealii::MatrixFree<dim, number>       &matrix_free,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    if (not is_band_fe_index(matrix_free.get_cell_range_category(cell_range, band_dof_idx)))
      return;

    FECellIntegrator<dim, 1, number> phi(matrix_free, cell_range, band_dof_idx, quad_idx);
    AssertThrow(phi.get_shape_info().data[0].n_q_points_1d ==
                  phi.get_shape_info().data[0].fe_degree + 1,
                dealii::ExcMessage("The cell-wise inverse mass matrix requires a Gauss "
                                   "quadrature with p+1 points per direction."));

    dealii::MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, 1, number> inverse(phi);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.read_dof_values(src);
        inverse.apply(phi.begin_dof_values(), phi.begin_dof_values());
        phi.set_dof_values(dst);
      }
  }

  template <int dim, typename number>
  void
  NarrowBandAdvectionDGOperator<dim, number>::apply_operator(
    const number                                           time,
    VectorType                                            &dst,
    const VectorType                                      &src,
    const std::function<void(unsigned int, unsigned int)> &post) const
  {
    AssertThrow(velocity_function, dealii::ExcMessage("No advection velocity has been set."));
    velocity_function->set_time(time);

    const auto &matrix_free = scratch_data.get_matrix_free();

    matrix_free.loop(&NarrowBandAdvectionDGOperator::local_apply_cell,
                     &NarrowBandAdvectionDGOperator::local_apply_face,
                     &NarrowBandAdvectionDGOperator::local_apply_boundary_face,
                     this,
                     dst,
                     src,
                     true /*zero_dst_vector*/,
                     dealii::MatrixFree<dim, number>::DataAccessOnFaces::values,
                     dealii::MatrixFree<dim, number>::DataAccessOnFaces::values);

    matrix_free.cell_loop(&NarrowBandAdvectionDGOperator::local_apply_inverse_mass_matrix,
                          this,
                          dst,
                          dst);

    if (post)
      post(0, dst.locally_owned_size());
  }

  template <int dim, typename number>
  number
  NarrowBandAdvectionDGOperator<dim, number>::compute_cfl_number(const number time,
                                                                 const number time_step) const
  {
    AssertThrow(velocity_function, dealii::ExcMessage("No advection velocity has been set."));
    velocity_function->set_time(time);

    const auto &matrix_free = scratch_data.get_matrix_free();
    const auto &cell_sizes  = scratch_data.get_cell_sizes();

    number max_cfl = 0.;

    for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
      {
        const unsigned int active_fe_index = matrix_free.get_cell_category(cell, band_dof_idx);
        if (not is_band_fe_index(active_fe_index))
          continue;

        FECellIntegrator<dim, 1, number> phi(
          matrix_free, band_dof_idx, quad_idx, 0, active_fe_index);
        phi.reinit(cell);

        const unsigned int fe_degree = phi.get_shape_info().data[0].fe_degree;

        for (const unsigned int q : phi.quadrature_point_indices())
          {
            const auto velocity =
              MeltPoolDG::VectorTools::evaluate_function_at_vectorized_points<dim, number>(
                *velocity_function, phi.quadrature_point(q));
            const auto cfl = velocity.norm() * time_step * (2. * fe_degree + 1.) / cell_sizes[cell];

            for (unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(cell); ++v)
              max_cfl = std::max(max_cfl, cfl[v]);
          }
      }

    return dealii::Utilities::MPI::max(max_cfl, matrix_free.get_task_info().communicator);
  }

  template class NarrowBandAdvectionDGOperator<1, double>;
  template class NarrowBandAdvectionDGOperator<2, double>;
  template class NarrowBandAdvectionDGOperator<3, double>;
} // namespace MeltPoolDG::LevelSet
