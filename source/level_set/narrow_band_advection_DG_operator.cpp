#include <deal.II/base/mpi.h>

#include <deal.II/matrix_free/operators.h>

#include <meltpooldg/level_set/narrow_band_advection_DG_operator.hpp>
#include <meltpooldg/level_set/narrow_band_utilities.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <algorithm>
#include <cmath>

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
  void
  NarrowBandAdvectionDGOperator<dim, number>::set_extension_velocity(const bool flag)
  {
    use_extension_velocity = flag;
  }

  template <int dim, typename number>
  dealii::Tensor<1, dim, typename NarrowBandAdvectionDGOperator<dim, number>::VectorizedArrayType>
  NarrowBandAdvectionDGOperator<dim, number>::evaluate_velocity(
    const dealii::Point<dim, VectorizedArrayType>     &point,
    const VectorizedArrayType                         &value,
    const dealii::Tensor<1, dim, VectorizedArrayType> &gradient) const
  {
    if (not use_extension_velocity)
      return MeltPoolDG::VectorTools::evaluate_function_at_vectorized_points<dim, number>(
        *velocity_function, point);

    // closest point y = x - d n on the zero level set with the normal n = grad phi / |grad phi|
    // and the distance d = phi / max(|grad phi|, 0.5)
    dealii::Tensor<1, dim, VectorizedArrayType> velocity;
    for (unsigned int v = 0; v < VectorizedArrayType::size(); ++v)
      {
        dealii::Point<dim>     y;
        dealii::Tensor<1, dim> grad;
        for (unsigned int d = 0; d < dim; ++d)
          {
            y[d]    = point[d][v];
            grad[d] = gradient[d][v];
          }

        if (const number norm = grad.norm(); norm > 1e-12)
          y -= (value[v] / std::max<number>(norm, 0.5)) * (grad / norm);

        for (unsigned int d = 0; d < dim; ++d)
          velocity[d][v] = velocity_function->value(y, d);
      }

    return velocity;
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
    FECellIntegrator<dim, 1, number> epsilon(matrix_free, cell_range, band_dof_idx, quad_idx);

    const bool has_viscosity = artificial_viscosity != nullptr;
    const auto integration_flags =
      has_viscosity ? dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients :
                      dealii::EvaluationFlags::values;

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(src,
                            use_extension_velocity ?
                              dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients :
                              dealii::EvaluationFlags::gradients);

        // cell-wise constant viscosity
        VectorizedArrayType epsilon_cell = 0.;
        if (has_viscosity)
          {
            epsilon.reinit(cell);
            epsilon.read_dof_values_plain(*artificial_viscosity);
            epsilon_cell = epsilon.begin_dof_values()[0];
          }

        for (const unsigned int q : phi.quadrature_point_indices())
          {
            const auto gradient = phi.get_gradient(q);
            const auto velocity =
              evaluate_velocity(phi.quadrature_point(q),
                                use_extension_velocity ? phi.get_value(q) : VectorizedArrayType(),
                                gradient);

            phi.submit_value(-(velocity * gradient), q);

            // - (epsilon grad phi, grad v)
            if (has_viscosity)
              phi.submit_gradient(-epsilon_cell * phi.get_gradient(q), q);
          }

        phi.integrate_scatter(integration_flags, dst);
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
    FEFaceIntegrator<dim, 1, number> epsilon_m(
      matrix_free, face_range, true, band_dof_idx, quad_idx);
    FEFaceIntegrator<dim, 1, number> epsilon_p(
      matrix_free, face_range, false, band_dof_idx, quad_idx);

    const bool has_viscosity = artificial_viscosity != nullptr;
    const auto flags         = has_viscosity ?
                                 dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients :
                                 dealii::EvaluationFlags::values;
    // the extension velocity needs the gradients of the traces
    const auto evaluation_flags =
      use_extension_velocity ? flags | dealii::EvaluationFlags::gradients : flags;

    // interior penalty factor (p+1)(p+2) / h_min of the viscous term
    const number fe_degree = phi_m.get_shape_info().data[0].fe_degree;
    const number penalty_factor =
      (fe_degree + 1.) * (fe_degree + 2.) / scratch_data.get_min_cell_size();

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_m.reinit(face);
        phi_p.reinit(face);
        phi_m.gather_evaluate(src, evaluation_flags);
        phi_p.gather_evaluate(src, evaluation_flags);

        if (has_viscosity)
          {
            epsilon_m.reinit(face);
            epsilon_p.reinit(face);
            epsilon_m.gather_evaluate(*artificial_viscosity, dealii::EvaluationFlags::values);
            epsilon_p.gather_evaluate(*artificial_viscosity, dealii::EvaluationFlags::values);
          }

        for (const unsigned int q : phi_m.quadrature_point_indices())
          {
            // extension velocity: from the averages of the traces
            const auto velocity =
              use_extension_velocity ?
                evaluate_velocity(phi_m.quadrature_point(q),
                                  0.5 * (phi_m.get_value(q) + phi_p.get_value(q)),
                                  0.5 * (phi_m.get_gradient(q) + phi_p.get_gradient(q))) :
                evaluate_velocity(phi_m.quadrature_point(q), {}, {});

            // normal velocity with respect to the outer normal of the interior (minus) cell
            const VectorizedArrayType normal_velocity = velocity * phi_m.normal_vector(q);
            const VectorizedArrayType jump            = phi_m.get_value(q) - phi_p.get_value(q);

            // upwind terms on the right-hand side: on the inflow part of the minus cell
            // (u.n_m < 0): (u.n_m)(phi_m - phi_p) v_m; on the inflow part of the plus cell
            // (u.n_p = -u.n_m < 0): (u.n_p)(phi_p - phi_m) v_p = (u.n_m)(phi_m - phi_p) v_p
            VectorizedArrayType value_flux_m =
              std::min(normal_velocity, VectorizedArrayType(0.)) * jump;
            VectorizedArrayType value_flux_p =
              std::max(normal_velocity, VectorizedArrayType(0.)) * jump;

            if (has_viscosity)
              {
                // SIPG: ({eps grad phi}.n, [v]) + ({eps grad v}.n, [phi]) - sigma ([phi], [v])
                const auto eps_m        = epsilon_m.get_value(q);
                const auto eps_p        = epsilon_p.get_value(q);
                const auto average_flux = 0.5 * (eps_m * phi_m.get_normal_derivative(q) +
                                                 eps_p * phi_p.get_normal_derivative(q));
                const auto sigma        = penalty_factor * std::max(eps_m, eps_p);

                value_flux_m += average_flux - sigma * jump;
                value_flux_p -= average_flux - sigma * jump;

                phi_m.submit_gradient((0.5 * eps_m * jump) * phi_m.normal_vector(q), q);
                phi_p.submit_gradient((0.5 * eps_p * jump) * phi_m.normal_vector(q), q);
              }

            phi_m.submit_value(value_flux_m, q);
            phi_p.submit_value(value_flux_p, q);
          }

        phi_m.integrate_scatter(flags, dst);
        phi_p.integrate_scatter(flags, dst);
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
                     artificial_viscosity != nullptr ?
                       dealii::MatrixFree<dim, number>::DataAccessOnFaces::gradients :
                       dealii::MatrixFree<dim, number>::DataAccessOnFaces::values,
                     // the extension velocity needs the gradients of the traces
                     artificial_viscosity != nullptr or use_extension_velocity ?
                       dealii::MatrixFree<dim, number>::DataAccessOnFaces::gradients :
                       dealii::MatrixFree<dim, number>::DataAccessOnFaces::values);

    matrix_free.cell_loop(&NarrowBandAdvectionDGOperator::local_apply_inverse_mass_matrix,
                          this,
                          dst,
                          dst);

    if (post)
      post(0, dst.locally_owned_size());
  }

  template <int dim, typename number>
  void
  NarrowBandAdvectionDGOperator<dim, number>::set_artificial_viscosity(const VectorType *viscosity)
  {
    artificial_viscosity = viscosity;
  }

  template <int dim, typename number>
  typename NarrowBandAdvectionDGOperator<dim, number>::ArtificialViscosityStatistics
  NarrowBandAdvectionDGOperator<dim, number>::compute_artificial_viscosity(
    const VectorType &level_set,
    VectorType       &viscosity,
    const number      time,
    const number      time_step,
    const number      factor,
    const number      threshold,
    const number      width,
    const bool        only_non_cut_cells) const
  {
    AssertThrow(velocity_function, dealii::ExcMessage("No advection velocity has been set."));
    AssertThrow(width > 0., dealii::ExcMessage("The indicator width must be positive."));
    velocity_function->set_time(time);

    const auto &matrix_free = scratch_data.get_matrix_free();
    const auto &cell_sizes  = scratch_data.get_cell_sizes();

    const auto &fe =
      matrix_free.get_dof_handler(band_dof_idx).get_fe_collection()[NarrowBand::ActiveFEIndex::cut];
    const NarrowBand::ModalSmoothnessIndicator<dim> indicator(fe);
    const number                                    fe_degree = fe.degree;

    viscosity = 0.;

    ArtificialViscosityStatistics statistics;

    for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
      {
        const unsigned int active_fe_index = matrix_free.get_cell_category(cell, band_dof_idx);
        if (not is_band_fe_index(active_fe_index) or
            (only_non_cut_cells and active_fe_index == NarrowBand::ActiveFEIndex::cut))
          continue;

        FECellIntegrator<dim, 1, number> phi(
          matrix_free, band_dof_idx, quad_idx, 0, active_fe_index);
        phi.reinit(cell);
        phi.read_dof_values_plain(level_set);

        const VectorizedArrayType ratio = indicator.compute_energy_ratio(phi.begin_dof_values());

        // maximum velocity over the quadrature points
        VectorizedArrayType max_speed = 0.;
        for (const unsigned int q : phi.quadrature_point_indices())
          max_speed =
            std::max(max_speed,
                     MeltPoolDG::VectorTools::evaluate_function_at_vectorized_points<dim, number>(
                       *velocity_function, phi.quadrature_point(q))
                       .norm());

        VectorizedArrayType epsilon = 0.;
        for (unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(cell); ++v)
          {
            const number s         = std::log10(std::max<number>(ratio[v], 1e-300));
            const number epsilon_0 = factor * cell_sizes[cell][v] * max_speed[v] / fe_degree;

            if (s > threshold + width)
              epsilon[v] = epsilon_0;
            else if (s >= threshold - width)
              epsilon[v] = 0.5 * epsilon_0 *
                           (1. + std::sin(0.5 * dealii::numbers::PI * (s - threshold) / width));

            if (epsilon[v] > 0.)
              {
                ++statistics.n_cells;
                statistics.max_diffusion_number =
                  std::max(statistics.max_diffusion_number,
                           epsilon[v] * time_step / (cell_sizes[cell][v] * cell_sizes[cell][v]));
              }
          }

        for (unsigned int i = 0; i < phi.get_shape_info().dofs_per_component_on_cell; ++i)
          phi.begin_dof_values()[i] = epsilon;
        phi.set_dof_values_plain(viscosity);
      }

    viscosity.update_ghost_values();

    const MPI_Comm comm = matrix_free.get_task_info().communicator;
    statistics.n_cells  = dealii::Utilities::MPI::sum(statistics.n_cells, comm);
    statistics.max_diffusion_number =
      dealii::Utilities::MPI::max(statistics.max_diffusion_number, comm);
    return statistics;
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
