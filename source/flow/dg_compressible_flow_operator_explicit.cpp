#include <meltpooldg/flow/compressible_flow_explicit_utils.hpp>
#include <meltpooldg/flow/compressible_flow_kernels.hpp>
#include <meltpooldg/flow/compressible_flow_types.hpp>
#include <meltpooldg/flow/compressible_flow_views.hpp>
#include <meltpooldg/flow/dg_compressible_flow_operator_explicit.hpp>
#include <meltpooldg/flow/dg_generic_convection_diffusion_worker.hpp>
#include <meltpooldg/linear_algebra/utilities_matrixfree.hpp>
#include <meltpooldg/time_integration/time_integrator_util.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>
#include <meltpooldg/utilities/preprocessor_directives.hpp>
#include <meltpooldg/utilities/vector_tools.templates.hpp>


namespace MeltPoolDG::Flow
{
  using namespace dealii;

  template <int dim, typename number, bool is_viscous>
  DGCompressibleFlowOperatorExplicit<dim, number, is_viscous>::DGCompressibleFlowOperatorExplicit(
    CompressibleFlowScratchData<dim, number> &flow_scratch_data)
    : flow_scratch_data(flow_scratch_data)
    , time_integrator(flow_scratch_data.flow_data.time_integrator)
  {
    time_integrator.configure_rhs(
      std::bind_front(&DGCompressibleFlowOperatorExplicit<dim, number, is_viscous>::apply_operator,
                      this));
  }

  template <int dim, typename number, bool is_viscous>
  void
  DGCompressibleFlowOperatorExplicit<dim, number, is_viscous>::reinit()
  {
    flow_scratch_data.reinit(time_integrator.required_solution_history_size());
    time_integrator.reinit(flow_scratch_data.solution_history.get_current_solution());
  }

  template <int dim, typename number, bool is_viscous>
  void
  DGCompressibleFlowOperatorExplicit<dim, number, is_viscous>::advance_time_step(number time,
                                                                                 number time_step)
  {
    std::function<void(number, number, VectorType &, const VectorType &)> pre_processing =
      [&](number time, number, VectorType &, const VectorType &) -> void {
      flow_scratch_data.boundary_conditions.update_boundary_conditions(time);
    };

    time_integrator.perform_time_step(
      time,
      time_step,
      flow_scratch_data.solution_history,
      pre_processing,
      std::function<void(number, number, VectorType &, const VectorType &)>());
  }

  template <int dim, typename number, bool is_viscous>
  void
  DGCompressibleFlowOperatorExplicit<dim, number, is_viscous>::apply_operator(
    const number                                           time,
    const number                                           time_step,
    VectorType                                            &dst,
    const VectorType                                      &src,
    const std::function<void(unsigned int, unsigned int)> &func) const
  {
    current_time_step = time_step;
    current_time = time;
    using local_applier_type =
      std::function<void(const dealii::MatrixFree<dim, number> &,
                         dealii::LinearAlgebra::distributed::Vector<number>       &dst,
                         const dealii::LinearAlgebra::distributed::Vector<number> &src,
                         const std::pair<unsigned int, unsigned int> &)>;

    flow_scratch_data.boundary_conditions.update_boundary_conditions(time);
    local_applier_type cell          = MPDG_LAMBDA_WRAPPER(this->local_apply_cell);
    local_applier_type face          = MPDG_LAMBDA_WRAPPER(this->local_apply_face);
    local_applier_type boundary_face = MPDG_LAMBDA_WRAPPER(this->local_apply_boundary_face);
    flow_scratch_data.scratch_data.get_matrix_free().loop(
      cell, face, boundary_face, dst, src, false);

    local_applier_type inverse =
      [dof_idx = flow_scratch_data.dof_idx,
       quad_idx =
         flow_scratch_data.quad_idx](const MatrixFree<dim, number>                    &matrix_free,
                                     LinearAlgebra::distributed::Vector<number>       &dst,
                                     const LinearAlgebra::distributed::Vector<number> &src,
                                     const std::pair<unsigned int, unsigned int>      &cell_range) {
        Utilities::MatrixFree::local_apply_inverse_mass_matrix<dim, dim + 2, number>(
          matrix_free, dst, src, cell_range, dof_idx, quad_idx);
      };
    flow_scratch_data.scratch_data.get_matrix_free().cell_loop(
      inverse, dst, dst, std::function<void(unsigned int, unsigned int)>(), func);
  }

  template <int dim, typename number, bool is_viscous>
  void
  DGCompressibleFlowOperatorExplicit<dim, number, is_viscous>::add_external_force(
    std::shared_ptr<ExternalFlowForce<dim, number>> external_force,
    std::shared_ptr<ExternalFlowForceJacobian<dim, number>>)
  {
    Assert(external_force != nullptr, dealii::ExcInternalError());
    external_forces.push_back(external_force);
  }

  template <int dim, typename number, bool is_viscous>
  void
  DGCompressibleFlowOperatorExplicit<dim, number, is_viscous>::local_apply_cell(
    const MatrixFree<dim, number>       &mf,
    VectorType                          &dst,
    const VectorType                    &src,
    const std::pair<unsigned, unsigned> &cell_range) const
  {
    FECellIntegrator<dim, dim + 2, number> phi(mf,
                                               flow_scratch_data.dof_idx,
                                               flow_scratch_data.quad_idx);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(src,
                            EvaluationFlags::values |
                              (is_viscous ? EvaluationFlags::gradients : EvaluationFlags::nothing));

        std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> cell_iterators =
          cells_in_cell_batch(mf, cell);

        for (const unsigned int q : phi.quadrature_point_indices())
          {
            CompressibleFlow::SourceType<dim, number> source;
            CompressibleFlow::FluxType<dim, number>   flux;

            if (is_viscous)
              flux = ConvectionDiffusionOperator::cell(
                phi.get_value(q),
                phi.get_gradient(q),
                CompressibleConvectiveFlux<dim, number>(flow_scratch_data.material.data),
                CompressibleDiffusiveFlux<dim, number>(flow_scratch_data.material.data));
            else
              flux = ConvectionOperator::cell(phi.get_value(q),
                                              CompressibleConvectiveFlux<dim, number>(
                                                flow_scratch_data.material.data));

            for (auto &external_force : external_forces)
              source += external_force->value(current_time_step,
                                              cell_iterators,
                                              phi.quadrature_point(q),
                                              phi.get_value(q));

                CompressibleFlow::ConservedVariablesType<dim, number> regularized_heat_source{};
                const VectorizedArray<double> interface_position = dealii::make_vectorized_array(0.);
                constexpr double epsilon = 10. * 3.125e-6;
                const dealii::Point<dim, VectorizedArray<double>> quad_point = phi.quadrature_point(q);
                const dealii::VectorizedArray<double> x = quad_point[0];
                auto delta = 1. / (sqrt(2.*std::numbers::pi) * epsilon) * std::exp(-((x-interface_position)/epsilon) * ((x-interface_position)/epsilon) / 2.);
                regularized_heat_source[2] = delta * 0.5 * (1. - std::cos(std::numbers::pi * current_time / 1.e-4)) * 2.0e9;
                if (current_time > 1.e-4)
                  regularized_heat_source[2] = 2.0e9;
                source += regularized_heat_source;

            phi.submit_value(source, q);
            phi.submit_gradient(flux, q);
          }

        phi.integrate_scatter((not external_forces.empty() ? EvaluationFlags::values :
                                                             EvaluationFlags::nothing) |
                                EvaluationFlags::gradients,
                              dst);
      }
  }

  template <int dim, typename number, bool is_viscous>
  void
  DGCompressibleFlowOperatorExplicit<dim, number, is_viscous>::local_apply_face(
    const MatrixFree<dim, number>               &mf,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceIntegrator<dim, dim + 2, number> phi_m(mf,
                                                 true,
                                                 flow_scratch_data.dof_idx,
                                                 flow_scratch_data.quad_idx);
    FEFaceIntegrator<dim, dim + 2, number> phi_p(mf,
                                                 false,
                                                 flow_scratch_data.dof_idx,
                                                 flow_scratch_data.quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_p.reinit(face);
        phi_p.gather_evaluate(src,
                              EvaluationFlags::values | (is_viscous ? EvaluationFlags::gradients :
                                                                      EvaluationFlags::nothing));

        phi_m.reinit(face);
        phi_m.gather_evaluate(src,
                              EvaluationFlags::values | (is_viscous ? EvaluationFlags::gradients :
                                                                      EvaluationFlags::nothing));

        const VectorizedArray<number> interior_penalty_parameter =
          is_viscous ?
            flow_scratch_data.material.data.dynamic_viscosity /
              flow_scratch_data.material.data.reference_density *
              std::max(phi_m.read_cell_data(flow_scratch_data.interior_penalty_parameter),
                       phi_p.read_cell_data(flow_scratch_data.interior_penalty_parameter)) :
            0.;

        for (const unsigned int q : phi_m.quadrature_point_indices())
          {
            CompressibleFlow::FaceFluxType<dim, number> flux_m;
            CompressibleFlow::FaceFluxType<dim, number> flux_p;

            if (is_viscous)
              {
                const auto flux = ConvectionDiffusionOperator::face(
                  phi_m.get_value(q),
                  phi_p.get_value(q),
                  phi_m.get_gradient(q),
                  phi_p.get_gradient(q),
                  phi_m.normal_vector(q),
                  interior_penalty_parameter,
                  CompressibleConvectiveFlux<dim, number>(flow_scratch_data.material.data),
                  CompressibleDiffusiveFlux<dim, number>(flow_scratch_data.material.data));

                flux_m = flux.inner_face_value;
                flux_p = flux.outer_face_value;

                phi_m.submit_gradient(flux.inner_face_gradient, q);
                phi_p.submit_gradient(flux.outer_face_gradient, q);
              }
            else
              {
                const auto flux = ConvectionOperator::face(phi_m.get_value(q),
                                                           phi_p.get_value(q),
                                                           phi_m.normal_vector(q),
                                                           CompressibleConvectiveFlux<dim, number>(
                                                             flow_scratch_data.material.data));

                flux_m = flux.inner_face_value;
                flux_p = flux.outer_face_value;
              }

            phi_m.submit_value(flux_m, q);
            phi_p.submit_value(flux_p, q);
          }

        phi_p.integrate_scatter(EvaluationFlags::values | (is_viscous ? EvaluationFlags::gradients :
                                                                        EvaluationFlags::nothing),
                                dst);
        phi_m.integrate_scatter(EvaluationFlags::values | (is_viscous ? EvaluationFlags::gradients :
                                                                        EvaluationFlags::nothing),
                                dst);
      }
  }

  template <int dim, typename number, bool is_viscous>
  void
  DGCompressibleFlowOperatorExplicit<dim, number, is_viscous>::local_apply_boundary_face(
    const MatrixFree<dim, number>       &mf,
    VectorType                          &dst,
    const VectorType                    &src,
    const std::pair<unsigned, unsigned> &face_range) const
  {
    FEFaceIntegrator<dim, dim + 2, number> phi_m(mf,
                                                 true,
                                                 flow_scratch_data.dof_idx,
                                                 flow_scratch_data.quad_idx);

    using DofValueAndGradientStateViewType = CompressibleFlow::DofValueAndGradientStateView<
      dim,
      number,
      const CompressibleFlow::ConservedVariablesType<dim, number>,
      const CompressibleFlow::ConservedVariablesGradientType<dim, number>>;

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_m.reinit(face);
        phi_m.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        const VectorizedArray<number> interior_penalty_parameter =
          is_viscous ? flow_scratch_data.material.data.dynamic_viscosity /
                         flow_scratch_data.material.data.reference_density *
                         phi_m.read_cell_data(flow_scratch_data.interior_penalty_parameter) :
                       0.;

        for (const unsigned int q : phi_m.quadrature_point_indices())
          {
            const auto w_m      = phi_m.get_value(q);
            const auto grad_w_m = phi_m.get_gradient(q);

            const auto [w_p, grad_w_p] =
              flow_scratch_data.boundary_conditions.get_boundary_face_value_and_gradient(
                phi_m.quadrature_point(q),
                phi_m.normal_vector(q),
                phi_m.boundary_id(),
                DofValueAndGradientStateViewType(w_m, grad_w_m, flow_scratch_data.material.data));

            CompressibleFlow::FaceFluxType<dim, number> flux_m;
            if (is_viscous)
              {
                const auto flux = ConvectionDiffusionOperator::face(
                  phi_m.get_value(q),
                  w_p,
                  phi_m.get_gradient(q),
                  grad_w_p,
                  phi_m.normal_vector(q),
                  interior_penalty_parameter,
                  CompressibleConvectiveFlux<dim, number>(flow_scratch_data.material.data),
                  CompressibleDiffusiveFlux<dim, number>(flow_scratch_data.material.data));

                flux_m = flux.inner_face_value;

                phi_m.submit_gradient(flux.inner_face_gradient, q);
              }
            else
              {
                const auto flux = ConvectionOperator::face(phi_m.get_value(q),
                                                           w_p,
                                                           phi_m.normal_vector(q),
                                                           CompressibleConvectiveFlux<dim, number>(
                                                             flow_scratch_data.material.data));

                flux_m = flux.inner_face_value;
              }

            phi_m.submit_value(flux_m, q);
          }

        phi_m.integrate_scatter(EvaluationFlags::values | (is_viscous ? EvaluationFlags::gradients :
                                                                        EvaluationFlags::nothing),
                                dst);
      }
  }

  template class DGCompressibleFlowOperatorExplicit<1, double, true>;
  template class DGCompressibleFlowOperatorExplicit<2, double, true>;
  template class DGCompressibleFlowOperatorExplicit<3, double, true>;
  template class DGCompressibleFlowOperatorExplicit<1, double, false>;
  template class DGCompressibleFlowOperatorExplicit<2, double, false>;
  template class DGCompressibleFlowOperatorExplicit<3, double, false>;
} // namespace MeltPoolDG::Flow
