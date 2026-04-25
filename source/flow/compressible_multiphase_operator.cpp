
#include <deal.II/base/vectorization.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/flow/compressible_flow_explicit_utils.hpp>
#include <meltpooldg/flow/compressible_flow_utils.hpp>
#include <meltpooldg/flow/compressible_multiphase_interface_kernels.hpp>
#include <meltpooldg/flow/compressible_multiphase_operator.hpp>
#include <meltpooldg/utilities/preprocessor_directives.hpp>

#include <cmath>


namespace MeltPoolDG::Multiphase
{
  using namespace dealii;

  template <int dim, typename number, bool is_viscous_gas, bool is_viscous_liquid>
  CompressibleMultiphaseOperator<dim, number, is_viscous_gas, is_viscous_liquid>::
    CompressibleMultiphaseOperator(
      MeltPoolDG::Flow::CompressibleMultiphaseScratchData<dim, number> &multiphase_scratch_data,
      const MappingInfoType                                            &mapping_info_interface_in,
      const MappingInfoVectorType                                      &mapping_info_cells_in,
      const MappingInfoVectorType                                      &mapping_info_faces_in)
    : multiphase_scratch_data(multiphase_scratch_data)
    , convective_terms_liquid(multiphase_scratch_data.flow_data,
                              multiphase_scratch_data.material_liquid)
    , convective_terms_gas(multiphase_scratch_data.flow_data, multiphase_scratch_data.material_gas)
    , viscous_terms_liquid(multiphase_scratch_data.material_liquid)
    , viscous_terms_gas(multiphase_scratch_data.material_gas)
    , mapping_info_interface(mapping_info_interface_in)
    , mapping_info_cells(mapping_info_cells_in)
    , mapping_info_faces(mapping_info_faces_in)
    , fe_point_temp(FE_DGQ<dim>(multiphase_scratch_data.flow_data.fe.degree), dim + 2)
    , n_dofs_per_cell(fe_point_temp.dofs_per_cell)
  {
    const auto &l     = multiphase_scratch_data.material_liquid.data;
    const auto &g     = multiphase_scratch_data.material_gas.data;
    const auto &pc_lg = multiphase_scratch_data.phase_change.liquid_gas;

    const number q_liquid =
      2. * l.dynamic_viscosity + l.thermal_conductivity / (l.specific_isobaric_heat / l.gamma);
    const number q_gas =
      2. * g.dynamic_viscosity + g.thermal_conductivity / (g.specific_isobaric_heat / g.gamma);

    visc_ave_weight_phase_liquid = q_liquid / (q_liquid + q_gas);
    visc_ave_weight_phase_gas    = 1. - visc_ave_weight_phase_liquid;

    if (multiphase_scratch_data.phase_coupling.evaporation_model == EvaporationModelType::Knight)
      {
        evaporation_model_knight = std::make_unique<Evaporation::EvaporationModelKnight<number>>(
          pc_lg.reference_pressure,
          pc_lg.boiling_temperature,
          pc_lg.latent_heat_of_vaporization,
          g.specific_gas_constant,
          g.gamma);
      }
    else
      AssertThrow(multiphase_scratch_data.phase_coupling.evaporation_model ==
                    EvaporationModelType::constant,
                  dealii::ExcMessage("The given evaporation model is not supported."));
  }

  template <int dim, typename number, bool is_viscous_gas, bool is_viscous_liquid>
  void
  CompressibleMultiphaseOperator<dim, number, is_viscous_gas, is_viscous_liquid>::vmult(
    VectorType       &dst,
    const VectorType &src) const
  {
    using local_applier_type =
      std::function<void(const dealii::MatrixFree<dim, number> &,
                         dealii::LinearAlgebra::distributed::Vector<number>       &dst,
                         const dealii::LinearAlgebra::distributed::Vector<number> &src,
                         const std::pair<unsigned int, unsigned int> &)>;

    local_applier_type cell          = MPDG_LAMBDA_WRAPPER(this->local_apply_cell_lhs);
    local_applier_type face          = MPDG_LAMBDA_WRAPPER(this->local_apply_face_lhs);
    local_applier_type boundary_face = MPDG_LAMBDA_WRAPPER(this->local_apply_boundary_face_lhs);
    multiphase_scratch_data.scratch_data.get_matrix_free().loop(
      cell,
      face,
      boundary_face,
      dst,
      src,
      true,
      MatrixFree<dim, number>::DataAccessOnFaces::gradients,
      MatrixFree<dim, number>::DataAccessOnFaces::gradients);
  }

  template <int dim, typename number, bool is_viscous_gas, bool is_viscous_liquid>
  void
  CompressibleMultiphaseOperator<dim, number, is_viscous_gas, is_viscous_liquid>::create_rhs(
    const number     &time,
    const number     &time_step,
    VectorType       &dst,
    const VectorType &src) const
  {
    Assert(time_step > 0., dealii::ExcMessage("Time step size must be larger than 0!"));
    inv_time_step = 1. / time_step;
using local_applier_type =
      std::function<void(const MatrixFree<dim, number> &,
                         LinearAlgebra::distributed::Vector<number> &,
                         const LinearAlgebra::distributed::Vector<number> &,
                         const std::pair<unsigned int, unsigned int> &)>;
    if (time+time_step>(1e-4-1e-11))
    //if (time>(0.05e-5-0.48*time_step) and time<(0.05e-5+0.48*time_step))
      {
        local_applier_type cell_quad          = MPDG_LAMBDA_WRAPPER(output_at_quad_points);
    multiphase_scratch_data.scratch_data.get_matrix_free().cell_loop(
    cell_quad,
  dst,
  src);

        /*multiphase_scratch_data.boundary_conditions.update_boundary_conditions(time);
        local_applier_type cell          = MPDG_LAMBDA_WRAPPER(local_apply_dummy);
        local_applier_type face          = MPDG_LAMBDA_WRAPPER(output_at_quad_points_face);
        local_applier_type boundary_face = MPDG_LAMBDA_WRAPPER(local_apply_dummy);
        multiphase_scratch_data.scratch_data.get_matrix_free().loop(
          cell,
          face,
          boundary_face,
          dst,
          src,
          true,
          MatrixFree<dim, number>::DataAccessOnFaces::gradients,
          MatrixFree<dim, number>::DataAccessOnFaces::gradients);*/

      }





    multiphase_scratch_data.boundary_conditions.update_boundary_conditions(time);
    local_applier_type cell          = MPDG_LAMBDA_WRAPPER(local_apply_cell_rhs);
    local_applier_type face          = MPDG_LAMBDA_WRAPPER(local_apply_face_rhs);
    local_applier_type boundary_face = MPDG_LAMBDA_WRAPPER(local_apply_boundary_face_rhs);
    multiphase_scratch_data.scratch_data.get_matrix_free().loop(
      cell,
      face,
      boundary_face,
      dst,
      src,
      true,
      MatrixFree<dim, number>::DataAccessOnFaces::gradients,
      MatrixFree<dim, number>::DataAccessOnFaces::gradients);

    dst *= time_step;
  }

  template <int dim, typename number, bool is_viscous_gas, bool is_viscous_liquid>
  void
  CompressibleMultiphaseOperator<dim, number, is_viscous_gas, is_viscous_liquid>::
    local_apply_cell_rhs(const dealii::MatrixFree<dim, number> &,
                         VectorType                          &dst,
                         const VectorType                    &src,
                         const std::pair<unsigned, unsigned> &cell_range) const
  {
    const auto cell_category =
      multiphase_scratch_data.scratch_data.get_cell_range_category(cell_range);

    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> constant_body_force;
    const Functions::ConstantFunction<dim>                 *constant_function =
      dynamic_cast<Functions::ConstantFunction<dim> *>(multiphase_scratch_data.body_force.get());

    if (constant_function)
      constant_body_force = VectorTools::evaluate_function_at_vectorized_points(
        *constant_function, dealii::Point<dim, dealii::VectorizedArray<number>>());

    const number laser_heat_source =
              update_laser_heat_source<number>(multiphase_scratch_data.phase_coupling,
                                               current_time);

    const dealii::VectorizedArray<number> interface_position = dealii::make_vectorized_array(0.);
    const number element_size = 3.125e-6;

    // lambda function for cell integral
    auto process_cell =
      [&]<bool is_gas_phase, bool is_viscous, typename IntegratorType>(IntegratorType &eval,
                                                                       const auto &convective_terms,
                                                                       const auto &viscous_terms) {
        for (const unsigned int q : eval.quadrature_point_indices())
          {
            const auto [force, grad_flux] =
              Flow::rhs_cell_integral_kernel<dim, number, IntegratorType, is_viscous>(
                eval,
                q,
                constant_function ? &constant_body_force : nullptr,
                convective_terms,
                viscous_terms,
                multiphase_scratch_data.body_force);

            ConservedVariablesType darcy_damping{};

            // TODO: use DarcyDampingOperation
            if (!is_gas_phase and
                multiphase_scratch_data.phase_change.solid_liquid.use_darcy_damping)
              {
                const auto w        = eval.get_value(q);
                const auto velocity = Flow::calculate_velocity<dim, number>(w);
                const auto temperature =
                  multiphase_scratch_data.material_liquid.eos_utils->calculate_temperature(w);

                const VectorizedArray<number> T_liquidus_vec = dealii::make_vectorized_array(
                  multiphase_scratch_data.phase_change.solid_liquid.liquidus_temperature);
                const VectorizedArray<number> T_solidus_vec = dealii::make_vectorized_array(
                  multiphase_scratch_data.phase_change.solid_liquid.solidus_temperature);

                VectorizedArray<number> liquid_fraction =
                  (temperature - T_solidus_vec) / (T_liquidus_vec - T_solidus_vec);

                // liquid fraction is bounded [0;1]
                liquid_fraction =
                  std::min(std::max(liquid_fraction,
                                    dealii::make_vectorized_array<VectorizedArray<number>>(0.)),
                           dealii::make_vectorized_array<VectorizedArray<number>>(1.));

                const VectorizedArray<number> solid_fraction = 1. - liquid_fraction;

                VectorizedArray<number> darcy_damping_coefficient =
                  -multiphase_scratch_data.darcy_damping.mushy_zone_morphology * solid_fraction *
                  solid_fraction /
                  (liquid_fraction * liquid_fraction * liquid_fraction +
                   multiphase_scratch_data.darcy_damping.avoid_div_zero_constant);

                darcy_damping_coefficient = dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
                      eval.quadrature_point(q)[0],
                      dealii::make_vectorized_array(-3.e-4),
                      darcy_damping_coefficient,
                      0. * darcy_damping_coefficient);

                // contribution to momentum equation
                darcy_damping[1] = darcy_damping_coefficient * velocity[0];

                // contribution to energy equation
                darcy_damping[2] = darcy_damping[1] * velocity[0];
              }

            // consider mass term
            ConservedVariablesType flux = eval.get_value(q) * inv_time_step;

            if (multiphase_scratch_data.phase_change.solid_liquid.use_darcy_damping)
              flux += darcy_damping;

            if (not is_gas_phase)
              {
                ConservedVariablesType regularized_heat_source{};
                double epsilon = 2. * 3.125e-6;
                const dealii::Point<dim, VectorizedArray<double>> quad_point = eval.quadrature_point(q);
                const dealii::VectorizedArray<double> x = quad_point[0];
                auto delta = 1. / (sqrt(2.*std::numbers::pi) * epsilon) * std::exp(-((x-interface_position)/epsilon) * ((x-interface_position)/epsilon) / 2.);
                delta = dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
                                      x,
                                      dealii::make_vectorized_array(-10.*epsilon),
                                      0. * delta,
                                      delta);
                delta = dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
                                      x,
                                      dealii::make_vectorized_array(10.*epsilon),
                                      0. * delta,
                                      delta);

                regularized_heat_source[2] = delta * laser_heat_source;
                flux += regularized_heat_source;
              }

            if (multiphase_scratch_data.body_force.get() != nullptr)
              flux += force;

            eval.submit_value(flux, q);
            eval.submit_gradient(grad_flux, q);
          }
      };

    switch (cell_category)
      {
          case CutUtil::CellCategory::liquid: {
            auto eval_liquid = create_cell_integrator(CutUtil::CellCategory::liquid, 0);
            for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
              {
                eval_liquid.reinit(cell);
                eval_liquid.gather_evaluate(src,
                                            EvaluationFlags::values |
                                              (is_viscous_liquid ? EvaluationFlags::gradients :
                                                                   EvaluationFlags::nothing));
                process_cell.template operator()<false, is_viscous_liquid>(eval_liquid,
                                                                           convective_terms_liquid,
                                                                           viscous_terms_liquid);
                eval_liquid.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients,
                                              dst);
              }
            break;
          }
          case CutUtil::CellCategory::gas: {
            auto eval_gas = create_cell_integrator(CutUtil::CellCategory::gas, dim + 2);
            for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
              {
                eval_gas.reinit(cell);
                eval_gas.gather_evaluate(src,
                                         EvaluationFlags::values |
                                           (is_viscous_gas ? EvaluationFlags::gradients :
                                                             EvaluationFlags::nothing));
                process_cell.template operator()<true, is_viscous_gas>(eval_gas,
                                                                       convective_terms_gas,
                                                                       viscous_terms_gas);
                eval_gas.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients,
                                           dst);
              }
            break;
          }
          case CutUtil::CellCategory::intersected: {
            constexpr unsigned int n_lanes = VectorizedArray<number>::size();

            auto eval_liquid_intersected =
              create_cell_integrator(CutUtil::CellCategory::intersected, 0);
            auto eval_gas_intersected =
              create_cell_integrator(CutUtil::CellCategory::intersected, dim + 2);

            dealii::FEPointEvaluation<dim + 2, dim, dim, dealii::VectorizedArray<number>>
              eval_point_liquid(*mapping_info_cells[0], fe_point_temp);
            dealii::FEPointEvaluation<dim + 2, dim, dim, dealii::VectorizedArray<number>>
              eval_point_interface_liquid(mapping_info_interface, fe_point_temp);
            dealii::FEPointEvaluation<dim + 2, dim, dim, dealii::VectorizedArray<number>>
              eval_point_gas(*mapping_info_cells[1], fe_point_temp);
            dealii::FEPointEvaluation<dim + 2, dim, dim, dealii::VectorizedArray<number>>
              eval_point_interface_gas(mapping_info_interface, fe_point_temp);

            // reset values for interface velocity
            level_set_advection_operator.clear_interface_velocity();

            // update current laser heat source
            const number laser_heat_source =
              update_laser_heat_source<number>(multiphase_scratch_data.phase_coupling,
                                               current_time);

            for (unsigned int cell_batch = cell_range.first; cell_batch < cell_range.second;
                 ++cell_batch)
              {
                eval_liquid_intersected.reinit(cell_batch);
                eval_liquid_intersected.read_dof_values(src);

                eval_gas_intersected.reinit(cell_batch);
                eval_gas_intersected.read_dof_values(src);

                for (unsigned int lane = 0;
                     lane < multiphase_scratch_data.scratch_data.get_matrix_free()
                              .n_active_entries_per_cell_batch(cell_batch);
                     ++lane)
                  {
                    // evaluate for domain integral in liquid phase
                    eval_point_liquid.reinit(cell_batch * n_lanes + lane);
                    eval_point_liquid.evaluate(
                      StridedArrayView<const number, n_lanes>(
                        &eval_liquid_intersected.begin_dof_values()[0][lane], n_dofs_per_cell),
                      EvaluationFlags::values | (is_viscous_liquid ? EvaluationFlags::gradients :
                                                                     EvaluationFlags::nothing));

                    // evaluate for interface integral in liquid phase
                    eval_point_interface_liquid.reinit(cell_batch * n_lanes + lane);
                    eval_point_interface_liquid.evaluate(
                      StridedArrayView<const number, n_lanes>(
                        &eval_liquid_intersected.begin_dof_values()[0][lane], n_dofs_per_cell),
                      EvaluationFlags::values | EvaluationFlags::gradients);

                    // evaluate for domain integral in gas phase
                    eval_point_gas.reinit(cell_batch * n_lanes + lane);
                    eval_point_gas.evaluate(
                      StridedArrayView<const number, n_lanes>(
                        &eval_gas_intersected.begin_dof_values()[0][lane], n_dofs_per_cell),
                      EvaluationFlags::values |
                        (is_viscous_gas ? EvaluationFlags::gradients : EvaluationFlags::nothing));

                    // evaluate for interface integral in gas phase
                    eval_point_interface_gas.reinit(cell_batch * n_lanes + lane);
                    eval_point_interface_gas.evaluate(
                      StridedArrayView<const number, n_lanes>(
                        &eval_gas_intersected.begin_dof_values()[0][lane], n_dofs_per_cell),
                      EvaluationFlags::values | EvaluationFlags::gradients);

                    // do domain integral in liquid phase
                    process_cell.template operator()<false, is_viscous_liquid>(
                      eval_point_liquid, convective_terms_liquid, viscous_terms_liquid);

                    eval_point_liquid.integrate(
                      StridedArrayView<number, n_lanes>(
                        &eval_liquid_intersected.begin_dof_values()[0][lane], n_dofs_per_cell),
                      EvaluationFlags::values | EvaluationFlags::gradients);

                    // do domain integral in gas phase
                    process_cell.template operator()<true, is_viscous_gas>(eval_point_gas,
                                                                           convective_terms_gas,
                                                                           viscous_terms_gas);

                    eval_point_gas.integrate(StridedArrayView<number, n_lanes>(
                                               &eval_gas_intersected.begin_dof_values()[0][lane],
                                               n_dofs_per_cell),
                                             EvaluationFlags::values | EvaluationFlags::gradients);

                    // do interface integral
                    if (multiphase_scratch_data.phase_coupling.type ==
                        InterfaceNumericalMethod::penalty)
                      {
                        // enumeration for conserved variables component indices
                        using Idx = std::conditional_t<
                          dim == 1,
                          Flow::Idx1D,
                          std::conditional_t<dim == 2,
                                             Flow::Idx2D,
                                             std::conditional_t<dim == 3, Flow::Idx3D, void>>>;

                        for (const unsigned int q :
                             eval_point_interface_liquid.quadrature_point_indices())
                          {
                            auto w_liquid      = eval_point_interface_liquid.get_value(q);
                            auto w_gas         = eval_point_interface_gas.get_value(q);
                            auto grad_w_liquid = eval_point_interface_liquid.get_gradient(q);
                            auto grad_w_gas    = eval_point_interface_gas.get_gradient(q);
                            // Outwards pointing normal vector with respect to the liquid domain
                            const auto normal = -eval_point_interface_liquid.normal_vector(q);

                            const auto [m_dot_evap, delta_T] =
                              update_evaporative_mass_flux_and_temperature_jump<dim, number>(
                                w_liquid,
                                w_gas,
                                normal,
                                multiphase_scratch_data,
                                evaporation_model_knight.get());

                            const auto [flux_liquid, flux_gas] =
                              calculate_convective_and_viscous_interface_flux_penalty<
                                dim,
                                number,
                                ConservedVariablesType,
                                ConservedVariablesGradType>(w_liquid,
                                                            w_gas,
                                                            grad_w_liquid,
                                                            grad_w_gas,
                                                            multiphase_scratch_data,
                                                            viscous_terms_liquid,
                                                            viscous_terms_gas,
                                                            m_dot_evap,
                                                            laser_heat_source);

                            // Compute the velocity at the interface using the difference in
                            // momentum and density between the liquid and gas phases
                            // (mass conservation across the interface).
                            // indices: [conserved variables component][vectorization index]

                            // TODO: temporal solution for dim=1; revise for dim>1!
                            const number interface_velocity =
                              (std::abs(w_liquid[Idx::density][0] - w_gas[Idx::density][0]) >
                                   1.e-12 ?
                                 (w_liquid[Idx::momentum_x][0] - w_gas[Idx::momentum_x][0]) /
                                   (w_liquid[Idx::density][0] - w_gas[Idx::density][0]) :
                                 w_liquid[Idx::momentum_x][0] / w_liquid[Idx::density][0]) *
                              normal[0][0];

                            level_set_advection_operator.set_interface_velocity(interface_velocity);

                            eval_point_interface_liquid.submit_value(-flux_liquid, q);
                            eval_point_interface_gas.submit_value(-flux_gas, q);
                          }
                      }
                    else if (multiphase_scratch_data.phase_coupling.type ==
                               InterfaceNumericalMethod::HLLP0_and_SIPG or
                             multiphase_scratch_data.phase_coupling.type ==
                               InterfaceNumericalMethod::HLLP0_and_penalty)
                      {
                        for (const unsigned int q :
                             eval_point_interface_gas.quadrature_point_indices())
                          {
                            const auto w_liquid      = eval_point_interface_liquid.get_value(q);
                            const auto w_gas         = eval_point_interface_gas.get_value(q);
                            const auto grad_w_liquid = eval_point_interface_liquid.get_gradient(q);
                            const auto grad_w_gas    = eval_point_interface_gas.get_gradient(q);
                            // Outwards pointing normal vector with respect to the liquid domain.
                            // (The sign depends on the level-set orientation. Currently, we use
                            // a positive level-set for the liquid phase.)
                            const auto normal = -eval_point_interface_liquid.normal_vector(q);

                            const auto [m_dot_evap, delta_T] =
                              update_evaporative_mass_flux_and_temperature_jump<dim, number>(
                                w_liquid,
                                w_gas,
                                normal,
                                multiphase_scratch_data,
                                evaporation_model_knight.get());

                            const auto [riemann_flux_liquid,
                                        riemann_flux_gas,
                                        velocity_interface_vec] =
                              calculate_convective_interface_flux_HLLP0<dim,
                                                                        number,
                                                                        ConservedVariablesType,
                                                                        ConservedVariablesGradType>(
                                w_liquid,
                                w_gas,
                                grad_w_gas,
                                grad_w_liquid,
                                normal,
                                convective_terms_liquid,
                                convective_terms_gas,
                                multiphase_scratch_data,
                                m_dot_evap,
                                laser_heat_source,
                                current_time);

                            ConservedVariablesType flux_liquid =
                              contract_tensor_with_vector<dim + 2, dim, number>(riemann_flux_liquid,
                                                                                normal);
                            ConservedVariablesType flux_gas =
                              contract_tensor_with_vector<dim + 2, dim, number>(riemann_flux_gas,
                                                                                -normal);
                            // TODO: consider more complex data structure for velocity for dim>1
                            // TODO: project interface velocity in normal direction for dim>1
                            // returns velocity with respect to the outward liquid phase pointing
                            // normal!
                            level_set_advection_operator.set_interface_velocity(
                              velocity_interface_vec[0] * normal[0][0]);

                            if (is_viscous_liquid or is_viscous_gas)
                              {
                                if (multiphase_scratch_data.phase_coupling.type ==
                                    InterfaceNumericalMethod::HLLP0_and_SIPG)
                                  {
                                    const auto [viscous_interface_flux_liquid,
                                                viscous_interface_flux_gas] =
                                      calculate_viscous_interface_flux<dim,
                                                                       number,
                                                                       ConservedVariablesType,
                                                                       ConservedVariablesGradType>(
                                        w_liquid,
                                        w_gas,
                                        grad_w_liquid,
                                        grad_w_gas,
                                        normal,
                                        visc_ave_weight_phase_liquid,
                                        visc_ave_weight_phase_gas,
                                        multiphase_scratch_data.phase_coupling.hllp0_and_sipg
                                          .interior_penalty_parameter_interface,
                                        viscous_terms_liquid,
                                        viscous_terms_gas,
                                        multiphase_scratch_data,
                                        multiphase_scratch_data.scratch_data.get_min_cell_size(),
                                        m_dot_evap,
                                        delta_T,
                                        laser_heat_source);

                                    flux_liquid -= viscous_interface_flux_liquid;
                                    // opposite normal direction for phase 2
                                    flux_gas += viscous_interface_flux_gas;
                                  }
                                else if (multiphase_scratch_data.phase_coupling.type ==
                                         InterfaceNumericalMethod::HLLP0_and_penalty)
                                  {
                                    const auto [viscous_interface_flux_liquid,
                                                viscous_interface_flux_gas] =
                                      calculate_viscous_interface_flux_method_3<
                                        dim,
                                        number,
                                        ConservedVariablesType,
                                        ConservedVariablesGradType>(
                                        w_liquid,
                                        w_gas,
                                        grad_w_liquid,
                                        grad_w_gas,
                                        normal,
                                        visc_ave_weight_phase_liquid,
                                        visc_ave_weight_phase_gas,
                                        viscous_terms_liquid,
                                        viscous_terms_gas,
                                        multiphase_scratch_data,
                                        multiphase_scratch_data.scratch_data.get_min_cell_size(),
                                        m_dot_evap,
                                        delta_T,
                                        laser_heat_source);

                                    flux_liquid += viscous_interface_flux_liquid;
                                    flux_gas += viscous_interface_flux_gas;
                                  }
                              }

                            eval_point_interface_liquid.submit_value(-flux_liquid, q);
                            eval_point_interface_gas.submit_value(-flux_gas, q);

                            if ((is_viscous_liquid or is_viscous_gas) and
                                multiphase_scratch_data.phase_coupling.type ==
                                  InterfaceNumericalMethod::HLLP0_and_SIPG)
                              {
                                const auto [numerical_flux_gradient_liquid,
                                            numerical_flux_gradient_gas] =
                                  calculate_viscous_interface_flux_gradient<
                                    dim,
                                    number,
                                    ConservedVariablesType,
                                    ConservedVariablesGradType>(w_liquid,
                                                                w_gas,
                                                                normal,
                                                                visc_ave_weight_phase_liquid,
                                                                visc_ave_weight_phase_gas,
                                                                viscous_terms_liquid,
                                                                viscous_terms_gas,
                                                                multiphase_scratch_data,
                                                                m_dot_evap,
                                                                delta_T);

                                eval_point_interface_liquid.submit_gradient(
                                  -numerical_flux_gradient_liquid, q);
                                eval_point_interface_gas.submit_gradient(
                                  -numerical_flux_gradient_gas, q);
                              }
                          }
                      }
                    else
                      Assert(false, dealii::ExcNotImplemented());

                    eval_point_interface_liquid.integrate(StridedArrayView<number, n_lanes>(
                                  &eval_liquid_intersected.begin_dof_values()[0][lane],
                                  n_dofs_per_cell),
                                  EvaluationFlags::values |
                                  (multiphase_scratch_data.phase_coupling.type
                                    == InterfaceNumericalMethod::HLLP0_and_SIPG ?
                                    EvaluationFlags::gradients: EvaluationFlags::nothing) , true
                                  /*specify flag 'true' for summing the integrated values
                                   *into the solution values*/);

                    eval_point_interface_gas.integrate(StridedArrayView<number, n_lanes>(
                                  &eval_gas_intersected.begin_dof_values()[0][lane],
                                  n_dofs_per_cell),
                                  EvaluationFlags::values |
                                  (multiphase_scratch_data.phase_coupling.type
                                    == InterfaceNumericalMethod::HLLP0_and_SIPG ?
                                    EvaluationFlags::gradients : EvaluationFlags::nothing), true
                                  /*specify flag 'true' for summing the integrated values
                                   *into the solution values*/);
                  }
                eval_liquid_intersected.distribute_local_to_global(dst);
                eval_gas_intersected.distribute_local_to_global(dst);
              }
            break;
          }
        default:
          break;
      }
  }

  template <int dim, typename number, bool is_viscous_gas, bool is_viscous_liquid>
  void
  CompressibleMultiphaseOperator<dim, number, is_viscous_gas, is_viscous_liquid>::
  output_at_quad_points(const MatrixFree<dim, number> &,
               VectorType &dst,
               const VectorType &src,
               const std::pair<unsigned int, unsigned int> &cell_range) const
  {
      auto eval_liquid = create_cell_integrator(CutUtil::CellCategory::liquid, 0);
      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          eval_liquid.reinit(cell);
          eval_liquid.gather_evaluate(src,
                                      EvaluationFlags::values |
                                        (is_viscous_liquid ? EvaluationFlags::gradients :
                                                             EvaluationFlags::nothing));
          const number laser_heat_source =
              update_laser_heat_source<number>(multiphase_scratch_data.phase_coupling,
                                               current_time);

          for (const unsigned int q : eval_liquid.quadrature_point_indices())
            {
              const auto w        = eval_liquid.get_value(q);
              const auto grad_w        = eval_liquid.get_gradient(q);
              const auto quad_point = eval_liquid.quadrature_point(q)[0];
              const auto velocity = Flow::calculate_velocity<dim, number>(w);
              const auto pressure =
                multiphase_scratch_data.material_liquid.eos_utils->calculate_thermodynamic_pressure(w);
              const auto temperature =
                multiphase_scratch_data.material_liquid.eos_utils->calculate_temperature(w);

              double epsilon = 2. * 3.125e-6;
              auto delta = 1. / (sqrt(2.*std::numbers::pi) * epsilon) * std::exp(-((quad_point)/epsilon) * ((quad_point)/epsilon) / 2.);
              delta = dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
                                    quad_point,
                                    dealii::make_vectorized_array(-10.*epsilon),
                                    0. * delta,
                                    delta);
              delta = dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
                                    quad_point,
                                    dealii::make_vectorized_array(10.*epsilon),
                                    0. * delta,
                                    delta);

              const auto regularized_heat_source = delta * laser_heat_source;

              const ConservedVariablesGradType convective_flux = convective_terms_liquid.calculate_convective_flux(w);
              const ConservedVariablesGradType viscous_flux = viscous_terms_liquid.calculate_viscous_flux(w,grad_w);

              std::cout << std::fixed << std::setprecision(12);

              for (unsigned int lane = 0; lane < multiphase_scratch_data.scratch_data.get_matrix_free()
                                               .n_active_entries_per_cell_batch(cell);
               ++lane)
                {
                  std::cout << "position: " << quad_point[lane]
                  << "; density: " << w[0][lane]
                  << "; momentum: " << w[1][lane]
                  << "; total energy: " << w[2][lane]
                  << ": temperature: " << temperature[lane]
                  << "; pressure: " << pressure[lane]
                  << "; velocity: " << w[1][lane] / w[0][lane]
                  << "; inner energy: " << w[1][lane] / w[0][lane]
                  << "; heat source: " << regularized_heat_source[lane]
                  << "; convective flux mass: " << convective_flux[0][0][lane]
                  << "; convective flux momentum: " << convective_flux[1][0][lane]
                  << "; convective flux total energy: " << convective_flux[2][0][lane]
                  << "; viscous flux momentum: " << viscous_flux[1][0][lane]
                  << "; viscous flux total energy: " << viscous_flux[2][0][lane]
                  << std::endl;
                }
            }
        }
  }

    template <int dim, typename number, bool is_viscous_gas, bool is_viscous_liquid>
  void
  CompressibleMultiphaseOperator<dim, number, is_viscous_gas, is_viscous_liquid>::
  output_at_quad_points_face(const MatrixFree<dim, number> &,
               VectorType &dst,
               const VectorType &src,
               const std::pair<unsigned int, unsigned int> &face_range) const
  {
    auto [eval_liquid_m, eval_liquid_p] =
            create_face_integrators(CutUtil::CellCategory::liquid, 0);
      for (unsigned int cell = face_range.first; cell < face_range.second; ++cell)
        {
          eval_liquid_m.reinit(cell);
          eval_liquid_m.gather_evaluate(src,
                                      EvaluationFlags::values |
                                        (is_viscous_liquid ? EvaluationFlags::gradients :
                                                             EvaluationFlags::nothing));

          eval_liquid_p.reinit(cell);
          eval_liquid_p.gather_evaluate(src,
                                      EvaluationFlags::values |
                                        (is_viscous_liquid ? EvaluationFlags::gradients :
                                                             EvaluationFlags::nothing));

          const auto interior_penalty_parameter = 0.5 *
      std::max(eval_liquid_m.read_cell_data(multiphase_scratch_data.interior_penalty_parameter),
               eval_liquid_p.read_cell_data(
                 multiphase_scratch_data.interior_penalty_parameter));

          for (const unsigned int q : eval_liquid_m.quadrature_point_indices())
            {
              const auto w_m        = eval_liquid_m.get_value(q);
              const auto normal = eval_liquid_m.normal_vector(q);
              const auto grad_w_m        = eval_liquid_m.get_gradient(q);
              const auto w_p        = eval_liquid_p.get_value(q);
              const auto grad_w_p        = eval_liquid_p.get_gradient(q);
              const auto quad_point = eval_liquid_m.quadrature_point(q)[0];

              const auto convective_numerical_flux = convective_terms_liquid.calculate_convective_numerical_flux(w_m,w_p,normal);
              const auto viscous_numerical_flux = viscous_terms_liquid.calculate_viscous_numerical_flux(w_m,w_p,grad_w_m,grad_w_p, normal,interior_penalty_parameter);

              std::cout << std::fixed << std::setprecision(12);

              for (unsigned int lane = 0; lane < multiphase_scratch_data.scratch_data.get_matrix_free()
                                               .n_active_entries_per_face_batch(cell);
               ++lane)
                {
                  std::cout << "position: " << quad_point[lane]
                  << "; convective numerical flux mass: " << convective_numerical_flux[0][lane]
                  << "; convective numerical flux momentum: " << convective_numerical_flux[1][lane]
                  << "; convective numerical flux total energy: " << convective_numerical_flux[2][lane]
                  << "; viscous numerical flux mass: " << viscous_numerical_flux[0][lane]
                  << "; viscous numerical flux momentum: " << viscous_numerical_flux[1][lane]
                  << "; viscous numerical flux total energy: " << viscous_numerical_flux[2][lane]
                  << std::endl;
                }
            }
        }
  }

   template <int dim, typename number, bool is_viscous_gas, bool is_viscous_liquid>
  void
  CompressibleMultiphaseOperator<dim, number, is_viscous_gas, is_viscous_liquid>::
  local_apply_dummy(const MatrixFree<dim, number> &,
               VectorType &dst,
               const VectorType &src,
               const std::pair<unsigned int, unsigned int> &face_range) const
  {
  }


  template <int dim, typename number, bool is_viscous_gas, bool is_viscous_liquid>
  void
  CompressibleMultiphaseOperator<dim, number, is_viscous_gas, is_viscous_liquid>::
    local_apply_face_rhs(const dealii::MatrixFree<dim, number> &,
                         VectorType                                  &dst,
                         const VectorType                            &src,
                         const std::pair<unsigned int, unsigned int> &face_range) const
  {
    const auto face_category =
      multiphase_scratch_data.scratch_data.get_face_range_category(face_range);
    const CutUtil::FaceType face_type = CutUtil::get_face_type(face_category);

    auto process_bulk_face_range = [&]<bool is_viscous>(auto &eval_m,
                                                        auto &eval_p,
                                                        auto &convective_terms,
                                                        auto &viscous_terms) {
      const auto eval_flags = EvaluationFlags::values |
                              (is_viscous ? EvaluationFlags::gradients : EvaluationFlags::nothing);
      for (unsigned int face = face_range.first; face < face_range.second; ++face)
        {
          eval_m.reinit(face);
          eval_p.reinit(face);

          eval_m.gather_evaluate(src, eval_flags);
          eval_p.gather_evaluate(src, eval_flags);

          const auto interior_penalty_parameter =
            is_viscous ?
              0.5 *
                std::max(eval_m.read_cell_data(multiphase_scratch_data.interior_penalty_parameter),
                         eval_p.read_cell_data(
                           multiphase_scratch_data.interior_penalty_parameter)) :
              0.;

          for (const unsigned int q : eval_m.quadrature_point_indices())
            {
              const auto [flux_m, flux_p, grad_flux_m, grad_flux_p] =
                MeltPoolDG::Flow::rhs_face_integral_kernel<dim,
                                                           number,
                                                           FEFaceIntegrator<dim, dim + 2, number>,
                                                           is_viscous>(
                  eval_m, eval_p, q, interior_penalty_parameter, convective_terms, viscous_terms);

              eval_m.submit_value(flux_m, q);
              eval_p.submit_value(flux_p, q);
              if (is_viscous)
                {
                  eval_m.submit_gradient(grad_flux_m, q);
                  eval_p.submit_gradient(grad_flux_p, q);
                }
            }
          eval_m.integrate_scatter(eval_flags, dst);
          eval_p.integrate_scatter(eval_flags, dst);
        }
    };

    auto process_intersected_face_range = [&]<bool is_viscous>(auto              &eval_m_int,
                                                               auto              &eval_p_int,
                                                               const unsigned int mapping_idx,
                                                               const auto        &convective_terms,
                                                               const auto        &viscous_terms) {
      FEFacePointEvaluation<dim + 2, dim, dim, VectorizedArray<number>> eval_point_m(
        *mapping_info_faces[mapping_idx], fe_point_temp);
      FEFacePointEvaluation<dim + 2, dim, dim, VectorizedArray<number>> eval_point_p(
        *mapping_info_faces[mapping_idx], fe_point_temp);

      const auto eval_flags = EvaluationFlags::values |
                              (is_viscous ? EvaluationFlags::gradients : EvaluationFlags::nothing);

      for (unsigned int face = face_range.first; face < face_range.second; ++face)
        {
          eval_m_int.reinit(face);
          eval_m_int.read_dof_values(src);
          eval_p_int.reinit(face);
          eval_p_int.read_dof_values(src);

          eval_m_int.project_to_face(eval_flags);
          eval_p_int.project_to_face(eval_flags);

          const auto face_info =
            multiphase_scratch_data.scratch_data.get_matrix_free().get_face_info(face);

          for (unsigned int lane = 0; lane < multiphase_scratch_data.scratch_data.get_matrix_free()
                                               .n_active_entries_per_face_batch(face);
               ++lane)
            {
              eval_point_m.reinit(face_info.cells_interior[lane],
                                  static_cast<int>(face_info.interior_face_no));
              eval_point_p.reinit(face_info.cells_exterior[lane],
                                  static_cast<int>(face_info.exterior_face_no));

              eval_point_m.evaluate_in_face(&eval_m_int.get_scratch_data().begin()[0][lane],
                                            eval_flags);

              eval_point_p.evaluate_in_face(&eval_p_int.get_scratch_data().begin()[0][lane],
                                            eval_flags);

              // factor 0.5 for interior face
              const dealii::VectorizedArray<number> interior_penalty_parameter =
                is_viscous ? 0.5 * std::max(eval_m_int.read_cell_data(
                                              multiphase_scratch_data.interior_penalty_parameter),
                                            eval_p_int.read_cell_data(
                                              multiphase_scratch_data.interior_penalty_parameter)) :
                             0.;

              for (const unsigned int q : eval_point_m.quadrature_point_indices())
                {
                  const auto [flux_m, flux_p, grad_flux_m, grad_flux_p] =
                    MeltPoolDG::Flow::rhs_face_integral_kernel<
                      dim,
                      number,
                      FEFacePointEvaluation<dim + 2, dim, dim, VectorizedArray<number>>,
                      is_viscous>(eval_point_m,
                                  eval_point_p,
                                  q,
                                  interior_penalty_parameter,
                                  convective_terms,
                                  viscous_terms);

                  eval_point_m.submit_value(flux_m, q);
                  eval_point_p.submit_value(flux_p, q);
                  if (is_viscous)
                    {
                      eval_point_m.submit_gradient(grad_flux_m, q);
                      eval_point_p.submit_gradient(grad_flux_p, q);
                    }
                }

              eval_point_m.integrate_in_face(&eval_m_int.get_scratch_data().begin()[0][lane],
                                             eval_flags);

              eval_point_p.integrate_in_face(&eval_p_int.get_scratch_data().begin()[0][lane],
                                             eval_flags);
            }

          eval_m_int.collect_from_face(eval_flags, eval_m_int.begin_dof_values());
          eval_p_int.collect_from_face(eval_flags, eval_p_int.begin_dof_values());

          eval_m_int.distribute_local_to_global(dst);
          eval_p_int.distribute_local_to_global(dst);
        }
    };

    switch (face_type)
      {
          case CutUtil::FaceType::inside_face_liquid: {
            auto [eval_liquid_m, eval_liquid_p] =
              create_face_integrators(CutUtil::CellCategory::liquid, 0);
            process_bulk_face_range.template operator()<is_viscous_liquid>(eval_liquid_m,
                                                                           eval_liquid_p,
                                                                           convective_terms_liquid,
                                                                           viscous_terms_liquid);
          }
          break;

          case CutUtil::FaceType::mixed_face_liquid_intersected: {
            auto eval_liquid_p_intersected =
              create_face_integrator(false, CutUtil::CellCategory::intersected, 0);
            auto eval_liquid_m = create_face_integrator(true, CutUtil::CellCategory::liquid, 0);
            process_bulk_face_range.template operator()<is_viscous_liquid>(
              eval_liquid_m,
              eval_liquid_p_intersected,
              convective_terms_liquid,
              viscous_terms_liquid);
          }
          break;

          case CutUtil::FaceType::mixed_face_intersected_liquid: {
            auto eval_liquid_m_intersected =
              create_face_integrator(true, CutUtil::CellCategory::intersected, 0);
            auto eval_liquid_p = create_face_integrator(false, CutUtil::CellCategory::liquid, 0);
            process_bulk_face_range.template operator()<is_viscous_liquid>(
              eval_liquid_m_intersected,
              eval_liquid_p,
              convective_terms_liquid,
              viscous_terms_liquid);
          }
          break;

          case CutUtil::FaceType::inside_face_gas: {
            auto [eval_gas_m, eval_gas_p] =
              create_face_integrators(CutUtil::CellCategory::gas, dim + 2);
            process_bulk_face_range.template operator()<is_viscous_gas>(eval_gas_m,
                                                                        eval_gas_p,
                                                                        convective_terms_gas,
                                                                        viscous_terms_gas);
          }
          break;

          case CutUtil::FaceType::mixed_face_gas_intersected: {
            auto eval_gas_m = create_face_integrator(true, CutUtil::CellCategory::gas, dim + 2);
            auto eval_gas_p_intersected =
              create_face_integrator(false, CutUtil::CellCategory::intersected, dim + 2);
            process_bulk_face_range.template operator()<is_viscous_gas>(eval_gas_m,
                                                                        eval_gas_p_intersected,
                                                                        convective_terms_gas,
                                                                        viscous_terms_gas);
          }
          break;

          case CutUtil::FaceType::mixed_face_intersected_gas: {
            auto eval_gas_p = create_face_integrator(false, CutUtil::CellCategory::gas, dim + 2);
            auto eval_gas_m_intersected =
              create_face_integrator(true, CutUtil::CellCategory::intersected, dim + 2);
            process_bulk_face_range.template operator()<is_viscous_gas>(eval_gas_m_intersected,
                                                                        eval_gas_p,
                                                                        convective_terms_gas,
                                                                        viscous_terms_gas);
          }
          break;

          case CutUtil::FaceType::intersected_face: {
            auto [eval_liquid_m_intersected, eval_liquid_p_intersected] =
              create_face_integrators(CutUtil::CellCategory::intersected, 0);
            auto [eval_gas_m_intersected, eval_gas_p_intersected] =
              create_face_integrators(CutUtil::CellCategory::intersected, dim + 2);
            process_intersected_face_range.template operator()<is_viscous_liquid>(
              eval_liquid_m_intersected,
              eval_liquid_p_intersected,
              0,
              convective_terms_liquid,
              viscous_terms_liquid);
            process_intersected_face_range.template operator()<is_viscous_gas>(
              eval_gas_m_intersected,
              eval_gas_p_intersected,
              1,
              convective_terms_gas,
              viscous_terms_gas);
            break;
          }
        default:
          break;
      }
  }

  template <int dim, typename number, bool is_viscous_gas, bool is_viscous_liquid>
  void
  CompressibleMultiphaseOperator<dim, number, is_viscous_gas, is_viscous_liquid>::
    local_apply_boundary_face_rhs(const dealii::MatrixFree<dim, number> &,
                                  VectorType                          &dst,
                                  const VectorType                    &src,
                                  const std::pair<unsigned, unsigned> &face_range) const
  {
    const auto face_category =
      multiphase_scratch_data.scratch_data.get_face_range_category(face_range);

    auto process_bulk_face_range = [&]<bool is_viscous, bool is_gas_phase>(auto &eval_m,
                                                                           auto &convective_terms,
                                                                           auto &viscous_terms,
                                                                           auto &material) {
      for (unsigned int face = face_range.first; face < face_range.second; ++face)
        {
          eval_m.reinit(face);
          eval_m.gather_evaluate(src,
                                 dealii::EvaluationFlags::values |
                                   dealii::EvaluationFlags::gradients);

          const dealii::VectorizedArray<number> interior_penalty_parameter =
            is_viscous ? eval_m.read_cell_data(multiphase_scratch_data.interior_penalty_parameter) :
                         0.;

          for (const unsigned int q : eval_m.quadrature_point_indices())
            {
              const auto [flux_m, grad_flux_m] =
                Flow::rhs_boundary_face_integral_kernel<dim,
                                                        number,
                                                        FEFaceIntegrator<dim, dim + 2, number>,
                                                        is_viscous,
                                                        is_gas_phase>(
                  eval_m,
                  q,
                  multiphase_scratch_data.scratch_data.get_matrix_free().get_boundary_id(face),
                  interior_penalty_parameter,
                  convective_terms,
                  viscous_terms,
                  material,
                  multiphase_scratch_data.boundary_conditions);
              eval_m.submit_value(flux_m, q);
              if (is_viscous)
                eval_m.submit_gradient(grad_flux_m, q);
            }

          eval_m.integrate_scatter(EvaluationFlags::values |
                                     (is_viscous ? EvaluationFlags::gradients :
                                                   EvaluationFlags::nothing),
                                   dst);
        }
    };

    auto process_intersected_face_range = [&]<bool is_viscous, bool is_gas_phase>(
                                            auto              &eval_m_int,
                                            const unsigned int mapping_idx,
                                            auto              &convective_terms,
                                            auto              &viscous_terms,
                                            auto              &material) {
      FEFacePointEvaluation<dim + 2, dim, dim, VectorizedArray<number>> eval_point_m_int(
        *mapping_info_faces[mapping_idx], fe_point_temp);

      for (unsigned int face = face_range.first; face < face_range.second; ++face)
        {
          eval_m_int.reinit(face);
          eval_m_int.read_dof_values(src);

          eval_m_int.project_to_face(EvaluationFlags::values | EvaluationFlags::gradients);

          const auto face_info =
            multiphase_scratch_data.scratch_data.get_matrix_free().get_face_info(face);

          for (unsigned int lane = 0; lane < multiphase_scratch_data.scratch_data.get_matrix_free()
                                               .n_active_entries_per_face_batch(face);
               ++lane)
            {
              eval_point_m_int.reinit(face_info.cells_interior[lane],
                                      static_cast<int>(face_info.interior_face_no));

              eval_point_m_int.evaluate_in_face(&eval_m_int.get_scratch_data().begin()[0][lane],
                                                EvaluationFlags::values |
                                                  EvaluationFlags::gradients);

              const dealii::VectorizedArray<number> interior_penalty_parameter =
                is_viscous ?
                  eval_m_int.read_cell_data(multiphase_scratch_data.interior_penalty_parameter) :
                  0.;

              for (const unsigned int q : eval_point_m_int.quadrature_point_indices())
                {
                  const auto [flux_m, grad_flux_m] =
                    MeltPoolDG::Flow::rhs_boundary_face_integral_kernel<
                      dim,
                      number,
                      FEFacePointEvaluation<dim + 2, dim, dim, VectorizedArray<number>>,
                      is_viscous,
                      is_gas_phase>(
                      eval_point_m_int,
                      q,
                      multiphase_scratch_data.scratch_data.get_matrix_free().get_boundary_id(face),
                      interior_penalty_parameter,
                      convective_terms,
                      viscous_terms,
                      material,
                      multiphase_scratch_data.boundary_conditions);

                  eval_point_m_int.submit_value(flux_m, q);
                  if (is_viscous)
                    eval_point_m_int.submit_gradient(grad_flux_m, q);
                }

              eval_point_m_int.integrate_in_face(&eval_m_int.get_scratch_data().begin()[0][lane],
                                                 dealii::EvaluationFlags::values |
                                                   dealii::EvaluationFlags::gradients);
            }

          eval_m_int.collect_from_face(dealii::EvaluationFlags::values |
                                         dealii::EvaluationFlags::gradients,
                                       eval_m_int.begin_dof_values());

          eval_m_int.distribute_local_to_global(dst);
        }
    };

    switch (face_category.first)
      {
          case CutUtil::CellCategory::liquid: {
            auto eval_liquid_m = create_face_integrator(true, CutUtil::CellCategory::liquid, 0);
            process_bulk_face_range.template operator()<is_viscous_liquid, false /*is_gas_phase*/>(
              eval_liquid_m,
              convective_terms_liquid,
              viscous_terms_liquid,
              multiphase_scratch_data.material_liquid);
            break;
          }
          case CutUtil::CellCategory::gas: {
            auto eval_gas_m = create_face_integrator(true, CutUtil::CellCategory::gas, dim + 2);
            process_bulk_face_range.template operator()<is_viscous_gas, true /*is_gas_phase*/>(
              eval_gas_m,
              convective_terms_gas,
              viscous_terms_gas,
              multiphase_scratch_data.material_gas);
            break;
          }
          case CutUtil::CellCategory::intersected: {
            auto eval_liquid_m_intersected =
              create_face_integrator(true, CutUtil::CellCategory::intersected, 0);
            auto eval_gas_m_intersected =
              create_face_integrator(true, CutUtil::CellCategory::intersected, dim + 2);
            process_intersected_face_range
              .template operator()<is_viscous_liquid, false /*is_gas_phase*/>(
                eval_liquid_m_intersected,
                0,
                convective_terms_liquid,
                viscous_terms_liquid,
                multiphase_scratch_data.material_liquid);
            process_intersected_face_range.template
            operator()<is_viscous_gas, true /*is_gas_phase*/>(eval_gas_m_intersected,
                                                              1,
                                                              convective_terms_gas,
                                                              viscous_terms_gas,
                                                              multiphase_scratch_data.material_gas);
            break;
          }
        default:
          break;
      }
  }

  template <int dim, typename number, bool is_viscous_gas, bool is_viscous_liquid>
  void
  CompressibleMultiphaseOperator<dim, number, is_viscous_gas, is_viscous_liquid>::
    local_apply_cell_lhs(const dealii::MatrixFree<dim, number> &,
                         VectorType                          &dst,
                         const VectorType                    &src,
                         const std::pair<unsigned, unsigned> &cell_range) const
  {
    const auto cell_category =
      multiphase_scratch_data.scratch_data.get_cell_range_category(cell_range);

    // Processing function for non-intersected cells
    auto process_bulk_cell_range = [&](auto &eval) {
      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          eval.reinit(cell);
          eval.gather_evaluate(src, dealii::EvaluationFlags::values);

          for (const unsigned int q : eval.quadrature_point_indices())
            eval.submit_value(eval.get_value(q), q);

          eval.integrate_scatter(dealii::EvaluationFlags::values, dst);
        }
    };

    // Processing function for intersected cells
    auto process_intersected_cell_range = [&](auto &eval_intersected, auto &eval_point) {
      constexpr unsigned int n_lanes = VectorizedArray<number>::size();

      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          eval_intersected.reinit(cell);
          eval_intersected.read_dof_values(src);

          for (unsigned int lane = 0; lane < multiphase_scratch_data.scratch_data.get_matrix_free()
                                               .n_active_entries_per_cell_batch(cell);
               ++lane)
            {
              eval_point.reinit(cell * n_lanes + lane);
              eval_point.evaluate(StridedArrayView<const number, n_lanes>(
                                    &eval_intersected.begin_dof_values()[0][lane], n_dofs_per_cell),
                                  dealii::EvaluationFlags::values);

              for (const unsigned int q : eval_point.quadrature_point_indices())
                eval_point.submit_value(eval_point.get_value(q), q);

              eval_point.integrate(
                StridedArrayView<number, n_lanes>(&eval_intersected.begin_dof_values()[0][lane],
                                                  n_dofs_per_cell),
                dealii::EvaluationFlags::values);
            }
          eval_intersected.distribute_local_to_global(dst);
        }
    };

    switch (cell_category)
      {
          case CutUtil::CellCategory::liquid: {
            auto eval_liquid = create_cell_integrator(CutUtil::CellCategory::liquid, 0);
            process_bulk_cell_range(eval_liquid);
          }
          break;

          case CutUtil::CellCategory::gas: {
            auto eval_gas = create_cell_integrator(CutUtil::CellCategory::gas, dim + 2);
            process_bulk_cell_range(eval_gas);
          }
          break;

          case CutUtil::CellCategory::intersected: {
            auto eval_liquid_intersected =
              create_cell_integrator(CutUtil::CellCategory::intersected, 0);
            auto eval_gas_intersected =
              create_cell_integrator(CutUtil::CellCategory::intersected, dim + 2);

            FEPointEvaluation<dim + 2, dim, dim, VectorizedArray<number>> eval_point_liquid(
              *mapping_info_cells[0], fe_point_temp);
            FEPointEvaluation<dim + 2, dim, dim, VectorizedArray<number>> eval_point_gas(
              *mapping_info_cells[1], fe_point_temp);

            process_intersected_cell_range(eval_liquid_intersected, eval_point_liquid);
            process_intersected_cell_range(eval_gas_intersected, eval_point_gas);
          }
          break;

        default:
          break;
      }
  }

  template <int dim, typename number, bool is_viscous_gas, bool is_viscous_liquid>
  void
  CompressibleMultiphaseOperator<dim, number, is_viscous_gas, is_viscous_liquid>::
    local_apply_face_lhs(const dealii::MatrixFree<dim, number> &,
                         VectorType                                  &dst,
                         const VectorType                            &src,
                         const std::pair<unsigned int, unsigned int> &face_range) const
  {
    const auto face_category =
      multiphase_scratch_data.scratch_data.get_face_range_category(face_range);
    const CutUtil::FaceType face_type = CutUtil::get_face_type(face_category);

    const number cell_side_length       = multiphase_scratch_data.scratch_data.get_min_cell_size();
    const number cell_side_length_pow_3 = Utilities::fixed_power<3>(cell_side_length);
    const number cell_side_length_pow_5 = (multiphase_scratch_data.flow_data.fe.degree == 2) ?
                                            Utilities::fixed_power<5>(cell_side_length) :
                                            0.;

    auto apply_ghost_penalty = [&](auto &eval_m, auto &eval_p) {
      EvaluationFlags::EvaluationFlags evaluation_flags =
        dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients |
        ((multiphase_scratch_data.flow_data.fe.degree == 2) ? dealii::EvaluationFlags::hessians :
                                                              dealii::EvaluationFlags::nothing);

      for (unsigned int face = face_range.first; face < face_range.second; ++face)
        {
          eval_m.reinit(face);
          eval_m.gather_evaluate(src, evaluation_flags);

          eval_p.reinit(face);
          eval_p.gather_evaluate(src, evaluation_flags);

          for (const unsigned int q : eval_m.quadrature_point_indices())
            {
              const auto w_minus = eval_m.get_value(q);
              const auto w_plus  = eval_p.get_value(q);

              const auto w_normal_grad_minus = eval_m.get_normal_derivative(q);
              const auto w_normal_grad_plus  = eval_p.get_normal_derivative(q);

              const auto ghost_penalty_term_0 =
                (w_minus - w_plus) *
                multiphase_scratch_data.cut.stabilization.ghost_penalty.gamma_M_degree_0 *
                cell_side_length;

              const auto ghost_penalty_term_1 =
                (w_normal_grad_minus - w_normal_grad_plus) *
                multiphase_scratch_data.cut.stabilization.ghost_penalty.gamma_M_degree_1 *
                cell_side_length_pow_3;

              if (multiphase_scratch_data.flow_data.fe.degree == 2)
                {
                  const auto w_normal_hessian_minus = eval_m.get_normal_hessian(q);
                  const auto w_normal_hessian_plus  = eval_p.get_normal_hessian(q);

                  const auto ghost_penalty_term_2 =
                    (w_normal_hessian_minus - w_normal_hessian_plus) *
                    multiphase_scratch_data.cut.stabilization.ghost_penalty.gamma_M_degree_2 *
                    cell_side_length_pow_5;

                  eval_m.submit_normal_hessian(ghost_penalty_term_2, q);
                  eval_p.submit_normal_hessian(-ghost_penalty_term_2, q);
                }
              eval_m.submit_normal_derivative(ghost_penalty_term_1, q);
              eval_p.submit_normal_derivative(-ghost_penalty_term_1, q);

              eval_m.submit_value(ghost_penalty_term_0, q);
              eval_p.submit_value(-ghost_penalty_term_0, q);
            }
          eval_m.integrate_scatter(evaluation_flags, dst);
          eval_p.integrate_scatter(evaluation_flags, dst);
        }
    };

    switch (face_type)
      {
          case CutUtil::FaceType::mixed_face_liquid_intersected: {
            auto eval_liquid_m = create_face_integrator(true, CutUtil::CellCategory::liquid, 0);
            auto eval_liquid_p_intersected =
              create_face_integrator(false, CutUtil::CellCategory::intersected, 0);
            apply_ghost_penalty(eval_liquid_m, eval_liquid_p_intersected);
          }
          break;

          case CutUtil::FaceType::mixed_face_intersected_liquid: {
            auto eval_liquid_p = create_face_integrator(false, CutUtil::CellCategory::liquid, 0);
            auto eval_liquid_m_intersected =
              create_face_integrator(true, CutUtil::CellCategory::intersected, 0);
            apply_ghost_penalty(eval_liquid_m_intersected, eval_liquid_p);
          }
          break;

          case CutUtil::FaceType::mixed_face_gas_intersected: {
            auto eval_gas_m = create_face_integrator(true, CutUtil::CellCategory::gas, dim + 2);
            auto eval_gas_p_intersected =
              create_face_integrator(false, CutUtil::CellCategory::intersected, dim + 2);
            apply_ghost_penalty(eval_gas_m, eval_gas_p_intersected);
          }
          break;

          case CutUtil::FaceType::mixed_face_intersected_gas: {
            auto eval_gas_m_intersected =
              create_face_integrator(true, CutUtil::CellCategory::intersected, dim + 2);
            auto eval_gas_p = create_face_integrator(false, CutUtil::CellCategory::gas, dim + 2);
            apply_ghost_penalty(eval_gas_m_intersected, eval_gas_p);
          }
          break;

          case CutUtil::FaceType::intersected_face: {
            auto [eval_liquid_m_intersected, eval_liquid_p_intersected] =
              create_face_integrators(CutUtil::CellCategory::intersected, 0);
            auto [eval_gas_m_intersected, eval_gas_p_intersected] =
              create_face_integrators(CutUtil::CellCategory::intersected, dim + 2);
            apply_ghost_penalty(eval_liquid_m_intersected, eval_liquid_p_intersected);
            apply_ghost_penalty(eval_gas_m_intersected, eval_gas_p_intersected);
          }
          break;

        default:
          break;
      }
  }

  template <int dim, typename number, bool is_viscous_gas, bool is_viscous_liquid>
  void
  CompressibleMultiphaseOperator<dim, number, is_viscous_gas, is_viscous_liquid>::
    local_apply_boundary_face_lhs(const dealii::MatrixFree<dim, number> &,
                                  VectorType &,
                                  const VectorType &,
                                  const std::pair<unsigned, unsigned> &) const
  {
    // nothing to do here
  }

  template class CompressibleMultiphaseOperator<1, double, true, true>;
  template class CompressibleMultiphaseOperator<2, double, true, true>;
  template class CompressibleMultiphaseOperator<3, double, true, true>;
  template class CompressibleMultiphaseOperator<1, double, true, false>;
  template class CompressibleMultiphaseOperator<2, double, true, false>;
  template class CompressibleMultiphaseOperator<3, double, true, false>;
  template class CompressibleMultiphaseOperator<1, double, false, true>;
  template class CompressibleMultiphaseOperator<2, double, false, true>;
  template class CompressibleMultiphaseOperator<3, double, false, true>;
  template class CompressibleMultiphaseOperator<1, double, false, false>;
  template class CompressibleMultiphaseOperator<2, double, false, false>;
  template class CompressibleMultiphaseOperator<3, double, false, false>;
} // namespace MeltPoolDG::Multiphase
