
#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/full_matrix.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <meltpooldg/compressible_flow/explicit_time_integration_utils.hpp>
#include <meltpooldg/compressible_flow/multiphase_interface_kernels.hpp>
#include <meltpooldg/compressible_flow/multiphase_operator.hpp>
#include <meltpooldg/compressible_flow/operation_scratch_data.hpp>
#include <meltpooldg/compressible_flow/utils.hpp>
#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>
#include <meltpooldg/utilities/preprocessor_directives.hpp>

#include <array>
#include <bitset>
#include <cmath>
#include <functional>
#include <optional>
#include <string>
#include <vector>


namespace MeltPoolDG::Multiphase
{
  using namespace dealii;

  namespace
  {
    /**
     * Invert the dense square matrix @p matrix with Gauss-Jordan elimination and partial pivoting.
     *
     * @return false if the matrix is (numerically) singular. In that case, @p inverse is undefined.
     */
    template <typename number>
    bool
    invert_dense_block(const FullMatrix<number> &matrix, FullMatrix<number> &inverse)
    {
      const unsigned int n = matrix.m();
      AssertDimension(n, matrix.n());

      FullMatrix<number> A(matrix);
      inverse.reinit(n, n);
      for (unsigned int i = 0; i < n; ++i)
        inverse(i, i) = 1.;

      number max_abs_diagonal = 0.;
      for (unsigned int i = 0; i < n; ++i)
        max_abs_diagonal = std::max(max_abs_diagonal, std::abs(A(i, i)));

      if (!(max_abs_diagonal > 0.) || !std::isfinite(max_abs_diagonal))
        return false;

      const number tolerance = 1e-14 * max_abs_diagonal;

      for (unsigned int k = 0; k < n; ++k)
        {
          // partial pivoting
          unsigned int pivot = k;
          for (unsigned int i = k + 1; i < n; ++i)
            if (std::abs(A(i, k)) > std::abs(A(pivot, k)))
              pivot = i;

          if (!(std::abs(A(pivot, k)) > tolerance))
            return false;

          if (pivot != k)
            {
              A.swap_row(k, pivot);
              inverse.swap_row(k, pivot);
            }

          const number inverse_pivot = 1. / A(k, k);
          for (unsigned int j = 0; j < n; ++j)
            {
              A(k, j) *= inverse_pivot;
              inverse(k, j) *= inverse_pivot;
            }

          for (unsigned int i = 0; i < n; ++i)
            if (i != k && A(i, k) != number(0.))
              {
                const number factor = A(i, k);
                for (unsigned int j = 0; j < n; ++j)
                  {
                    A(i, j) -= factor * A(k, j);
                    inverse(i, j) -= factor * inverse(k, j);
                  }
              }
        }

      return true;
    }

    /**
     * Return whether the cell category @p category (active FE index) contains DoFs of the phase
     * @p phase (0: liquid, 1: gas).
     */
    inline bool
    category_has_phase(const unsigned int category, const unsigned int phase)
    {
      return category == CutUtil::CellCategory::intersected ||
             category == (phase == 0 ? CutUtil::CellCategory::liquid : CutUtil::CellCategory::gas);
    }

    /**
     * Call @p face_worker(face_no, subface_no) for every (sub)face of the active cell @p cell at
     * which the ghost-penalty stabilization of phase @p phase acts, i.e., faces to a neighbor
     * containing the phase where at least one of the two cells is intersected (same face selection
     * as CutUtil::face_type_has_ghost_penalty()). For faces to a neighbor on the same level or a
     * coarser neighbor, subface_no is dealii::numbers::invalid_unsigned_int. Boundary faces (except
     * for periodic ones) are skipped.
     */
    template <int dim, typename CellIteratorType, typename FaceWorker>
    void
    for_each_ghost_penalty_face(const CellIteratorType &cell,
                                const unsigned int      phase,
                                const FaceWorker       &face_worker)
    {
      const unsigned int category = cell->active_fe_index();

      const auto is_ghost_penalty_face = [&](const unsigned int neighbor_category) {
        return category_has_phase(neighbor_category, phase) &&
               (category == CutUtil::CellCategory::intersected ||
                neighbor_category == CutUtil::CellCategory::intersected);
      };

      for (const unsigned int f : cell->face_indices())
        {
          const bool is_periodic = cell->has_periodic_neighbor(f);
          if (cell->at_boundary(f) && !is_periodic)
            continue;

          const auto neighbor = cell->neighbor_or_periodic_neighbor(f);

          if (!neighbor->has_children())
            {
              // neighbor on the same level or coarser
              if (is_ghost_penalty_face(neighbor->active_fe_index()))
                face_worker(f, dealii::numbers::invalid_unsigned_int);
            }
          else if constexpr (dim == 1)
            {
              // a face is a point: only the category of the adjacent active child matters
              auto child = neighbor;
              while (child->has_children())
                child = child->child(1 - f);

              if (is_ghost_penalty_face(child->active_fe_index()))
                face_worker(f, dealii::numbers::invalid_unsigned_int);
            }
          else
            {
              // finer neighbor: visit the subfaces adjacent to ghost-penalty neighbors
              const unsigned int n_subfaces = is_periodic ?
                                                GeometryInfo<dim>::max_children_per_face :
                                                cell->face(f)->n_children();
              for (unsigned int subface = 0; subface < n_subfaces; ++subface)
                {
                  const auto child = is_periodic ?
                                       cell->periodic_neighbor_child_on_subface(f, subface) :
                                       cell->neighbor_child_on_subface(f, subface);

                  if (is_ghost_penalty_face(child->active_fe_index()))
                    face_worker(f, subface);
                }
            }
        }
    }
  } // namespace

  template <int dim, typename number, bool is_viscous_gas, bool is_viscous_liquid>
  CompressibleMultiphaseOperator<dim, number, is_viscous_gas, is_viscous_liquid>::
    CompressibleMultiphaseOperator(
      MeltPoolDG::CompressibleFlow::MultiphaseOperationScratchData<dim, number>
                                  &multiphase_scratch_data,
      const MappingInfoType       &mapping_info_interface_in,
      const MappingInfoVectorType &mapping_info_cells_in,
      const MappingInfoVectorType &mapping_info_faces_in)
    : multiphase_scratch_data(multiphase_scratch_data)
    , mapping_info_interface(mapping_info_interface_in)
    , mapping_info_cells(mapping_info_cells_in)
    , mapping_info_faces(mapping_info_faces_in)
    , fe_point_temp(FE_DGQ<dim>(multiphase_scratch_data.flow_data.fe.degree),
                    CompressibleFlow::n_conserved_variables<dim>)
    , n_dofs_per_cell(fe_point_temp.dofs_per_cell)
    , darcy_damping_model(multiphase_scratch_data.darcy_damping)
  {
    const auto &l     = multiphase_scratch_data.material_liquid;
    const auto &g     = multiphase_scratch_data.material_gas;
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
  CompressibleMultiphaseOperator<dim, number, is_viscous_gas, is_viscous_liquid>::
    add_external_force(
      std::shared_ptr<CompressibleFlow::ExternalFlowForce<dim, number>> external_force)
  {
    Assert(external_force != nullptr, dealii::ExcInternalError());
    external_forces.push_back(external_force);
  }

  template <int dim, typename number, bool is_viscous_gas, bool is_viscous_liquid>
  template <typename EvaluatorType>
  inline void
  CompressibleMultiphaseOperator<dim, number, is_viscous_gas, is_viscous_liquid>::
    ghost_penalty_face_integral(EvaluatorType     &eval_m,
                                EvaluatorType     &eval_p,
                                const unsigned int q,
                                const number       cell_side_length,
                                const number       cell_side_length_pow_3,
                                const number       cell_side_length_pow_5) const
  {
    const auto w_minus = eval_m.get_value(q);
    const auto w_plus  = eval_p.get_value(q);

    const auto w_normal_grad_minus = eval_m.get_normal_derivative(q);
    const auto w_normal_grad_plus  = eval_p.get_normal_derivative(q);

    const auto ghost_penalty_term_0 =
      (w_minus - w_plus) *
      multiphase_scratch_data.cut.stabilization.ghost_penalty.gamma_M_degree_0 * cell_side_length;

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

  template <int dim, typename number, bool is_viscous_gas, bool is_viscous_liquid>
  void
  CompressibleMultiphaseOperator<dim, number, is_viscous_gas, is_viscous_liquid>::create_rhs(
    const number     &time,
    const number     &time_step_in,
    VectorType       &dst,
    const VectorType &src) const
  {
    Assert(time_step_in > 0., dealii::ExcMessage("Time step size must be larger than 0!"));
    inv_time_step = 1. / time_step_in;
    time_step     = time_step_in;

    using local_applier_type =
      std::function<void(const MatrixFree<dim, number> &,
                         LinearAlgebra::distributed::Vector<number> &,
                         const LinearAlgebra::distributed::Vector<number> &,
                         const std::pair<unsigned int, unsigned int> &)>;

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

    // lambda function for cell integral
    auto process_cell = [&]<bool is_gas_phase, bool is_viscous, typename IntegratorType>(
                          IntegratorType &eval, const auto &material, const auto &cell) {
      for (const unsigned int q : eval.quadrature_point_indices())
        {
          FlowSourceType flux;
          FlowFluxType   grad_flux;

          if (is_viscous)
            grad_flux = ConvectionDiffusionOperator::cell(eval.get_value(q),
                                                          eval.get_gradient(q),
                                                          ConvectiveKernel(material),
                                                          DiffusiveKernel(material));
          else
            grad_flux = ConvectionOperator::cell(eval.get_value(q), ConvectiveKernel(material));

          // consider mass term
          flux = eval.get_value(q) * inv_time_step;

          ConservedVariablesType darcy_damping{};

          if (!is_gas_phase and multiphase_scratch_data.phase_change.solid_liquid.use_darcy_damping)
            {
              const auto w = eval.get_value(q);

              DofStateView liquid_state(w, multiphase_scratch_data.material_liquid);

              const auto velocity    = liquid_state.velocity();
              const auto temperature = liquid_state.temperature();

              const VectorizedArray<number> T_liquidus_vec(
                multiphase_scratch_data.phase_change.solid_liquid.liquidus_temperature);
              const VectorizedArray<number> T_solidus_vec(
                multiphase_scratch_data.phase_change.solid_liquid.solidus_temperature);

              VectorizedArray<number> solid_fraction =
                (T_liquidus_vec - temperature) / (T_liquidus_vec - T_solidus_vec);

              // solid fraction is bounded [0;1]
              solid_fraction =
                std::min(std::max(solid_fraction,
                                  dealii::make_vectorized_array<VectorizedArray<number>>(0.)),
                         dealii::make_vectorized_array<VectorizedArray<number>>(1.));

              const VectorizedArray<number> darcy_damping_coefficient =
                darcy_damping_model.compute_darcy_damping_coefficient(solid_fraction);

              using Idx = CompressibleFlow::ConservedVariableIndex<dim>;

              // contribution to momentum equation
              for (unsigned int i = 0; i < dim; ++i)
                darcy_damping[Idx::momentum + i] = darcy_damping_coefficient * velocity[i];

              // contribution to energy equation
              darcy_damping[Idx::energy] = 0.0;
              for (unsigned int i = 0; i < dim; ++i)
                darcy_damping[Idx::energy] += darcy_damping[Idx::momentum + i] * velocity[i];

              flux += darcy_damping;
            }

          for (auto &external_force : external_forces)
            flux +=
              external_force->value(time_step, cell, eval.quadrature_point(q), eval.get_value(q));

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
                process_cell.template operator()<false, is_viscous_liquid>(
                  eval_liquid, multiphase_scratch_data.material_liquid, cell);
                eval_liquid.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients,
                                              dst);
              }
            break;
          }
          case CutUtil::CellCategory::gas: {
            auto eval_gas = create_cell_integrator(CutUtil::CellCategory::gas,
                                                   CompressibleFlow::n_conserved_variables<dim>);
            for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
              {
                eval_gas.reinit(cell);
                eval_gas.gather_evaluate(src,
                                         EvaluationFlags::values |
                                           (is_viscous_gas ? EvaluationFlags::gradients :
                                                             EvaluationFlags::nothing));
                process_cell.template operator()<true, is_viscous_gas>(
                  eval_gas, multiphase_scratch_data.material_gas, cell);
                eval_gas.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients,
                                           dst);
              }
            break;
          }
          case CutUtil::CellCategory::intersected: {
            constexpr unsigned int n_lanes = VectorizedArray<number>::size();

            EvaluationFlags::EvaluationFlags evaluation_flags_domain_eval =
              EvaluationFlags::values |
              (is_viscous_liquid ? EvaluationFlags::gradients : EvaluationFlags::nothing);
            EvaluationFlags::EvaluationFlags evaluation_flags_interface_int =
              EvaluationFlags::values | EvaluationFlags::gradients;

            auto eval_liquid_intersected =
              create_cell_integrator(CutUtil::CellCategory::intersected, 0);
            auto eval_gas_intersected =
              create_cell_integrator(CutUtil::CellCategory::intersected,
                                     CompressibleFlow::n_conserved_variables<dim>);

            PointDomainEval eval_point_liquid(*mapping_info_cells[0], fe_point_temp);
            PointDomainEval eval_point_interface_liquid(mapping_info_interface, fe_point_temp);
            PointDomainEval eval_point_gas(*mapping_info_cells[1], fe_point_temp);
            PointDomainEval eval_point_interface_gas(mapping_info_interface, fe_point_temp);

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
                    CutUtil::evaluate_intersected_domain(eval_point_liquid,
                                                         eval_liquid_intersected,
                                                         evaluation_flags_domain_eval,
                                                         cell_batch,
                                                         lane,
                                                         n_dofs_per_cell);

                    // evaluate for interface integral in liquid phase
                    CutUtil::evaluate_intersected_domain(eval_point_interface_liquid,
                                                         eval_liquid_intersected,
                                                         evaluation_flags_domain_eval,
                                                         cell_batch,
                                                         lane,
                                                         n_dofs_per_cell);

                    // evaluate for domain integral in gas phase
                    CutUtil::evaluate_intersected_domain(eval_point_gas,
                                                         eval_gas_intersected,
                                                         evaluation_flags_domain_eval,
                                                         cell_batch,
                                                         lane,
                                                         n_dofs_per_cell);

                    // evaluate for interface integral in gas phase
                    CutUtil::evaluate_intersected_domain(eval_point_interface_gas,
                                                         eval_gas_intersected,
                                                         evaluation_flags_domain_eval,
                                                         cell_batch,
                                                         lane,
                                                         n_dofs_per_cell);

                    // do domain integral in liquid phase
                    process_cell.template operator()<false, is_viscous_liquid>(
                      eval_point_liquid, multiphase_scratch_data.material_liquid, cell_batch);

                    eval_point_liquid.integrate(
                      StridedArrayView<number, n_lanes>(
                        &eval_liquid_intersected.begin_dof_values()[0][lane], n_dofs_per_cell),
                      evaluation_flags_interface_int);

                    // do domain integral in gas phase
                    process_cell.template operator()<true, is_viscous_gas>(
                      eval_point_gas, multiphase_scratch_data.material_gas, cell_batch);

                    eval_point_gas.integrate(StridedArrayView<number, n_lanes>(
                                               &eval_gas_intersected.begin_dof_values()[0][lane],
                                               n_dofs_per_cell),
                                             evaluation_flags_interface_int);

                    // do interface integral
                    if (multiphase_scratch_data.phase_coupling.type ==
                        InterfaceNumericalMethod::penalty)
                      {
                        // enumeration for conserved variables component indices
                        using Idx = std::conditional_t<
                          dim == 1,
                          CompressibleFlow::Idx1D,
                          std::conditional_t<
                            dim == 2,
                            CompressibleFlow::Idx2D,
                            std::conditional_t<dim == 3, CompressibleFlow::Idx3D, void>>>;

                        for (const unsigned int q :
                             eval_point_interface_liquid.quadrature_point_indices())
                          {
                            auto w_liquid      = eval_point_interface_liquid.get_value(q);
                            auto w_gas         = eval_point_interface_gas.get_value(q);
                            auto grad_w_liquid = eval_point_interface_liquid.get_gradient(q);
                            auto grad_w_gas    = eval_point_interface_gas.get_gradient(q);
                            // Outwards pointing normal vector with respect to the liquid domain
                            const auto normal = -eval_point_interface_liquid.normal_vector(q);

                            DofStateView liquid_state(w_liquid,
                                                      multiphase_scratch_data.material_liquid);
                            DofStateView gas_state(w_gas, multiphase_scratch_data.material_gas);

                            DofValueAndGradientStateView liquid_value_and_gradient_state(
                              w_liquid, grad_w_liquid, multiphase_scratch_data.material_liquid);
                            DofValueAndGradientStateView gas_value_and_gradient_state(
                              w_gas, grad_w_gas, multiphase_scratch_data.material_gas);

                            const auto [m_dot_evap, delta_T] =
                              update_evaporative_mass_flux_and_temperature_jump<dim,
                                                                                number,
                                                                                DofStateView>(
                                liquid_state,
                                gas_state,
                                normal,
                                multiphase_scratch_data,
                                evaporation_model_knight.get());

                            const auto [flux_liquid, flux_gas] =
                              calculate_convective_and_viscous_interface_flux_penalty<
                                dim,
                                number,
                                ConservedVariablesType>(liquid_value_and_gradient_state,
                                                        gas_value_and_gradient_state,
                                                        multiphase_scratch_data,
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

                            DofStateView liquid_state(w_liquid,
                                                      multiphase_scratch_data.material_liquid);
                            DofStateView gas_state(w_gas, multiphase_scratch_data.material_gas);

                            DofValueAndGradientStateView liquid_value_and_gradient_state(
                              w_liquid, grad_w_liquid, multiphase_scratch_data.material_liquid);
                            DofValueAndGradientStateView gas_value_and_gradient_state(
                              w_gas, grad_w_gas, multiphase_scratch_data.material_gas);

                            const auto [m_dot_evap, delta_T] =
                              update_evaporative_mass_flux_and_temperature_jump<dim,
                                                                                number,
                                                                                DofStateView>(
                                liquid_state,
                                gas_state,
                                normal,
                                multiphase_scratch_data,
                                evaporation_model_knight.get());

                            const auto [riemann_flux_liquid,
                                        riemann_flux_gas,
                                        velocity_interface_vec] =
                              calculate_convective_interface_flux_HLLP0<dim,
                                                                        number,
                                                                        ConservedVariablesType,
                                                                        ConservedVariablesGradType,
                                                                        DofStateView>(
                                liquid_state,
                                gas_state,
                                normal,
                                ConvectiveKernel(multiphase_scratch_data.material_liquid),
                                ConvectiveKernel(multiphase_scratch_data.material_gas),
                                m_dot_evap);

                            ConservedVariablesType flux_liquid = contract_tensor_with_vector<
                              CompressibleFlow::n_conserved_variables<dim>,
                              dim,
                              number>(riemann_flux_liquid, normal);
                            ConservedVariablesType flux_gas = contract_tensor_with_vector<
                              CompressibleFlow::n_conserved_variables<dim>,
                              dim,
                              number>(riemann_flux_gas, -normal);
                            // TODO: consider more complex data structure for velocity for dim>1
                            // TODO: project interface velocity in normal direction for dim>1
                            // returns velocity with respect to the outward liquid phase pointing
                            // normal!
                            level_set_advection_operator.set_interface_velocity(
                              velocity_interface_vec[q]);

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
                                                                       ConservedVariablesGradType,
                                                                       DofValueAndGradientStateView,
                                                                       DofValueView,
                                                                       DofPrimitiveValueView,
                                                                       DofPrimitiveStateView>(
                                        liquid_value_and_gradient_state,
                                        gas_value_and_gradient_state,
                                        normal,
                                        visc_ave_weight_phase_liquid,
                                        visc_ave_weight_phase_gas,
                                        multiphase_scratch_data.phase_coupling.hllp0_and_sipg
                                          .interior_penalty_parameter_interface,
                                        DiffusiveKernel(multiphase_scratch_data.material_liquid),
                                        DiffusiveKernel(multiphase_scratch_data.material_gas),
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
                                        ConservedVariablesGradType,
                                        DofValueAndGradientStateView>(
                                        liquid_value_and_gradient_state,
                                        gas_value_and_gradient_state,
                                        normal,
                                        visc_ave_weight_phase_liquid,
                                        visc_ave_weight_phase_gas,
                                        DiffusiveKernel(multiphase_scratch_data.material_liquid),
                                        DiffusiveKernel(multiphase_scratch_data.material_gas),
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
                                    ConservedVariablesGradType,
                                    DofStateView,
                                    DofValueView,
                                    DofPrimitiveValueView,
                                    DofPrimitiveStateView>(
                                    liquid_state,
                                    gas_state,
                                    normal,
                                    visc_ave_weight_phase_liquid,
                                    visc_ave_weight_phase_gas,
                                    DiffusiveKernel(multiphase_scratch_data.material_liquid),
                                    DiffusiveKernel(multiphase_scratch_data.material_gas),
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
                                                        auto &material) {
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
              0.5 * material.dynamic_viscosity / material.reference_density *
                std::max(eval_m.read_cell_data(multiphase_scratch_data.interior_penalty_parameter),
                         eval_p.read_cell_data(
                           multiphase_scratch_data.interior_penalty_parameter)) :
              0.;

          for (const unsigned int q : eval_m.quadrature_point_indices())
            {
              CompressibleFlow::FaceFluxType<dim, number> flux_m;
              CompressibleFlow::FaceFluxType<dim, number> flux_p;

              if (is_viscous)
                {
                  const auto flux = ConvectionDiffusionOperator::face(eval_m.get_value(q),
                                                                      eval_p.get_value(q),
                                                                      eval_m.get_gradient(q),
                                                                      eval_p.get_gradient(q),
                                                                      eval_m.normal_vector(q),
                                                                      interior_penalty_parameter,
                                                                      ConvectiveKernel(material),
                                                                      DiffusiveKernel(material));

                  flux_m = flux.inner_face_value;
                  flux_p = flux.outer_face_value;

                  eval_m.submit_gradient(flux.inner_face_gradient, q);
                  eval_p.submit_gradient(flux.outer_face_gradient, q);
                }
              else
                {
                  const auto flux = ConvectionOperator::face(eval_m.get_value(q),
                                                             eval_p.get_value(q),
                                                             eval_m.normal_vector(q),
                                                             ConvectiveKernel(material));

                  flux_m = flux.inner_face_value;
                  flux_p = flux.outer_face_value;
                }

              eval_m.submit_value(flux_m, q);
              eval_p.submit_value(flux_p, q);
            }
          eval_m.integrate_scatter(eval_flags, dst);
          eval_p.integrate_scatter(eval_flags, dst);
        }
    };

    auto process_intersected_face_range = [&]<bool is_viscous>(auto              &eval_m_int,
                                                               auto              &eval_p_int,
                                                               const unsigned int mapping_idx,
                                                               auto              &material) {
      PointFaceEval eval_point_m(*mapping_info_faces[mapping_idx], fe_point_temp);
      PointFaceEval eval_point_p(*mapping_info_faces[mapping_idx], fe_point_temp);

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
                is_viscous ? 0.5 * material.dynamic_viscosity / material.reference_density *
                               std::max(eval_m_int.read_cell_data(
                                          multiphase_scratch_data.interior_penalty_parameter),
                                        eval_p_int.read_cell_data(
                                          multiphase_scratch_data.interior_penalty_parameter)) :
                             0.;

              for (const unsigned int q : eval_point_m.quadrature_point_indices())
                {
                  const auto w_m      = eval_point_m.get_value(q);
                  const auto w_p      = eval_point_p.get_value(q);
                  const auto grad_w_m = eval_point_m.get_gradient(q);
                  const auto grad_w_p = eval_point_p.get_gradient(q);
                  const auto normal   = eval_point_m.normal_vector(q);

                  CompressibleFlow::FaceFluxType<dim, number> flux_m;
                  CompressibleFlow::FaceFluxType<dim, number> flux_p;

                  if (is_viscous)
                    {
                      const auto flux =
                        ConvectionDiffusionOperator::face(w_m,
                                                          w_p,
                                                          grad_w_m,
                                                          grad_w_p,
                                                          normal,
                                                          interior_penalty_parameter,
                                                          ConvectiveKernel(material),
                                                          DiffusiveKernel(material));

                      flux_m = flux.inner_face_value;
                      flux_p = flux.outer_face_value;

                      eval_point_m.submit_gradient(flux.inner_face_gradient, q);
                      eval_point_p.submit_gradient(flux.outer_face_gradient, q);
                    }
                  else
                    {
                      const auto flux =
                        ConvectionOperator::face(w_m, w_p, normal, ConvectiveKernel(material));

                      flux_m = flux.inner_face_value;
                      flux_p = flux.outer_face_value;
                    }

                  eval_point_m.submit_value(flux_m, q);
                  eval_point_p.submit_value(flux_p, q);
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
            process_bulk_face_range.template operator()<is_viscous_liquid>(
              eval_liquid_m, eval_liquid_p, multiphase_scratch_data.material_liquid);
          }
          break;

          case CutUtil::FaceType::mixed_face_liquid_intersected: {
            auto eval_liquid_p_intersected =
              create_face_integrator(false, CutUtil::CellCategory::intersected, 0);
            auto eval_liquid_m = create_face_integrator(true, CutUtil::CellCategory::liquid, 0);
            process_bulk_face_range.template operator()<is_viscous_liquid>(
              eval_liquid_m, eval_liquid_p_intersected, multiphase_scratch_data.material_liquid);
          }
          break;

          case CutUtil::FaceType::mixed_face_intersected_liquid: {
            auto eval_liquid_m_intersected =
              create_face_integrator(true, CutUtil::CellCategory::intersected, 0);
            auto eval_liquid_p = create_face_integrator(false, CutUtil::CellCategory::liquid, 0);
            process_bulk_face_range.template operator()<is_viscous_liquid>(
              eval_liquid_m_intersected, eval_liquid_p, multiphase_scratch_data.material_liquid);
          }
          break;

          case CutUtil::FaceType::inside_face_gas: {
            auto [eval_gas_m, eval_gas_p] =
              create_face_integrators(CutUtil::CellCategory::gas,
                                      CompressibleFlow::n_conserved_variables<dim>);
            process_bulk_face_range.template operator()<is_viscous_gas>(
              eval_gas_m, eval_gas_p, multiphase_scratch_data.material_gas);
          }
          break;

          case CutUtil::FaceType::mixed_face_gas_intersected: {
            auto eval_gas_m = create_face_integrator(true,
                                                     CutUtil::CellCategory::gas,
                                                     CompressibleFlow::n_conserved_variables<dim>);
            auto eval_gas_p_intersected =
              create_face_integrator(false,
                                     CutUtil::CellCategory::intersected,
                                     CompressibleFlow::n_conserved_variables<dim>);
            process_bulk_face_range.template operator()<is_viscous_gas>(
              eval_gas_m, eval_gas_p_intersected, multiphase_scratch_data.material_gas);
          }
          break;

          case CutUtil::FaceType::mixed_face_intersected_gas: {
            auto eval_gas_p = create_face_integrator(false,
                                                     CutUtil::CellCategory::gas,
                                                     CompressibleFlow::n_conserved_variables<dim>);
            auto eval_gas_m_intersected =
              create_face_integrator(true,
                                     CutUtil::CellCategory::intersected,
                                     CompressibleFlow::n_conserved_variables<dim>);
            process_bulk_face_range.template operator()<is_viscous_gas>(
              eval_gas_m_intersected, eval_gas_p, multiphase_scratch_data.material_gas);
          }
          break;

          case CutUtil::FaceType::intersected_face: {
            auto [eval_liquid_m_intersected, eval_liquid_p_intersected] =
              create_face_integrators(CutUtil::CellCategory::intersected, 0);
            auto [eval_gas_m_intersected, eval_gas_p_intersected] =
              create_face_integrators(CutUtil::CellCategory::intersected,
                                      CompressibleFlow::n_conserved_variables<dim>);
            process_intersected_face_range.template operator()<is_viscous_liquid>(
              eval_liquid_m_intersected,
              eval_liquid_p_intersected,
              0,
              multiphase_scratch_data.material_liquid);
            process_intersected_face_range.template operator()<is_viscous_gas>(
              eval_gas_m_intersected,
              eval_gas_p_intersected,
              1,
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
    local_apply_boundary_face_rhs(const dealii::MatrixFree<dim, number> &,
                                  VectorType                          &dst,
                                  const VectorType                    &src,
                                  const std::pair<unsigned, unsigned> &face_range) const
  {
    const auto face_category =
      multiphase_scratch_data.scratch_data.get_face_range_category(face_range);

    auto process_bulk_face_range = [&]<bool is_viscous, bool is_gas_phase>(auto &eval_m,
                                                                           auto &material) {
      for (unsigned int face = face_range.first; face < face_range.second; ++face)
        {
          eval_m.reinit(face);
          eval_m.gather_evaluate(src,
                                 dealii::EvaluationFlags::values |
                                   dealii::EvaluationFlags::gradients);

          const dealii::VectorizedArray<number> interior_penalty_parameter =
            is_viscous ?
              material.dynamic_viscosity / material.reference_density *
                eval_m.read_cell_data(multiphase_scratch_data.interior_penalty_parameter) :
              0.;

          for (const unsigned int q : eval_m.quadrature_point_indices())
            {
              const auto w_m      = eval_m.get_value(q);
              const auto grad_w_m = eval_m.get_gradient(q);

              const auto [w_p, grad_w_p] =
                multiphase_scratch_data.boundary_conditions.get_boundary_face_value_and_gradient(
                  eval_m.quadrature_point(q),
                  eval_m.normal_vector(q),
                  eval_m.boundary_id(),
                  w_m,
                  grad_w_m,
                  material,
                  is_gas_phase);

              CompressibleFlow::FaceFluxType<dim, number> flux_m;
              if (is_viscous)
                {
                  const auto flux = ConvectionDiffusionOperator::face(eval_m.get_value(q),
                                                                      w_p,
                                                                      eval_m.get_gradient(q),
                                                                      grad_w_p,
                                                                      eval_m.normal_vector(q),
                                                                      interior_penalty_parameter,
                                                                      ConvectiveKernel(material),
                                                                      DiffusiveKernel(material));

                  flux_m = flux.inner_face_value;

                  eval_m.submit_gradient(flux.inner_face_gradient, q);
                }
              else
                {
                  const auto flux = ConvectionOperator::face(eval_m.get_value(q),
                                                             w_p,
                                                             eval_m.normal_vector(q),
                                                             ConvectiveKernel(material));

                  flux_m = flux.inner_face_value;
                }

              eval_m.submit_value(flux_m, q);
            }

          eval_m.integrate_scatter(EvaluationFlags::values |
                                     (is_viscous ? EvaluationFlags::gradients :
                                                   EvaluationFlags::nothing),
                                   dst);
        }
    };

    auto process_intersected_face_range =
      [&]<bool is_viscous, bool is_gas_phase>(auto              &eval_m_int,
                                              const unsigned int mapping_idx,
                                              auto              &material) {
        PointFaceEval eval_point_m(*mapping_info_faces[mapping_idx], fe_point_temp);

        for (unsigned int face = face_range.first; face < face_range.second; ++face)
          {
            eval_m_int.reinit(face);
            eval_m_int.read_dof_values(src);

            eval_m_int.project_to_face(EvaluationFlags::values | EvaluationFlags::gradients);

            const auto face_info =
              multiphase_scratch_data.scratch_data.get_matrix_free().get_face_info(face);

            for (unsigned int lane = 0;
                 lane < multiphase_scratch_data.scratch_data.get_matrix_free()
                          .n_active_entries_per_face_batch(face);
                 ++lane)
              {
                eval_point_m.reinit(face_info.cells_interior[lane],
                                    static_cast<int>(face_info.interior_face_no));

                eval_point_m.evaluate_in_face(&eval_m_int.get_scratch_data().begin()[0][lane],
                                              EvaluationFlags::values | EvaluationFlags::gradients);

                const dealii::VectorizedArray<number> interior_penalty_parameter =
                  is_viscous ?
                    eval_m_int.read_cell_data(multiphase_scratch_data.interior_penalty_parameter) :
                    0.;

                for (const unsigned int q : eval_point_m.quadrature_point_indices())
                  {
                    const auto w_m      = eval_point_m.get_value(q);
                    const auto grad_w_m = eval_point_m.get_gradient(q);
                    const auto normal   = eval_point_m.normal_vector(q);

                    const auto [w_p, grad_w_p] =
                      multiphase_scratch_data.boundary_conditions
                        .get_boundary_face_value_and_gradient(eval_point_m.quadrature_point(q),
                                                              normal,
                                                              eval_m_int.boundary_id(),
                                                              w_m,
                                                              grad_w_m,
                                                              material,
                                                              is_gas_phase);

                    CompressibleFlow::FaceFluxType<dim, number> flux_m;
                    if (is_viscous)
                      {
                        const auto flux =
                          ConvectionDiffusionOperator::face(w_m,
                                                            w_p,
                                                            grad_w_m,
                                                            grad_w_p,
                                                            normal,
                                                            interior_penalty_parameter,
                                                            ConvectiveKernel(material),
                                                            DiffusiveKernel(material));

                        flux_m = flux.inner_face_value;

                        eval_point_m.submit_gradient(flux.inner_face_gradient, q);
                      }
                    else
                      {
                        const auto flux =
                          ConvectionOperator::face(w_m, w_p, normal, ConvectiveKernel(material));

                        flux_m = flux.inner_face_value;
                      }

                    eval_point_m.submit_value(flux_m, q);
                  }

                eval_point_m.integrate_in_face(&eval_m_int.get_scratch_data().begin()[0][lane],
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
              eval_liquid_m, multiphase_scratch_data.material_liquid);
            break;
          }
          case CutUtil::CellCategory::gas: {
            auto                             eval_gas_m = create_face_integrator(true,
                                                     CutUtil::CellCategory::gas,
                                                     CompressibleFlow::n_conserved_variables<dim>);
            process_bulk_face_range.template operator()<is_viscous_gas, true /*is_gas_phase*/>(
              eval_gas_m, multiphase_scratch_data.material_gas);
            break;
          }
          case CutUtil::CellCategory::intersected: {
            auto eval_liquid_m_intersected =
              create_face_integrator(true, CutUtil::CellCategory::intersected, 0);
            auto eval_gas_m_intersected =
              create_face_integrator(true,
                                     CutUtil::CellCategory::intersected,
                                     CompressibleFlow::n_conserved_variables<dim>);
            process_intersected_face_range
              .template operator()<is_viscous_liquid, false /*is_gas_phase*/>(
                eval_liquid_m_intersected, 0, multiphase_scratch_data.material_liquid);
            process_intersected_face_range
              .template operator()<is_viscous_gas, true /*is_gas_phase*/>(
                eval_gas_m_intersected, 1, multiphase_scratch_data.material_gas);
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
              CutUtil::evaluate_intersected_domain(
                eval_point, eval_intersected, EvaluationFlags::values, cell, lane, n_dofs_per_cell);

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
            auto eval_gas = create_cell_integrator(CutUtil::CellCategory::gas,
                                                   CompressibleFlow::n_conserved_variables<dim>);
            process_bulk_cell_range(eval_gas);
          }
          break;

          case CutUtil::CellCategory::intersected: {
            auto eval_liquid_intersected =
              create_cell_integrator(CutUtil::CellCategory::intersected, 0);
            auto eval_gas_intersected =
              create_cell_integrator(CutUtil::CellCategory::intersected,
                                     CompressibleFlow::n_conserved_variables<dim>);

            PointDomainEval eval_point_liquid(*mapping_info_cells[0], fe_point_temp);
            PointDomainEval eval_point_gas(*mapping_info_cells[1], fe_point_temp);

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

    // TODO: use local face size and not globally minimum cell size for ghost penalty scaling
    const number cell_side_length       = multiphase_scratch_data.scratch_data.get_min_cell_size();
    const number cell_side_length_pow_3 = dealii::Utilities::fixed_power<3>(cell_side_length);
    const number cell_side_length_pow_5 = (multiphase_scratch_data.flow_data.fe.degree == 2) ?
                                            dealii::Utilities::fixed_power<5>(cell_side_length) :
                                            0.;

    auto apply_ghost_penalty = [&](const CutUtil::CellCategory category_m,
                                   const CutUtil::CellCategory category_p,
                                   const unsigned int          component) {
      EvaluationFlags::EvaluationFlags evaluation_flags =
        dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients |
        ((multiphase_scratch_data.flow_data.fe.degree == 2) ? dealii::EvaluationFlags::hessians :
                                                              dealii::EvaluationFlags::nothing);

      auto eval_m = create_face_integrator(true, category_m, component);
      auto eval_p = create_face_integrator(false, category_p, component);

      for (unsigned int face = face_range.first; face < face_range.second; ++face)
        {
          eval_m.reinit(face);
          eval_m.gather_evaluate(src, evaluation_flags);

          eval_p.reinit(face);
          eval_p.gather_evaluate(src, evaluation_flags);

          for (const unsigned int q : eval_m.quadrature_point_indices())
            ghost_penalty_face_integral(
              eval_m, eval_p, q, cell_side_length, cell_side_length_pow_3, cell_side_length_pow_5);

          eval_m.integrate_scatter(evaluation_flags, dst);
          eval_p.integrate_scatter(evaluation_flags, dst);
        }
    };

    switch (face_type)
      {
        case CutUtil::FaceType::mixed_face_liquid_intersected:
          apply_ghost_penalty(CutUtil::CellCategory::liquid, CutUtil::CellCategory::intersected, 0);
          break;

        case CutUtil::FaceType::mixed_face_intersected_liquid:
          apply_ghost_penalty(CutUtil::CellCategory::intersected, CutUtil::CellCategory::liquid, 0);
          break;

        case CutUtil::FaceType::mixed_face_gas_intersected:
          apply_ghost_penalty(CutUtil::CellCategory::gas,
                              CutUtil::CellCategory::intersected,
                              CompressibleFlow::n_conserved_variables<dim>);
          break;

        case CutUtil::FaceType::mixed_face_intersected_gas:
          apply_ghost_penalty(CutUtil::CellCategory::intersected,
                              CutUtil::CellCategory::gas,
                              CompressibleFlow::n_conserved_variables<dim>);
          break;

        case CutUtil::FaceType::intersected_face:
          apply_ghost_penalty(CutUtil::CellCategory::intersected,
                              CutUtil::CellCategory::intersected,
                              0);

          apply_ghost_penalty(CutUtil::CellCategory::intersected,
                              CutUtil::CellCategory::intersected,
                              CompressibleFlow::n_conserved_variables<dim>);
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

  template <int dim, typename number, bool is_viscous_gas, bool is_viscous_liquid>
  void
  CompressibleMultiphaseOperator<dim, number, is_viscous_gas, is_viscous_liquid>::
    compute_inverse_diagonal_from_matrixfree(VectorType &diagonal) const
  {
    multiphase_scratch_data.scratch_data.initialize_dof_vector(diagonal,
                                                               multiphase_scratch_data.dof_idx);

    dealii::TrilinosWrappers::SparseMatrix dummy;
    internal_compute_diagonal_or_system_matrix(diagonal, dummy, true);

    // invert
    const double linfty_norm = std::max(1.0, diagonal.linfty_norm());
    for (auto &i : diagonal)
      i = std::abs(i) > 1.0e-16 * linfty_norm ? 1.0 / i : 1.0;
  }

  template <int dim, typename number, bool is_viscous_gas, bool is_viscous_liquid>
  void
  CompressibleMultiphaseOperator<dim, number, is_viscous_gas, is_viscous_liquid>::
    compute_system_matrix_from_matrixfree(
      dealii::TrilinosWrappers::SparseMatrix &system_matrix) const
  {
    system_matrix = 0.0;

    VectorType dummy;
    internal_compute_diagonal_or_system_matrix(dummy, system_matrix, false);
  }

  template <int dim, typename number, bool is_viscous_gas, bool is_viscous_liquid>
  void
  CompressibleMultiphaseOperator<dim, number, is_viscous_gas, is_viscous_liquid>::
    internal_compute_diagonal_or_system_matrix(
      VectorType                             &diagonal,
      dealii::TrilinosWrappers::SparseMatrix &system_matrix,
      const bool                              do_diagonal) const
  {
    const auto &matrix_free = multiphase_scratch_data.scratch_data.get_matrix_free();

    enum class Phase
    {
      liquid,
      gas
    };

    const EvaluationFlags::EvaluationFlags face_evaluation_flags =
      EvaluationFlags::values | EvaluationFlags::gradients |
      (multiphase_scratch_data.flow_data.fe.degree == 2 ? EvaluationFlags::hessians :
                                                          EvaluationFlags::nothing);

    // TODO: use local face size and not globally minimum cell size for ghost penalty scaling
    const number cell_side_length       = multiphase_scratch_data.scratch_data.get_min_cell_size();
    const number cell_side_length_pow_3 = dealii::Utilities::fixed_power<3>(cell_side_length);
    const number cell_side_length_pow_5 = (multiphase_scratch_data.flow_data.fe.degree == 2) ?
                                            dealii::Utilities::fixed_power<5>(cell_side_length) :
                                            0.;

    // assemble each phase separately (0: liquid, 1: gas)
    for (const unsigned int phase : {0, 1})
      {
        const bool         is_liquid       = phase == 0;
        const unsigned int first_component = phase * CompressibleFlow::n_conserved_variables<dim>;
        const unsigned int bulk_category =
          is_liquid ? CutUtil::CellCategory::liquid : CutUtil::CellCategory::gas;

        // cells: (cut) mass matrix
        dealii::MatrixFreeTools::internal::
          ComputeMatrixScratchData<dim, dealii::VectorizedArray<number>, false /*is_face_*/>
            data_cell;

        data_cell.dof_numbers               = {multiphase_scratch_data.dof_idx};
        data_cell.quad_numbers              = {multiphase_scratch_data.quad_idx};
        data_cell.n_components              = {CompressibleFlow::n_conserved_variables<dim>};
        data_cell.first_selected_components = {first_component};
        data_cell.batch_type                = {0}; // 0 for cell

        data_cell.op_create = [&](const std::pair<unsigned int, unsigned int> &cell_range) {
          std::vector<
            std::unique_ptr<dealii::FEEvaluationData<dim, dealii::VectorizedArray<number>, false>>>
            eval_data;

          const auto cell_category = matrix_free.get_cell_range_category(cell_range);

          if (cell_category == bulk_category or cell_category == CutUtil::CellCategory::intersected)
            eval_data.emplace_back(std::make_unique<DomainEval<>>(matrix_free,
                                                                  cell_range,
                                                                  multiphase_scratch_data.dof_idx,
                                                                  multiphase_scratch_data.quad_idx,
                                                                  first_component));

          return eval_data;
        };

        data_cell.op_reinit = [](auto &evaluators, const unsigned int cell_batch) {
          for (auto &eval : evaluators)
            static_cast<DomainEval<> &>(*eval).reinit(cell_batch);
        };

        PointDomainEval<> eval_point(*mapping_info_cells[phase], fe_point_temp);

        data_cell.op_compute = [&](auto &evaluators) {
          auto &eval = static_cast<DomainEval<> &>(*evaluators[0]);

          if (eval.get_active_fe_index() == CutUtil::CellCategory::intersected)
            {
              const unsigned int cell_batch = eval.get_current_cell_index();

              for (unsigned int lane = 0;
                   lane < matrix_free.n_active_entries_per_cell_batch(cell_batch);
                   ++lane)
                {
                  CutUtil::evaluate_intersected_domain(
                    eval_point, eval, EvaluationFlags::values, cell_batch, lane, n_dofs_per_cell);

                  for (const unsigned int q : eval_point.quadrature_point_indices())
                    eval_point.submit_value(eval_point.get_value(q), q);

                  eval_point.integrate(StridedArrayView<number, VectorizedArray<number>::size()>(
                                         &eval.begin_dof_values()[0][lane], n_dofs_per_cell),
                                       dealii::EvaluationFlags::values);
                }
            }
          else // bulk cell of the current phase
            {
              eval.evaluate(EvaluationFlags::values);

              for (const unsigned int q : eval.quadrature_point_indices())
                eval.submit_value(eval.get_value(q), q);

              eval.integrate(dealii::EvaluationFlags::values);
            }
        };

        // interior faces: ghost penalty
        dealii::MatrixFreeTools::internal::
          ComputeMatrixScratchData<dim, dealii::VectorizedArray<number>, true /*is_face_*/>
            data_face;

        data_face.dof_numbers  = {multiphase_scratch_data.dof_idx, multiphase_scratch_data.dof_idx};
        data_face.quad_numbers = {multiphase_scratch_data.quad_idx,
                                  multiphase_scratch_data.quad_idx};
        data_face.n_components = {CompressibleFlow::n_conserved_variables<dim>,
                                  CompressibleFlow::n_conserved_variables<dim>};
        data_face.first_selected_components = {first_component, first_component};
        data_face.batch_type                = {1, 2}; // 1 for interior face, 2 for exterior face

        data_face.op_create = [&](const std::pair<unsigned int, unsigned int> &face_range) {
          std::vector<
            std::unique_ptr<dealii::FEEvaluationData<dim, dealii::VectorizedArray<number>, true>>>
            eval_data;

          const auto              face_category = matrix_free.get_face_range_category(face_range);
          const CutUtil::FaceType face_type     = CutUtil::get_face_type(face_category);

          if (face_type_has_ghost_penalty(face_type, is_liquid))
            {
              eval_data.emplace_back(std::make_unique<FaceEval<>>(matrix_free,
                                                                  face_range,
                                                                  true /*interior*/,
                                                                  multiphase_scratch_data.dof_idx,
                                                                  multiphase_scratch_data.quad_idx,
                                                                  first_component));
              eval_data.emplace_back(std::make_unique<FaceEval<>>(matrix_free,
                                                                  face_range,
                                                                  false /*exterior*/,
                                                                  multiphase_scratch_data.dof_idx,
                                                                  multiphase_scratch_data.quad_idx,
                                                                  first_component));
            }

          return eval_data;
        };

        data_face.op_reinit = [](auto &evaluators, const unsigned int face_batch) {
          for (auto &eval : evaluators)
            static_cast<FaceEval<> &>(*eval).reinit(face_batch);
        };

        data_face.op_compute = [&](auto &evaluators) {
          auto &eval_minus = static_cast<FaceEval<> &>(*evaluators[0]);
          auto &eval_plus  = static_cast<FaceEval<> &>(*evaluators[1]);

          eval_minus.evaluate(face_evaluation_flags);
          eval_plus.evaluate(face_evaluation_flags);

          for (const unsigned int q : eval_minus.quadrature_point_indices())
            ghost_penalty_face_integral(eval_minus,
                                        eval_plus,
                                        q,
                                        cell_side_length,
                                        cell_side_length_pow_3,
                                        cell_side_length_pow_5);

          eval_minus.integrate(face_evaluation_flags);
          eval_plus.integrate(face_evaluation_flags);
        };

        if (do_diagonal)
          {
            std::vector<VectorType *> diagonal_components(1, &diagonal);
            dealii::MatrixFreeTools::internal::
              compute_diagonal<dim, number, dealii::VectorizedArray<number>>(matrix_free,
                                                                             data_cell,
                                                                             data_face,
                                                                             {} /*data_boundary*/,
                                                                             diagonal,
                                                                             diagonal_components);
          }
        else // compute matrix
          {
            dealii::MatrixFreeTools::internal::compute_matrix<dim,
                                                              number,
                                                              dealii::VectorizedArray<number>>(
              matrix_free,
              multiphase_scratch_data.scratch_data.get_constraint(multiphase_scratch_data.dof_idx),
              data_cell,
              data_face,
              {} /*data_boundary*/,
              system_matrix);
          }
      }
  }

  template <int dim, typename number, bool is_viscous_gas, bool is_viscous_liquid>
  void
  CompressibleMultiphaseOperator<dim, number, is_viscous_gas, is_viscous_liquid>::
    compute_inverse_block_diagonal_from_matrixfree(BlockJacobiData &block_inverses) const
  {
    const auto        &matrix_free = multiphase_scratch_data.scratch_data.get_matrix_free();
    const unsigned int dof_idx     = multiphase_scratch_data.dof_idx;
    const unsigned int quad_idx    = multiphase_scratch_data.quad_idx;
    const unsigned int fe_degree   = multiphase_scratch_data.flow_data.fe.degree;

    constexpr unsigned int n_lanes       = VectorizedArray<number>::size();
    constexpr unsigned int n_components  = CompressibleFlow::n_conserved_variables<dim>;
    const unsigned int     n_scalar_dofs = dealii::Utilities::pow(fe_degree + 1, dim);

    // The inverse mass matrix of the bulk cells is applied via sum factorization in
    // apply_inverse_block_diagonal(), which is exact only for as many quadrature points as DoFs.
    AssertThrow(matrix_free.get_quadrature(quad_idx, CutUtil::CellCategory::liquid).size() ==
                  n_scalar_dofs,
                ExcMessage("The block-Jacobi preconditioner requires a cell quadrature rule with "
                           "fe_degree + 1 points per coordinate direction."));

    // one block per cell and phase (0: liquid, 1: gas)
    block_inverses.reinit(matrix_free.n_cell_batches(), n_lanes, 2, n_scalar_dofs);

    // ghost-penalty parameters, identical to local_apply_face_lhs()
    // TODO: use local face size and not globally minimum cell size for ghost penalty scaling
    const number h                 = multiphase_scratch_data.scratch_data.get_min_cell_size();
    const number h_pow3            = dealii::Utilities::fixed_power<3>(h);
    const number h_pow5            = dealii::Utilities::fixed_power<5>(h);
    const auto  &ghost_penalty     = multiphase_scratch_data.cut.stabilization.ghost_penalty;
    const bool   with_hessian_term = (fe_degree == 2);

    // All conserved variables share the same scalar block. Thus, scalar evaluators are sufficient.
    const FE_DGQ<dim> fe_scalar(fe_degree);

    // (a) face integrals: ghost-penalty contributions of the cell to itself
    UpdateFlags face_update_flags =
      update_values | update_gradients | update_JxW_values | update_normal_vectors;
    if (with_hessian_term)
      face_update_flags = face_update_flags | update_hessians;

    const Quadrature<dim - 1> &face_quadrature =
      matrix_free.get_face_quadrature(quad_idx, CutUtil::CellCategory::liquid);
    const Mapping<dim> &mapping = multiphase_scratch_data.scratch_data.get_mapping();

    FEFaceValues<dim> fe_face_values(mapping, fe_scalar, face_quadrature, face_update_flags);

    // subfaces (hanging faces) do not exist in 1d
    std::optional<FESubfaceValues<dim>> fe_subface_values;
    if constexpr (dim > 1)
      fe_subface_values.emplace(mapping, fe_scalar, face_quadrature, face_update_flags);

    std::vector<number> phi(n_scalar_dofs), normal_grad_phi(n_scalar_dofs),
      normal_hessian_phi(n_scalar_dofs, 0.);

    // Add the self-coupling part of the ghost-penalty bilinear form on one face (see
    // ghost_penalty_face_integral()). Test and trial functions both belong to the current cell,
    // so the neighbor's shape functions and the orientation of the normal vector do not matter.
    const auto add_ghost_penalty_self_block = [&](const FEFaceValuesBase<dim> &fe_values,
                                                  FullMatrix<number>          &block) {
      for (const unsigned int q : fe_values.quadrature_point_indices())
        {
          const Tensor<1, dim> normal = fe_values.normal_vector(q);

          for (unsigned int i = 0; i < n_scalar_dofs; ++i)
            {
              phi[i]             = fe_values.shape_value(i, q);
              normal_grad_phi[i] = fe_values.shape_grad(i, q) * normal;
              if (with_hessian_term)
                normal_hessian_phi[i] = normal * (fe_values.shape_hessian(i, q) * normal);
            }

          const number JxW = fe_values.JxW(q);
          const number c_0 = ghost_penalty.gamma_M_degree_0 * h * JxW;
          const number c_1 = ghost_penalty.gamma_M_degree_1 * h_pow3 * JxW;
          const number c_2 = with_hessian_term ? ghost_penalty.gamma_M_degree_2 * h_pow5 * JxW : 0.;

          for (unsigned int i = 0; i < n_scalar_dofs; ++i)
            for (unsigned int j = 0; j < n_scalar_dofs; ++j)
              block(i, j) += c_0 * phi[i] * phi[j] + c_1 * normal_grad_phi[i] * normal_grad_phi[j] +
                             c_2 * normal_hessian_phi[i] * normal_hessian_phi[j];
        }
    };

    // Add the ghost-penalty contributions of all faces of @p cell for @p phase to @p block (see
    // for_each_ghost_penalty_face()). The block is set to zero before the first contribution is
    // added and is not touched if the cell has no ghost-penalty face. Return whether the cell has a
    // ghost-penalty face.
    const auto add_ghost_penalty_faces = [&](const typename DoFHandler<dim>::cell_iterator &cell,
                                             const unsigned int                             phase,
                                             FullMatrix<number> &block) -> bool {
      // FEValues objects are set up for a scalar element; reinitialize them with a triangulation
      // iterator to bypass the check against the finite element of the DoFHandler
      const typename Triangulation<dim>::cell_iterator tria_cell(cell);

      bool has_ghost_penalty_face = false;

      for_each_ghost_penalty_face<dim>(
        cell, phase, [&](const unsigned int face_no, const unsigned int subface_no) {
          if (!has_ghost_penalty_face)
            block.reinit(n_scalar_dofs, n_scalar_dofs);
          has_ghost_penalty_face = true;

          if (subface_no == dealii::numbers::invalid_unsigned_int)
            {
              fe_face_values.reinit(tria_cell, face_no);
              add_ghost_penalty_self_block(fe_face_values, block);
            }
          else
            {
              // subfaces (hanging faces) do not exist in 1d
              if constexpr (dim > 1)
                {
                  fe_subface_values->reinit(tria_cell, face_no, subface_no);
                  add_ghost_penalty_self_block(*fe_subface_values, block);
                }
              else
                Assert(false, ExcInternalError());
            }
        });

      return has_ghost_penalty_face;
    };

    // (b) cell integrals: mass matrix of bulk cells (vectorized over the cells of a batch) and cut
    // mass matrix of intersected cells (one cell at a time)
    std::array<FECellIntegrator<dim, 1, number>, 2> eval_bulk = {
      {FECellIntegrator<dim, 1, number>(
         matrix_free, dof_idx, quad_idx, 0, CutUtil::CellCategory::liquid),
       FECellIntegrator<dim, 1, number>(
         matrix_free, dof_idx, quad_idx, n_components, CutUtil::CellCategory::gas)}};

    std::array<PointDomainEval<1>, 2> eval_point_intersected = {
      {PointDomainEval<1>(*mapping_info_cells[0], fe_scalar),
       PointDomainEval<1>(*mapping_info_cells[1], fe_scalar)}};

    AlignedVector<VectorizedArray<number>>  bulk_mass_matrix(n_scalar_dofs * n_scalar_dofs);
    std::vector<number>                     unit_vector(n_scalar_dofs, 0.);
    std::vector<number>                     column(n_scalar_dofs, 0.);
    std::array<FullMatrix<number>, n_lanes> blocks;
    std::array<bool, n_lanes>               needs_dense_block;
    FullMatrix<number>                      inverse_block(n_scalar_dofs, n_scalar_dofs);
    unsigned int                            n_singular_blocks = 0;

    // invert the block and store it
    const auto store_inverse_block = [&](const unsigned int  cell_batch,
                                         const unsigned int  lane,
                                         const unsigned int  phase,
                                         FullMatrix<number> &block) {
      if (!invert_dense_block(block, inverse_block))
        {
          // Fall back to the inverse diagonal. This is only expected for a singular system matrix,
          // e.g., if a cell has a vanishing cut and no ghost-penalty stabilization.
          ++n_singular_blocks;

          number max_abs_diagonal = 0.;
          for (unsigned int i = 0; i < n_scalar_dofs; ++i)
            max_abs_diagonal = std::max(max_abs_diagonal, std::abs(block(i, i)));

          inverse_block = 0.;
          for (unsigned int i = 0; i < n_scalar_dofs; ++i)
            inverse_block(i, i) = std::abs(block(i, i)) > 1e-14 * max_abs_diagonal ?
                                    1. / block(i, i) :
                                    (max_abs_diagonal > 0. ? 1. / max_abs_diagonal : 1.);
        }

      number *data = block_inverses.add_block(cell_batch, lane, phase);
      for (unsigned int i = 0; i < n_scalar_dofs; ++i)
        for (unsigned int j = 0; j < n_scalar_dofs; ++j)
          data[i * n_scalar_dofs + j] = inverse_block(i, j);
    };

    for (unsigned int cell_batch = 0; cell_batch < matrix_free.n_cell_batches(); ++cell_batch)
      {
        const unsigned int category       = matrix_free.get_cell_category(cell_batch);
        const unsigned int n_active_lanes = matrix_free.n_active_entries_per_cell_batch(cell_batch);

        for (const unsigned int phase : {0u, 1u})
          {
            if (!category_has_phase(category, phase))
              continue;

            // 1) ghost-penalty contributions; find the cells that need a dense block
            bool any_dense_block = false;
            for (unsigned int lane = 0; lane < n_active_lanes; ++lane)
              {
                const bool has_ghost_penalty_face =
                  add_ghost_penalty_faces(matrix_free.get_cell_iterator(cell_batch, lane, dof_idx),
                                          phase,
                                          blocks[lane]);

                needs_dense_block[lane] =
                  (category == CutUtil::CellCategory::intersected) || has_ghost_penalty_face;
                any_dense_block = any_dense_block || needs_dense_block[lane];

                // intersected cell without ghost-penalty face: start from a zero block
                if (needs_dense_block[lane] && !has_ghost_penalty_face)
                  blocks[lane].reinit(n_scalar_dofs, n_scalar_dofs);
              }

            // pure bulk cells: the inverse mass matrix is applied via sum factorization
            if (!any_dense_block)
              continue;

            // 2) add the (cut) mass matrix, invert and store the blocks
            if (category == CutUtil::CellCategory::intersected)
              {
                auto &eval_point = eval_point_intersected[phase];

                for (unsigned int lane = 0; lane < n_active_lanes; ++lane)
                  {
                    eval_point.reinit(cell_batch * n_lanes + lane);

                    for (unsigned int j = 0; j < n_scalar_dofs; ++j)
                      {
                        std::fill(unit_vector.begin(), unit_vector.end(), number(0.));
                        unit_vector[j] = 1.;

                        eval_point.evaluate(ArrayView<const number>(unit_vector.data(),
                                                                    unit_vector.size()),
                                            EvaluationFlags::values);
                        for (const unsigned int q : eval_point.quadrature_point_indices())
                          eval_point.submit_value(eval_point.get_value(q), q);
                        eval_point.integrate(ArrayView<number>(column.data(), column.size()),
                                             EvaluationFlags::values);

                        for (unsigned int i = 0; i < n_scalar_dofs; ++i)
                          blocks[lane](i, j) += column[i];
                      }

                    store_inverse_block(cell_batch, lane, phase, blocks[lane]);
                  }
              }
            else
              {
                auto &eval = eval_bulk[phase];
                eval.reinit(cell_batch);

                for (unsigned int j = 0; j < n_scalar_dofs; ++j)
                  {
                    for (unsigned int i = 0; i < n_scalar_dofs; ++i)
                      eval.begin_dof_values()[i] = (i == j) ? number(1.) : number(0.);

                    eval.evaluate(EvaluationFlags::values);
                    for (const unsigned int q : eval.quadrature_point_indices())
                      eval.submit_value(eval.get_value(q), q);
                    eval.integrate(EvaluationFlags::values);

                    for (unsigned int i = 0; i < n_scalar_dofs; ++i)
                      bulk_mass_matrix[j * n_scalar_dofs + i] = eval.begin_dof_values()[i];
                  }

                for (unsigned int lane = 0; lane < n_active_lanes; ++lane)
                  if (needs_dense_block[lane])
                    {
                      for (unsigned int i = 0; i < n_scalar_dofs; ++i)
                        for (unsigned int j = 0; j < n_scalar_dofs; ++j)
                          blocks[lane](i, j) += bulk_mass_matrix[j * n_scalar_dofs + i][lane];

                      store_inverse_block(cell_batch, lane, phase, blocks[lane]);
                    }
              }
          }
      }

    n_singular_blocks =
      dealii::Utilities::MPI::sum(n_singular_blocks,
                                  multiphase_scratch_data.scratch_data.get_mpi_comm());
    if (n_singular_blocks > 0)
      Journal::print_line(multiphase_scratch_data.scratch_data.get_pcout(1),
                          "WARNING: " + std::to_string(n_singular_blocks) +
                            " singular cell block(s) replaced by their inverse diagonal",
                          "block-Jacobi preconditioner");
  }

  template <int dim, typename number, bool is_viscous_gas, bool is_viscous_liquid>
  void
  CompressibleMultiphaseOperator<dim, number, is_viscous_gas, is_viscous_liquid>::
    apply_inverse_block_diagonal(const BlockJacobiData &block_inverses,
                                 VectorType            &dst,
                                 const VectorType      &src) const
  {
    const auto &matrix_free = multiphase_scratch_data.scratch_data.get_matrix_free();

    Assert(block_inverses.n_cell_batches() == matrix_free.n_cell_batches(),
           ExcMessage("The block-Jacobi data does not match the MatrixFree object. Update the "
                      "preconditioner after the DoF layout has changed."));

    constexpr unsigned int n_components  = CompressibleFlow::n_conserved_variables<dim>;
    const unsigned int     n_scalar_dofs = block_inverses.block_size();
    const unsigned int     n_dofs        = n_components * n_scalar_dofs;

    using InverseMassType =
      MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, n_components, number>;

    // evaluators for all combinations of cell category and phase
    DomainEval<> eval_liquid = create_cell_integrator(CutUtil::CellCategory::liquid, 0);
    DomainEval<> eval_liquid_intersected =
      create_cell_integrator(CutUtil::CellCategory::intersected, 0);
    DomainEval<> eval_gas_intersected =
      create_cell_integrator(CutUtil::CellCategory::intersected, n_components);
    DomainEval<> eval_gas = create_cell_integrator(CutUtil::CellCategory::gas, n_components);

    // exact inverse of the mass matrix of bulk cells via sum factorization (requires
    // n_q_points_1d = fe_degree + 1, which is the case for FE_DGQ)
    const InverseMassType inverse_mass_liquid(eval_liquid);
    const InverseMassType inverse_mass_gas(eval_gas);

    AlignedVector<VectorizedArray<number>> src_values(n_dofs);

    const auto process_cell_batch = [&](DomainEval<>          &eval,
                                        const InverseMassType *inverse_mass,
                                        const unsigned int     cell_batch,
                                        const unsigned int     phase) {
      eval.reinit(cell_batch);
      eval.read_dof_values(src);

      VectorizedArray<number> *values = eval.begin_dof_values();
      for (unsigned int i = 0; i < n_dofs; ++i)
        src_values[i] = values[i];

      // bulk cells: inverse mass matrix (vectorized over all cells of the batch)
      if (inverse_mass != nullptr)
        inverse_mass->apply(src_values.data(), values);

      // cells with a dense block (intersected cells, cells at ghost-penalty faces)
      for (unsigned int lane = 0; lane < matrix_free.n_active_entries_per_cell_batch(cell_batch);
           ++lane)
        {
          const number *block = block_inverses.get_block(cell_batch, lane, phase);

          if (block == nullptr)
            {
              Assert(inverse_mass != nullptr, ExcInternalError());
              continue;
            }

          for (unsigned int c = 0; c < n_components; ++c)
            {
              const VectorizedArray<number> *in  = src_values.data() + c * n_scalar_dofs;
              VectorizedArray<number>       *out = values + c * n_scalar_dofs;

              for (unsigned int i = 0; i < n_scalar_dofs; ++i)
                {
                  const number *block_row = block + i * n_scalar_dofs;
                  number        sum       = 0.;
                  for (unsigned int j = 0; j < n_scalar_dofs; ++j)
                    sum += block_row[j] * in[j][lane];
                  out[i][lane] = sum;
                }
            }
        }

      // DG: every DoF belongs to exactly one cell and phase, so the values can be set directly
      eval.set_dof_values(dst);
    };

    for (unsigned int cell_batch = 0; cell_batch < matrix_free.n_cell_batches(); ++cell_batch)
      switch (matrix_free.get_cell_category(cell_batch))
        {
          case CutUtil::CellCategory::liquid:
            process_cell_batch(eval_liquid, &inverse_mass_liquid, cell_batch, 0);
            break;
          case CutUtil::CellCategory::intersected:
            process_cell_batch(eval_liquid_intersected, nullptr, cell_batch, 0);
            process_cell_batch(eval_gas_intersected, nullptr, cell_batch, 1);
            break;
          case CutUtil::CellCategory::gas:
            process_cell_batch(eval_gas, &inverse_mass_gas, cell_batch, 1);
            break;
          default:
            DEAL_II_NOT_IMPLEMENTED();
        }
  }

  template <int dim, typename number, bool is_viscous_gas, bool is_viscous_liquid>
  void
  CompressibleMultiphaseOperator<dim, number, is_viscous_gas, is_viscous_liquid>::compute_band(
    BandData &band) const
  {
    const auto        &matrix_free = multiphase_scratch_data.scratch_data.get_matrix_free();
    const unsigned int dof_idx     = multiphase_scratch_data.dof_idx;
    const unsigned int quad_idx    = multiphase_scratch_data.quad_idx;
    const unsigned int fe_degree   = multiphase_scratch_data.flow_data.fe.degree;

    constexpr unsigned int n_lanes      = VectorizedArray<number>::size();
    constexpr unsigned int n_components = CompressibleFlow::n_conserved_variables<dim>;
    const unsigned int     n_dofs       = n_components * dealii::Utilities::pow(fe_degree + 1, dim);

    // The inverse mass matrix of the cells outside the band is applied via sum factorization in
    // apply_inverse_mass_matrix_outside_band(), which is exact only for as many quadrature points
    // as DoFs.
    AssertThrow(matrix_free.get_quadrature(quad_idx, CutUtil::CellCategory::liquid).size() ==
                  dealii::Utilities::pow(fe_degree + 1, dim),
                ExcMessage("The band solver requires a cell quadrature rule with fe_degree + 1 "
                           "points per coordinate direction."));

    // 1) band cells: one entry per cell and phase (0: liquid, 1: gas)
    band.reinit(matrix_free.n_cell_batches(), n_lanes, 2);

    for (unsigned int cell_batch = 0; cell_batch < matrix_free.n_cell_batches(); ++cell_batch)
      {
        const unsigned int category = matrix_free.get_cell_category(cell_batch);

        for (unsigned int lane = 0; lane < matrix_free.n_active_entries_per_cell_batch(cell_batch);
             ++lane)
          {
            const auto cell = matrix_free.get_cell_iterator(cell_batch, lane, dof_idx);

            for (const unsigned int phase : {0u, 1u})
              {
                if (!category_has_phase(category, phase))
                  continue;

                // intersected cells always belong to the band, bulk cells only if they share a
                // ghost-penalty face of their phase with an intersected cell
                bool is_band_cell = (category == CutUtil::CellCategory::intersected);
                if (!is_band_cell)
                  for_each_ghost_penalty_face<dim>(cell,
                                                   phase,
                                                   [&](const unsigned int, const unsigned int) {
                                                     is_band_cell = true;
                                                   });

                if (is_band_cell)
                  band.add(cell_batch, lane, phase);
              }
          }
      }

    // 2) DoFs of the band: mark the DoFs of all band cells (and phases) with 1
    VectorType band_marker;
    multiphase_scratch_data.scratch_data.initialize_dof_vector(band_marker,
                                                               multiphase_scratch_data.dof_idx);

    DomainEval<> eval_liquid = create_cell_integrator(CutUtil::CellCategory::liquid, 0);
    DomainEval<> eval_liquid_intersected =
      create_cell_integrator(CutUtil::CellCategory::intersected, 0);
    DomainEval<> eval_gas_intersected =
      create_cell_integrator(CutUtil::CellCategory::intersected, n_components);
    DomainEval<> eval_gas = create_cell_integrator(CutUtil::CellCategory::gas, n_components);

    const auto mark_cell_batch =
      [&](DomainEval<> &eval, const unsigned int cell_batch, const unsigned int phase) {
        std::bitset<n_lanes> band_lanes;
        for (unsigned int lane = 0; lane < matrix_free.n_active_entries_per_cell_batch(cell_batch);
             ++lane)
          band_lanes[lane] = band.contains(cell_batch, lane, phase);

        if (band_lanes.none())
          return;

        eval.reinit(cell_batch);
        for (unsigned int i = 0; i < n_dofs; ++i)
          eval.begin_dof_values()[i] = number(1.);
        eval.set_dof_values_plain(band_marker, 0, band_lanes);
      };

    for (const unsigned int cell_batch : band.get_cell_batches())
      switch (matrix_free.get_cell_category(cell_batch))
        {
          case CutUtil::CellCategory::liquid:
            mark_cell_batch(eval_liquid, cell_batch, 0);
            break;
          case CutUtil::CellCategory::intersected:
            mark_cell_batch(eval_liquid_intersected, cell_batch, 0);
            mark_cell_batch(eval_gas_intersected, cell_batch, 1);
            break;
          case CutUtil::CellCategory::gas:
            mark_cell_batch(eval_gas, cell_batch, 1);
            break;
          default:
            DEAL_II_NOT_IMPLEMENTED();
        }

    // collect the local indices of the marked DoFs
    std::vector<unsigned int> &band_dof_indices = band.get_dof_indices();
    band_dof_indices.clear();
    for (unsigned int i = 0; i < band_marker.locally_owned_size(); ++i)
      if (band_marker.local_element(i) != number(0.))
        band_dof_indices.push_back(i);
  }

  template <int dim, typename number, bool is_viscous_gas, bool is_viscous_liquid>
  void
  CompressibleMultiphaseOperator<dim, number, is_viscous_gas, is_viscous_liquid>::
    apply_inverse_mass_matrix_outside_band(const BandData   &band,
                                           VectorType       &dst,
                                           const VectorType &src) const
  {
    const auto &matrix_free = multiphase_scratch_data.scratch_data.get_matrix_free();

    Assert(band.n_cell_batches() == matrix_free.n_cell_batches(),
           ExcMessage("The band data does not match the MatrixFree object. Update the band "
                      "solver after the DoF layout has changed."));

    constexpr unsigned int n_lanes      = VectorizedArray<number>::size();
    constexpr unsigned int n_components = CompressibleFlow::n_conserved_variables<dim>;

    using InverseMassType =
      MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, n_components, number>;

    DomainEval<> eval_liquid = create_cell_integrator(CutUtil::CellCategory::liquid, 0);
    DomainEval<> eval_gas    = create_cell_integrator(CutUtil::CellCategory::gas, n_components);

    // exact inverse of the cell mass matrix via sum factorization (requires
    // n_q_points_1d = fe_degree + 1, checked in compute_band())
    const InverseMassType inverse_mass_liquid(eval_liquid);
    const InverseMassType inverse_mass_gas(eval_gas);

    const auto process_cell_batch = [&](DomainEval<>          &eval,
                                        const InverseMassType &inverse_mass,
                                        const unsigned int     cell_batch,
                                        const unsigned int     phase) {
      // lanes (cells) outside the band
      std::bitset<n_lanes> lanes_outside_band;
      for (unsigned int lane = 0; lane < matrix_free.n_active_entries_per_cell_batch(cell_batch);
           ++lane)
        lanes_outside_band[lane] = !band.contains(cell_batch, lane, phase);

      if (lanes_outside_band.none())
        return;

      eval.reinit(cell_batch);
      eval.read_dof_values(src);
      inverse_mass.apply(eval.begin_dof_values(), eval.begin_dof_values());

      // DG: every DoF belongs to exactly one cell and phase, so the values can be set directly
      eval.set_dof_values(dst, 0, lanes_outside_band);
    };

    for (unsigned int cell_batch = 0; cell_batch < matrix_free.n_cell_batches(); ++cell_batch)
      switch (matrix_free.get_cell_category(cell_batch))
        {
          case CutUtil::CellCategory::liquid:
            process_cell_batch(eval_liquid, inverse_mass_liquid, cell_batch, 0);
            break;
          case CutUtil::CellCategory::intersected:
            // intersected cells always belong to the band
            break;
          case CutUtil::CellCategory::gas:
            process_cell_batch(eval_gas, inverse_mass_gas, cell_batch, 1);
            break;
          default:
            DEAL_II_NOT_IMPLEMENTED();
        }
  }

  template <int dim, typename number, bool is_viscous_gas, bool is_viscous_liquid>
  void
  CompressibleMultiphaseOperator<dim, number, is_viscous_gas, is_viscous_liquid>::vmult_band(
    const BandData   &band,
    VectorType       &dst,
    const VectorType &src) const
  {
    const auto &matrix_free = multiphase_scratch_data.scratch_data.get_matrix_free();

    Assert(band.n_cell_batches() == matrix_free.n_cell_batches(),
           ExcMessage("The band data does not match the MatrixFree object. Update the band "
                      "solver after the DoF layout has changed."));

    using local_applier_type = std::function<void(const dealii::MatrixFree<dim, number> &,
                                                  VectorType &,
                                                  const VectorType &,
                                                  const std::pair<unsigned int, unsigned int> &)>;

    // cells: only the cell batches of the band are processed; contiguous runs of band batches are
    // passed to local_apply_cell_lhs() at once
    local_applier_type cell = [&](const dealii::MatrixFree<dim, number>       &mf,
                                  VectorType                                  &cell_dst,
                                  const VectorType                            &cell_src,
                                  const std::pair<unsigned int, unsigned int> &cell_range) {
      unsigned int first = cell_range.first;
      while (first < cell_range.second)
        {
          if (!band.contains_cell_batch(first))
            {
              ++first;
              continue;
            }

          unsigned int last = first + 1;
          while (last < cell_range.second && band.contains_cell_batch(last))
            ++last;

          this->local_apply_cell_lhs(mf, cell_dst, cell_src, std::make_pair(first, last));
          first = last;
        }
    };

    // faces: the operator only acts on ghost-penalty faces, which all lie within the band
    local_applier_type face          = MPDG_LAMBDA_WRAPPER(this->local_apply_face_lhs);
    local_applier_type boundary_face = MPDG_LAMBDA_WRAPPER(this->local_apply_boundary_face_lhs);

    // dst is not zeroed: only the band entries are relevant, which are set to zero by the caller
    matrix_free.loop(cell,
                     face,
                     boundary_face,
                     dst,
                     src,
                     false /*zero_dst_vector*/,
                     MatrixFree<dim, number>::DataAccessOnFaces::gradients,
                     MatrixFree<dim, number>::DataAccessOnFaces::gradients);
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
