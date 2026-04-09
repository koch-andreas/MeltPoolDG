#include <meltpooldg/heat/heat_diffuse_operator.hpp>
//

#include <deal.II/base/array_view.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/vector_operation.h>

#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/tools.h>

#include <meltpooldg/core/exceptions.hpp>
#include <meltpooldg/core/material.templates.hpp>
#include <meltpooldg/level_set/level_set_tools.hpp>
#include <meltpooldg/phase_change/evaporative_cooling.templates.hpp>
#include <meltpooldg/utilities/dealii_tensor.hpp>
#include <meltpooldg/utilities/dof_tools.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/physical_constants.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>
#include <meltpooldg/utilities/vector_tools.templates.hpp>

#include <algorithm>
#include <bitset>
#include <cmath>


namespace MeltPoolDG::Heat
{
  using namespace dealii;

  template <int dim, typename number>
  HeatDiffuseMultiPhaseOperator<dim, number>::HeatDiffuseMultiPhaseOperator(
    const ScratchData<dim, dim, number>                               &scratch_data_in,
    const std::shared_ptr<const BoundaryConditionManager<dim, number>> heat_bc_manager,
    const HeatData<number>                                            &data_in,
    const Material<number>                                            &material,
    const unsigned int                                                 heat_dof_idx_in,
    const unsigned int                                                 heat_quad_idx_in,
    const unsigned int                                                 heat_no_bc_dof_idx_in,
    const VectorType                                                  &temperature_in,
    const VectorType                                                  &temperature_old_in,
    const VectorType                                                  &heat_source_in,
    const unsigned int                                                 vel_dof_idx_in,
    const VectorType                                                  *velocity_in,
    const unsigned int                                                 ls_dof_idx_in,
    const VectorType                                                  *level_set_as_heaviside_in,
    const bool                                                         do_solidification_in)
    : scratch_data(scratch_data_in)
    , data(data_in)
    , material(material)
    , heat_dof_idx(heat_dof_idx_in)
    , heat_quad_idx(heat_quad_idx_in)
    , heat_no_bc_dof_idx(heat_no_bc_dof_idx_in)
    , temperature(temperature_in)
    , temperature_old(temperature_old_in)
    , heat_source(heat_source_in)
    , vel_dof_idx(vel_dof_idx_in)
    , velocity(velocity_in)
    , ls_dof_idx(ls_dof_idx_in)
    , level_set_as_heaviside(level_set_as_heaviside_in)
    , do_solidification(do_solidification_in)
    , do_update_ghosts(6)
  {
    material.get_data().check_parameters_heat_transfer(level_set_as_heaviside != nullptr,
                                                       do_solidification);

    if (heat_bc_manager)
      {
        bc_convection_indices = heat_bc_manager->get_indices_of_type("convection");
        bc_radiation_indices  = heat_bc_manager->get_indices_of_type("radiation");
        neumann_bc            = heat_bc_manager->get_bc_of_type("neumann");
      }
  }

  template <int dim, typename number>
  void
  HeatDiffuseMultiPhaseOperator<dim, number>::register_surface_mesh(
    const std::vector<std::tuple<const typename Triangulation<dim, dim>::cell_iterator /*cell*/,
                                 std::vector<Point<dim>> /*quad_points*/,
                                 std::vector<number> /*weights*/
                                 >> &surface_mesh_info_in)
  {
    surface_mesh_info = &surface_mesh_info_in;
  }

  template <int dim, typename number>
  void
  HeatDiffuseMultiPhaseOperator<dim, number>::register_evaporative_mass_flux(
    VectorType                                 *evaporative_mass_flux_in,
    const unsigned int                          mass_flux_dof_idx_in,
    const Evaporation::EvaporationData<number> &evapor_data)
  {
    delta_phase_weighted = create_phase_weighted_delta_approximation(
      evapor_data.evaporative_cooling.delta_approximation_phase_weighted);

    mass_flux_type = evapor_data.evaporative_cooling.model;
    AssertThrow(surface_mesh_info or
                  mass_flux_type != Evaporation::EvaporCoolingInterfaceFluxType::sharp,
                ExcMessage("If you would like to use a sharp flux model, you first"
                           "need to register the surface mesh."));

    if (evaporative_mass_flux_in)
      {
        do_level_set_temperature_gradient_interpolation = scratch_data.is_FE_Q_iso_Q_1(ls_dof_idx);

        if (do_level_set_temperature_gradient_interpolation)
          ls_to_heat_grad_interpolation_matrix =
            DoFTools::create_dof_interpolation_matrix<dim, number>(
              scratch_data.get_dof_handler(heat_dof_idx),
              scratch_data.get_dof_handler(ls_dof_idx),
              true /* do_matrix_free */);

        evaporative_mass_flux = evaporative_mass_flux_in;

        mass_flux_dof_idx = mass_flux_dof_idx_in;
      }

    evaporative_cooling = std::make_unique<Evaporation::EvaporativeCooling<number>>(
      evapor_data,
      material.get_data(),
      (evaporative_mass_flux) ? false : true /*setup_internal_mass_flux_operator*/);

    if (evapor_data.evaporative_cooling.consider_enthalpy_transport_vapor_mass_flux == "true")
      {
        const auto &material_data = material.get_data();
        AssertThrow(!do_solidification or (material_data.solid.specific_heat_capacity ==
                                           material_data.liquid.specific_heat_capacity),
                    ExcMessage("The equation for specific enthalpy for evaporative cooling "
                               "assumes equality between the solid and liquid "
                               "phase heat capacity! Abort..."));
      }
  }

  template <int dim, typename number>
  void
  HeatDiffuseMultiPhaseOperator<dim, number>::pre()
  {
    unsigned int i = 0;

    if ((do_update_ghosts[i++] = not temperature.has_ghost_elements()))
      MeltPoolDG::VectorTools::update_ghost_values(temperature);
    if ((do_update_ghosts[i++] = not heat_source.has_ghost_elements()))
      MeltPoolDG::VectorTools::update_ghost_values(heat_source);
    if (velocity and (do_update_ghosts[i++] = not velocity->has_ghost_elements()))
      MeltPoolDG::VectorTools::update_ghost_values(*velocity);
    if (level_set_as_heaviside and
        (do_update_ghosts[i++] = not level_set_as_heaviside->has_ghost_elements()))
      MeltPoolDG::VectorTools::update_ghost_values(*level_set_as_heaviside);
    if (do_solidification and (do_update_ghosts[i++] = not temperature_old.has_ghost_elements()))
      MeltPoolDG::VectorTools::update_ghost_values(temperature_old);
    if (evaporative_mass_flux and
        (do_update_ghosts[i++] = not evaporative_mass_flux->has_ghost_elements()))
      MeltPoolDG::VectorTools::update_ghost_values(*evaporative_mass_flux);
  }

  template <int dim, typename number>
  void
  HeatDiffuseMultiPhaseOperator<dim, number>::post()
  {
    unsigned int i = 0;

    if (do_update_ghosts[i++])
      MeltPoolDG::VectorTools::zero_out_ghost_values(temperature);
    if (do_update_ghosts[i++])
      MeltPoolDG::VectorTools::zero_out_ghost_values(heat_source);
    if (velocity and do_update_ghosts[i++])
      MeltPoolDG::VectorTools::zero_out_ghost_values(*velocity);
    if (level_set_as_heaviside and do_update_ghosts[i++])
      MeltPoolDG::VectorTools::zero_out_ghost_values(*level_set_as_heaviside);
    if (do_solidification and do_update_ghosts[i++])
      MeltPoolDG::VectorTools::zero_out_ghost_values(temperature_old);
    if (evaporative_mass_flux and do_update_ghosts[i++])
      MeltPoolDG::VectorTools::zero_out_ghost_values(*evaporative_mass_flux);
  }

  template <int dim, typename number>
  void
  HeatDiffuseMultiPhaseOperator<dim, number>::vmult(VectorType &dst, const VectorType &src) const
  {
    scratch_data.get_matrix_free().template loop<VectorType, VectorType>(
      [&](const auto &matrix_free, auto &dst, const auto &src, auto cell_range) {
        tangent_cell_loop(matrix_free, dst, src, cell_range);
      }, // cell loop
      [&]([[maybe_unused]] const auto &matrix_free,
          [[maybe_unused]] auto       &dst,
          [[maybe_unused]] const auto &src,
          [[maybe_unused]] auto        face_range) { /*do nothing*/ }, // internal face loop
      [&](const auto &matrix_free, auto &dst, const auto &src, auto face_range) {
        tangent_boundary_loop(matrix_free, dst, src, face_range); // boundary face loop
      },
      dst,
      src,
      true /*zero dst vector*/);
  }

  template <int dim, typename number>
  void
  HeatDiffuseMultiPhaseOperator<dim, number>::tangent_cell_loop(
    const MatrixFree<dim, number>        &matrix_free,
    VectorType                           &dst,
    const VectorType                     &src,
    std::pair<unsigned int, unsigned int> cell_range) const
  {
    FECellIntegrator<dim, 1, number>   eval(matrix_free, heat_dof_idx, heat_quad_idx);
    FECellIntegrator<dim, dim, number> vel_eval(matrix_free, vel_dof_idx, heat_quad_idx);
    FECellIntegrator<dim, 1, number>   heaviside_eval(matrix_free, ls_dof_idx, heat_quad_idx);
    FECellIntegrator<dim, 1, number>   heaviside_interpolated_eval(matrix_free,
                                                                 heat_dof_idx,
                                                                 heat_quad_idx);
    FECellIntegrator<dim, 1, number>   T_new_eval(matrix_free, heat_dof_idx, heat_quad_idx);
    FECellIntegrator<dim, 1, number>   T_old_eval(matrix_free, heat_no_bc_dof_idx, heat_quad_idx);

    std::unique_ptr<FECellIntegrator<dim, 1, number>> mass_flux_eval;
    if (evaporative_mass_flux and
        mass_flux_type == Evaporation::EvaporCoolingInterfaceFluxType::regularized)
      {
        mass_flux_eval = std::make_unique<FECellIntegrator<dim, 1, number>>(matrix_free,
                                                                            mass_flux_dof_idx,
                                                                            heat_quad_idx);
      }

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval.reinit(cell);
        eval.read_dof_values(src);

        tangent_local_cell_operation(eval,
                                     T_new_eval,
                                     T_old_eval,
                                     vel_eval,
                                     heaviside_eval,
                                     heaviside_interpolated_eval,
                                     mass_flux_eval,
                                     true);

        eval.distribute_local_to_global(dst);
      }
  }

  template <int dim, typename number>
  void
  HeatDiffuseMultiPhaseOperator<dim, number>::tangent_boundary_loop(
    const MatrixFree<dim, number>        &matrix_free,
    VectorType                           &dst,
    const VectorType                     &src,
    std::pair<unsigned int, unsigned int> face_range) const
  {
    FEFaceIntegrator<dim, 1, number> face_eval(matrix_free,
                                               true /*is_interior_face*/,
                                               heat_dof_idx,
                                               heat_quad_idx);
    FEFaceIntegrator<dim, 1, number> T_face_eval(matrix_free,
                                                 true /*is_interior_face*/,
                                                 heat_dof_idx,
                                                 heat_quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        face_eval.reinit(face);
        face_eval.read_dof_values(src);

        tangent_local_boundary_operation(face_eval, T_face_eval, true /*do reinit faces*/);

        face_eval.distribute_local_to_global(dst);
      }
  }

  template <int dim, typename number>
  void
  HeatDiffuseMultiPhaseOperator<dim, number>::compute_inverse_diagonal_from_matrixfree(
    VectorType &diagonal) const
  {
    scratch_data.initialize_dof_vector(diagonal, heat_dof_idx);

    SparseMatrixType dummy;
    internal_compute_diagonal_or_system_matrix(diagonal, dummy, true);

    // ... and invert it
    for (auto &i : diagonal)
      i = std::abs(i) > 1.0e-10 ? 1.0 / i : 1.0;
  }

  template <int dim, typename number>
  void
  HeatDiffuseMultiPhaseOperator<dim, number>::compute_system_matrix_from_matrixfree(
    SparseMatrixType &system_matrix) const
  {
    system_matrix = 0.0;

    VectorType dummy;
    internal_compute_diagonal_or_system_matrix(dummy, system_matrix, false);
  }

  template <int dim, typename number>
  void
  HeatDiffuseMultiPhaseOperator<dim, number>::internal_compute_diagonal_or_system_matrix(
    [[maybe_unused]] VectorType       &diagonal,
    [[maybe_unused]] SparseMatrixType &system_matrix,
    const bool                         do_diagonal) const
  {
    const auto &matrix_free = scratch_data.get_matrix_free();

    FECellIntegrator<dim, dim, number> vel_eval(matrix_free, vel_dof_idx, heat_quad_idx);
    FECellIntegrator<dim, 1, number>   heaviside_eval(matrix_free, ls_dof_idx, heat_quad_idx);
    FECellIntegrator<dim, 1, number>   heaviside_interpolated_eval(matrix_free,
                                                                 heat_dof_idx,
                                                                 heat_quad_idx);
    FECellIntegrator<dim, 1, number>   T_new_eval(matrix_free, heat_dof_idx, heat_quad_idx);
    FECellIntegrator<dim, 1, number>   T_old_eval(matrix_free, heat_no_bc_dof_idx, heat_quad_idx);
    std::unique_ptr<FECellIntegrator<dim, 1, number>> mass_flux_eval;
    if (evaporative_mass_flux and
        mass_flux_type == Evaporation::EvaporCoolingInterfaceFluxType::regularized)
      {
        mass_flux_eval = std::make_unique<FECellIntegrator<dim, 1, number>>(matrix_free,
                                                                            mass_flux_dof_idx,
                                                                            heat_quad_idx);
      }

    FEFaceIntegrator<dim, 1, number> T_face_eval(matrix_free,
                                                 true /*is_interior_face*/,
                                                 heat_dof_idx,
                                                 heat_quad_idx);

    unsigned int old_cell_index = dealii::numbers::invalid_unsigned_int;

    if (do_diagonal)
      {
        // compute diagonal ...
        MatrixFreeTools::template compute_diagonal<dim, -1, 0, 1, number, VectorizedArray<number>>(
          matrix_free,
          diagonal,
          [&](auto &eval) {
            const unsigned int current_cell_index = eval.get_current_cell_index();

            tangent_local_cell_operation(eval,
                                         T_new_eval,
                                         T_old_eval,
                                         vel_eval,
                                         heaviside_eval,
                                         heaviside_interpolated_eval,
                                         mass_flux_eval,
                                         old_cell_index != current_cell_index);

            old_cell_index = current_cell_index;
          },
          // face operation
          {},
          // boundary operation
          [&](auto &face_eval) {
            tangent_local_boundary_operation(face_eval,
                                             T_face_eval,
                                             true /*old_face_index != current_face_index*/);
          },
          heat_dof_idx,
          heat_quad_idx);
      }
    else
      {
        MatrixFreeTools::template compute_matrix<dim, -1, 0, 1, number, VectorizedArray<number>>(
          matrix_free,
          scratch_data.get_constraint(heat_dof_idx),
          system_matrix,
          [&](auto &eval) {
            const unsigned int current_cell_index = eval.get_current_cell_index();

            tangent_local_cell_operation(eval,
                                         T_new_eval,
                                         T_old_eval,
                                         vel_eval,
                                         heaviside_eval,
                                         heaviside_interpolated_eval,
                                         mass_flux_eval,
                                         old_cell_index != current_cell_index);

            old_cell_index = current_cell_index;
          },
          // face operation
          {},
          // boundary operation
          [&](auto &face_eval) {
            tangent_local_boundary_operation(face_eval,
                                             T_face_eval,
                                             true /*old_face_index != current_face_index*/);
          },
          heat_dof_idx,
          heat_quad_idx);
      }
  }

  template <int dim, typename number>
  void
  HeatDiffuseMultiPhaseOperator<dim, number>::rhs_cell_loop(
    const MatrixFree<dim, number>        &matrix_free,
    VectorType                           &dst,
    const VectorType                     &src,
    std::pair<unsigned int, unsigned int> cell_range) const
  {
    FECellIntegrator<dim, 1, number>   T_new_eval(matrix_free, heat_dof_idx, heat_quad_idx);
    FECellIntegrator<dim, 1, number>   T_old_eval(matrix_free, heat_no_bc_dof_idx, heat_quad_idx);
    FECellIntegrator<dim, 1, number>   heat_source_eval(matrix_free,
                                                      heat_no_bc_dof_idx,
                                                      heat_quad_idx);
    FECellIntegrator<dim, dim, number> vel_eval(matrix_free, vel_dof_idx, heat_quad_idx);
    FECellIntegrator<dim, 1, number>   heaviside_eval(matrix_free, ls_dof_idx, heat_quad_idx);
    FECellIntegrator<dim, 1, number>   heaviside_interpolated_eval(matrix_free,
                                                                 heat_dof_idx,
                                                                 heat_quad_idx);

    auto &heaviside_eval_used = do_level_set_temperature_gradient_interpolation ?
                                  heaviside_interpolated_eval :
                                  heaviside_eval;

    std::unique_ptr<FECellIntegrator<dim, 1, number>> mass_flux_eval;

    if (evaporative_mass_flux and
        mass_flux_type == Evaporation::EvaporCoolingInterfaceFluxType::regularized)
      {
        mass_flux_eval = std::make_unique<FECellIntegrator<dim, 1, number>>(matrix_free,
                                                                            mass_flux_dof_idx,
                                                                            heat_quad_idx);
      }

    //@todo: only in case of certain verbosity level or if variable is requested
    q_vapor.resize_fast(scratch_data.get_matrix_free().n_cell_batches() * T_new_eval.n_q_points);
    conductivity_at_q.resize_fast(scratch_data.get_matrix_free().n_cell_batches() *
                                  T_new_eval.n_q_points);
    rho_cp_at_q.resize_fast(scratch_data.get_matrix_free().n_cell_batches() *
                            T_new_eval.n_q_points);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        T_new_eval.reinit(cell);
        T_new_eval.read_dof_values_plain(temperature);
        T_new_eval.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

        T_old_eval.reinit(cell);
        T_old_eval.read_dof_values_plain(src); // = temperature_old
        T_old_eval.evaluate(EvaluationFlags::values);

        heat_source_eval.reinit(cell);
        heat_source_eval.read_dof_values_plain(heat_source);
        heat_source_eval.evaluate(EvaluationFlags::values);

        if (velocity)
          {
            vel_eval.reinit(cell);
            vel_eval.read_dof_values_plain(*velocity);
            vel_eval.evaluate(EvaluationFlags::values);
          }

        if (level_set_as_heaviside)
          {
            heaviside_eval.reinit(cell);
            heaviside_eval.read_dof_values_plain(*level_set_as_heaviside);

            if (evaporative_mass_flux)
              {
                if (do_level_set_temperature_gradient_interpolation)
                  {
                    heaviside_interpolated_eval.reinit(cell);

                    DoFTools::compute_gradient_at_interpolated_dof_values<dim>(
                      heaviside_eval,
                      heaviside_interpolated_eval,
                      ls_to_heat_grad_interpolation_matrix);
                  }

                heaviside_eval_used.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
              }
            else
              heaviside_eval.evaluate(EvaluationFlags::values);
          }

        if (mass_flux_eval)
          {
            mass_flux_eval->reinit(cell);
            mass_flux_eval->read_dof_values_plain(*evaporative_mass_flux);
            mass_flux_eval->evaluate(EvaluationFlags::values);
          }


        for (unsigned int q = 0; q < T_new_eval.n_q_points; ++q)
          {
            const auto [rho_cp, conductivity] =
              get_material_parameters(T_new_eval, heaviside_eval_used, q);

            conductivity_at_q[cell * T_new_eval.n_q_points + q] = conductivity;
            rho_cp_at_q[cell * T_new_eval.n_q_points + q]       = rho_cp;

            auto val = this->time_increment_inv * rho_cp *
                         (T_new_eval.get_value(q) - T_old_eval.get_value(q)) -
                       heat_source_eval.get_value(q);

            auto val_grad = conductivity * T_new_eval.get_gradient(q);

            if (velocity)
              val += rho_cp * scalar_product(T_new_eval.get_gradient(q), vel_eval.get_value(q));

            if (mass_flux_eval)
              {
                VectorizedArray<number> weight(1.0);
                if (delta_phase_weighted)
                  weight = delta_phase_weighted->compute_weight(heaviside_eval_used.get_value(q));

                const auto cooling =
                  mass_flux_eval ?
                    evaporative_cooling->compute_evaporative_cooling(mass_flux_eval->get_value(q),
                                                                     T_new_eval.get_value(q)) :
                    evaporative_cooling->compute_evaporative_cooling(T_new_eval.get_value(q));

                const auto temp = -cooling * heaviside_eval_used.get_gradient(q).norm() * weight;

                // contribution to heat source for output purposes
                q_vapor[cell * T_new_eval.n_q_points + q] = -temp;

                val += temp;
              }

            T_new_eval.submit_value(-1.0 * val,
                                    q); // -1 residual is moved to rhs
            T_new_eval.submit_gradient(-1.0 * val_grad,
                                       q); // -1 since residual is moved to rhs
          }
        T_new_eval.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
  }

  template <int dim, typename number>
  void
  HeatDiffuseMultiPhaseOperator<dim, number>::rhs_cut_cell_loop(VectorType &dst) const
  {
    // evaluate the evaporative cooling term as surface integral
    if (mass_flux_type == Evaporation::EvaporCoolingInterfaceFluxType::sharp)
      {
        Assert(evaporative_mass_flux,
               ExcMessage("You need to register the evaporative mass flux."));

        evapor_heat_source = 0;

        FEPointEvaluation<1, dim> mass_flux_surface_eval(scratch_data.get_mapping(),
                                                         scratch_data.get_fe(mass_flux_dof_idx),
                                                         update_values);

        FEPointEvaluation<1, dim> T_surface_eval(scratch_data.get_mapping(),
                                                 scratch_data.get_fe(heat_dof_idx),
                                                 update_values);

        std::vector<number>                  buffer;
        std::vector<types::global_dof_index> local_dof_indices;

        const int n_dofs_mass_flux = scratch_data.get_fe(mass_flux_dof_idx).n_dofs_per_cell();
        const int n_dofs_heat      = scratch_data.get_fe(heat_dof_idx).n_dofs_per_cell();

        if (surface_mesh_info->size() > 0)
          {
            for (const auto &[cell, quad_points, weights] : *surface_mesh_info)
              {
                const unsigned int n_points = quad_points.size();

                const ArrayView<const Point<dim>> unit_points(quad_points.data(), n_points);
                const ArrayView<const number>     JxW(weights.data(), n_points);

                mass_flux_surface_eval.reinit(cell, unit_points);
                T_surface_eval.reinit(cell, unit_points);

                const auto T_cell =
                  TriaIterator<DoFCellAccessor<dim, dim, false>>(&scratch_data.get_triangulation(),
                                                                 cell->level(),
                                                                 cell->index(),
                                                                 &scratch_data.get_dof_handler(
                                                                   heat_dof_idx));

                const auto mass_flux_cell =
                  TriaIterator<DoFCellAccessor<dim, dim, false>>(&scratch_data.get_triangulation(),
                                                                 cell->level(),
                                                                 cell->index(),
                                                                 &scratch_data.get_dof_handler(
                                                                   mass_flux_dof_idx));

                // gather evaluate mass_flux

                if (evaporative_mass_flux)
                  {
                    local_dof_indices.resize(n_dofs_mass_flux);
                    buffer.resize(n_dofs_mass_flux);
                    mass_flux_cell->get_dof_indices(local_dof_indices);
                    scratch_data.get_constraint(mass_flux_dof_idx)
                      .get_dof_values(*evaporative_mass_flux,
                                      local_dof_indices.begin(),
                                      buffer.begin(),
                                      buffer.end());
                    mass_flux_surface_eval.evaluate(buffer, EvaluationFlags::values);
                  }

                // gather evaluate temperature
                buffer.resize(n_dofs_heat);
                local_dof_indices.resize(n_dofs_heat);
                T_cell->get_dof_indices(local_dof_indices);
                scratch_data.get_constraint(heat_dof_idx)
                  .get_dof_values(temperature,
                                  local_dof_indices.begin(),
                                  buffer.begin(),
                                  buffer.end());
                T_surface_eval.evaluate(buffer, EvaluationFlags::values);

                for (unsigned int q = 0; q < n_points; ++q)
                  {
                    const number cooling =
                      evaporative_mass_flux ?
                        evaporative_cooling->compute_evaporative_cooling(
                          mass_flux_surface_eval.get_value(q), T_surface_eval.get_value(q)) :
                        evaporative_cooling->compute_evaporative_cooling(
                          T_surface_eval.get_value(q));

                    T_surface_eval.submit_value(cooling * JxW[q],
                                                q); // *(-1) for the residual
                  }
                T_surface_eval.test_and_sum(buffer,
                                            EvaluationFlags::values); //

                scratch_data.get_constraint(heat_dof_idx)
                  .distribute_local_to_global(buffer, local_dof_indices, evapor_heat_source);
              }
          }

        evapor_heat_source.compress(VectorOperation::add);
        scratch_data.get_constraint(heat_dof_idx).set_zero(evapor_heat_source);

        dst += evapor_heat_source;
      }
    else if (mass_flux_type == Evaporation::EvaporCoolingInterfaceFluxType::sharp_conforming)
      {
        Assert(evaporative_mass_flux,
               ExcMessage("You need to register the evaporative mass flux."));
        evapor_heat_source = 0;
        // step 1: set material ID of cells
        LevelSet::Tools::set_material_id_from_level_set(scratch_data,
                                                        ls_dof_idx,
                                                        *level_set_as_heaviside);

        // step 2: evaluate and fill rhs
        FEFaceIntegrator<dim, 1, number> T_eval(scratch_data.get_matrix_free(),
                                                true /*is_interior_face*/,
                                                heat_dof_idx,
                                                heat_quad_idx);
        FEFaceIntegrator<dim, 1, number> mass_flux_eval(scratch_data.get_matrix_free(),
                                                        true /*is_interior_face*/,
                                                        mass_flux_dof_idx,
                                                        heat_quad_idx);

        std::pair<unsigned int, unsigned int> face_range = {
          0, scratch_data.get_matrix_free().n_inner_face_batches()};

        for (unsigned int face = face_range.first; face < face_range.second; ++face)
          {
            T_eval.reinit(face);
            if (evaporative_cooling)
              T_eval.gather_evaluate(temperature, EvaluationFlags::values);

            if (evaporative_mass_flux)
              {
                mass_flux_eval.reinit(face);
                mass_flux_eval.gather_evaluate(*evaporative_mass_flux, EvaluationFlags::values);
              }

            // collect lanes that need to be processed
            std::bitset<VectorizedArray<number>::size()> active_lanes;
            for (unsigned int v = 0;
                 v < scratch_data.get_matrix_free().n_active_entries_per_face_batch(face);
                 ++v)
              {
                const auto face_iter_inner =
                  scratch_data.get_matrix_free().get_face_iterator(face, v, true);
                const auto face_iter_outer =
                  scratch_data.get_matrix_free().get_face_iterator(face, v, false);

                // check if surrounding cells have different materials
                active_lanes[v] = static_cast<bool>(face_iter_inner.first->material_id() !=
                                                    face_iter_outer.first->material_id());
              }

            // loop over quadrature points
            for (unsigned int q = 0; q < T_eval.n_q_points; ++q)
              {
                const auto cooling =
                  evaporative_mass_flux ?
                    evaporative_cooling->compute_evaporative_cooling(mass_flux_eval.get_value(q),
                                                                     T_eval.get_value(q)) :
                    evaporative_cooling->compute_evaporative_cooling(T_eval.get_value(q));
                T_eval.submit_value(cooling,
                                    q); // *(-1) for the residual
              }
            T_eval.integrate(EvaluationFlags::values);
            T_eval.distribute_local_to_global(evapor_heat_source, 0, active_lanes);
          }
        evapor_heat_source.compress(VectorOperation::add);
        scratch_data.get_constraint(heat_dof_idx).set_zero(evapor_heat_source);
        dst += evapor_heat_source;
      }
  }

  template <int dim, typename number>
  void
  HeatDiffuseMultiPhaseOperator<dim, number>::rhs_boundary_loop(
    const MatrixFree<dim, number>        &matrix_free,
    VectorType                           &dst,
    [[maybe_unused]] const VectorType    &src,
    std::pair<unsigned int, unsigned int> face_range) const
  {
    FEFaceIntegrator<dim, 1, number> T_eval(matrix_free,
                                            true /* is interior face*/,
                                            heat_dof_idx,
                                            heat_quad_idx);
    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        const types::boundary_id bc_index = matrix_free.get_boundary_id(face);

        bool do_neumann = neumann_bc.find(bc_index) != neumann_bc.end();

        bool do_radiation =
          std::find(bc_radiation_indices.begin(), bc_radiation_indices.end(), bc_index) !=
          bc_radiation_indices.end();
        bool do_convection =
          std::find(bc_convection_indices.begin(), bc_convection_indices.end(), bc_index) !=
          bc_convection_indices.end();

        AssertThrow(not(do_neumann and (do_radiation or do_convection)),
                    ExcMessage(
                      "It is not allowed to specify both Neumann and radiation and/or convection"
                      " boundary conditions at the same face."));

        if (not do_neumann and not do_radiation and not do_convection)
          continue;

        T_eval.reinit(face);
        T_eval.read_dof_values_plain(temperature);
        T_eval.evaluate(EvaluationFlags::values);

        for (unsigned int q = 0; q < T_eval.n_q_points; ++q)
          {
            const auto T_values = T_eval.get_value(q);

            VectorizedArray<number> val = 0;
            if (do_neumann)
              {
                auto quad_point = T_eval.quadrature_point(q);

                val -= MeltPoolDG::VectorTools::evaluate_function_at_vectorized_points(
                         *neumann_bc.at(bc_index), quad_point) *
                       T_eval.normal_vector(q);
              }
            if (do_radiation)
              val -= data.radiation.emissivity * PhysicalConstants::stefan_boltzmann_constant *
                     (Utilities::fixed_power<4>(T_values) -
                      Utilities::fixed_power<4>(data.radiation.temperature_infinity));
            if (do_convection)
              val -= data.convection.convection_coefficient *
                     (T_values - data.convection.temperature_infinity);

            T_eval.submit_value(val, q);
          }
        T_eval.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

  template <int dim, typename number>
  void
  HeatDiffuseMultiPhaseOperator<dim, number>::create_rhs(VectorType       &dst,
                                                         const VectorType &src) const
  {
    AssertThrowZeroTimeIncrement(this->time_increment);

    scratch_data.get_matrix_free().template loop<VectorType, VectorType>(
      [&](const auto &matrix_free, auto &dst, const auto &src, auto cell_range) {
        rhs_cell_loop(matrix_free, dst, src, cell_range);
      },
      [&]([[maybe_unused]] const auto &matrix_free,
          [[maybe_unused]] auto       &dst,
          [[maybe_unused]] const auto &src,
          [[maybe_unused]] auto        face_range) { /*do nothing*/ }, // face loop
      [&](const auto &matrix_free, auto &dst, const auto &src, auto face_range) {
        rhs_boundary_loop(matrix_free, dst, src, face_range);
      },
      dst,
      src,
      false /*zero dst vector*/); // should not be zeroed out in case of boundary conditions
                                  //
    rhs_cut_cell_loop(dst);
  }

  template <int dim, typename number>
  void
  HeatDiffuseMultiPhaseOperator<dim, number>::attach_vectors(
    std::vector<VectorType *> & /*vectors*/)
  {
    // none
  }

  template <int dim, typename number>
  void
  HeatDiffuseMultiPhaseOperator<dim, number>::distribute_constraints()
  {
    // none
  }

  template <int dim, typename number>
  void
  HeatDiffuseMultiPhaseOperator<dim, number>::reinit()
  {
    // TODO: only if output variable is requested
    if (mass_flux_type == Evaporation::EvaporCoolingInterfaceFluxType::sharp or
        mass_flux_type == Evaporation::EvaporCoolingInterfaceFluxType::sharp_conforming)
      scratch_data.initialize_dof_vector(evapor_heat_source, heat_dof_idx);
  }

  template <int dim, typename number>
  void
  HeatDiffuseMultiPhaseOperator<dim, number>::attach_output_vectors(
    GenericDataOut<dim, number> &data_out) const
  {
    /**
     * write conductivity vector to dof vector
     */
    if (data_out.is_requested("conductivity"))
      {
        scratch_data.initialize_dof_vector(conductivity_vec, heat_no_bc_dof_idx);

        if (not q_vapor.empty() and scratch_data.is_hex_mesh())
          MeltPoolDG::VectorTools::fill_dof_vector_from_cell_operation<dim, 1>(
            conductivity_vec,
            scratch_data.get_matrix_free(),
            heat_no_bc_dof_idx,
            heat_quad_idx,
            [&](const unsigned int cell,
                const unsigned int quad) -> const VectorizedArray<number> & {
              return conductivity_at_q[cell * scratch_data.get_n_q_points(heat_quad_idx) + quad];
            });

        scratch_data.get_constraint(heat_no_bc_dof_idx).distribute(conductivity_vec);

        data_out.add_data_vector(scratch_data.get_dof_handler(heat_no_bc_dof_idx),
                                 conductivity_vec,
                                 "conductivity");
      }

    if (data_out.is_requested("rho_cp"))
      {
        scratch_data.initialize_dof_vector(rho_cp_vec, heat_no_bc_dof_idx);

        if (!q_vapor.empty() and scratch_data.is_hex_mesh())
          MeltPoolDG::VectorTools::fill_dof_vector_from_cell_operation<dim, 1>(
            rho_cp_vec,
            scratch_data.get_matrix_free(),
            heat_no_bc_dof_idx,
            heat_quad_idx,
            [&](const unsigned int cell,
                const unsigned int quad) -> const VectorizedArray<number> & {
              return rho_cp_at_q[cell * scratch_data.get_n_q_points(heat_quad_idx) + quad];
            });

        scratch_data.get_constraint(heat_no_bc_dof_idx).distribute(rho_cp_vec);

        data_out.add_data_vector(scratch_data.get_dof_handler(heat_no_bc_dof_idx),
                                 rho_cp_vec,
                                 "rho_cp");
      }

    /**
     * write evaporative mass flux to dof vector
     */
    if (data_out.is_requested("evaporative_heat_source_projected"))
      {
        if (evaporative_mass_flux)
          {
            scratch_data.initialize_dof_vector(evapor_heat_source_projected, heat_dof_idx);
            if (mass_flux_type == Evaporation::EvaporCoolingInterfaceFluxType::sharp or
                mass_flux_type == Evaporation::EvaporCoolingInterfaceFluxType::sharp_conforming)
              {
                VectorTools::project_vector<1, dim>(scratch_data.get_mapping(),
                                                    scratch_data.get_dof_handler(heat_dof_idx),
                                                    scratch_data.get_constraint(heat_dof_idx),
                                                    scratch_data.get_quadrature(heat_quad_idx),
                                                    evapor_heat_source,
                                                    evapor_heat_source_projected);
                data_out.add_data_vector(scratch_data.get_dof_handler(heat_dof_idx),
                                         evapor_heat_source,
                                         "evaporative_heat_source");
              }
            else
              {
                if (!q_vapor.empty() and scratch_data.is_hex_mesh())
                  MeltPoolDG::VectorTools::fill_dof_vector_from_cell_operation<dim, 1>(
                    evapor_heat_source_projected,
                    scratch_data.get_matrix_free(),
                    heat_no_bc_dof_idx,
                    heat_quad_idx,
                    [&](const unsigned int cell,
                        const unsigned int quad) -> const VectorizedArray<number> & {
                      return q_vapor[cell * scratch_data.get_n_q_points(heat_quad_idx) + quad];
                    });
              }

            Journal::print_formatted_norm<number>(
              scratch_data.get_pcout(3),
              [&]() -> number {
                return VectorTools::compute_norm<dim, number>(evapor_heat_source_projected,
                                                              scratch_data,
                                                              heat_no_bc_dof_idx,
                                                              heat_quad_idx);
              },
              "int(mDot*hV)",
              "heat_transfer_operator",
              15 /*precision*/
            );

            scratch_data.get_constraint(heat_no_bc_dof_idx)
              .distribute(evapor_heat_source_projected);

            data_out.add_data_vector(scratch_data.get_dof_handler(heat_no_bc_dof_idx),
                                     evapor_heat_source_projected,
                                     "evaporative_heat_source_projected");
          }
      }
  }

  template <int dim, typename number>
  void
  HeatDiffuseMultiPhaseOperator<dim, number>::tangent_local_cell_operation(
    FECellIntegrator<dim, 1, number>                        &eval,
    FECellIntegrator<dim, 1, number>                        &T_new_eval,
    FECellIntegrator<dim, 1, number>                        &T_old_eval,
    FECellIntegrator<dim, dim, number>                      &vel_eval,
    FECellIntegrator<dim, 1, number>                        &heaviside_eval,
    FECellIntegrator<dim, 1, number>                        &heaviside_interpolated_eval,
    const std::unique_ptr<FECellIntegrator<dim, 1, number>> &mass_flux_eval,
    const bool                                               do_reinit_cells) const
  {
    auto &heaviside_eval_used = do_level_set_temperature_gradient_interpolation ?
                                  heaviside_interpolated_eval :
                                  heaviside_eval;

    eval.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    if (do_reinit_cells)
      {
        if (velocity)
          {
            vel_eval.reinit(eval.get_current_cell_index());
            vel_eval.read_dof_values_plain(*velocity);
            vel_eval.evaluate(EvaluationFlags::values);
          }

        if (level_set_as_heaviside)
          {
            heaviside_eval.reinit(eval.get_current_cell_index());
            heaviside_eval.read_dof_values_plain(*level_set_as_heaviside);

            if (evaporative_mass_flux)
              {
                if (do_level_set_temperature_gradient_interpolation)
                  {
                    heaviside_interpolated_eval.reinit(eval.get_current_cell_index());

                    DoFTools::compute_gradient_at_interpolated_dof_values<dim>(
                      heaviside_eval,
                      heaviside_interpolated_eval,
                      ls_to_heat_grad_interpolation_matrix);

                    heaviside_interpolated_eval.evaluate(EvaluationFlags::values |
                                                         EvaluationFlags::gradients);
                  }
                else
                  heaviside_eval.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
              }
            else
              heaviside_eval.evaluate(EvaluationFlags::values);
          }

        if (do_solidification)
          {
            T_old_eval.reinit(eval.get_current_cell_index());
            T_old_eval.read_dof_values_plain(temperature_old);
            T_old_eval.evaluate(EvaluationFlags::values);

            T_new_eval.reinit(eval.get_current_cell_index());
            T_new_eval.read_dof_values_plain(temperature);
            T_new_eval.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
          }
        if (mass_flux_eval)
          {
            mass_flux_eval->reinit(eval.get_current_cell_index());
            mass_flux_eval->read_dof_values_plain(*evaporative_mass_flux);
            mass_flux_eval->evaluate(EvaluationFlags::values);
          }
      }

    for (unsigned int q = 0; q < eval.n_q_points; ++q)
      {
        const auto [rho_cp, conductivity, d_rho_cp_d_T, d_conductivity_d_T] =
          get_material_parameters_and_derivatives(T_new_eval, heaviside_eval_used, q);

        auto val = this->time_increment_inv * rho_cp * eval.get_value(q);

        auto val_grad = conductivity * eval.get_gradient(q);

        if (velocity)
          {
            val += rho_cp * scalar_product(eval.get_gradient(q), vel_eval.get_value(q));
            // todo: add term containing ∇·u  in case of evaporation
          }

        if (do_solidification)
          {
            val += this->time_increment_inv * d_rho_cp_d_T *
                   (T_new_eval.get_value(q) - T_old_eval.get_value(q)) * eval.get_value(q);
            val_grad += d_conductivity_d_T * T_new_eval.get_gradient(q) * eval.get_value(q);
          }

        if (velocity and do_solidification)
          {
            val +=
              d_rho_cp_d_T * T_new_eval.get_gradient(q) * vel_eval.get_value(q) * eval.get_value(q);
            // todo: add term containing ∇·u  in case of evaporation
          }

        if (mass_flux_eval)
          {
            VectorizedArray<number> weight(1.0);
            if (delta_phase_weighted)
              weight = delta_phase_weighted->compute_weight(heaviside_eval_used.get_value(q));

            const auto cooling_dertivative =
              evaporative_cooling->compute_evaporative_cooling_derivative_constant_mass_flux(
                mass_flux_eval->get_value(q));

            val += -cooling_dertivative * heaviside_eval_used.get_gradient(q).norm() * weight *
                   eval.get_value(q);
          }
        // TODO: add derivative if evaporative cooling without mDOt shall be computed

        eval.submit_value(val, q);
        eval.submit_gradient(val_grad, q);
      }
    eval.integrate(EvaluationFlags::values | EvaluationFlags::gradients);

    // evaluate the evaporative cooling term as surface integral
    if (evaporative_mass_flux and surface_mesh_info)
      {
        // TODO
      }
  }

  template <int dim, typename number>
  void
  HeatDiffuseMultiPhaseOperator<dim, number>::tangent_local_boundary_operation(
    FEFaceIntegrator<dim, 1, number> &face_eval,
    FEFaceIntegrator<dim, 1, number> &T_face_eval,
    const bool                        do_reinit_face) const
  {
    face_eval.evaluate(EvaluationFlags::values);

    if (do_reinit_face)
      {
        T_face_eval.reinit(face_eval.get_current_cell_index());
        T_face_eval.gather_evaluate(temperature, EvaluationFlags::values);
      }

    const types::boundary_id bc_index =
      scratch_data.get_matrix_free().get_boundary_id(face_eval.get_current_cell_index());

    const bool do_radiation =
      std::find(bc_radiation_indices.begin(), bc_radiation_indices.end(), bc_index) !=
      bc_radiation_indices.end();

    const bool do_convection =
      std::find(bc_convection_indices.begin(), bc_convection_indices.end(), bc_index) !=
      bc_convection_indices.end();

    for (unsigned int q = 0; q < face_eval.n_q_points; ++q)
      {
        const auto inc_temperature_vals_at_q = face_eval.get_value(q);

        VectorizedArray<number> val = 0;

        if (do_convection)
          val += data.convection.convection_coefficient * inc_temperature_vals_at_q;
        if (do_radiation)
          val += 4. * data.radiation.emissivity * PhysicalConstants::stefan_boltzmann_constant *
                 pow<number>(T_face_eval.get_value(q), 3) * inc_temperature_vals_at_q;

        face_eval.submit_value(val, q);
      }
    face_eval.integrate(EvaluationFlags::values);
  }



  template <int dim, typename number>
  std::tuple<VectorizedArray<number>, VectorizedArray<number>>
  HeatDiffuseMultiPhaseOperator<dim, number>::get_material_parameters(
    const FECellIntegrator<dim, 1, number> &T_eval,
    const FECellIntegrator<dim, 1, number> &heaviside_eval,
    const unsigned int                      q) const
  {
    if (not data.diffuse.use_volume_specific_thermal_capacity_for_phase_interpolation)
      {
        const auto material_values = material.template compute_parameters<VectorizedArray<number>>(
          heaviside_eval,
          T_eval,
          MaterialUpdateFlags::thermal_conductivity | MaterialUpdateFlags::specific_heat_capacity |
            MaterialUpdateFlags::density,
          q);
        return {material_values.specific_heat_capacity * material_values.density,
                material_values.thermal_conductivity};
      }
    // special interpolation of volumetric heat capacity
    else
      {
        const auto material_values = material.template compute_parameters<VectorizedArray<number>>(
          heaviside_eval,
          T_eval,
          MaterialUpdateFlags::thermal_conductivity |
            MaterialUpdateFlags::volume_specific_heat_capacity,
          q);
        return {material_values.volume_specific_heat_capacity,
                material_values.thermal_conductivity};
      }
  }



  template <int dim, typename number>
  std::tuple<VectorizedArray<number>,
             VectorizedArray<number>,
             VectorizedArray<number>,
             VectorizedArray<number>>
  HeatDiffuseMultiPhaseOperator<dim, number>::get_material_parameters_and_derivatives(
    const FECellIntegrator<dim, 1, number> &T_eval,
    const FECellIntegrator<dim, 1, number> &heaviside_eval,
    const unsigned int                      q) const
  {
    if (not data.diffuse.use_volume_specific_thermal_capacity_for_phase_interpolation)
      {
        const auto material_values = material.template compute_parameters<VectorizedArray<number>>(
          heaviside_eval,
          T_eval,
          MaterialUpdateFlags::thermal_conductivity | MaterialUpdateFlags::specific_heat_capacity |
            MaterialUpdateFlags::density | MaterialUpdateFlags::d_thermal_conductivity_d_T |
            MaterialUpdateFlags::d_specific_heat_capacity_d_T | MaterialUpdateFlags::d_density_d_T,
          q);

        return {/* rho cp */ material_values.density * material_values.specific_heat_capacity,
                /* conductivity */ material_values.thermal_conductivity,
                /* d_rho_cp_d_T */ material_values.d_specific_heat_capacity_d_T *
                    material_values.density +
                  material_values.d_density_d_T * material_values.specific_heat_capacity,
                /* d_conductivity_d_T */ material_values.d_thermal_conductivity_d_T};
      }
    else
      {
        // special interpolation of volumetric heat capacity
        const auto material_values = material.template compute_parameters<VectorizedArray<number>>(
          heaviside_eval,
          T_eval,
          MaterialUpdateFlags::thermal_conductivity |
            MaterialUpdateFlags::volume_specific_heat_capacity |
            MaterialUpdateFlags::d_thermal_conductivity_d_T |
            MaterialUpdateFlags::d_volume_specific_heat_capacity_d_T,
          q);
        return {/* rho cp */ material_values.volume_specific_heat_capacity,
                /* conductivity */ material_values.thermal_conductivity,
                /* d_rho_cp_d_T */ material_values.d_volume_specific_heat_capacity_d_T,
                /* d_conductivity_d_T */ material_values.d_thermal_conductivity_d_T};
      }
  }



  template class HeatDiffuseMultiPhaseOperator<1, double>;
  template class HeatDiffuseMultiPhaseOperator<2, double>;
  template class HeatDiffuseMultiPhaseOperator<3, double>;
} // namespace MeltPoolDG::Heat
