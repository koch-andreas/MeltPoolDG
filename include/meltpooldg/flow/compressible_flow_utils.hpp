/**
 * @brief Type definitions and helper functions for the compressible flow implementations.
 */

#pragma once

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/flow/compressible_flow_types.hpp>
#include <meltpooldg/utilities/better_enum.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>

namespace MeltPoolDG::Flow
{
  // Forward declaration
  template <int dim, typename number>
  class CompressibleFlowMaterial;

  /// Index sets for the components of the compressible Navier-Stokes equations.
  BETTER_ENUM(Idx1D, char, density, momentum_x, energy);
  BETTER_ENUM(Idx2D, char, density, momentum_x, momentum_y, energy);
  BETTER_ENUM(Idx3D, char, density, momentum_x, momentum_y, momentum_z, energy);

  /**
   * @brief Calculate the velocity from the conserved variables by computing u = (ρu)/ρ.
   *
   * @param conserved_variables Current values of the conserved variables.
   *
   * @return Current velocity.
   */
  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<1, dim, dealii::VectorizedArray<number>>
    calculate_velocity(
      const CompressibleFlow::ConservedVariablesType<dim, number> &conserved_variables);

  /**
   * @brief Calculate the velocity gradient.
   *
   * Calculate the gradient of the  velocity from the conserved variables and their gradients by
   * computing grad(u) = 1/ρ * (grad(ρu) - u*grad(ρ)).
   *
   * @param conserved_variables Current values of the conserved variables.
   * @param grad_conserved_variables Current gradient of the conserved variables.
   *
   * @return Current gradient of the velocity.
   */
  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<2, dim, dealii::VectorizedArray<number>>
    calculate_grad_velocity(
      const CompressibleFlow::ConservedVariablesType<dim, number> &conserved_variables,
      const CompressibleFlow::ConservedVariablesGradientType<dim, number>
        &grad_conserved_variables);

  /**
   * @brief This function computes the local values of the internal penalty parameter used in the
   * viscous numerical flux.
   *
   * @param array_penalty_parameter Array in which the values of the penalty parameter are stored.
   * @param matrix_free Matrix-free object providing the required geometrical data.
   * @param domain_representation_type Numerical operator type (cut or fitted_mesh).
   * @param dof_index Index of the relevant dof handler in the matrix-free object.
   * @param scaling_factor Additional scaling factor to scale the penalty parameter.
   */
  template <int dim, typename Number>
  void
  calculate_penalty_parameter(
    dealii::AlignedVector<dealii::VectorizedArray<Number>> &array_penalty_parameter,
    const dealii::MatrixFree<dim, Number>                  &matrix_free,
    const std::string                                      &domain_representation_type,
    const unsigned int                                      dof_index      = 0,
    const Number                                            scaling_factor = 1.0);

  /**
   * @brief Update the primitive variable solution according to the current solution vector.
   *
   * @param solution_primitive_variables Vector where the solution in primitive variables is stored.
   * @param solution Current solution vector in conservative variable formulation.
   * @param dof_idx Index of the relevant dof handler in the matrix-free object.
   * @param quad_idx Relevant quadrature index of the flow solver.
   * @param material_liquid Pointer to the material object for liquid phase.
   * @param material_gas Pointer to the material object for the gas phase.
   *
   * @note The second material object is only required for the two-phase case.
   */
  template <int dim, typename number>
  void
  update_primitive_variables_solution(
    dealii::LinearAlgebra::distributed::Vector<number>       &solution_primitive_variables,
    const dealii::LinearAlgebra::distributed::Vector<number> &solution,
    const ScratchData<dim, dim, number>                      &scratch_data,
    const unsigned int                                        dof_idx,
    const unsigned int                                        quad_idx,
    const CompressibleFlowMaterial<dim, number>              *material_liquid,
    const CompressibleFlowMaterial<dim, number>              *material_gas = nullptr);

  /**
   * @brief An abstract interface for defining external forces acting on the fluid that must be
   * evaluated and incorporated during the cell loop of an explicit time integration scheme.
   *
   * This struct serves as a base for user-defined external fluid force models. Any derived class
   * must implement the core functions:
   * @p value(): Invoked at each quadrature point to compute the contribution of the external force.
   */
  template <int dim, typename number>
  struct ExternalFlowForce
  {
    virtual ~ExternalFlowForce() = default;

    /**
     * This function computes the value of the external force contribution for the balance of mass,
     * momentum, and energy equations at the given set of points. The returned value then contains
     * the body force as given in the respective governing equations and can directly be added to
     * the right-hand side.
     *
     * @param time_step_size Size of the current time step.
     * @param cells Container holding an iterator to the cells associated with the provided points.
     * @param points Coordinates of the points at which the external force is evaluated.
     * @param w Conserved variables at the corresponding points.
     *
     * @return The computed contribution of the external force to be added to the conservation
     * equations.
     */
    virtual CompressibleFlow::ConservedVariablesType<dim, number>
    value(number                                                              time_step_size,
          const std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> &cells,
          const dealii::Point<dim, dealii::VectorizedArray<number>>          &points,
          const CompressibleFlow::ConservedVariablesType<dim, number>        &w) = 0;
  };


  template <int dim, typename number>
  struct ExternalFlowForceJacobian
  {
    virtual ~ExternalFlowForceJacobian() = default;

    /**
     * This function computes the jacobian of the external force contribution for the balance of
     * mass, momentum, and energy equations at the given set of points. The returned value then
     * contains the body force as given in the respective governing equations and can directly be
     * added to the right-hand side.
     *
     * @param time_step_size Size of the current time step.
     * @param cells Container holding an iterator to the cells associated with the provided points.
     * If required this can also include an iterator with invalid state.
     * @param points Coordinates of the points at which the external force is evaluated.
     * @param w Conserved variables at the corresponding points.
     * @param delta_w Increment in conserved variables at the corresponding points.
     *
     * @return The computed contribution of the external force to be added to the conservation
     * equations.
     */
    virtual CompressibleFlow::ConservedVariablesType<dim, number>
    value(number                                                              time_step_size,
          const std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> &cells,
          const dealii::Point<dim, dealii::VectorizedArray<number>>          &points,
          const CompressibleFlow::ConservedVariablesType<dim, number>        &w,
          const CompressibleFlow::ConservedVariablesType<dim, number>        &delta_w) = 0;
  };

  /********************************************************************************************
   * Inlined function definitions
   * *************************************************************************************+****/
  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<1, dim, dealii::VectorizedArray<number>>
    calculate_velocity(
      const CompressibleFlow::ConservedVariablesType<dim, number> &conserved_variables)
  {
    const dealii::VectorizedArray<number> inverse_density =
      dealii::VectorizedArray<number>(1.) / conserved_variables[0];

    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> velocity;
    for (unsigned int d = 0; d < dim; ++d)
      velocity[d] = conserved_variables[1 + d] * inverse_density;

    return velocity;
  }

  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<2, dim, dealii::VectorizedArray<number>>
    calculate_grad_velocity(
      const CompressibleFlow::ConservedVariablesType<dim, number>         &conserved_variables,
      const CompressibleFlow::ConservedVariablesGradientType<dim, number> &grad_conserved_variables)
  {
    const dealii::VectorizedArray<number> inverse_density =
      dealii::VectorizedArray<number>(1.) / conserved_variables[0];
    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> velocity =
      calculate_velocity<dim, number>(conserved_variables);

    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> grad_rho;
    for (unsigned int d = 0; d < dim; ++d)
      grad_rho[d] = grad_conserved_variables[0][d];

    dealii::Tensor<2, dim, dealii::VectorizedArray<number>> grad_rho_velocity;
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int e = 0; e < dim; ++e)
        grad_rho_velocity[d][e] = grad_conserved_variables[1 + d][e];

    dealii::Tensor<2, dim, dealii::VectorizedArray<number>> grad_velocity;
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int e = 0; e < dim; ++e)
        grad_velocity[d][e] =
          inverse_density * (grad_rho_velocity[d][e] - velocity[d] * grad_rho[e]);

    return grad_velocity;
  }

  template <int dim, typename Number>
  void
  calculate_penalty_parameter(
    dealii::AlignedVector<dealii::VectorizedArray<Number>> &array_penalty_parameter,
    const dealii::MatrixFree<dim, Number>                  &matrix_free,
    const std::string                                      &domain_representation_type,
    const unsigned int                                      dof_index,
    const Number                                            scaling_factor)
  {
    const unsigned int n_cells = matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();
    array_penalty_parameter.resize(n_cells);

    dealii::Mapping<dim> const       &mapping = *matrix_free.get_mapping_info().mapping;
    dealii::FiniteElement<dim> const &fe      = matrix_free.get_dof_handler(dof_index).get_fe();
    unsigned int const                degree  = fe.degree;

    // use penalty factor for hypercube elements according to K. Hillewaert, Development of the
    // discontinuous Galerkin method for high-resolution, large scale CFD and acoustics in
    // industrial geometries, PhD thesis, Univ. de Louvain, 2013.
    const Number fac = scaling_factor * (degree + 1.0) * (degree + 1.0);

    auto const reference_cells =
      matrix_free.get_dof_handler(dof_index).get_triangulation().get_reference_cells();
    AssertThrow(reference_cells.size() == 1, dealii::ExcMessage("No mixed meshes allowed."));

    auto const quadrature = reference_cells[0].template get_gauss_type_quadrature(degree + 1);
    dealii::FEValues<dim> fe_values(mapping, fe, quadrature, dealii::update_JxW_values);

    auto const face_quadrature =
      reference_cells[0].face_reference_cell(0).template get_gauss_type_quadrature(degree +
                                                                                            1);
    dealii::FEFaceValues<dim> fe_face_values(mapping,
                                             fe,
                                             face_quadrature,
                                             dealii::update_JxW_values);

    if (domain_representation_type == "fitted")
      {
        for (unsigned int i = 0; i < n_cells; ++i)
          {
            for (unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(i); ++v)
              {
                typename dealii::DoFHandler<dim>::cell_iterator cell =
                  matrix_free.get_cell_iterator(i, v, dof_index);
                fe_values.reinit(cell);

                // calculate cell volume
                Number volume = 0;
                for (unsigned int q = 0; q < quadrature.size(); ++q)
                  {
                    volume += fe_values.JxW(q);
                  }

                // calculate surface area
                Number surface_area = 0;
                for (unsigned int const f : cell->face_indices())
                  {
                    fe_face_values.reinit(cell, f);
                    Number const factor =
                      (cell->at_boundary(f) and not(cell->has_periodic_neighbor(f))) ? 1. : 0.5;
                    for (unsigned int q = 0; q < face_quadrature.size(); ++q)
                      {
                        surface_area += fe_face_values.JxW(q) * factor;
                      }
                  }

                array_penalty_parameter[i][v] = surface_area / volume * fac;
              }
          }
      }
    else if (domain_representation_type == "cut")
      {
        for (unsigned int i = 0; i < n_cells; ++i)
          {
            for (unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(i); ++v)
              {
                typename dealii::DoFHandler<dim>::cell_iterator cell =
                  matrix_free.get_cell_iterator(i, v, dof_index);

                // simplified computation for hypercube elements
                array_penalty_parameter[i][v] = 2. * fac / cell->minimum_vertex_distance();
              }
          }
      }
    else
      AssertThrow(false,
                  dealii::ExcMessage("The domain representation type '" +
                                     domain_representation_type + "' is not supported."));
  }

  template <int dim, typename number>
  void
  update_primitive_variables_solution(
    dealii::LinearAlgebra::distributed::Vector<number>       &solution_primitive_variables,
    const dealii::LinearAlgebra::distributed::Vector<number> &solution,
    const ScratchData<dim, dim, number>                      &scratch_data,
    const unsigned int                                        dof_idx,
    const unsigned int                                        quad_idx,
    const CompressibleFlowMaterial<dim, number>              *material_liquid,
    const CompressibleFlowMaterial<dim, number>              *material_gas)
  {
    const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free =
      scratch_data.get_matrix_free();
    unsigned int n_support_points_per_cell = scratch_data.get_n_dofs_per_cell(dof_idx) / (dim + 2);

    auto process_cell = [&](CutUtil::CellCategory                        category,
                            unsigned int                                 first_component,
                            const CompressibleFlowMaterial<dim, number> *material,
                            const unsigned int                           cell_batch) {
      if (!material)
        return;
      auto eval = FECellIntegrator<dim, dim + 2, number>(
        matrix_free, dof_idx, quad_idx, first_component, category);
      eval.reinit(cell_batch);
      eval.read_dof_values(solution);

      for (unsigned int i = 0; i < n_support_points_per_cell; ++i)
        {
          const auto &u_cons = eval.get_dof_value(i);
          auto u_prim = material->eos_utils->convert_conservative_into_primitive_variables(u_cons);
          eval.submit_dof_value(u_prim, i);
        }

      eval.set_dof_values(solution_primitive_variables);
    };

    if (!matrix_free.get_dof_info(dof_idx).cell_active_fe_index.empty())
      {
        // using hp::FECollection (cutFEM/DG)
        for (unsigned int cell_batch = 0; cell_batch < matrix_free.n_cell_batches(); ++cell_batch)
          {
            const auto cell_category = matrix_free.get_cell_category(cell_batch);

            if (cell_category == CutUtil::CellCategory::liquid)
              {
                process_cell(CutUtil::CellCategory::liquid, 0, material_liquid, cell_batch);
              }
            else if (cell_category == CutUtil::CellCategory::gas)
              {
                process_cell(CutUtil::CellCategory::gas, dim + 2, material_gas, cell_batch);
              }
            else if (cell_category == CutUtil::CellCategory::intersected)
              {
                process_cell(CutUtil::CellCategory::intersected, 0, material_liquid, cell_batch);
                process_cell(CutUtil::CellCategory::intersected, dim + 2, material_gas, cell_batch);
              }
          }
      }
    else
      {
        // not using hp::FECollection (fitted mesh)
        for (unsigned int cell_batch = 0; cell_batch < matrix_free.n_cell_batches(); ++cell_batch)
          {
            process_cell(CutUtil::CellCategory::liquid, 0, material_liquid, cell_batch);
          }
      }
  }
} // namespace MeltPoolDG::Flow
