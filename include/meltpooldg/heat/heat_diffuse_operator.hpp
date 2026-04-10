#pragma once

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/types.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/full_matrix.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/core/boundary_conditions.hpp>
#include <meltpooldg/core/material.hpp>
#include <meltpooldg/core/operator_base.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/heat/heat_data.hpp>
#include <meltpooldg/level_set/delta_approximation_phase_weighted.hpp>
#include <meltpooldg/phase_change/evaporation_data.hpp>
#include <meltpooldg/phase_change/evaporative_cooling.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>

#include <map>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>


namespace MeltPoolDG::Heat
{
  /*
   * This operator computes the residual and its consistent tangent of the discretized heat
   * equation with temperature dependent material properties:
   * ρ^(n) = ρ(T^(n)), c_p^(n) = c_p(T^(n)), k^(n) = k(T^(n))
   *
   *                  1  /                                                     \
   *  R(T_b^(n+1)) = --- | N_a, ρ^(n+1) c_p^(n+1) N_b ( T_b^(n+1) - T_b^(n)) ) |
   *                 dt  \                                                     /
   *                                                                            Ω
   *                 /                               \
   *               + | ∇N_a, k^(n+1) ∇N_b T_b^(n+1)) |
   *                 \                               /
   *                                                  Ω
   *                 /                                           \
   *               + | N_a, ρ^(n+1) c_p^(n+1) ∇N_b T_b^(n+1) · u |
   *                 \                                           /
   *                                                              Ω
   *                 /                                          \
   *               + | N_a, ρ^(n+1) c_p^(n+1) N_b T_b^(n+1) ∇·u |    (this term is not yet considered @todo)
   *                 \                                          /
   *                                                             Ω
   *                 /      _   \     /      _  \
   *               - | N_a, q_s |  -  | N_a, q  | = 0
   *                 \          /     \         /
   *                             Ω               Γ
   *                                              N
   *
   *
   *  dR(T^(n+1))    1  /                            \
   *  ----------- = --- | N_a, ρ^(n+1) c_p^(n+1) N_b |
   *  dT_b^(n+1)     dt \                            /
   *                                                  Ω
   *                1  /       d ρ c_p |                              \
   *              + -- | N_a, ---------| N_b ( T_b^(n+1) - T_b^(n)) ) |
   *                dt \         d T   |                              /
   *                                   |(n+1)                          Ω
   *                /                       d k |                \
   *              + | ∇N_a, k^(n+1) ∇N_b + -----| ∇N_b T_b^(n+1) |
   *                \                       d T |                /
   *                                            |(n+1)            Ω
   *                /                                             \
   *              + | N_a, ρ^(n+1) c_p^(n+1) ( ∇N_b u + N_b ∇·u ) |
   *                \                                             /
   *                                                               Ω
   *                /       d ρ c_p |                                            \
   *              + | N_a, ---------| ( ∇N_b T_b^(n+1) · u + N_b T_b^(n+1) ∇·u ) |
   *                \         d T   |                                            /
   *                                |(n+1)                                        Ω
   *                            _                      _
   *                /        d q_s     \    /        d q       \
   *              - | N_a, ---------   |  - | N_a, ---------   |
   *                \       dT_b^(n+1) /    \       dT_b^(n+1) /
   *                                    Ω                       Γ
   *                                                             N
   *
   * with shape functions N_a and N_b, nodal temperature values T_b^(n+1), the density ρ, the
   * specific heat capacity c_p and the conductivity k, source/sink terms q_s and prescribed
   * fluxes q along Neumann boundaries. The heat flux may result from radiative losses
   *
   *  _
   *  q = σ ϵ (T^4-T∞^4)
   *
   * with the Stefan-Boltzmann constant σ, the emissivity ϵ and the temperature of the surroundings
   * T∞ as well as convective losses
   *
   *  _
   *  q = α (T-T∞)
   *
   * with the convection coefficient denoted as α.
   *
   * We assume that the density and the specific heat capacity do not dependent on the temperature.
   *
   */

  template <int dim, typename number>
  class HeatDiffuseMultiPhaseOperator : public OperatorMatrixFree<dim, number>
  {
  private:
    // to avoid compiler warnings regarding hidden overriden functions
    using OperatorMatrixFree<dim, number>::create_rhs;
    using OperatorMatrixFree<dim, number>::compute_inverse_diagonal_from_matrixfree;
    using OperatorMatrixFree<dim, number>::vmult;

    using VectorType       = typename OperatorMatrixFree<dim, number>::VectorType;
    using SparseMatrixType = typename OperatorMatrixFree<dim, number>::SparseMatrixType;

    const ScratchData<dim, dim, number> &scratch_data;
    const HeatData<number>              &data;
    const Material<number>              &material;
    const unsigned int                   heat_dof_idx;
    const unsigned int                   heat_quad_idx;
    const unsigned int                   heat_no_bc_dof_idx;

    const VectorType &temperature;
    const VectorType &temperature_old;
    std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
                                            neumann_bc; //@todo find a nice way to provide BC
    std::vector<dealii::types::boundary_id> bc_radiation_indices;
    std::vector<dealii::types::boundary_id> bc_convection_indices;

    const VectorType &heat_source;

    // optional: flow velocity for internal convection
    const unsigned int vel_dof_idx;
    const VectorType  *velocity;

    // optional: level set heaviside field for two phase flow
    const unsigned int ls_dof_idx;
    const VectorType  *level_set_as_heaviside;

    // optional: two phase flow with evaporation
    VectorType  *evaporative_mass_flux = nullptr;
    unsigned int mass_flux_dof_idx     = dealii::numbers::invalid_unsigned_int;
    std::unique_ptr<Evaporation::EvaporativeCooling<number>> evaporative_cooling;

    // optional: melting/solidification effects
    const bool do_solidification;

    // interpolation of the level set space to the temperature space
    dealii::FullMatrix<number> ls_to_heat_grad_interpolation_matrix;
    bool                       do_level_set_temperature_gradient_interpolation = false;

    /*
     * contribution to heat source due to evaporation;
     * just for output purposes
     */
    mutable VectorType                                             evapor_heat_source;
    mutable VectorType                                             evapor_heat_source_projected;
    mutable dealii::AlignedVector<dealii::VectorizedArray<number>> q_vapor;

    // phase weighted delta function, only used for evaporative cooling
    std::unique_ptr<const LevelSet::DeltaApproximationBase<number>> delta_phase_weighted;

    mutable dealii::AlignedVector<dealii::VectorizedArray<number>> conductivity_at_q;
    mutable VectorType                                             conductivity_vec;

    mutable dealii::AlignedVector<dealii::VectorizedArray<number>> rho_cp_at_q;
    mutable VectorType                                             rho_cp_vec;

    // for efficient ghost value update
    mutable std::vector<bool> do_update_ghosts;

  public:
    HeatDiffuseMultiPhaseOperator(
      const ScratchData<dim, dim, number>                               &scratch_data_in,
      const std::shared_ptr<const BoundaryConditionManager<dim, number>> heat_bc_manager,
      const HeatData<number>                                            &data_in,
      const Material<number>                                            &material,
      const unsigned int                                                 heat_dof_idx_in,
      const unsigned int                                                 heat_quad_idx_in,
      const unsigned int                                                 heat_no_bc_dof_idx,
      const VectorType                                                  &temperature_in,
      const VectorType                                                  &temperature_old_in,
      const VectorType                                                  &heat_source_in,
      const unsigned int                                                 vel_dof_idx_in = 0,
      const VectorType                                                  *velocity_in    = nullptr,
      const unsigned int                                                 ls_dof_idx_in  = 0,
      const VectorType *level_set_as_heaviside_in                                       = nullptr,
      const bool        do_solidifiaction_in                                            = false);

    void
    register_evaporative_mass_flux(VectorType        *evaporative_mass_flux_in,
                                   const unsigned int mass_flux_dof_idx_in,
                                   const Evaporation::EvaporationData<number> &evapor_data);

    void
    register_surface_mesh(
      const std::vector<
        std::tuple<const typename dealii::Triangulation<dim, dim>::cell_iterator /*cell*/,
                   std::vector<dealii::Point<dim>> /*quad_points*/,
                   std::vector<number> /*weights*/
                   >> &surface_mesh_info);

    void
    pre() final;

    void
    post() final;
    /*
     *    matrix-free implementation
     */
    void
    vmult(VectorType &dst, const VectorType &src /*solution_update*/) const final;

    void
    tangent_cell_loop(const dealii::MatrixFree<dim, number> &matrix_free,
                      VectorType                            &dst,
                      const VectorType                      &src,
                      std::pair<unsigned int, unsigned int>  cell_range) const;

    /*
     * compute the tangent of Robin-type boundary conditions for convection and radiation
     */
    void
    tangent_boundary_loop(const dealii::MatrixFree<dim, number> &matrix_free,
                          VectorType                            &dst,
                          const VectorType                      &src,
                          std::pair<unsigned int, unsigned int>  face_range) const;

    void
    compute_inverse_diagonal_from_matrixfree(VectorType &diagonal) const final;

    void
    compute_system_matrix_from_matrixfree(SparseMatrixType &system_matrix) const final;

    void
    rhs_cell_loop(const dealii::MatrixFree<dim, number> &matrix_free,
                  VectorType                            &dst,
                  const VectorType                      &src, /* temperature_old*/
                  std::pair<unsigned int, unsigned int>  cell_range) const;

    /*
     * compute the RHS due to Neumann and Robin-type boundary conditions for convection and
     * radiation
     *
     * @todo: add equations
     */
    void
    rhs_boundary_loop(const dealii::MatrixFree<dim, number> &matrix_free,
                      VectorType                            &dst,
                      [[maybe_unused]] const VectorType     &src,
                      std::pair<unsigned int, unsigned int>  face_range) const;

    void
    rhs_cut_cell_loop(VectorType &dst) const;

    /**
     * -R(T)
     */
    void
    create_rhs(VectorType &dst, const VectorType &src /*temperature_old*/) const final;

    /**
     * attach vector for solution transfer for AMR
     */
    void
    attach_vectors(std::vector<VectorType *> &vectors);

    void
    distribute_constraints();

    void
    reinit() final;

    void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const;

  private:
    /**
     * This function executes the local cell operation for computing the tangent.
     *
     * @note The function assumes that @p eval has been already initialized
     *   and the dof-values of @p eval are already set a priori. Afterwards, the
     *   dof-values held by @p eval can be written back to the global vector via
     *   eval.distribute_local_to_global(dst).
     */
    void
    tangent_local_cell_operation(
      FECellIntegrator<dim, 1, number>                        &eval,
      FECellIntegrator<dim, 1, number>                        &T_new_eval,
      FECellIntegrator<dim, 1, number>                        &T_old_eval,
      FECellIntegrator<dim, dim, number>                      &vel_eval,
      FECellIntegrator<dim, 1, number>                        &heaviside_eval,
      FECellIntegrator<dim, 1, number>                        &heaviside_interpolated_eval,
      const std::unique_ptr<FECellIntegrator<dim, 1, number>> &mass_flux_eval,
      bool                                                     do_reinit_cells) const;

    /**
     * This function executes the local boundary operation for computing the tangent.
     *
     * @note The function assumes that @p face_eval has been already initialized
     *   and the dof-values of @p face_eval are already set a priori. Afterwards, the
     *   dof-values held by @p face_eval can be written back to the global vector via
     *   face_eval.distribute_local_to_global(dst).
     */
    void
    tangent_local_boundary_operation(FEFaceIntegrator<dim, 1, number> &face_eval,
                                     FEFaceIntegrator<dim, 1, number> &T_face_eval,
                                     bool                              do_reinit_face) const;
    /**
     * The setup for dealii::MatrixFreeTools::internal::compute_diagonal and
     * dealii::MatrixFreeTools::internal::compute_matrix is identical. To avoid duplicate code this
     * internal function can handle both operations. Choose which operation to perform using
     * @param do_diagonal: `true` for compute_diagonal and `false` for compute_matrix.
     */
    void
    internal_compute_diagonal_or_system_matrix([[maybe_unused]] VectorType       &diagonal,
                                               [[maybe_unused]] SparseMatrixType &system_matrix,
                                               const bool do_diagonal) const;

    /**
     * Determine the material parameters. This function takes two-phase flow and solidification
     * effects into account.
     *
     * The values of @p rho_cp and @p conductivity must be set to the material.gas values
     * initially. This function only modifies their values if necessary. I.e. in case of no
     * two-phase flow and no solidification this function does nothing.
     */
    std::tuple<dealii::VectorizedArray<number>, dealii::VectorizedArray<number>>
    get_material_parameters(const FECellIntegrator<dim, 1, number> &T_eval,
                            const FECellIntegrator<dim, 1, number> &heaviside_eval,
                            unsigned int                            q) const;

    /**
     * Determine the material parameters and their temperature derivatives. This function takes
     * two-phase flow and solidification effects into account.
     *
     * The values of @p rho_cp, @p conductivity, @p d_rho_cp_d_T and @p d_conductivity_dT must be set
     * to the material.gas values initially (0 for the derivatives). This function only modifies
     * their values if necessary. I.e. in case of no two-phase flow and no solidification this
     * function does nothing.
     *
     * @note The derivatives @p d_rho_cp_d_T and @p d_conductivity_dT are only non-zero in the case of solidification
     * and if the temperature is between the solidus- and liquidus temperature.
     */
    std::tuple<dealii::VectorizedArray<number>,
               dealii::VectorizedArray<number>,
               dealii::VectorizedArray<number>,
               dealii::VectorizedArray<number>>
    get_material_parameters_and_derivatives(const FECellIntegrator<dim, 1, number> &T_eval,
                                            const FECellIntegrator<dim, 1, number> &heaviside_eval,
                                            unsigned int                            q) const;

    const std::vector<
      std::tuple<const typename dealii::Triangulation<dim, dim>::cell_iterator /*cell*/,
                 std::vector<dealii::Point<dim>> /*quad_points*/,
                 std::vector<number> /*weights*/
                 >> *surface_mesh_info = nullptr;

    Evaporation::EvaporCoolingInterfaceFluxType evaporative_cooling_interface_flux_type =
      Evaporation::EvaporCoolingInterfaceFluxType::none;
  };
} // namespace MeltPoolDG::Heat
