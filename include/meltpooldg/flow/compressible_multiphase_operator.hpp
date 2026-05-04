#pragma once

#include <deal.II/fe/fe_system.h>

#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/flow/compressible_flow_convective_kernels.hpp>
#include <meltpooldg/flow/compressible_flow_types.hpp>
#include <meltpooldg/flow/compressible_flow_viscous_kernels.hpp>
#include <meltpooldg/flow/compressible_multiphase_level_set_advection.hpp>
#include <meltpooldg/phase_change/evaporation_model_knight.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::Multiphase
{
  /**
   * @brief Operator for the matrix-free of a compressible multiphase flow cutDG formulation.
   *
   * @tparam dim Dimension of the considered simulation case.
   * @tparam number Floating point format type.
   * @tparam is_viscous_gas Indicates whether the gas phase is viscous.
   * @tparam is_viscous_liquid Indicates whether the liquid phase is viscous.
   */
  template <int dim, typename number, bool is_viscous_gas = true, bool is_viscous_liquid = true>
  class CompressibleMultiphaseOperator
  {
  public:
    using VectorType            = dealii::LinearAlgebra::distributed::Vector<number>;
    using MappingInfoType       = CutUtil::MappingInfoType<dim, number>;
    using MappingInfoVectorType = CutUtil::MappingInfoVectorType<dim, number>;

    using ConservedVariablesType = CompressibleFlow::ConservedVariablesType<dim, number>;
    using ConservedVariablesGradType =
      CompressibleFlow::ConservedVariablesGradientType<dim, number>;

    /**
     * @brief Constructor.
     *
     * Initializes the operators, integrators, and mapping objects needed to compute
     * the matrix-free evaluation of a compressible multiphase cutDG formulation.
     *
     * @param multiphase_scratch_data Flow scratch data object holding all relevant compressible flow data
     * required by the operator.
     * @param mapping_info_interface_in dealii::NonMatching::MappingInfo object, provides the mapping
     * information computation and mapping data storage of the interface.
     * @param mapping_info_cells_in Vector of dealii::NonMatching::MappingInfo objects, provides
     * the mapping information computation and mapping data storage of the cells on the
     * inner subdomain and the outer subdomain, respectively.
     * @param mapping_info_faces_in Vector of dealii::NonMatching::MappingInfo objects, provides
     * the mapping information computation and mapping data storage of the faces on the
     * inner subdomain and the outer subdomain, respectively.
     */
    explicit CompressibleMultiphaseOperator(
      Flow::CompressibleMultiphaseScratchData<dim, number> &multiphase_scratch_data,
      const MappingInfoType                                &mapping_info_interface_in,
      const MappingInfoVectorType                          &mapping_info_cells_in,
      const MappingInfoVectorType                          &mapping_info_faces_in);

    /**
     * @brief Local applier for the cell integrals in the right-hand side evaluation.
     *
     * @param matrix_free Matrix-free object which contains all relevant data for matrix free evaluation.
     * @param dst Destination vector, in which the result is written.
     * @param src Source vector for the operator evaluation.
     * @param cell_range Considered cell range.
     */
    void
    local_apply_cell_rhs(const dealii::MatrixFree<dim, number>       &matrix_free,
                         VectorType                                  &dst,
                         const VectorType                            &src,
                         const std::pair<unsigned int, unsigned int> &cell_range) const;

   void
    local_apply_cell_p(const dealii::MatrixFree<dim, number>       &matrix_free,
                         VectorType                                  &dst,
                         const VectorType                            &src,
                         const std::pair<unsigned int, unsigned int> &cell_range) const;

 void
 p_postprocessing(
    VectorType       &dst,
   const VectorType &src) const;

   void
    local_apply_face_p(const dealii::MatrixFree<dim, number>       &matrix_free,
                         VectorType                                  &dst,
                         const VectorType                            &src,
                         const std::pair<unsigned int, unsigned int> &cell_range) const;

   void
    local_apply_boundary_face_p(const dealii::MatrixFree<dim, number>       &matrix_free,
                         VectorType                                  &dst,
                         const VectorType                            &src,
                         const std::pair<unsigned int, unsigned int> &cell_range) const;

  void
   output_at_quad_points(const dealii::MatrixFree<dim, number> &,
              VectorType &dst,
              const VectorType &src,
              const std::pair<unsigned int, unsigned int> &cell_range) const;

   void
  output_at_quad_points_face(const dealii::MatrixFree<dim, number> &,
             VectorType &dst,
             const VectorType &src,
             const std::pair<unsigned int, unsigned int> &face_range) const;

   void
  local_apply_dummy(const dealii::MatrixFree<dim, number> &,
             VectorType &dst,
             const VectorType &src,
             const std::pair<unsigned int, unsigned int> &face_range) const;

    /**
     * @brief Local applier for the face integrals in the right-hand side evaluation.
     *
     * @param matrix_free Matrix-free object which contains all relevant data for matrix free evaluation.
     * @param dst Destination vector, in which the result is written.
     * @param src Source vector for the operator evaluation.
     * @param face_range Considered face range.
     */
    void
    local_apply_face_rhs(const dealii::MatrixFree<dim, number>       &matrix_free,
                         VectorType                                  &dst,
                         const VectorType                            &src,
                         const std::pair<unsigned int, unsigned int> &face_range) const;

    /**
     * @brief Local applier for the boundary face integral in the right-hand side evaluation.
     *
     * @param matrix_free Matrix-free object which contains all relevant data for matrix free evaluation.
     * @param dst Destination vector, in which the result is written.
     * @param src Source vector for the operator evaluation.
     * @param face_range Considered face range.
     */
    void
    local_apply_boundary_face_rhs(const dealii::MatrixFree<dim, number>       &matrix_free,
                                  VectorType                                  &dst,
                                  const VectorType                            &src,
                                  const std::pair<unsigned int, unsigned int> &face_range) const;

    /**
     * @brief Local applier for the cell integrals in the left-hand side matrix-vector product evaluation.
     *
     * @param matrix_free Matrix-free object which contains all relevant data for matrix free evaluation.
     * @param dst Destination vector, in which the result is written.
     * @param src Source vector for the operator evaluation.
     * @param cell_range Considered cell range.
     */
    void
    local_apply_cell_lhs(const dealii::MatrixFree<dim, number>       &matrix_free,
                         VectorType                                  &dst,
                         const VectorType                            &src,
                         const std::pair<unsigned int, unsigned int> &cell_range) const;

    /**
     * @brief Local applier for the face integrals in the left-hand side matrix-vector product evaluation.
     *
     * @param matrix_free Matrix-free object which contains all relevant data for matrix free evaluation.
     * @param dst Destination vector, in which the result is written.
     * @param src Source vector for the operator evaluation.
     * @param face_range Considered face range.
     */
    void
    local_apply_face_lhs(const dealii::MatrixFree<dim, number>       &matrix_free,
                         VectorType                                  &dst,
                         const VectorType                            &src,
                         const std::pair<unsigned int, unsigned int> &face_range) const;

    /**
     * @brief Local applier for the boundary face integrals in the left-hand side matrix-vector product
     * evaluation.
     *
     * @param matrix_free Matrix-free object which contains all relevant data for matrix free evaluation.
     * @param dst Destination vector, in which the result is written.
     * @param src Source vector for the operator evaluation.
     * @param face_range Considered face range.
     */
    void
    local_apply_boundary_face_lhs(const dealii::MatrixFree<dim, number>       &matrix_free,
                                  VectorType                                  &dst,
                                  const VectorType                            &src,
                                  const std::pair<unsigned int, unsigned int> &face_range) const;

    /**
     * @brief Function for the matrix-free right-hand side vector evaluation.
     *
     * @param time Current simulation time.
     * @param time_step Current time step size.
     * @param dst Vector where the computed right-hand side rhs(src) is stored.
     * @param src The solution vector at the current time.
     */
    void
    create_rhs(const number     &time,
               const number     &time_step,
               VectorType       &dst,
               const VectorType &src) const;

    /**
     * @brief Function for the matrix-free matrix-vector product evaluation.
     *
     * @param dst Vector where the computed evaluated vector-matrix product is stored.
     * @param src The solution vector at the current time.
     */
    void
    vmult(VectorType &dst, const VectorType &src) const;

    /**
     * @brief Set the current time.
     *
     * @param current_time_in Current time.
     */
    void
    set_current_time(const number &current_time_in)
    {
      current_time = current_time_in;
    };

  private:
    /// Scratch data for multiphase case
    Flow::CompressibleMultiphaseScratchData<dim, number> &multiphase_scratch_data;

    /// Object for the convective term evaluations for the liquid phase
    const Flow::CompressibleFlowConvectiveKernels<dim, number> convective_terms_liquid;

    /// Object for the convective term evaluations for the gas phase
    const Flow::CompressibleFlowConvectiveKernels<dim, number> convective_terms_gas;

    /// Object for the viscous term evaluations for the liquid phase
    const Flow::CompressibleFlowViscousKernels<dim, number> viscous_terms_liquid;

    /// Object for the viscous term evaluations for the gas phase
    const Flow::CompressibleFlowViscousKernels<dim, number> viscous_terms_gas;

    /// Mapping information for integration over the phase interface
    const MappingInfoType &mapping_info_interface;

    /// Mapping information for integration over cut cells
    const MappingInfoVectorType &mapping_info_cells;

    /// Mapping information for integration over cut faces
    const MappingInfoVectorType &mapping_info_faces;

    /// FESystem object, required by FEPointEvaluation
    dealii::FESystem<dim> fe_point_temp;

    /// Number of DoFs per cell
    const unsigned int n_dofs_per_cell;

    /// Weighting factors for Nitsche-type viscous interface flux
    number visc_ave_weight_phase_liquid;
    number visc_ave_weight_phase_gas;

    /// Inverse time step size
    mutable number inv_time_step = 0.;

    /// Current time
    mutable number current_time = 0.;

    /// Object for the analytical advection of the level set field for 1D simulations
    mutable LevelSetAdvection<dim, number> level_set_advection_operator;

    /// Pointer to the object for the evaluation of the Knight evaporation model
    std::unique_ptr<Evaporation::EvaporationModelKnight<number>> evaporation_model_knight;

    /**
     * @brief Wrapper for the generation of a FECellIntegrator object.
     *
     * @param category Category of the considered cell range (liquid/intersected/gas).
     * @param offset Offset for the first selected component in a FESystem.
     *
     * @return Generated FECellIntegrator object.
     */
    FECellIntegrator<dim, dim + 2, number>
    create_cell_integrator(const CutUtil::CellCategory category,
                           const unsigned int          offset = 0) const
    {
      return FECellIntegrator<dim, dim + 2, number>(
        multiphase_scratch_data.scratch_data.get_matrix_free(),
        multiphase_scratch_data.dof_idx,
        multiphase_scratch_data.quad_idx,
        offset,
        category);
    }

    /**
     * @brief Wrapper for the generation of a FEFaceIntegrator object.
     *
     * @param is_inner_face This selects which of the two cells of an internal face the current
     * evaluator will be based upon. The interior face is the main face along which the normal
     * vectors are oriented.
     * @param category Category of the considered cell adjacent to the face
     * (liquid/intersected/gas).
     * @param offset Offset for the first selected component in a FESystem.
     *
     * @return Generated FEFaceIntegrator object.
     */
    FEFaceIntegrator<dim, dim + 2, number>
    create_face_integrator(const bool                  is_inner_face,
                           const CutUtil::CellCategory category,
                           const unsigned int          offset = 0) const
    {
      return FEFaceIntegrator<dim, dim + 2, number>(
        multiphase_scratch_data.scratch_data.get_matrix_free(),
        is_inner_face,
        multiphase_scratch_data.dof_idx,
        multiphase_scratch_data.quad_idx,
        offset,
        category);
    };

    /**
     * @brief Wrapper for the generation of a pair of FEFaceIntegrator objects for the "interior"
     * face and the "exterior" face.
     *
     * The "interior" face is the main face along which the normal vectors are oriented.
     *
     * @param category Category of the considered cell adjacent to the face
     * (liquid/intersected/gas).
     * @param offset Offset for the first selected component in a FESystem.
     *
     * @return Pair of FEFaceIntegrator objects. The first one corresponds to the "interior" face
     * and the second one to the "exterior" face.
     */
    std::pair<FEFaceIntegrator<dim, dim + 2, number>, FEFaceIntegrator<dim, dim + 2, number>>
    create_face_integrators(const CutUtil::CellCategory category,
                            const unsigned int          offset = 0) const
    {
      return {create_face_integrator(true, category, offset),
              create_face_integrator(false, category, offset)};
    };
  };
} // namespace MeltPoolDG::Multiphase
