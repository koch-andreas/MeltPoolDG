#pragma once

#include <deal.II/base/function.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/core/scratch_data.hpp>

#include <functional>
#include <memory>
#include <utility>

namespace MeltPoolDG::LevelSet
{
  /**
   * @brief Matrix-free upwind DG discretization of the level-set advection equation in
   * non-conservative form on the narrow band.
   *
   * The level-set field lives on the band DoFHandler (see NarrowBand::ActiveFEIndex), i.e., on
   * the cut cells and their face neighbors $\mathcal{T}^B$. Since the (extended) advection
   * velocity $\boldsymbol{u}_h$ is in general not divergence-free, the non-conservative form
   * $\partial_t \phi + \boldsymbol{u}_h \cdot \nabla \phi = 0$ is discretized with the upwind DG
   * scheme: find $\phi_h(t) \in V^p_\mathrm{DG}(\mathcal{T}^B)$ such that for all
   * $v \in V^p_\mathrm{DG}(\mathcal{T}^B)$
   *
   * $\sum_{K \in \mathcal{T}^B} \left[ (\partial_t \phi_h, v)_K
   *    + (\boldsymbol{u}_h \cdot \nabla \phi_h, v)_K
   *    - \int_{\partial K^-} (\boldsymbol{u}_h \cdot \boldsymbol{n}_K)
   *      (\phi_h^\mathrm{int} - \phi_h^\mathrm{ext}) v \, \mathrm{d}s \right] = 0$,
   *
   * where $\partial K^- = \{\boldsymbol{x} \in \partial K: \boldsymbol{u}_h \cdot
   * \boldsymbol{n}_K < 0\}$ is the inflow boundary of $K$. On the outer boundary of the band and
   * on the domain boundary, no exterior data is available and $\phi_h^\mathrm{ext} =
   * \phi_h^\mathrm{int}$ is used, i.e., these faces do not contribute.
   *
   * Optionally, an artificial viscosity term $\nabla \cdot (\epsilon \nabla \phi)$ with a
   * cell-wise constant viscosity $\epsilon_K \geq 0$ is added (see
   * compute_artificial_viscosity()), which is discretized by the symmetric interior penalty
   * method on the faces between two band cells, i.e., the residual contains
   *
   * $- \sum_K (\epsilon \nabla \phi_h, \nabla v)_K + \sum_F \int_F \left( \{\epsilon \nabla
   * \phi_h\} \cdot \boldsymbol{n} [v] + \{\epsilon \nabla v\} \cdot \boldsymbol{n} [\phi_h] -
   * \sigma_F [\phi_h][v] \right) \mathrm{d}s$
   *
   * with $\sigma_F = (p+1)(p+2) \max(\epsilon^-, \epsilon^+) / h_\mathrm{min}$. At the outer
   * boundary of the band and on the domain boundary, the natural boundary condition
   * $\epsilon \nabla \phi \cdot \boldsymbol{n} = 0$ is used.
   *
   * The operator evaluates $M^{-1} r(\phi_h)$ with the block-diagonal mass matrix $M$ and the
   * residual $r(\phi_h) = -\sum_K (\boldsymbol{u}_h \cdot \nabla \phi_h, v)_K + \sum_K
   * \int_{\partial K^-} \ldots$, which is the right-hand side of the semi-discrete system for
   * explicit time integration.
   *
   * @note The quadrature rule of @p quad_idx needs to be a Gauss rule with $p+1$ points per
   * direction, which is required by the cell-wise inverse mass matrix.
   */
  template <int dim, typename number>
  class NarrowBandAdvectionDGOperator
  {
  public:
    using VectorType          = dealii::LinearAlgebra::distributed::Vector<number>;
    using VectorizedArrayType = dealii::VectorizedArray<number>;

    /**
     * @brief Constructor.
     *
     * @param scratch_data Scratch data.
     * @param band_dof_idx Index of the band DoFHandler in @p scratch_data.
     * @param quad_idx Index of the quadrature rule in @p scratch_data.
     */
    NarrowBandAdvectionDGOperator(const ScratchData<dim, dim, number> &scratch_data,
                                  const unsigned int                   band_dof_idx,
                                  const unsigned int                   quad_idx);

    /**
     * @brief Set the advection velocity as (time-dependent) function.
     */
    void
    set_velocity_function(const std::shared_ptr<dealii::Function<dim>> &velocity_function);

    /**
     * @brief Advect the level set with the extension velocity u(y(x)), i.e., the velocity function
     * at the closest point y(x) = x - phi(x) grad phi(x) / |grad phi(x)|^2 on the zero level set
     * (exact for a signed distance function), instead of u(x).
     *
     * The closest point is computed from the level set to which the operator is applied, i.e.,
     * from the current stage of the time integration, and the operator becomes nonlinear. On the
     * faces, the averages of the traces of phi and grad phi are used, such that the upwind flux
     * sees a single velocity. |grad phi| is bounded from below by 0.5 in the distance estimate
     * phi / |grad phi| (e.g., at the ridge of a thin filament), and y = x for grad phi = 0. The
     * extension velocity is constant along the normals of the interface: the level set of the band
     * remains a signed distance, while the interface moves with the velocity function.
     */
    void
    set_extension_velocity(const bool flag);

    /**
     * @brief Evaluate the right-hand side $M^{-1} r(\phi_h)$ of the semi-discrete system.
     *
     * The signature is compatible with the explicit time integrators of MeltPoolDG.
     *
     * @param time Time at which the velocity is evaluated.
     * @param dst Result vector.
     * @param src Level-set field.
     * @param post Function, which is applied to all locally owned DoF ranges of @p dst after the
     * evaluation.
     */
    void
    apply_operator(const number                                           time,
                   VectorType                                            &dst,
                   const VectorType                                      &src,
                   const std::function<void(unsigned int, unsigned int)> &post = {}) const;

    /**
     * @brief Statistics of the artificial viscosity.
     */
    struct ArtificialViscosityStatistics
    {
      /// Number of cells with a positive viscosity (summed over all processes).
      unsigned int n_cells = 0;

      /// Maximum diffusion number $\epsilon_K \Delta t / h_K^2$ (over all processes).
      number max_diffusion_number = 0.;
    };

    /**
     * @brief Compute the cell-wise constant artificial viscosity from the modal smoothness
     * indicator (see NarrowBand::ModalSmoothnessIndicator and
     * NarrowBandData::ArtificialViscosity) of the level set.
     *
     * @param level_set Level-set field on the band.
     * @param viscosity Viscosity field on the band (constant per cell, ghost values are updated).
     * @param time Time at which the velocity is evaluated.
     * @param time_step Time-step size (statistics).
     * @param factor Factor c of the maximum viscosity c h max|u| / p.
     * @param threshold Threshold s_0 of the indicator log10(E_p / E_var).
     * @param width Half width kappa of the transition around the threshold.
     * @param only_non_cut_cells Only cells, which are not cut, get a viscosity.
     */
    ArtificialViscosityStatistics
    compute_artificial_viscosity(const VectorType &level_set,
                                 VectorType       &viscosity,
                                 const number      time,
                                 const number      time_step,
                                 const number      factor,
                                 const number      threshold,
                                 const number      width,
                                 const bool        only_non_cut_cells) const;

    /**
     * @brief Set the artificial viscosity field (see compute_artificial_viscosity()), which is used
     * by apply_operator() with updated ghost values. A null pointer disables the viscosity.
     */
    void
    set_artificial_viscosity(const VectorType *viscosity);

    /**
     * @brief Compute the CFL number $\Delta t (2p+1) \max |\boldsymbol{u}_h| / h$ over the
     * locally owned cells of the narrow band, where $\max |\boldsymbol{u}_h|$ is taken over the
     * quadrature points and $h$ is the cell size.
     *
     * @param time Time at which the velocity is evaluated.
     * @param time_step Time step size.
     */
    number
    compute_cfl_number(const number time, const number time_step) const;

  private:
    /**
     * @brief Check whether an active FE index belongs to the narrow band, i.e., holds level-set
     * DoFs.
     */
    static bool
    is_band_fe_index(const unsigned int active_fe_index);

    void
    local_apply_cell(const dealii::MatrixFree<dim, number>       &matrix_free,
                     VectorType                                  &dst,
                     const VectorType                            &src,
                     const std::pair<unsigned int, unsigned int> &cell_range) const;

    void
    local_apply_face(const dealii::MatrixFree<dim, number>       &matrix_free,
                     VectorType                                  &dst,
                     const VectorType                            &src,
                     const std::pair<unsigned int, unsigned int> &face_range) const;

    void
    local_apply_boundary_face(const dealii::MatrixFree<dim, number> &,
                              VectorType &,
                              const VectorType &,
                              const std::pair<unsigned int, unsigned int> &) const
    {
      // phi_ext = phi_int on the domain boundary: no contribution
    }

    void
    local_apply_inverse_mass_matrix(const dealii::MatrixFree<dim, number>       &matrix_free,
                                    VectorType                                  &dst,
                                    const VectorType                            &src,
                                    const std::pair<unsigned int, unsigned int> &cell_range) const;

    /// Scratch data.
    const ScratchData<dim, dim, number> &scratch_data;

    /// Index of the band DoFHandler.
    const unsigned int band_dof_idx;

    /// Index of the quadrature rule.
    const unsigned int quad_idx;

    /// Advection velocity.
    std::shared_ptr<dealii::Function<dim>> velocity_function;

    /// Use the extension velocity u(y(x)) (see set_extension_velocity()).
    bool use_extension_velocity = false;

    /**
     * @brief Velocity at the points @p point: the velocity function at @p point or, for the
     * extension velocity, at the closest point computed from the level set @p value and its
     * gradient @p gradient.
     */
    dealii::Tensor<1, dim, VectorizedArrayType>
    evaluate_velocity(const dealii::Point<dim, VectorizedArrayType>     &point,
                      const VectorizedArrayType                         &value,
                      const dealii::Tensor<1, dim, VectorizedArrayType> &gradient) const;

    /// Artificial viscosity field (constant per cell); nullptr if disabled.
    const VectorType *artificial_viscosity = nullptr;
  };
} // namespace MeltPoolDG::LevelSet
