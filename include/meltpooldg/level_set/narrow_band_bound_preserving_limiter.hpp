#pragma once

#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/level_set/narrow_band_data.hpp>

#include <array>
#include <vector>

namespace MeltPoolDG::LevelSet
{
  /**
   * @brief Bound-preserving limiter for the advection of the narrow-band DG level set (Zhang &
   * Shu, 2010, https://doi.org/10.1016/j.jcp.2009.12.030).
   *
   * The exact solution of the advection equation preserves the values of the level set along the
   * characteristics, i.e., no new local extrema and no new zero crossings arise. The upwind DG
   * discretization with $p \geq 1$ violates this property near under-resolved features (e.g., the
   * tip of a thin filament): the over- and undershoots may create spurious zero crossings, i.e.,
   * spurious interfaces.
   *
   * At the beginning of a time step, compute_bounds() determines for each limited cell $K$ the
   * minimum $m_K$ and maximum $M_K$ of the level set $\phi^n$ on an approximation of the domain of
   * dependence of $K$ for the time step: $K$ and either all its face neighbors in the band, the
   * face neighbors across inflow faces ($\boldsymbol{u} \cdot \boldsymbol{n}_K < 0$ at any
   * Gauss-Lobatto point of the face), or the strips of width $|\boldsymbol{u} \cdot
   * \boldsymbol{n}_K| \Delta t$ of these neighbors along the inflow faces (the exact domain of
   * dependence for a velocity normal to the face), which are extended by $|\boldsymbol{u}| \Delta
   * t$ beyond the edges of the face (polynomial extrapolation of the neighbor towards the diagonal
   * neighbors, which are passed by the characteristics of oblique flows). The bounds $[a_K, b_K]$
   * are
   * - the local maximum principle: $[m_K - r h_K, M_K + r h_K]$ with the relaxation $r$; cells with
   *   an inflow face to an inactive cell or at the boundary of the domain are not limited, since
   *   their domain of dependence is unknown,
   * - the sign preservation: $[\min(m_K, \epsilon h_K), \infty)$ if $m_K > 0$ and $(-\infty,
   *   \max(M_K, -\epsilon h_K)]$ if $M_K < 0$; cells with values of both signs in their domain of
   *   dependence are not limited. Inactive cells and the boundary of the domain do not contribute
   *   to $m_K$ and $M_K$, since the inactive cells are not cut, i.e., they have the sign of the
   *   adjacent band cell.
   *
   * The sign preservation only acts, if the advection would create a zero crossing in a cell,
   * which is not reached by the interface within the time step. In contrast, the local maximum
   * principle also clips smooth extrema of the level set (e.g., the ridge of a thin filament or the
   * center of a small circle within the band), which deteriorates the accuracy.
   *
   * After each stage of the SSP Runge-Kutta scheme, limit() scales the level set of each limited
   * cell around its cell mean $\bar{\phi}_K$,
   *
   * $\phi \leftarrow \bar{\phi}_K + \theta (\phi - \bar{\phi}_K)$, $\theta = \min\left(1,
   * \frac{b_K - \bar{\phi}_K}{\max_K \phi - \bar{\phi}_K}, \frac{\bar{\phi}_K - a_K}{\bar{\phi}_K
   * - \min_K \phi}\right)$,
   *
   * where the minimum and maximum are evaluated at the Gauss-Lobatto points with $p+2$ points per
   * direction. The cell mean is preserved. If the cell mean itself violates the bounds (e.g., for
   * time steps above the CFL condition of the limiter), the cell is not modified.
   *
   * @note The limiter requires nodal elements (FE_DGQ) of the same degree on all cells of the
   * band and parallelepiped cells.
   */
  template <int dim, typename number>
  class NarrowBandBoundPreservingLimiter
  {
  public:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    /**
     * @brief Statistics of the limiter.
     */
    struct Statistics
    {
      /// Number of cells, whose level set was scaled.
      unsigned int n_limited_cells = 0;

      /// Number of cells, whose cell mean violates the bounds (not modified).
      unsigned int n_mean_violations = 0;

      Statistics &
      operator+=(const Statistics &other)
      {
        n_limited_cells += other.n_limited_cells;
        n_mean_violations += other.n_mean_violations;
        return *this;
      }
    };

    /**
     * @brief Constructor.
     *
     * @param data Parameters of the limiter.
     * @param mapping Mapping.
     */
    NarrowBandBoundPreservingLimiter(const typename NarrowBandData<number>::AdvectionLimiter &data,
                                     const dealii::Mapping<dim> &mapping);

    /**
     * @brief Determine the limited cells and their bounds from the level set at the beginning of
     * the time step.
     *
     * @param dof_handler Band DoFHandler.
     * @param level_set Level-set vector of the band DoFHandler.
     * @param velocity Advection velocity (dim components).
     * @param time Time at which the velocity is evaluated.
     * @param time_step Time-step size (width of the upwind strips).
     *
     * @return Number of limited cells, i.e., cells with bounds (summed over all processes).
     */
    unsigned int
    compute_bounds(const dealii::DoFHandler<dim> &dof_handler,
                   const VectorType              &level_set,
                   dealii::Function<dim>         &velocity,
                   const number                   time,
                   const number                   time_step);

    /**
     * @brief Limit the level set of the limited cells to their bounds.
     *
     * @param dof_handler Band DoFHandler (the same as in compute_bounds()).
     * @param level_set Level-set vector of the band DoFHandler.
     *
     * @return Statistics (summed over all processes).
     */
    Statistics
    limit(const dealii::DoFHandler<dim> &dof_handler, VectorType &level_set) const;

  private:
    /**
     * @brief Values of the level set of a cell at the sample points (Gauss-Lobatto points).
     */
    void
    evaluate_at_sample_points(const dealii::Vector<number> &dof_values,
                              std::vector<number>          &sample_values) const;

    /// Parameters of the limiter.
    const typename NarrowBandData<number>::AdvectionLimiter &data;

    /// Mapping.
    const dealii::Mapping<dim> &mapping;

    /// Values of the shape functions at the sample points (rows: sample points).
    dealii::FullMatrix<number> sample_shape_values;

    /// Bounds [a_K, b_K] of the locally owned limited cells, indexed by the active cell index.
    std::vector<std::array<number, 2>> bounds;

    /// Flags of the locally owned limited cells, indexed by the active cell index.
    std::vector<bool> is_limited_cell;
  };
} // namespace MeltPoolDG::LevelSet
