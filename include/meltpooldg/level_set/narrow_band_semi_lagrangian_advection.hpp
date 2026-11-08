#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <vector>

namespace MeltPoolDG::LevelSet
{
  /**
   * @brief Semi-Lagrangian advection of a DG level-set field, which is only defined on the cut
   * cells.
   *
   * The level set $\phi_h^n$ at $t^n$ is only given on the cut cells $\mathcal{T}^{\Gamma,n}$ (cut
   * DoFHandler). The level set $\phi_h^{n+1}$ at $t^{n+1} = t^n + \Delta t$ is evaluated on the
   * candidate cells (band DoFHandler: the cut cells and their vertex neighbors, which contain the
   * interface at $t^{n+1}$ if it moves less than one cell per time step) by nodal interpolation of
   * the back-traced field,
   *
   * $\phi_h^{n+1}(\boldsymbol{x}_i) = \phi^n(\boldsymbol{X}(t^n; \boldsymbol{x}_i, t^{n+1}))$,
   *
   * where the foot point $\boldsymbol{x}_i^* = \boldsymbol{X}(t^n; \boldsymbol{x}_i, t^{n+1})$ of
   * the characteristic through the node $\boldsymbol{x}_i$ is computed by the classical
   * fourth-order Runge-Kutta method backward in time. The old level set at the foot point is
   * - the polynomial of the cut cell containing $\boldsymbol{x}_i^*$, or
   * - if $\boldsymbol{x}_i^*$ is located in a cell $K$, which is not cut, the scaled distance to
   *   the discrete interface $\Gamma_h^n$ of the cut cells with the sign $s_K$ of $K$:
   *   $s_K |\nabla \phi_h(\boldsymbol{y})| d$, where $\boldsymbol{y}$ is the closest point on
   *   $\Gamma_h^n$ and $d = |\boldsymbol{x}_i^* - \boldsymbol{y}| + |\phi_h(\boldsymbol{y})| /
   *   |\nabla \phi_h(\boldsymbol{y})|$ the estimate of the distance (for a converged closest
   *   point, this is the magnitude of the first-order Taylor expansion from $\boldsymbol{y}$).
   * Since $K$ is not cut, $\phi^n$ has a single sign on $K$. The signs of the cells, which are not
   * cut, are maintained by the caller: a cell can only change its sign, if the interface passes
   * through it, i.e., if it is a candidate cell.
   *
   * Thus, no level set outside the cut cells is stored or extended. Only the material that enters
   * the candidate cells from the cut cells carries the interface; the level set in the other parts
   * of the candidate cells is a signed distance of the correct sign. In particular, open ends of
   * the discrete interface (e.g., at an under-resolved tip) are not prolonged by polynomial
   * extrapolation, but only move with the flow.
   *
   * The cut cells of all processes are gathered on all processes (the interface is small compared
   * to the domain).
   *
   * @note The semi-Lagrangian advection requires parallelepiped cells, nodal tensor-product
   * elements (FE_DGQ) and a triangulation without hanging nodes. The foot points of the nodes of a
   * candidate cell need to be located in the cell or its vertex neighbors (displacement below one
   * cell per time step); otherwise, the sign of the Taylor expansion at the foot point is used.
   */
  template <int dim, typename number>
  class NarrowBandSemiLagrangianAdvection
  {
  public:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    /**
     * @brief Statistics of a time step (summed or maximized over all processes).
     */
    struct Statistics
    {
      /// Number of nodes.
      unsigned int n_nodes = 0;

      /// Number of nodes, whose foot point is located outside of the cut cells.
      unsigned int n_nodes_outside = 0;

      /// Number of nodes, whose foot point could not be located in the vertex neighborhood of the
      /// cell.
      unsigned int n_nodes_not_located = 0;

      /// Maximum distance between a node and its foot point relative to the cell size.
      number max_displacement = 0.;
    };

    /**
     * @brief Constructor.
     *
     * @param pcout Output stream for the statistics.
     */
    explicit NarrowBandSemiLagrangianAdvection(const dealii::ConditionalOStream &pcout);

    /**
     * @brief Evaluate the level set at $t^{n+1}$ on the candidate cells.
     *
     * @param dof_handler_cut Cut DoFHandler (FE_DGQ on the cut cells at $t^n$).
     * @param level_set_cut Level set at $t^n$ on the cut cells.
     * @param dof_handler_band Band DoFHandler of the candidate cells (all cells with DoFs).
     * @param level_set_band Level set at $t^{n+1}$ on the candidate cells (output).
     * @param cell_signs Signs (+1 or -1) of the level set on the locally owned and ghost cells,
     * which are not cut at $t^n$, indexed by the active cell index (see
     * NarrowBandLevelSetOperation).
     * @param velocity Advection velocity (dim components). Its time is restored on exit.
     * @param time Time $t^n$.
     * @param time_step Time-step size $\Delta t$.
     */
    Statistics
    advect(const dealii::DoFHandler<dim> &dof_handler_cut,
           const VectorType              &level_set_cut,
           const dealii::DoFHandler<dim> &dof_handler_band,
           VectorType                    &level_set_band,
           const std::vector<int>        &cell_signs,
           dealii::Function<dim>         &velocity,
           const number                   time,
           const number                   time_step) const;

  private:
    /// Output stream.
    const dealii::ConditionalOStream pcout;
  };
} // namespace MeltPoolDG::LevelSet
