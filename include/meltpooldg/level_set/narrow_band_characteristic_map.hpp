#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <array>
#include <set>
#include <vector>

namespace MeltPoolDG::LevelSet
{
  /**
   * @brief Characteristic map of the sub-cycling of the semi-Lagrangian narrow band: displacement
   * of the nodes since the reference time $t_r$ as a continuous field on the candidate cells of
   * the reference.
   *
   * With sub-cycling (NarrowBandData::level_set_update_cfl), the level set at $t^{n+1}$ is
   * evaluated from the reference level set $\phi_r$ at the foot points of the nodes at $t_r$,
   * $\phi_h^{n+1}(\boldsymbol{x}_i) = \phi_r(\boldsymbol{X}(t_r; \boldsymbol{x}_i, t^{n+1}))$.
   * Instead of integrating the characteristics backward over all time steps since the reference,
   * which requires the velocity at all these times, the backward map is stored as the
   * displacement
   *
   * $\boldsymbol{D}^n(\boldsymbol{x}) = \boldsymbol{x} - \boldsymbol{X}(t_r; \boldsymbol{x},
   * t^n)$,
   *
   * a continuous FE_Q field (dim scalar components) on the candidate cells of the reference
   * (the cut cells of the reference and their vertex neighbors). The map is the identity at the
   * reference ($\boldsymbol{D} = 0$, see reset()) and is composed in each time step with the
   * backward characteristic of the time step (advance()):
   *
   * $\boldsymbol{D}^{n+1}(\boldsymbol{x}_j) = \boldsymbol{x}_j - \boldsymbol{y}_j +
   * \boldsymbol{D}^n_h(\boldsymbol{y}_j), \quad \boldsymbol{y}_j = \boldsymbol{X}(t^n;
   * \boldsymbol{x}_j, t^{n+1})$
   *
   * for each node $\boldsymbol{x}_j$ of the continuous field (classical fourth-order Runge-Kutta
   * step backward in time, which only evaluates the velocity in $[t^n, t^{n+1}]$). Since
   * $\boldsymbol{y}_j$ is close to the node $\boldsymbol{x}_j$, where the interpolation error of
   * $\boldsymbol{D}^n_h$ vanishes, the interpolation error of a time step is proportional to the
   * displacement $|\boldsymbol{x}_j - \boldsymbol{y}_j|$ of the time step. Thus, the
   * interpolation errors do not accumulate with the number of time steps, but with the
   * displacement since the reference.
   *
   * Each node is updated once (by the process owning it), such that the field and the foot points
   * $\boldsymbol{x} - \boldsymbol{D}_h(\boldsymbol{x})$ of the nodes of the candidate cells
   * (foot_point()) are the same for all cells sharing a node.
   *
   * The point $\boldsymbol{y}_j$ is located in the cell of the node or its vertex neighbors (the
   * map is only kept for a displacement below one cell since the reference). If it lies outside
   * of the candidate cells of the reference, $\boldsymbol{D}^n_h$ is extrapolated from the closest
   * candidate cell in this neighborhood (smooth displacement).
   *
   * @note The map requires parallelepiped cells (see NarrowBandSemiLagrangianAdvection).
   */
  template <int dim, typename number>
  class NarrowBandCharacteristicMap
  {
  public:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    /**
     * @brief Statistics of a time step (summed or maximized over all processes).
     */
    struct Statistics
    {
      /// Number of nodes of the continuous field.
      unsigned int n_nodes = 0;

      /// Number of nodes, whose foot point of the time step is located outside of the candidate
      /// cells of the reference (extrapolation of the displacement).
      unsigned int n_nodes_extrapolated = 0;

      /// Maximum displacement of the time step relative to the cell size.
      number max_step_displacement = 0.;

      /// Maximum displacement since the reference relative to the cell size.
      number max_displacement = 0.;
    };

    /**
     * @brief Constructor.
     *
     * @param triangulation Triangulation of the narrow band.
     * @param fe_degree Polynomial degree of the continuous field (degree of the level set).
     * @param pcout Output stream for the statistics.
     */
    NarrowBandCharacteristicMap(const dealii::Triangulation<dim> &triangulation,
                                const unsigned int                fe_degree,
                                const dealii::ConditionalOStream &pcout);

    /**
     * @brief Identity map ($\boldsymbol{D} = 0$) on the cells with DoFs of @p dof_handler_band
     * (candidate cells of the reference).
     */
    void
    reset(const dealii::DoFHandler<dim> &dof_handler_band);

    /**
     * @brief Composition of the map at $t^n$ with the backward characteristic of the time step
     * from $t^{n+1} = t^n + \Delta t$ to $t^n$.
     *
     * @param velocity Advection velocity (dim components), evaluated in $[t^n, t^{n+1}]$. Its time
     * is restored on exit.
     * @param time Time $t^n$.
     * @param time_step Time-step size $\Delta t$.
     */
    Statistics
    advance(dealii::Function<dim> &velocity, const number time, const number time_step);

    /**
     * @brief Foot point $\boldsymbol{x} - \boldsymbol{D}_h(\boldsymbol{x})$ at the time of the
     * reference of the point with the unit coordinates @p unit_point in a cell with DoFs (locally
     * owned or ghost cell).
     */
    dealii::Point<dim>
    foot_point(const typename dealii::Triangulation<dim>::cell_iterator &cell,
               const dealii::Point<dim>                                 &unit_point) const;

    /**
     * @brief Displacement $\boldsymbol{D}_h$ at the point with the unit coordinates
     * @p unit_point in a cell with DoFs (locally owned or ghost cell).
     */
    dealii::Tensor<1, dim>
    displacement(const typename dealii::Triangulation<dim>::cell_iterator &cell,
                 const dealii::Point<dim>                                 &unit_point) const;

    /**
     * @brief Restart: DoF values of the displacement on a locally owned cell (all components,
     * component by component; empty for cells without DoFs).
     */
    std::vector<number>
    get_cell_values(const typename dealii::Triangulation<dim>::cell_iterator &cell) const;

    /**
     * @brief Restart: set the DoF values of the displacement on a locally owned cell (see
     * get_cell_values()). After all cells are set, finish_set_cell_values() has to be called.
     */
    void
    set_cell_values(const typename dealii::Triangulation<dim>::cell_iterator &cell,
                    const std::vector<number>                                &values);

    /**
     * @brief Restart: update the ghost values after set_cell_values().
     */
    void
    finish_set_cell_values();

    /**
     * @brief DoFHandler of the continuous field (FE_Q on the candidate cells of the reference).
     */
    const dealii::DoFHandler<dim> &
    get_dof_handler() const;

    /**
     * @brief Components of the displacement (with ghost values).
     */
    const std::array<VectorType, dim> &
    get_displacement() const;

  private:
    /**
     * @brief Displacement $\boldsymbol{D}_h$ at the point @p point, which is located in the cell
     * @p cell (a cell with DoFs) or one of its vertex neighbors; if it is not located in a cell
     * with DoFs, the displacement is extrapolated from the closest cell with DoFs among them.
     *
     * @param extrapolated Set to true, if the displacement is extrapolated.
     */
    dealii::Tensor<1, dim>
    evaluate_in_neighborhood(const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
                             const dealii::Point<dim>                                     &point,
                             bool &extrapolated) const;

    /// Output stream.
    const dealii::ConditionalOStream pcout;

    /// FE_Q on the candidate cells of the reference, FE_Nothing (not dominating) otherwise.
    const dealii::hp::FECollection<dim> fe_collection;

    /// DoFHandler of the continuous field.
    dealii::DoFHandler<dim> dof_handler;

    /// Components of the displacement (locally owned and locally relevant DoFs).
    std::array<VectorType, dim> displacement_values;

    /// Cells sharing a vertex, indexed by the vertex index.
    std::vector<std::set<typename dealii::Triangulation<dim>::active_cell_iterator>>
      vertex_to_cells;

    /// Whether reset() has been called.
    bool is_initialized = false;
  };
} // namespace MeltPoolDG::LevelSet
