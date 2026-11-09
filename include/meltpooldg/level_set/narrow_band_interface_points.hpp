#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/non_matching/immersed_surface_quadrature.h>

#include <functional>
#include <vector>

namespace MeltPoolDG::LevelSet
{
  /**
   * @brief Points of the interface, which are advected with the velocity (front tracking), for the
   * interface penalty term of the reinitialization of the narrow-band level set.
   *
   * The points are seeded at the quadrature points of the zero level set $\Gamma_h$ of the
   * initial level set on the cut cells (seed()), with the weights and normal vectors of the
   * surface quadrature in real space. Each time step, they are moved along the characteristics
   * $\mathrm{d}\boldsymbol{x} / \mathrm{d}t = \boldsymbol{u}(\boldsymbol{x}, t)$ from $t^n$ to
   * $t^{n+1}$ by the classical fourth-order Runge-Kutta method (advect()), i.e., independently of
   * the interpolation of the advected level set. After the advection, every point is assigned to
   * the cell of the band that contains it (locate()), and the point set is repaired locally:
   * - points outside of the band are removed,
   * - points, which are not supported by the advected level set $\tilde\phi$, are removed:
   *   $|\tilde\phi(\boldsymbol{x})| > c_s h$ (support tolerance $c_s$, default 0.2) or
   *   $\boldsymbol{n} \cdot \nabla \tilde\phi(\boldsymbol{x}) \leq 0$ (e.g., parts of the
   *   interface, which are not resolved by the level set and have been lost by it, such as the
   *   tip of a thin filament); thus, the points do not reintroduce parts of the interface, which
   *   the level set does not contain,
   * - the quadrature points of the zero level set of the advected level set, whose distance to
   *   all points exceeds $h / n$ ($n$ points per direction of the surface quadrature), are added
   *   as new points (gaps of the point set due to the stretching of the interface),
   * - in cells with more than $2 n^{d-1}$ points ($n$ points per direction of the surface
   *   quadrature), the closest pairs of points with the same orientation are merged by removing
   *   one point and adding its weight to the other one (compression of the interface).
   * The reinitialization then enforces $\phi_h = 0$ weakly at the points
   * (get_surface_quadrature()) instead of at the zero level set of the advected level set in the
   * cut cells containing usable points (see below). The cut cells are still determined from the
   * advected level set, and points in cells, which are not cut, are not used: the points only
   * correct the position of the zero level set, but never create new parts of it. Since the
   * points are not reseeded from the reinitialized level set, the fitting errors of the
   * reinitialization do not accumulate in the positions of the points.
   *
   * The weights and the normal vectors of the points are advected together with the points
   * (Nanson's formula for the weighted normal vector $\boldsymbol{m} = w \boldsymbol{n}$:
   * $\mathrm{d}\boldsymbol{m} / \mathrm{d}t = (\nabla \cdot \boldsymbol{u}) \boldsymbol{m} -
   * (\nabla \boldsymbol{u})^T \boldsymbol{m}$, where the velocity gradient is approximated by
   * central differences), i.e., the weights are the area elements of the advected surface. A point
   * is not used in the surface quadrature, if a point of opposite orientation
   * ($\boldsymbol{n}_i \cdot \boldsymbol{n}_j < 0$) is closer than $c_u h$ (unresolved distance
   * $c_u$, default 2), since the interface is not resolved there (e.g., both sides of a thin
   * filament or a sharp tip). In cut cells without usable points, the reinitialization uses the
   * zero level set of the advected level set as before. The points are kept and used again, once
   * they are separated by the flow. Only points with the same orientation are merged.
   *
   * The points of all processes are gathered on all processes (as the cut cells in the
   * semi-Lagrangian advection), and every process advects all points. A point on the common
   * boundary of several cells is assigned to the cell with the smallest CellId.
   *
   * @note Requires parallelepiped cells and a triangulation without hanging nodes (as the
   * semi-Lagrangian advection).
   */
  template <int dim, typename number>
  class NarrowBandInterfacePoints
  {
  public:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    /**
     * @brief Statistics of the assignment of the points to the cells (summed over all processes).
     */
    struct Statistics
    {
      /// Number of points (after the repair of the point set).
      unsigned int n_points = 0;

      /// Number of points, which are not located in a cell of the band (removed).
      unsigned int n_points_outside = 0;

      /// Number of points, which are not supported by the advected level set (removed).
      unsigned int n_unsupported_points = 0;

      /// Number of new points seeded in the gaps of the point set.
      unsigned int n_inserted_points = 0;

      /// Number of points removed by merging in cells with too many points.
      unsigned int n_merged_points = 0;

      /// Number of usable points in cells, which are not cut by the zero level set of the
      /// advected level set (not used).
      unsigned int n_points_in_non_cut_cells = 0;

      /// Number of cut cells without usable points (zero level set used).
      unsigned int n_cut_cells_without_points = 0;

      /// Number of points with a point of opposite orientation (n_i . n_j < 0) closer than the
      /// unresolved distance, which are not used in the surface quadrature.
      unsigned int n_unresolved_points = 0;
    };

    /**
     * @brief Constructor.
     *
     * @param pcout Output stream for the statistics.
     * @param unresolved_distance Distance relative to the cell size, below which points of
     * opposite orientation are not used (unresolved interface).
     * @param support_tolerance Maximum magnitude of the advected level set at a supported point
     * relative to the cell size.
     */
    explicit NarrowBandInterfacePoints(const dealii::ConditionalOStream &pcout,
                                       const double                      unresolved_distance = 2.0,
                                       const double                      support_tolerance   = 0.2);

    /**
     * @brief Seed the points on the zero level set of a level set (all previous points are
     * removed).
     *
     * @param dof_handler DoFHandler of the level set (FE_DGQ on the cells with DoFs).
     * @param level_set Level set.
     * @param n_points_1d Number of Gauss points per direction of the surface quadrature.
     */
    void
    seed(const dealii::DoFHandler<dim> &dof_handler,
         const VectorType              &level_set,
         const unsigned int             n_points_1d);

    /**
     * @brief Move the points along the characteristics from $t^n$ to $t^{n+1} = t^n + \Delta t$
     * (classical fourth-order Runge-Kutta method).
     *
     * @param velocity Advection velocity (dim components). Its time is restored on exit.
     * @param time Time $t^n$.
     * @param time_step Time-step size $\Delta t$.
     */
    void
    advect(dealii::Function<dim> &velocity, const number time, const number time_step);

    /**
     * @brief Assign the points to the cells of the band and repair the point set (removal of the
     * points outside of the band and of the points, which are not supported by the advected level
     * set, seeding in the gaps of the point set, merging in cells with too many points).
     *
     * @param dof_handler_band Band DoFHandler (all cells with DoFs).
     * @param level_set_band Advected level set on the band.
     * @param n_points_1d Number of Gauss points per direction of the surface quadrature of new
     * points.
     * @param is_cut_cell Cut cells of the advected level set, indexed by the active cell index
     * (locally owned cells).
     */
    Statistics
    locate(const dealii::DoFHandler<dim> &dof_handler_band,
           const VectorType              &level_set_band,
           const unsigned int             n_points_1d,
           const std::vector<bool>       &is_cut_cell);

    /**
     * @brief Return true, if the points have been assigned to the cells since the last seeding or
     * advection.
     */
    bool
    is_located() const;

    /**
     * @brief Surface quadrature (unit points, reference weights and reference normal vectors) of
     * the points assigned to a locally owned cell, such that the mapped weights JxW are the
     * weights of the points in real space.
     */
    dealii::NonMatching::ImmersedSurfaceQuadrature<dim>
    get_surface_quadrature(const typename dealii::Triangulation<dim>::cell_iterator &cell) const;

    /**
     * @brief Return the points (all processes).
     */
    const std::vector<dealii::Point<dim>> &
    get_points() const;

    /**
     * @brief Return the weights of the points in real space (all processes).
     */
    const std::vector<number> &
    get_weights() const;

  private:
    /// Output stream.
    const dealii::ConditionalOStream pcout;

    /// Positions of the points.
    std::vector<dealii::Point<dim>> points;

    /// Weights of the points in real space (area elements, advected with the velocity).
    std::vector<number> weights;

    /// Unit normal vectors in real space (advected with the velocity).
    std::vector<dealii::Tensor<1, dim>> normals;

    /**
     * @brief Append the quadrature points of the zero level set of a locally owned cell (positions,
     * weights and normal vectors in real space) to @p data. If @p accept is given, only the points
     * accepted by it are appended.
     */
    template <typename CellIteratorType, typename QuadratureGeneratorType>
    static void
    pack_surface_points(const CellIteratorType                                &cell,
                        QuadratureGeneratorType                               &quadrature_generator,
                        std::vector<double>                                   &data,
                        const std::function<bool(const dealii::Point<dim> &)> &accept = {});

    /**
     * @brief Append the points packed by pack_surface_points() on all processes.
     */
    void
    append_points(const std::vector<double> &local_data, const MPI_Comm mpi_comm);

    /**
     * @brief Assign the points to the locally owned cells of the band (points_of_cell).
     */
    void
    assign_points_to_cells(const dealii::DoFHandler<dim> &dof_handler_band);

    /// Indices of the points assigned to the locally owned cells, indexed by the active cell
    /// index.
    std::vector<std::vector<unsigned int>> points_of_cell;

    /// Points of the locally owned cells with a point of opposite orientation closer than the
    /// unresolved distance (not used in the surface quadrature).
    std::vector<bool> is_unresolved;

    /// Distance relative to the cell size, below which points of opposite orientation are not
    /// used (unresolved interface).
    const double unresolved_distance;

    /// Maximum magnitude of the advected level set at a supported point relative to the cell
    /// size.
    const double support_tolerance;

    /// Length scale of the finite differences of the velocity (minimum cell size at the seeding).
    double length_scale = 1.;

    /// The points are assigned to the cells.
    bool located = false;
  };
} // namespace MeltPoolDG::LevelSet
