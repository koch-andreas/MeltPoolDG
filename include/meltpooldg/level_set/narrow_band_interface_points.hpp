#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/cell_id.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/non_matching/immersed_surface_quadrature.h>

#include <functional>
#include <optional>
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
   *   as new points (gaps of the point set due to the stretching of the interface); alternatively,
   *   the new points are taken from the zero level set of the level set at the end of the time
   *   step, i.e., after the reinitialization (insert_points()),
   * - in cells with more than $2 n^{d-1}$ points ($n$ points per direction of the surface
   *   quadrature), the closest pairs of points with the same orientation are merged by removing
   *   one point and adding its weight to the other one (compression of the interface),
   * - merge rule (topology change, merge distance $c_m > 0$): the points of two parts of the
   *   interface, which have collided, are removed. The points move forward in time and cannot
   *   represent the disappearance of the parts of the interface, which collide (e.g., the facing
   *   sides of two merging droplets, whose points stop at the plane of the collision inside of
   *   the merged phase), while the advected level set, which is evaluated along the backward
   *   characteristics, already contains the merged interface. With the approach
   *   $a = \Delta t \max(0, (\boldsymbol{u}(\boldsymbol{x} - h \boldsymbol{n}) -
   *   \boldsymbol{u}(\boldsymbol{x})) \cdot \boldsymbol{n})$ of the flow on the outer side
   *   of a point within one time step (nonzero for converging flows, zero for rigid-body
   *   motions), a supported point in a converging flow (the positive part of the normal velocity
   *   difference $(\boldsymbol{u}(\boldsymbol{x} - h \boldsymbol{n}) -
   *   \boldsymbol{u}(\boldsymbol{x})) \cdot \boldsymbol{n}$ exceeds half of the local speed,
   *   e.g., two parts of the interface pushed against each other; not the points of thin gaps or
   *   filaments in stretching or rotating flows, which the level set does not resolve) is
   *   removed, (a) if the advected level set at the probe
   *   $\boldsymbol{x} - \delta \boldsymbol{n}$, $\delta = c_m h + a$, on its outer side
   *   exceeds the value at the point by more than $\delta / 4$ (for a point on the interface,
   *   it decreases by about $\delta$), i.e., the point lies inside of the phase or at a gap
   *   thinner than about $\delta$, or (b) if a point of a different orientation
   *   ($\boldsymbol{n}_i \cdot \boldsymbol{n}_j < 1/2$) is closer than the sum of the
   *   approaches of the two points and each of them lies on the outer side of the other one
   *   (back to back, i.e., the gap between the two parts has closed within the time step). Thin
   *   filaments (both sides face each other) and gaps wider than $\delta$ are kept. After the
   *   removal, the advected level set determines the interface in these cells, and new points
   *   are inserted from the merged interface.
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
   * Alternatively, the level set of the cut cells around unresolved points is given by the
   * L2-projection of the signed distance to all points evaluated at the subcell scale
   * (project_signed_distance()), i.e., by an exact redistancing, which resolves the medial axis of
   * a thin filament, and its best approximation by the polynomials of the cells. The projection
   * is either prescribed in the reinitialization of the other cut cells or replaces the
   * reinitialized level set (NarrowBandData::AdvectedPoints::subcell_coupling).
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

      /// Number of points removed by the merge rule (collision of two parts of the interface).
      unsigned int n_collided_points = 0;

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
     * @param merge_distance Merge distance $c_m$ relative to the cell size of the merge rule
     * (zero: no merge rule), see the class documentation.
     */
    explicit NarrowBandInterfacePoints(const dealii::ConditionalOStream &pcout,
                                       const double                      unresolved_distance = 2.0,
                                       const double                      support_tolerance   = 0.2,
                                       const double                      merge_distance      = 0.);

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
     * @param insert If false, no new points are seeded in the gaps of the point set, and the
     * statistics are not printed; the gaps are then filled by insert_points() from the level set
     * at the end of the time step.
     */
    Statistics
    locate(const dealii::DoFHandler<dim> &dof_handler_band,
           const VectorType              &level_set_band,
           const unsigned int             n_points_1d,
           const std::vector<bool>       &is_cut_cell,
           const bool                     insert = true);

    /**
     * @brief Seed new points in the gaps of the point set after locate(): the quadrature points of
     * the zero level set of @p level_set on the locally owned cells with DoFs of
     * @p dof_handler_cut (e.g., the reinitialized level set of the cut cells), whose distance to
     * all points exceeds $h / n$, are added. Then, all points are assigned to the cells of
     * @p dof_handler_band again, the unresolved points are updated, and the statistics of the time
     * step are printed. Thus, the gaps are filled from the corrected interface instead of the
     * zero level set of the advected level set.
     *
     * @param dof_handler_cut DoFHandler of the level set (FE_DGQ on the cut cells).
     * @param level_set Level set on the cut cells.
     * @param dof_handler_band Band DoFHandler (all cells with DoFs).
     * @param n_points_1d Number of Gauss points per direction of the surface quadrature of new
     * points.
     */
    Statistics
    insert_points(const dealii::DoFHandler<dim> &dof_handler_cut,
                  const VectorType              &level_set,
                  const dealii::DoFHandler<dim> &dof_handler_band,
                  const unsigned int             n_points_1d);

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
     * @brief Statistics of the projection of the subcell distance (summed over all processes).
     */
    struct ProjectionStatistics
    {
      /// Number of cells, whose level set is replaced by the projection.
      unsigned int n_cells = 0;

      /// Number of unresolved cells, which keep their level set, since the projection would change
      /// the sign of the advected level set away from the points.
      unsigned int n_rejected_cells = 0;

      /// Number of unresolved cells (projected or rejected), whose distance is only computed to
      /// the points of the orientation of the cell (orientation filter).
      unsigned int n_filtered_cells = 0;

      /// Number of projected cells, whose projection is pinned to zero at the points of the cell
      /// (pinning penalty).
      unsigned int n_pinned_cells = 0;
    };

    /**
     * @brief Subcell redistancing of the unresolved cells: replace the level set of the locally
     * owned cells with DoFs, which are closer than @p buffer times the cell size to a point with a
     * point of opposite orientation closer than the unresolved distance (e.g., both sides of a
     * thin filament), by the L2-projection of the signed distance to the points (see
     * signed_distance()).
     *
     * The signed distance is evaluated at the points of the iterated Gauss quadrature with
     * $p + 1$ points and @p n_subdivisions subdivisions per direction, which resolves the kink of
     * the distance on the medial axis of a thin filament. Thus, the conflict of the
     * reinitialization in these cells (the penalty term enforces two close zeros, while the
     * polynomial of the cell cannot represent a distance function with a ridge between them)
     * is replaced by the exact redistancing at the subcell scale and the best approximation by
     * the polynomials of the cell. A filament is kept, as long as the polynomials of the cells can
     * represent its two zeros (for $p = 2$, down to a width of about $0.2h$).
     *
     * The points only correct the position of the interface: beyond the edges of the patches
     * (e.g., beyond the open end of a chain of points at the tip of a filament, whose cap has
     * been removed), the signed distance takes the sign of @p level_set_advected, and a cell keeps
     * its level set, if the signed distance has a different sign than @p level_set_advected at a
     * quadrature point, which is farther than a quarter of the cell size from the points (e.g.,
     * one side of a part of a filament, which the level set has lost).
     *
     * @param mapping Mapping.
     * @param dof_handler DoFHandler of the level set (FE_DGQ on the cut cells).
     * @param level_set_advected Advected level set before the reinitialization (sign check).
     * @param level_set Level set, which is modified on the projected cells (the reinitialized
     * level set, or the advected level set, if the projected cells are prescribed in the
     * reinitialization).
     * @param buffer Distance relative to the cell size, within which the cells around unresolved
     * points are projected.
     * @param n_subdivisions Number of subdivisions per direction of the quadrature.
     * @param projected_dofs Optional mask of the DoFs of the projected cells (set to one on these
     * DoFs, unchanged otherwise).
     * @param orientation_filter If true, the distance in a cell, whose points all have the same
     * orientation (pairwise $\boldsymbol{n}_i \cdot \boldsymbol{n}_j > 0$), is only computed to
     * the points of this orientation, i.e., to the branch of the interface in this cell, and not to
     * a close branch of opposite orientation in a neighbor cell (e.g., the other side of a narrow
     * gap or of a thin filament, whose sides lie in different cells): the kink of the distance on
     * the medial axis then lies on a face (jump of the level set) instead of inside the cell. Cells
     * with points of both orientations or without points use all points. Since the level set
     * between the branches then only represents the distance to the branch of each cell, the
     * filter is only suited for interfaces at rest.
     * @param pinning_penalty Dimensionless penalty parameter $c$ of the pinned projection (zero:
     * plain L2-projection). In a cell, whose points all have the same orientation (one branch of
     * the interface in the cell), the projection minimizes
     * $\int_K (\phi - d)^2 \,\mathrm{d}x + c h \sum_k w_k \phi(\boldsymbol{x}_k)^2$, where the
     * sum runs over the points $\boldsymbol{x}_k$ of the cell with the weights $w_k$ and $d$ is the
     * signed distance to all points. The penalty keeps the zero level set of the cell on its
     * points, while the distance to a close branch of opposite orientation in a neighbor cell
     * (e.g., the other side of a narrow gap) still determines the level set away from the points,
     * i.e., the kink of the distance on the medial axis only perturbs the level set away from the
     * interface. Cells with points of both orientations or without points are not pinned, since the
     * polynomials of the cell may not be able to vanish at the points of both branches (the
     * penalty would then merge the branches). Since the penalty also keeps fragments of strongly
     * stretched thin filaments, which the L2-projection would remove, the pinning is mainly suited
     * for interfaces at rest or weakly deformed interfaces.
     */
    ProjectionStatistics
    project_signed_distance(const dealii::Mapping<dim>    &mapping,
                            const dealii::DoFHandler<dim> &dof_handler,
                            const VectorType              &level_set_advected,
                            VectorType                    &level_set,
                            const double                   buffer,
                            const unsigned int             n_subdivisions,
                            VectorType                    *projected_dofs     = nullptr,
                            const bool                     orientation_filter = false,
                            const double                   pinning_penalty    = 0.) const;

    /**
     * @brief Function called for each cell of the subcell distance (see for_each_unresolved_cell())
     * with the cell, the FEValues of the iterated Gauss quadrature on the cell, the signed distance
     * at its quadrature points, the indices of the points of the cell and whether all points of
     * the cell have the same orientation.
     */
    using UnresolvedCellFunction =
      std::function<void(const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
                         const dealii::FEValues<dim>                                  &fe_values,
                         const std::vector<double>                                    &distances,
                         const std::vector<unsigned int>                              &cell_points,
                         const bool one_orientation)>;

    /**
     * @brief Selection of the cells of the subcell distance and evaluation of the signed distance
     * at the subcell scale (see project_signed_distance()), without the projection: @p function is
     * called for each locally owned cell of @p dof_handler with DoFs, which is closer than
     * @p buffer times the cell size to an unresolved point and passes the sign check against
     * @p level_set_advected. This allows other projections of the signed distance, e.g., onto a
     * continuous finite element space. The returned statistics do not contain pinned cells.
     */
    ProjectionStatistics
    for_each_unresolved_cell(const dealii::Mapping<dim>    &mapping,
                             const dealii::DoFHandler<dim> &dof_handler,
                             const VectorType              &level_set_advected,
                             const double                   buffer,
                             const unsigned int             n_subdivisions,
                             const bool                     orientation_filter,
                             const UnresolvedCellFunction  &function) const;

    /**
     * @brief Signed distance of @p x to the interface represented by the points: each point is a
     * flat patch (a segment of the length of its weight in 2D, a disk of the area of its weight
     * in 3D, at most of radius @p max_radius) orthogonal to its normal vector. The distance is the
     * minimum distance to the patches of the 8 (2D) or 24 (3D) nearest points (the nearest point
     * and its neighbors on up to two close branches of the interface, as the candidates of the
     * closest-point search of the semi-Lagrangian advection), its sign is the side of the closest
     * patch (positive in the direction of the normal vector).
     */
    number
    signed_distance(const dealii::Point<dim> &x, const double max_radius) const;

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

    /**
     * @brief Return the unit normal vectors of the points in real space (all processes).
     */
    const std::vector<dealii::Tensor<1, dim>> &
    get_normals() const;

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
     * @brief Append the quadrature points of the zero level set of @p level_set on the locally
     * owned cells with DoFs of @p dof_handler (only the cells marked in @p cells, if given), whose
     * distance to all points exceeds h / @p n_points_1d, to @p data (see pack_surface_points()).
     * The ghost values of @p level_set must be up to date.
     */
    void
    pack_gap_points(const dealii::DoFHandler<dim> &dof_handler,
                    const VectorType              &level_set,
                    const unsigned int             n_points_1d,
                    const std::vector<bool>       *cells,
                    std::vector<double>           &data) const;

    /**
     * @brief Cell of the band containing a point.
     */
    struct Location
    {
      dealii::CellId cell_id;
      unsigned int   active_cell_index;
      bool           is_locally_owned;
    };

    /**
     * @brief Cells of the band (locally owned or ghost cells with DoFs) containing the points
     * @p query_points: on the common boundary of several cells, the cell with the smallest CellId
     * (unique on all processes); empty for points outside of these cells.
     */
    static std::vector<std::optional<Location>>
    find_cells(const dealii::DoFHandler<dim>         &dof_handler_band,
               const std::vector<dealii::Point<dim>> &query_points);

    /**
     * @brief Assign the points to the locally owned cells of the band (points_of_cell).
     */
    void
    assign_points_to_cells(const dealii::DoFHandler<dim> &dof_handler_band);

    /**
     * @brief Merge rule: set @p status to 3 for the supported points (status 1), which belong to
     * two parts of the interface, which have collided (see the class documentation).
     *
     * @param dof_handler_band Band DoFHandler.
     * @param level_set_band Advected level set on the band (with ghost values).
     * @param values_at_points Advected level set at the points (all processes).
     * @param cell_sizes Cell sizes of the cells containing the points (all processes).
     * @param status Status of the points (all processes): 0: outside of the band, 1: supported,
     * 2: not supported, 3: collided.
     */
    void
    mark_collided_points(const dealii::DoFHandler<dim> &dof_handler_band,
                         const VectorType              &level_set_band,
                         const std::vector<double>     &values_at_points,
                         const std::vector<double>     &cell_sizes,
                         std::vector<unsigned int>     &status) const;

    /**
     * @brief Mark the unresolved points of the locally owned cells and update the statistics of
     * the cells in last_statistics (number of points, points in cells, which are not cut, cut
     * cells without usable points, unresolved points).
     */
    void
    update_unresolved_points(const dealii::Triangulation<dim> &triangulation,
                             const std::vector<bool>          &is_cut_cell,
                             const MPI_Comm                    mpi_comm);

    /**
     * @brief Print last_statistics.
     */
    void
    print_statistics() const;

    /// Statistics of the last time step.
    Statistics last_statistics;

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

    /// Merge distance relative to the cell size (zero: no merge rule).
    const double merge_distance;

    /// Length scale of the finite differences of the velocity (minimum cell size at the seeding).
    double length_scale = 1.;

    /// Approach of the flow on the outer side of each point within the last time step (merge
    /// rule, computed in advect(), empty after the seeding).
    std::vector<double> approach;

    /// Convergence ratio of the flow on the outer side of each point: positive part of the normal
    /// velocity difference over the length scale divided by the speed (merge rule, computed in
    /// advect(), empty after the seeding).
    std::vector<double> convergence;

    /// The points are assigned to the cells.
    bool located = false;
  };
} // namespace MeltPoolDG::LevelSet
