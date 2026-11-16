#pragma once

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/non_matching/immersed_surface_quadrature.h>

#include <meltpooldg/level_set/narrow_band_interface_points.hpp>
#include <meltpooldg/level_set/reinitialization_data.hpp>

#include <functional>
#include <vector>

namespace MeltPoolDG::LevelSet
{
  /**
   * @brief Continuous (CG) variant of the reinitialization of the semi-Lagrangian narrow band.
   *
   * The level set on the cut cells is represented by an FE_Q element of degree $p$ (continuous
   * across the faces between cut cells, unconstrained at the faces to all other cells). The
   * discontinuous storage of the narrow band (cut DoFHandler with FE_DGQ) is kept: the
   * semi-Lagrangian step evaluates a single-valued function at the nodes, i.e., it maps a
   * continuous level set to a continuous level set, and the reinitialization and the subcell
   * distance are computed in the FE_Q space and interpolated back into the FE_DGQ space (exact,
   * since the polynomial spaces of the cells coincide).
   *
   * Reinitialization: fixed-point iteration of Basting & Kuzmin (2013) as in the DG
   * reinitialization on the cut cells, without face terms,
   *
   * $(\nabla \phi^{k+1}, \nabla v) + \gamma (\phi^{k+1}, v)_\Gamma = (g(|\nabla \phi^k|) \nabla
   * \phi^k, \nabla v)$, $g(s) = 1/s$ for $s > 1$, $g(s) = 2 - s$ otherwise,
   *
   * where $\Gamma$ is given by the surface quadrature function (advected points) or, in cells
   * without points, by the zero level set of the level set before the reinitialization. The
   * matrix is assembled once per reinitialization and factorized by a direct solver.
   *
   * Subcell distance: L2-projection of the signed distance to the points (evaluated at the subcell
   * quadrature points, see NarrowBandInterfacePoints::for_each_unresolved_cell()) onto the FE_Q
   * space on the union of the unresolved cells, with natural boundary conditions at the boundary
   * of this region. Since the projected cells share their boundary nodes with the adjacent cut
   * cells, the projection is a coupled problem on the region (instead of the cell-wise projection
   * of the DG level set).
   */
  template <int dim, typename number>
  class NarrowBandContinuousReinitialization
  {
  public:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;
    using SurfaceQuadratureFunction =
      std::function<dealii::NonMatching::ImmersedSurfaceQuadrature<dim>(
        const typename dealii::Triangulation<dim>::cell_iterator &)>;

    /**
     * @brief Statistics of the last reinitialization.
     */
    struct Statistics
    {
      /// Number of DoFs of the continuous level set.
      dealii::types::global_dof_index n_dofs = 0;

      /// Number of fixed-point iterations.
      unsigned int n_iterations = 0;

      /// Whether the fixed-point iteration converged.
      bool converged = true;

      /// Number of prescribed (projected) DoFs.
      dealii::types::global_dof_index n_prescribed_dofs = 0;
    };

    /**
     * @brief Constructor.
     *
     * @param mapping Mapping.
     * @param triangulation Triangulation.
     * @param reinit_data Parameters of the reinitialization (interface penalty parameter and
     * fixed-point solver).
     * @param fe_degree Polynomial degree.
     * @param fe_collection_cut FE collection of the cut DoFHandler (FE_DGQ for the cut cells,
     * FE_Nothing otherwise).
     * @param pcout Output stream.
     */
    NarrowBandContinuousReinitialization(const dealii::Mapping<dim>          &mapping,
                                         const dealii::Triangulation<dim>    &triangulation,
                                         const ReinitializationData<number>  &reinit_data,
                                         const unsigned int                   fe_degree,
                                         const dealii::hp::FECollection<dim> &fe_collection_cut,
                                         const dealii::ConditionalOStream    &pcout);

    /**
     * @brief Distribute the DoFs of the continuous level set on the cut cells of
     * @p dof_handler_cut.
     */
    void
    distribute_dofs(const dealii::DoFHandler<dim> &dof_handler_cut);

    /**
     * @brief Set the continuous level set to the nodal values of the discontinuous level set
     * @p level_set_dg (average of the values of the adjacent cut cells at shared nodes).
     */
    void
    interpolate_from_discontinuous(const dealii::DoFHandler<dim> &dof_handler_cut,
                                   const VectorType              &level_set_dg);

    /**
     * @brief Write the continuous level set into the discontinuous level set @p level_set_dg on
     * the locally owned cut cells.
     */
    void
    interpolate_to_discontinuous(const dealii::DoFHandler<dim> &dof_handler_cut,
                                 VectorType                    &level_set_dg) const;

    /**
     * @brief Subcell distance: L2-projection of the signed distance to @p interface_points onto
     * the continuous level set on the union of the unresolved cells (natural boundary conditions
     * at the boundary of the region; all other DoFs keep their values). The DoFs of the projected
     * cells are marked as prescribed for reinitialize().
     *
     * @param level_set_advected_dg Advected (discontinuous) level set for the sign check.
     */
    typename NarrowBandInterfacePoints<dim, number>::ProjectionStatistics
    project_subcell_distance(const NarrowBandInterfacePoints<dim, number> &interface_points,
                             const dealii::DoFHandler<dim>                &dof_handler_cut,
                             const VectorType                             &level_set_advected_dg,
                             const double                                  buffer,
                             const unsigned int                            n_subdivisions);

    /**
     * @brief Reinitialization of the continuous level set (fixed-point iteration). If
     * @p keep_prescribed_dofs is true, the DoFs marked by project_subcell_distance() keep their
     * values (Dirichlet values of the other DoFs). If the iteration does not converge, the level
     * set before the reinitialization is restored.
     *
     * @return Statistics of the reinitialization.
     */
    Statistics
    reinitialize(const SurfaceQuadratureFunction &surface_quadrature_function,
                 const bool                       keep_prescribed_dofs);

    /**
     * @brief Continuous level set (with ghost values).
     */
    const VectorType &
    get_level_set() const;

  private:
    /**
     * @brief Right-hand side $(g(|\nabla \phi|) \nabla \phi, \nabla v)$ of the fixed-point
     * iteration for the level set @p phi (with ghost values), with the inhomogeneities of the
     * constraints.
     */
    void
    assemble_rhs(const VectorType &phi, VectorType &rhs) const;

    /**
     * @brief Mark the DoFs of the given locally owned cells in @p mask (with ghost values).
     */
    void
    mark_dofs(const std::vector<bool> &cells, VectorType &mask) const;

    const dealii::Mapping<dim>         &mapping;
    const ReinitializationData<number> &reinit_data;
    const unsigned int                  fe_degree;
    const dealii::ConditionalOStream    pcout;

    /// FE_Q element of the cut cells and FE collection (FE_Nothing for the other cells).
    const dealii::FE_Q<dim>             fe;
    const dealii::hp::FECollection<dim> fe_collection;

    /// DoFHandler and constraints (none without hanging nodes) of the continuous level set.
    dealii::DoFHandler<dim>           dof_handler;
    dealii::AffineConstraints<number> hanging_node_constraints;
    dealii::IndexSet                  locally_owned_dofs;
    dealii::IndexSet                  locally_relevant_dofs;

    /// Continuous level set (with ghost values).
    VectorType level_set;

    /// DoFs with prescribed values (1) from the subcell distance (with ghost values).
    VectorType prescribed_dofs;

    /// Constraints, matrix and cell matrices of the current fixed-point iteration.
    dealii::AffineConstraints<number>       constraints;
    dealii::TrilinosWrappers::SparseMatrix  system_matrix;
    std::vector<dealii::FullMatrix<double>> cell_matrices;
  };
} // namespace MeltPoolDG::LevelSet
