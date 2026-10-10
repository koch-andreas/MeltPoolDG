/**
 * @brief Band solver for the ghost-penalty stabilized mass matrix of the cutDG compressible
 * multiphase flow solver: all cells away from the interface are inverted exactly and only the
 * coupled band of cells around the interface is solved with the CG solver of LinearSolver.
 */

#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>

#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/linear_algebra/linear_solver_data.hpp>

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

namespace MeltPoolDG::Multiphase
{
  /**
   * Return whether the cell category @p category (active FE index) contains DoFs of the phase
   * @p phase (0: liquid, 1: gas).
   */
  inline bool
  category_has_phase(const unsigned int category, const unsigned int phase)
  {
    return category == CutUtil::CellCategory::intersected ||
           category == (phase == 0 ? CutUtil::CellCategory::liquid : CutUtil::CellCategory::gas);
  }

  /**
   * Return whether the active cell @p cell has at least one (sub)face at which the ghost-penalty
   * stabilization of phase @p phase (0: liquid, 1: gas) acts, i.e., a face to a neighbor
   * containing the phase where at least one of the two cells is intersected (same face selection
   * as CutUtil::face_type_has_ghost_penalty()). Boundary faces (except for periodic ones) are
   * skipped.
   */
  template <int dim, typename CellIteratorType>
  bool
  has_ghost_penalty_face(const CellIteratorType &cell, const unsigned int phase)
  {
    const unsigned int category = cell->active_fe_index();

    const auto is_ghost_penalty_face = [&](const unsigned int neighbor_category) {
      return category_has_phase(neighbor_category, phase) &&
             (category == CutUtil::CellCategory::intersected ||
              neighbor_category == CutUtil::CellCategory::intersected);
    };

    for (const unsigned int f : cell->face_indices())
      {
        const bool is_periodic = cell->has_periodic_neighbor(f);
        if (cell->at_boundary(f) && !is_periodic)
          continue;

        const auto neighbor = cell->neighbor_or_periodic_neighbor(f);

        if (!neighbor->has_children())
          {
            // neighbor on the same level or coarser
            if (is_ghost_penalty_face(neighbor->active_fe_index()))
              return true;
          }
        else if constexpr (dim == 1)
          {
            // a face is a point: only the category of the adjacent active child matters
            auto child = neighbor;
            while (child->has_children())
              child = child->child(1 - f);

            if (is_ghost_penalty_face(child->active_fe_index()))
              return true;
          }
        else
          {
            // finer neighbor: check the subfaces
            const unsigned int n_subfaces = is_periodic ?
                                              dealii::GeometryInfo<dim>::max_children_per_face :
                                              cell->face(f)->n_children();
            for (unsigned int subface = 0; subface < n_subfaces; ++subface)
              {
                const auto child = is_periodic ?
                                     cell->periodic_neighbor_child_on_subface(f, subface) :
                                     cell->neighbor_child_on_subface(f, subface);

                if (is_ghost_penalty_face(child->active_fe_index()))
                  return true;
              }
          }
      }

    return false;
  }

  /**
   * Return whether the active cell @p cell belongs to the band, i.e., whether it is intersected or
   * shares a ghost-penalty face of its phase with an intersected cell.
   */
  template <int dim, typename CellIteratorType>
  bool
  is_band_cell(const CellIteratorType &cell)
  {
    const unsigned int category = cell->active_fe_index();

    if (category == CutUtil::CellCategory::intersected)
      return true;

    return has_ghost_penalty_face<dim>(cell, category == CutUtil::CellCategory::liquid ? 0 : 1);
  }

  /**
   * Set up the constraints of the band solver: all DoFs outside the band (see is_band_cell()) are
   * constrained to zero, the DoFs of the band remain unconstrained. A MatrixFree object set up with
   * these constraints (as an additional DoF index for the same DoFHandler) restricts the system
   * matrix to the band, i.e., the matrix-free gather and scatter operations skip all DoFs outside
   * the band.
   *
   * The constraints only depend on the cell categories (active FE indices) and have to be
   * recomputed whenever the DoFs are distributed anew, before the MatrixFree object is built.
   *
   * @param dof_handler DoFHandler of the compressible multiphase flow solution.
   * @param constraints Output constraints; cleared and reinitialized for the locally relevant DoFs.
   */
  template <int dim, typename number>
  void
  make_band_constraints(const dealii::DoFHandler<dim>     &dof_handler,
                        dealii::AffineConstraints<number> &constraints)
  {
    const dealii::IndexSet &locally_owned_dofs = dof_handler.locally_owned_dofs();
    const dealii::IndexSet  locally_relevant_dofs =
      dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);

    // mark the DoFs of all locally owned band cells; the ghost values provide the information for
    // the DoFs of ghost cells, whose neighbors might not be known on this process
    dealii::LinearAlgebra::distributed::Vector<number> band_marker(
      locally_owned_dofs, locally_relevant_dofs, dof_handler.get_mpi_communicator());

    std::vector<dealii::types::global_dof_index> dof_indices;
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned() && is_band_cell<dim>(cell))
        {
          dof_indices.resize(cell->get_fe().n_dofs_per_cell());
          cell->get_dof_indices(dof_indices);
          for (const auto i : dof_indices)
            band_marker(i) = number(1.);
        }
    band_marker.update_ghost_values();

    constraints.clear();
    constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    for (const auto i : locally_relevant_dofs)
      if (band_marker(i) == number(0.))
        constraints.constrain_dof_to_zero(i);
    constraints.close();
  }

  /**
   * Description of the "band" of a cutDG mass-type operator, i.e., of the cells whose degrees of
   * freedom are coupled to the degrees of freedom of other cells (intersected cells and cells
   * sharing a ghost-penalty face with them, see is_band_cell()).
   *
   * A band cell is identified by the matrix-free cell batch and the SIMD lane within that batch. In
   * addition, the object stores the local indices (within the locally owned range of a vector) of
   * all DoFs of the band.
   *
   * @note The cells are addressed via the matrix-free cell batch numbering. Thus, the data becomes
   * invalid whenever the MatrixFree object is reinitialized and has to be recomputed.
   */
  class BandData
  {
  public:
    /**
     * Initialize the internal data structures and remove all cells from the band.
     *
     * @param n_cell_batches_in Number of cell batches of the MatrixFree object.
     * @param n_lanes_in Number of SIMD lanes per cell batch.
     */
    void
    reinit(const unsigned int n_cell_batches_in, const unsigned int n_lanes_in)
    {
      n_cell_batches_ = n_cell_batches_in;
      n_lanes         = n_lanes_in;

      cell_in_band.assign(static_cast<std::size_t>(n_cell_batches_) * n_lanes, false);
      batch_in_band.assign(n_cell_batches_, false);
      dof_indices.clear();
      initialized = true;
    }

    /**
     * Remove all cells and release the memory.
     */
    void
    clear()
    {
      cell_in_band.clear();
      cell_in_band.shrink_to_fit();
      batch_in_band.clear();
      batch_in_band.shrink_to_fit();
      dof_indices.clear();
      dof_indices.shrink_to_fit();
      n_cell_batches_ = 0;
      initialized     = false;
    }

    /**
     * Return whether reinit() has been called since the last call to clear().
     */
    bool
    is_initialized() const
    {
      return initialized;
    }

    /**
     * Add the given cell to the band.
     */
    void
    add(const unsigned int cell_batch, const unsigned int lane)
    {
      cell_in_band[index(cell_batch, lane)] = true;
      batch_in_band[cell_batch]             = true;
    }

    /**
     * Return whether the given cell belongs to the band.
     */
    bool
    contains(const unsigned int cell_batch, const unsigned int lane) const
    {
      return cell_in_band[index(cell_batch, lane)];
    }

    /**
     * Return whether at least one cell of the given cell batch belongs to the band.
     */
    bool
    contains_cell_batch(const unsigned int cell_batch) const
    {
      AssertIndexRange(cell_batch, n_cell_batches_);
      return batch_in_band[cell_batch];
    }

    /**
     * Local indices (within the locally owned range of a vector) of all DoFs of the band.
     */
    std::vector<unsigned int> &
    get_dof_indices()
    {
      return dof_indices;
    }

    /**
     * Local indices (within the locally owned range of a vector) of all DoFs of the band.
     */
    const std::vector<unsigned int> &
    get_dof_indices() const
    {
      return dof_indices;
    }

    /**
     * Number of cell batches the data has been set up for.
     */
    unsigned int
    n_cell_batches() const
    {
      return n_cell_batches_;
    }

  private:
    std::size_t
    index(const unsigned int cell_batch, const unsigned int lane) const
    {
      AssertIndexRange(cell_batch, n_cell_batches_);
      AssertIndexRange(lane, n_lanes);
      return static_cast<std::size_t>(cell_batch) * n_lanes + lane;
    }

    unsigned int n_cell_batches_ = 0;
    unsigned int n_lanes         = 0;
    bool         initialized     = false;

    // Flag for each (cell batch, lane) whether the cell belongs to the band
    std::vector<bool> cell_in_band;

    // Flag for each cell batch whether it contains at least one band cell
    std::vector<bool> batch_in_band;

    // Local indices of all DoFs of the band
    std::vector<unsigned int> dof_indices;
  };

  /**
   * A concept for matrix-free operators that can be solved with the BandSolver.
   *
   * The system matrix of the operator has to decouple exactly into
   * - cells outside the band, whose diagonal blocks are mass matrices that the operator can invert
   *   cell-wise, and
   * - the band, i.e., cells coupled to other cells (e.g., via ghost-penalty face terms).
   * In particular, the system matrix must not couple a cell inside the band with a cell outside
   * the band.
   */
  template <typename OperatorType, typename VectorType>
  concept BandSolverOperatorType = requires(const OperatorType &op,
                                            BandData           &band,
                                            const BandData     &const_band,
                                            VectorType         &dst,
                                            const VectorType   &src) {
    /**
     * Determine the cells and the local DoF indices of the band.
     */
    op.compute_band(band);

    /**
     * Initialize a vector with the layout required by vmult_band(), i.e., the layout of the DoF
     * index with band constraints (see make_band_constraints()).
     */
    op.initialize_band_dof_vector(dst);

    /**
     * Apply the exact (cell-wise) inverse of the system matrix to the entries of @p src outside
     * the band and write the result into the corresponding entries of @p dst. The band entries
     * of @p dst remain untouched.
     */
    op.apply_inverse_mass_matrix_outside_band(const_band, dst, src);

    /**
     * Add the action of the system matrix restricted to the band to @p dst (vectors initialized
     * with initialize_band_dof_vector()). Entries outside the band are neither read nor written.
     */
    op.vmult_band(const_band, dst, src);
  };

  /**
   * Solver for the linear system of the cutDG compressible multiphase flow solver, i.e., the cut DG
   * mass matrix with face-based ghost-penalty stabilization (see CompressibleMultiphaseOperator).
   *
   * The solver
   * 1. applies the exact inverse of the cell mass matrices to the cells away from the interface,
   * 2. solves the remaining (small) band system with a linear solver
   *
   * The band system is formulated on compressed vectors, which only contain the band DoFs (with a
   * contiguous global numbering of the band DoFs). The operator of the band system copies the
   * entries of a compressed vector into a vector with the layout of the full system, applies the
   * system matrix restricted to the band and copies the band entries of the result back. The
   * system matrix restricted to the band is evaluated on an additional DoF index in which all DoFs
   * outside the band are constrained (see make_band_constraints()). Thus, the vector operations of
   * the CG solver and the operator applications only touch the band.
   *
   * The CG solver starts from a zero initial guess and stops as soon as the residual satisfies
   * |r| <= max(rel_tolerance * |b_band|, abs_tolerance), where the relative tolerance is bounded
   * from below by min_rel_tolerance to prevent iterating below the attainable accuracy in double
   * precision.
   *
   * @note The cost of the band solve scales with the number of cells in the band, i.e., with the
   * size of the interface, and not with the total number of cells.
   */
  template <typename number, typename VectorType>
  class BandSolver
  {
  public:
    /**
     * Determine the band of the operator. This has to be done whenever the MatrixFree object of
     * the operator has been reinitialized, e.g., after the DoF layout has changed.
     *
     * @param op Operator providing the band operations.
     */
    template <BandSolverOperatorType<VectorType> OperatorType>
    void
    update(const OperatorType &op)
    {
      op.compute_band(band);

      // the band may have changed: the vectors have to be set up anew
      vectors_valid = false;
    }

    /**
     * Reset the internal data structures. They are rebuilt by the next call to update().
     */
    void
    reinit()
    {
      band.clear();

      x_band = VectorType();
      b_band = VectorType();
      p_full = VectorType();
      q_full = VectorType();

      vectors_valid = false;
    }

    /**
     * Return whether the band has been determined since the last call to reinit().
     */
    bool
    is_initialized() const
    {
      return band.is_initialized();
    }

    /**
     * Return the band data.
     */
    const BandData &
    get_band() const
    {
      return band;
    }

    /**
     * Solve the linear system A x = b.
     *
     * @param op Operator providing the band operations.
     * @param x Solution vector. All locally owned entries are overwritten; the initial value is not
     * used.
     * @param b Right-hand side vector.
     * @param data Parameters of the CG solver (tolerances, maximum number of iterations, monitor
     * type).
     * @param identifier Optional identifier that is printed if the solver fails.
     *
     * @return Number of CG iterations of the band solve.
     */
    template <BandSolverOperatorType<VectorType> OperatorType>
    unsigned int
    solve(const OperatorType             &op,
          VectorType                     &x,
          const VectorType               &b,
          const LinearSolverData<number> &data,
          const std::string              &identifier = "")
    {
      Assert(band.is_initialized(),
             dealii::ExcMessage("The band solver has not been updated after its last "
                                "reinitialization. Call update() before solve()."));

      const std::vector<unsigned int> &band_dof_indices = band.get_dof_indices();

      if (!vectors_valid)
        setup_vectors(op, b);

      AssertDimension(p_full.locally_owned_size(), b.locally_owned_size());

      // 1) band system: compressed right-hand side, zero initial guess
      for (unsigned int k = 0; k < band_dof_indices.size(); ++k)
        b_band.local_element(k) = b.local_element(band_dof_indices[k]);

      x_band = number(0.);

      // do not iterate below the attainable accuracy in double precision
      LinearSolverData<number> band_solver_data = data;
      band_solver_data.rel_tolerance =
        std::max(band_solver_data.rel_tolerance, number(min_rel_tolerance));

      const BandOperator<OperatorType> band_operator{op, band, p_full, q_full};

      const int n_iterations = LinearSolver::solve<VectorType>(band_operator,
                                                               x_band,
                                                               b_band,
                                                               band_solver_data,
                                                               dealii::PreconditionIdentity(),
                                                               identifier);

      // 2) assemble the solution; only the locally owned entries of x are written
      const bool x_has_ghost_elements = x.has_ghost_elements();
      if (x_has_ghost_elements)
        x.zero_out_ghost_values();

      // exact cell-wise inverse for the cells away from the interface
      op.apply_inverse_mass_matrix_outside_band(band, x, b);

      // band entries
      for (unsigned int k = 0; k < band_dof_indices.size(); ++k)
        x.local_element(band_dof_indices[k]) = x_band.local_element(k);

      if (x_has_ghost_elements)
        x.update_ghost_values();

      return n_iterations;
    }

  private:
    /**
     * Operator of the band system acting on compressed vectors (band DoFs only), providing vmult()
     * for LinearSolver::solve().
     */
    template <typename OperatorType>
    struct BandOperator
    {
      const OperatorType &op;
      const BandData     &band;

      // Work vectors with the layout of the DoF index with band constraints
      VectorType &p_full;
      VectorType &q_full;

      void
      vmult(VectorType &dst, const VectorType &src) const
      {
        const std::vector<unsigned int> &band_dof_indices = band.get_dof_indices();

        // expand: band entries of the full vector (the constrained entries outside the band are
        // ignored by vmult_band())
        for (unsigned int k = 0; k < band_dof_indices.size(); ++k)
          {
            p_full.local_element(band_dof_indices[k]) = src.local_element(k);
            q_full.local_element(band_dof_indices[k]) = number(0.);
          }

        // q = A_band p
        op.vmult_band(band, q_full, p_full);

        // compress: band entries of the result
        for (unsigned int k = 0; k < band_dof_indices.size(); ++k)
          dst.local_element(k) = q_full.local_element(band_dof_indices[k]);
      }
    };

    /**
     * Set up the compressed vectors (band DoFs only, contiguous global numbering of the band DoFs)
     * and the work vectors of the band operator (layout of the DoF index with band constraints).
     */
    template <typename OperatorType>
    void
    setup_vectors(const OperatorType &op, const VectorType &full_vector)
    {
      const MPI_Comm mpi_comm = full_vector.get_mpi_communicator();

      const dealii::types::global_dof_index n_locally_owned_band_dofs =
        band.get_dof_indices().size();
      // contiguous global numbering of the band DoFs (ordered by MPI rank)
      const std::vector<dealii::types::global_dof_index> n_band_dofs_per_rank =
        dealii::Utilities::MPI::all_gather(mpi_comm, n_locally_owned_band_dofs);
      const unsigned int rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm);

      dealii::types::global_dof_index offset = 0, n_band_dofs = 0;
      for (unsigned int r = 0; r < n_band_dofs_per_rank.size(); ++r)
        {
          if (r < rank)
            offset += n_band_dofs_per_rank[r];
          n_band_dofs += n_band_dofs_per_rank[r];
        }

      dealii::IndexSet locally_owned_band_dofs(n_band_dofs);
      locally_owned_band_dofs.add_range(offset, offset + n_locally_owned_band_dofs);

      x_band.reinit(locally_owned_band_dofs, mpi_comm);
      b_band.reinit(locally_owned_band_dofs, mpi_comm);

      op.initialize_band_dof_vector(p_full);
      op.initialize_band_dof_vector(q_full);

      vectors_valid = true;
    }

    // Lower bound for the relative tolerance of the band solve (roundoff level)
    static constexpr double min_rel_tolerance = 1e-14;

    // Cells and DoFs of the band
    BandData band;

    // Compressed solution and right-hand side vectors of the band system (band DoFs only)
    VectorType x_band, b_band;

    // Work vectors of the band operator with the layout of the DoF index with band constraints
    VectorType p_full, q_full;

    // Flag indicating whether the vectors match the current band
    bool vectors_valid = false;
  };
} // namespace MeltPoolDG::Multiphase
