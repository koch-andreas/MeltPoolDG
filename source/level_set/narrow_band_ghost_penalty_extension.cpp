#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <meltpooldg/level_set/narrow_band_ghost_penalty_extension.hpp>
#include <meltpooldg/level_set/narrow_band_utilities.hpp>
#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/utilities/journal.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace MeltPoolDG::LevelSet
{
  template <int dim, typename number>
  NarrowBandGhostPenaltyExtension<dim, number>::NarrowBandGhostPenaltyExtension(
    const typename NarrowBandData<number>::Extension &data,
    const dealii::Mapping<dim>                       &mapping,
    const dealii::ConditionalOStream                 &pcout)
    : data(data)
    , mapping(mapping)
    , pcout(pcout)
  {}

  template <int dim, typename number>
  unsigned int
  NarrowBandGhostPenaltyExtension<dim, number>::extend(const dealii::DoFHandler<dim> &dof_handler,
                                                       VectorType &level_set) const
  {
    using namespace NarrowBand;

    const auto &fe_collection = dof_handler.get_fe_collection();
    AssertThrow(dof_handler.has_hp_capabilities() and fe_collection.size() == 3 and
                  fe_collection[ActiveFEIndex::cut].n_dofs_per_cell() > 0 and
                  fe_collection[ActiveFEIndex::neighbor].n_dofs_per_cell() > 0,
                dealii::ExcMessage("The ghost-penalty extension requires the band DoFHandler of "
                                   "the narrow-band level set."));
    AssertThrow(not dof_handler.get_triangulation().has_hanging_nodes(),
                dealii::ExcMessage("The narrow-band level set does not support hanging nodes."));

    const unsigned int fe_degree = fe_collection[ActiveFEIndex::cut].degree;
    AssertThrow(fe_degree >= 1 and fe_degree <= 3, dealii::ExcNotImplemented());

    const MPI_Comm mpi_comm = dof_handler.get_mpi_communicator();

    const dealii::IndexSet &locally_owned_dofs = dof_handler.locally_owned_dofs();
    const dealii::IndexSet  locally_relevant_dofs =
      dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);

    // The values of the cut cells are prescribed by inhomogeneous constraints. The values of the
    // cut cells of the ghost layer are required as well.
    VectorType level_set_relevant(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
    level_set_relevant.copy_locally_owned_data_from(level_set);
    level_set_relevant.update_ghost_values();

    dealii::AffineConstraints<number> constraints(locally_owned_dofs, locally_relevant_dofs);
    {
      std::vector<dealii::types::global_dof_index> dof_indices;
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (cell->is_artificial() or cell->active_fe_index() != ActiveFEIndex::cut)
            continue;

          dof_indices.resize(cell->get_fe().n_dofs_per_cell());
          cell->get_dof_indices(dof_indices);
          for (const auto dof : dof_indices)
            constraints.add_constraint(dof, {}, level_set_relevant(dof));
        }
    }
    constraints.close();

    // Small cut cells, which do not serve as sources of the extension: the faces between a small
    // cut cell and a neighbor cell are not evaluated, if the neighbor cell is connected to a cut
    // cell with a sufficient volume fraction via ghost-penalty faces (anchored neighbor cell).
    // Otherwise, the system would be singular. The flags of the neighbor cells are stored in the
    // DoFs of the neighbor cells to exchange them with the other processes.
    const bool        exclude_small_cut_cells = data.minimum_source_volume_fraction > 0.;
    std::vector<bool> is_small_cut_cell;
    VectorType        is_anchored(locally_owned_dofs, locally_relevant_dofs, mpi_comm);

    std::vector<dealii::types::global_dof_index> cell_dof_indices;

    const auto is_small = [&](const auto &cell) {
      return exclude_small_cut_cells and cell->active_fe_index() == ActiveFEIndex::cut and
             is_small_cut_cell[cell->active_cell_index()];
    };
    const auto is_anchored_neighbor = [&](const auto &cell) {
      if (cell->active_fe_index() != ActiveFEIndex::neighbor)
        return false;
      cell_dof_indices.resize(cell->get_fe().n_dofs_per_cell());
      cell->get_dof_indices(cell_dof_indices);
      return is_anchored(cell_dof_indices[0]) > 0.5;
    };

    if (exclude_small_cut_cells)
      {
        const std::vector<number> volume_fractions =
          compute_cut_cell_volume_fractions(dof_handler, level_set_relevant, fe_degree);

        is_small_cut_cell.assign(volume_fractions.size(), false);
        for (const auto &cell : dof_handler.active_cell_iterators())
          if (not cell->is_artificial() and cell->active_fe_index() == ActiveFEIndex::cut)
            is_small_cut_cell[cell->active_cell_index()] =
              volume_fractions[cell->active_cell_index()] < data.minimum_source_volume_fraction;

        const auto set_anchored = [&](const auto &cell) {
          cell_dof_indices.resize(cell->get_fe().n_dofs_per_cell());
          cell->get_dof_indices(cell_dof_indices);
          for (const auto dof : cell_dof_indices)
            is_anchored(dof) = 1.;
        };

        // (a) neighbor cells sharing a face with a cut cell with a sufficient volume fraction
        for (const auto &cell : dof_handler.active_cell_iterators())
          if (cell->is_locally_owned() and cell->active_fe_index() == ActiveFEIndex::neighbor)
            for (const unsigned int f : cell->face_indices())
              if (not cell->at_boundary(f) and
                  cell->neighbor(f)->active_fe_index() == ActiveFEIndex::cut and
                  not is_small(cell->neighbor(f)))
                {
                  set_anchored(cell);
                  break;
                }
        is_anchored.update_ghost_values();

        // (b) neighbor cells connected to an anchored neighbor cell via a ghost-penalty face
        if (data.ghost_penalty_faces == NarrowBandGhostPenaltyFaces::all_band_faces)
          while (true)
            {
              bool changed = false;
              for (const auto &cell : dof_handler.active_cell_iterators())
                if (cell->is_locally_owned() and
                    cell->active_fe_index() == ActiveFEIndex::neighbor and
                    not is_anchored_neighbor(cell))
                  for (const unsigned int f : cell->face_indices())
                    if (not cell->at_boundary(f) and is_anchored_neighbor(cell->neighbor(f)))
                      {
                        set_anchored(cell);
                        changed = true;
                        break;
                      }

              is_anchored.update_ghost_values();
              if (not dealii::Utilities::MPI::logical_or(changed, mpi_comm))
                break;
            }
      }

    // set up the sparse matrix
    dealii::TrilinosWrappers::SparsityPattern dsp(locally_owned_dofs,
                                                  locally_owned_dofs,
                                                  locally_relevant_dofs,
                                                  mpi_comm);
    dealii::DoFTools::make_flux_sparsity_pattern(dof_handler,
                                                 dsp,
                                                 constraints,
                                                 false /*keep_constrained_dofs*/);
    dsp.compress();

    dealii::TrilinosWrappers::SparseMatrix system_matrix;
    system_matrix.reinit(dsp);

    VectorType rhs(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
    VectorType solution(locally_owned_dofs, locally_relevant_dofs, mpi_comm);

    // assemble the ghost-penalty terms
    dealii::UpdateFlags update_flags = dealii::update_values | dealii::update_gradients |
                                       dealii::update_JxW_values | dealii::update_normal_vectors;
    if (fe_degree >= 2)
      update_flags |= dealii::update_hessians;
    if (fe_degree >= 3)
      update_flags |= dealii::update_3rd_derivatives;

    // Note: FEInterfaceValues keeps pointers to the collections; thus, they must outlive it.
    const dealii::hp::MappingCollection<dim> mapping_collection(mapping);
    const dealii::hp::QCollection<dim - 1> face_quadrature(dealii::QGauss<dim - 1>(fe_degree + 1));

    dealii::FEInterfaceValues<dim> fe_interface_values(mapping_collection,
                                                       fe_collection,
                                                       face_quadrature,
                                                       update_flags);

    const dealii::FEValuesExtractors::Scalar level_set_extractor(0);
    constexpr unsigned int                   invalid = dealii::numbers::invalid_unsigned_int;

    std::array<number, 4> gamma{};
    for (unsigned int k = 0; k <= fe_degree; ++k)
      gamma[k] = data.get_ghost_penalty_parameter(k);

    dealii::FullMatrix<number>                   local_matrix;
    dealii::Vector<number>                       local_rhs;
    std::array<std::vector<number>, 4>           normal_derivative_jumps;
    std::vector<dealii::types::global_dof_index> interface_dof_indices;

    unsigned int n_excluded_faces = 0;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or cell->active_fe_index() == ActiveFEIndex::inactive)
          continue;

        for (const unsigned int f : cell->face_indices())
          {
            if (cell->at_boundary(f))
              continue;

            const auto neighbor = cell->neighbor(f);
            Assert(neighbor->is_active(), dealii::ExcNotImplemented());

            if (neighbor->active_fe_index() == ActiveFEIndex::inactive)
              continue;

            // ghost-penalty faces
            const bool cell_is_neighbor = cell->active_fe_index() == ActiveFEIndex::neighbor;
            const bool neighbor_is_neighbor =
              neighbor->active_fe_index() == ActiveFEIndex::neighbor;

            if (not cell_is_neighbor and not neighbor_is_neighbor)
              continue;

            if (data.ghost_penalty_faces == NarrowBandGhostPenaltyFaces::cut_cell_faces and
                cell_is_neighbor and neighbor_is_neighbor)
              continue;

            // assemble each face only once: faces between two locally owned cells are assembled
            // from the cell with the smaller id, faces at the boundary of the subdomain by the
            // process with the smaller subdomain id
            if (neighbor->is_locally_owned() and neighbor->id() < cell->id())
              continue;
            if (not neighbor->is_locally_owned() and
                neighbor->subdomain_id() < cell->subdomain_id())
              continue;

            // faces between small cut cells and anchored neighbor cells
            if (exclude_small_cut_cells and ((is_small(cell) and is_anchored_neighbor(neighbor)) or
                                             (is_small(neighbor) and is_anchored_neighbor(cell))))
              {
                ++n_excluded_faces;
                continue;
              }

            fe_interface_values.reinit(
              cell, f, invalid, neighbor, cell->neighbor_of_neighbor(f), invalid);

            const unsigned int n_interface_dofs = fe_interface_values.n_current_interface_dofs();

            const number h =
              std::min(cell->minimum_vertex_distance(), neighbor->minimum_vertex_distance());

            // weights gamma_k h^(2k+1) of the jumps of the k-th normal derivative
            std::array<number, 4> weights{};
            for (unsigned int k = 0; k <= fe_degree; ++k)
              weights[k] = gamma[k] * std::pow(h, 2. * k + 1.);

            local_matrix.reinit(n_interface_dofs, n_interface_dofs);
            local_rhs.reinit(n_interface_dofs);
            for (unsigned int k = 0; k <= fe_degree; ++k)
              normal_derivative_jumps[k].resize(n_interface_dofs);

            const auto &fe_face = fe_interface_values[level_set_extractor];

            for (const unsigned int q : fe_interface_values.quadrature_point_indices())
              {
                const dealii::Tensor<1, dim> &n = fe_interface_values.normal_vector(q);

                for (unsigned int i = 0; i < n_interface_dofs; ++i)
                  {
                    normal_derivative_jumps[0][i] = fe_face.jump_in_values(i, q);
                    normal_derivative_jumps[1][i] = fe_face.jump_in_gradients(i, q) * n;
                    if (fe_degree >= 2)
                      normal_derivative_jumps[2][i] = n * (fe_face.jump_in_hessians(i, q) * n);
                    if (fe_degree >= 3)
                      {
                        const auto third_derivative = fe_face.jump_in_third_derivatives(i, q);
                        number     value            = 0.;
                        for (unsigned int a = 0; a < dim; ++a)
                          for (unsigned int b = 0; b < dim; ++b)
                            for (unsigned int c = 0; c < dim; ++c)
                              value += third_derivative[a][b][c] * n[a] * n[b] * n[c];
                        normal_derivative_jumps[3][i] = value;
                      }
                  }

                const number JxW = fe_interface_values.JxW(q);

                for (unsigned int i = 0; i < n_interface_dofs; ++i)
                  for (unsigned int j = 0; j < n_interface_dofs; ++j)
                    {
                      number value = 0.;
                      for (unsigned int k = 0; k <= fe_degree; ++k)
                        value += weights[k] * normal_derivative_jumps[k][i] *
                                 normal_derivative_jumps[k][j];
                      local_matrix(i, j) += value * JxW;
                    }
              }

            interface_dof_indices = fe_interface_values.get_interface_dof_indices();
            constraints.distribute_local_to_global(
              local_matrix, local_rhs, interface_dof_indices, system_matrix, rhs);
          }
      }

    system_matrix.compress(dealii::VectorOperation::add);
    rhs.compress(dealii::VectorOperation::add);

    // Rows of cut cells without ghost-penalty faces remain empty. Set their diagonal entries to
    // one to obtain a regular system. Their values are set by the constraints afterwards.
    for (const auto dof : locally_owned_dofs)
      if (system_matrix.diag_element(dof) == 0.)
        {
          Assert(constraints.is_constrained(dof), dealii::ExcInternalError());
          system_matrix.set(dof, dof, 1.0);
        }
    system_matrix.compress(dealii::VectorOperation::insert);

    // solve the system
    const auto solve = [&](const auto &preconditioner) -> unsigned int {
      return LinearSolver::solve<VectorType>(
        system_matrix, solution, rhs, data.linear_solver, preconditioner, "narrow_band_extension");
    };

    unsigned int n_iterations = 0;
    switch (data.linear_solver.preconditioner_type)
      {
          case PreconditionerType::Identity: {
            n_iterations = solve(dealii::PreconditionIdentity());
            break;
          }
          case PreconditionerType::Diagonal: {
            dealii::TrilinosWrappers::PreconditionJacobi preconditioner;
            preconditioner.initialize(system_matrix);
            n_iterations = solve(preconditioner);
            break;
          }
          case PreconditionerType::ILU: {
            dealii::TrilinosWrappers::PreconditionILU preconditioner;
            preconditioner.initialize(system_matrix);
            n_iterations = solve(preconditioner);
            break;
          }
          case PreconditionerType::AMG: {
            dealii::TrilinosWrappers::PreconditionAMG preconditioner;
            preconditioner.initialize(system_matrix);
            n_iterations = solve(preconditioner);
            break;
          }
        default:
          AssertThrow(false, dealii::ExcNotImplemented());
      }

    constraints.distribute(solution);

    level_set.zero_out_ghost_values();
    level_set.copy_locally_owned_data_from(solution);

    const bool         use_sign_limiter = data.sign_limiter_tolerance >= 0.;
    const unsigned int n_limited_cells =
      use_sign_limiter ?
        limit_sign_changes(mapping, dof_handler, level_set, data.sign_limiter_tolerance) :
        0;

    // Note: the number of excluded faces requires a collective operation.
    n_excluded_faces = dealii::Utilities::MPI::sum(n_excluded_faces, mpi_comm);

    {
      std::ostringstream str;
      str << "extension to "
          << NarrowBand::n_cells_with_active_fe_index(dof_handler, ActiveFEIndex::neighbor)
          << " neighbor cells: " << n_iterations
          << (exclude_small_cut_cells ? " iterations" : " linear iterations");
      if (use_sign_limiter)
        str << ", " << n_limited_cells << (exclude_small_cut_cells ? " limited" : " limited cells");
      if (exclude_small_cut_cells)
        str << ", " << n_excluded_faces << " excl. faces";
      Journal::print_line(pcout, str.str(), "narrow_band");
    }

    return n_iterations;
  }

  template class NarrowBandGhostPenaltyExtension<1, double>;
  template class NarrowBandGhostPenaltyExtension<2, double>;
  template class NarrowBandGhostPenaltyExtension<3, double>;
} // namespace MeltPoolDG::LevelSet
