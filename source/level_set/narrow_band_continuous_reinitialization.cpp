#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/vector.h>

#include <deal.II/non_matching/fe_immersed_values.h>
#include <deal.II/non_matching/quadrature_generator.h>

#include <meltpooldg/level_set/narrow_band_continuous_reinitialization.hpp>

#include <algorithm>
#include <cmath>

namespace MeltPoolDG::LevelSet
{
  namespace
  {
    /**
     * FE collection with the structure of @p fe_collection_cut: FE_Q for the FE indices with DoFs
     * and a non-dominating FE_Nothing otherwise (no constraints at the faces to other cells).
     */
    template <int dim>
    dealii::hp::FECollection<dim>
    create_fe_collection(const dealii::hp::FECollection<dim> &fe_collection_cut,
                         const unsigned int                   fe_degree)
    {
      dealii::hp::FECollection<dim> fe_collection;
      for (unsigned int i = 0; i < fe_collection_cut.size(); ++i)
        if (fe_collection_cut[i].n_dofs_per_cell() == 0)
          fe_collection.push_back(dealii::FE_Nothing<dim>());
        else
          fe_collection.push_back(dealii::FE_Q<dim>(fe_degree));
      return fe_collection;
    }

    /**
     * Coefficient g(s) = 1/s for s > 1 and g(s) = 2 - s otherwise of the fixed-point iteration
     * (Basting & Kuzmin, 2013).
     */
    inline double
    rhs_coefficient(const double gradient_norm)
    {
      return gradient_norm > 1. ? 1. / gradient_norm : 2. - gradient_norm;
    }

    /**
     * Binomial coefficient.
     */
    inline double
    binomial(const unsigned int n, const unsigned int k)
    {
      double value = 1.;
      for (unsigned int i = 1; i <= k; ++i)
        value = value * (n - k + i) / i;
      return value;
    }

    /**
     * Data of a face of the reference cell for the closed-interface correction.
     */
    struct FaceData
    {
      /// Cell DoFs on the face.
      std::vector<unsigned int> dofs;

      /// Whether a DoF of the face (index into dofs) is a vertex of the face.
      std::vector<bool> is_vertex;

      /// Bernstein coefficients of the restriction to the face from the values at the face DoFs
      /// (inverse of the matrix of the tensor-product Bernstein polynomials at the support
      /// points of the face).
      dealii::FullMatrix<double> bernstein_coefficients;

      /// Values of the (multi)linear interpolant of the vertex values at the face DoFs (columns:
      /// face DoFs, which are vertices, in the order of dofs).
      dealii::FullMatrix<double> linear_interpolation;
    };

    /**
     * Face data of the 2 dim faces of the reference cell for the element FE_Q(p): the face of
     * number f lies at x_{f/2} = f%2, its coordinates are the other coordinates of the cell.
     */
    template <int dim>
    std::vector<FaceData>
    create_face_data(const dealii::FE_Q<dim> &fe)
    {
      const unsigned int p              = fe.degree;
      const auto        &support_points = fe.get_unit_support_points();
      constexpr double   tolerance      = 1e-10;

      std::vector<FaceData> face_data(2 * dim);
      for (unsigned int f = 0; f < 2 * dim; ++f)
        {
          const unsigned int direction = f / 2;
          const double       side      = f % 2;
          FaceData          &data      = face_data[f];

          // DoFs on the face and their face coordinates
          std::vector<std::vector<double>> coordinates;
          for (unsigned int i = 0; i < support_points.size(); ++i)
            if (std::abs(support_points[i][direction] - side) < tolerance)
              {
                data.dofs.push_back(i);
                std::vector<double> xi;
                for (unsigned int d = 0; d < dim; ++d)
                  if (d != direction)
                    xi.push_back(support_points[i][d]);
                coordinates.push_back(xi);
              }
          const unsigned int n_face_dofs = data.dofs.size();
          AssertDimension(n_face_dofs, dealii::Utilities::pow(p + 1, dim - 1));

          // matrix of the tensor-product Bernstein polynomials
          // B_k(xi) = prod_e binom(p, k_e) xi_e^k_e (1 - xi_e)^(p - k_e) at the face DoFs
          dealii::FullMatrix<double> bernstein(n_face_dofs, n_face_dofs);
          for (unsigned int j = 0; j < n_face_dofs; ++j)
            for (unsigned int k = 0; k < n_face_dofs; ++k)
              {
                double       value = 1.;
                unsigned int index = k;
                for (unsigned int e = 0; e + 1 < dim; ++e, index /= p + 1)
                  {
                    const unsigned int k_e  = index % (p + 1);
                    const double       xi_e = coordinates[j][e];
                    value *= binomial(p, k_e) * std::pow(xi_e, k_e) * std::pow(1. - xi_e, p - k_e);
                  }
                bernstein(j, k) = value;
              }
          data.bernstein_coefficients = bernstein;
          data.bernstein_coefficients.gauss_jordan();

          // vertices of the face and (multi)linear interpolation of their values
          std::vector<unsigned int> vertex_bits;
          data.is_vertex.assign(n_face_dofs, false);
          for (unsigned int j = 0; j < n_face_dofs; ++j)
            {
              bool         is_vertex = true;
              unsigned int bits      = 0;
              for (unsigned int e = 0; e + 1 < dim; ++e)
                if (std::abs(coordinates[j][e] - 1.) < tolerance)
                  bits |= 1u << e;
                else if (std::abs(coordinates[j][e]) > tolerance)
                  is_vertex = false;
              if (is_vertex)
                {
                  data.is_vertex[j] = true;
                  vertex_bits.push_back(bits);
                }
            }
          AssertDimension(static_cast<unsigned int>(vertex_bits.size()), 1u << (dim - 1));

          data.linear_interpolation.reinit(n_face_dofs, vertex_bits.size());
          for (unsigned int j = 0; j < n_face_dofs; ++j)
            for (unsigned int v = 0; v < vertex_bits.size(); ++v)
              {
                double value = 1.;
                for (unsigned int e = 0; e + 1 < dim; ++e)
                  value *=
                    (vertex_bits[v] & (1u << e)) ? coordinates[j][e] : 1. - coordinates[j][e];
                data.linear_interpolation(j, v) = value;
              }
        }

      return face_data;
    }
  } // namespace

  template <int dim, typename number>
  NarrowBandContinuousReinitialization<dim, number>::NarrowBandContinuousReinitialization(
    const dealii::Mapping<dim>          &mapping,
    const dealii::Triangulation<dim>    &triangulation,
    const ReinitializationData<number>  &reinit_data,
    const unsigned int                   fe_degree,
    const dealii::hp::FECollection<dim> &fe_collection_cut,
    const dealii::ConditionalOStream    &pcout)
    : mapping(mapping)
    , reinit_data(reinit_data)
    , fe_degree(fe_degree)
    , pcout(pcout)
    , fe(fe_degree)
    , fe_collection(create_fe_collection<dim>(fe_collection_cut, fe_degree))
    , dof_handler(triangulation)
  {}

  template <int dim, typename number>
  void
  NarrowBandContinuousReinitialization<dim, number>::distribute_dofs(
    const dealii::DoFHandler<dim> &dof_handler_cut)
  {
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        cell->set_active_fe_index(
          cell->as_dof_handler_iterator(dof_handler_cut)->active_fe_index());

    dof_handler.distribute_dofs(fe_collection);

    locally_owned_dofs    = dof_handler.locally_owned_dofs();
    locally_relevant_dofs = dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);

    hanging_node_constraints.clear();
    hanging_node_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    dealii::DoFTools::make_hanging_node_constraints(dof_handler, hanging_node_constraints);
    hanging_node_constraints.close();

    const MPI_Comm mpi_comm = dof_handler.get_mpi_communicator();
    level_set.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
    prescribed_dofs.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_comm);

    cell_matrices.clear();
    cell_matrices.resize(dof_handler.get_triangulation().n_active_cells());
  }

  template <int dim, typename number>
  void
  NarrowBandContinuousReinitialization<dim, number>::interpolate_from_discontinuous(
    const dealii::DoFHandler<dim> &dof_handler_cut,
    const VectorType              &level_set_dg)
  {
    VectorType sum(locally_owned_dofs, locally_relevant_dofs, dof_handler.get_mpi_communicator());
    VectorType count(sum);

    dealii::FullMatrix<double>                   interpolation;
    dealii::Vector<double>                       values_dg;
    std::vector<dealii::types::global_dof_index> dof_indices(fe.n_dofs_per_cell());

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
          continue;

        const auto  cell_dg = cell->as_dof_handler_iterator(dof_handler_cut);
        const auto &fe_dg   = cell_dg->get_fe();
        if (interpolation.m() != fe.n_dofs_per_cell() or
            interpolation.n() != fe_dg.n_dofs_per_cell())
          {
            interpolation.reinit(fe.n_dofs_per_cell(), fe_dg.n_dofs_per_cell());
            for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
              for (unsigned int j = 0; j < fe_dg.n_dofs_per_cell(); ++j)
                interpolation(i, j) = fe_dg.shape_value(j, fe.get_unit_support_points()[i]);
          }

        values_dg.reinit(fe_dg.n_dofs_per_cell());
        cell_dg->get_dof_values(level_set_dg, values_dg);
        cell->get_dof_indices(dof_indices);
        for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
          {
            double value = 0.;
            for (unsigned int j = 0; j < fe_dg.n_dofs_per_cell(); ++j)
              value += interpolation(i, j) * values_dg[j];
            sum(dof_indices[i]) += value;
            count(dof_indices[i]) += 1.;
          }
      }

    sum.compress(dealii::VectorOperation::add);
    count.compress(dealii::VectorOperation::add);

    level_set.zero_out_ghost_values();
    for (const auto i : locally_owned_dofs)
      level_set(i) = count(i) > 0. ? sum(i) / count(i) : 0.;
    level_set.update_ghost_values();
  }

  template <int dim, typename number>
  void
  NarrowBandContinuousReinitialization<dim, number>::interpolate_to_discontinuous(
    const dealii::DoFHandler<dim> &dof_handler_cut,
    VectorType                    &level_set_dg) const
  {
    dealii::FullMatrix<double> interpolation;
    dealii::Vector<double>     values, values_dg;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
          continue;

        const auto  cell_dg = cell->as_dof_handler_iterator(dof_handler_cut);
        const auto &fe_dg   = cell_dg->get_fe();
        if (interpolation.m() != fe_dg.n_dofs_per_cell() or
            interpolation.n() != fe.n_dofs_per_cell())
          {
            interpolation.reinit(fe_dg.n_dofs_per_cell(), fe.n_dofs_per_cell());
            for (unsigned int i = 0; i < fe_dg.n_dofs_per_cell(); ++i)
              for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
                interpolation(i, j) = fe.shape_value(j, fe_dg.get_unit_support_points()[i]);
          }

        values.reinit(fe.n_dofs_per_cell());
        cell->get_dof_values(level_set, values);
        values_dg.reinit(fe_dg.n_dofs_per_cell());
        interpolation.vmult(values_dg, values);
        cell_dg->set_dof_values(values_dg, level_set_dg);
      }
  }

  template <int dim, typename number>
  void
  NarrowBandContinuousReinitialization<dim, number>::mark_dofs(const std::vector<bool> &cells,
                                                               VectorType              &mask) const
  {
    mask.reinit(locally_owned_dofs, locally_relevant_dofs, dof_handler.get_mpi_communicator());
    std::vector<dealii::types::global_dof_index> dof_indices(fe.n_dofs_per_cell());
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned() and cell->get_fe().n_dofs_per_cell() > 0 and
          cells[cell->active_cell_index()])
        {
          cell->get_dof_indices(dof_indices);
          for (const auto i : dof_indices)
            mask(i) = 1.;
        }
    mask.compress(dealii::VectorOperation::add);
    for (const auto i : locally_owned_dofs)
      mask(i) = mask(i) > 0. ? 1. : 0.;
    mask.update_ghost_values();
  }

  template <int dim, typename number>
  typename NarrowBandInterfacePoints<dim, number>::ProjectionStatistics
  NarrowBandContinuousReinitialization<dim, number>::project_subcell_distance(
    const NarrowBandInterfacePoints<dim, number> &interface_points,
    const dealii::DoFHandler<dim>                &dof_handler_cut,
    const VectorType                             &level_set_advected_dg,
    const double                                  buffer,
    const unsigned int                            n_subdivisions)
  {
    const MPI_Comm     mpi_comm = dof_handler.get_mpi_communicator();
    const unsigned int n_dofs   = fe.n_dofs_per_cell();

    struct LocalSystem
    {
      dealii::FullMatrix<double>                   matrix;
      dealii::Vector<double>                       rhs;
      std::vector<dealii::types::global_dof_index> dof_indices;
    };
    std::vector<LocalSystem> local_systems;
    std::vector<bool>        is_projected(dof_handler.get_triangulation().n_active_cells(), false);

    dealii::FEValues<dim> fe_values(mapping,
                                    fe,
                                    dealii::QIterated<dim>(dealii::QGauss<1>(fe_degree + 1),
                                                           n_subdivisions),
                                    dealii::update_values | dealii::update_JxW_values);

    const auto statistics = interface_points.for_each_unresolved_cell(
      mapping,
      dof_handler_cut,
      level_set_advected_dg,
      buffer,
      n_subdivisions,
      false,
      [&](const typename dealii::DoFHandler<dim>::active_cell_iterator &cell_dg,
          const dealii::FEValues<dim> &,
          const std::vector<double> &distances,
          const std::vector<unsigned int> &,
          const bool) {
        const auto cell = cell_dg->as_dof_handler_iterator(dof_handler);
        fe_values.reinit(cell);
        AssertDimension(fe_values.n_quadrature_points, distances.size());

        LocalSystem system;
        system.matrix.reinit(n_dofs, n_dofs);
        system.rhs.reinit(n_dofs);
        system.dof_indices.resize(n_dofs);
        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
          for (unsigned int i = 0; i < n_dofs; ++i)
            {
              const double value_i = fe_values.shape_value(i, q) * fe_values.JxW(q);
              system.rhs[i] += value_i * distances[q];
              for (unsigned int j = 0; j < n_dofs; ++j)
                system.matrix(i, j) += value_i * fe_values.shape_value(j, q);
            }
        cell->get_dof_indices(system.dof_indices);
        local_systems.push_back(std::move(system));
        is_projected[cell->active_cell_index()] = true;
      });

    mark_dofs(is_projected, prescribed_dofs);
    if (statistics.n_cells == 0)
      return statistics;

    // all DoFs outside of the projected region keep their values
    constraints.clear();
    constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    for (const auto i : locally_relevant_dofs)
      if (prescribed_dofs(i) == 0. and not constraints.is_constrained(i))
        {
          constraints.add_line(i);
          constraints.set_inhomogeneity(i, level_set(i));
        }
    constraints.close();

    dealii::DynamicSparsityPattern dsp(locally_relevant_dofs);
    dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, true);
    dealii::SparsityTools::distribute_sparsity_pattern(dsp,
                                                       locally_owned_dofs,
                                                       mpi_comm,
                                                       locally_relevant_dofs);
    system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_comm);

    VectorType rhs(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
    for (const auto &system : local_systems)
      constraints.distribute_local_to_global(
        system.matrix, system.rhs, system.dof_indices, system_matrix, rhs);
    system_matrix.compress(dealii::VectorOperation::add);
    rhs.compress(dealii::VectorOperation::add);

    // unit diagonal for the constrained rows without contributions
    for (const auto i : locally_owned_dofs)
      if (constraints.is_constrained(i) and system_matrix.diag_element(i) == 0.)
        system_matrix.set(i, i, 1.);
    system_matrix.compress(dealii::VectorOperation::insert);

    VectorType                             solution(rhs);
    dealii::TrilinosWrappers::SolverDirect solver;
    solver.initialize(system_matrix);
    solver.solve(solution, rhs);
    constraints.distribute(solution);

    level_set.zero_out_ghost_values();
    level_set.copy_locally_owned_data_from(solution);
    level_set.update_ghost_values();

    return statistics;
  }

  template <int dim, typename number>
  void
  NarrowBandContinuousReinitialization<dim, number>::assemble_rhs(const VectorType &phi,
                                                                  VectorType       &rhs) const
  {
    rhs = 0.;

    const unsigned int                           n_dofs = fe.n_dofs_per_cell();
    dealii::FEValues<dim>                        fe_values(mapping,
                                    fe,
                                    dealii::QGauss<dim>(fe_degree + 1),
                                    dealii::update_gradients | dealii::update_JxW_values);
    std::vector<dealii::Tensor<1, dim>>          gradients(fe_values.n_quadrature_points);
    dealii::Vector<double>                       cell_rhs(n_dofs);
    std::vector<dealii::types::global_dof_index> dof_indices(n_dofs);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
          continue;

        fe_values.reinit(cell);
        fe_values.get_function_gradients(phi, gradients);
        cell_rhs = 0.;
        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
          {
            const auto flux =
              rhs_coefficient(gradients[q].norm()) * gradients[q] * fe_values.JxW(q);
            for (unsigned int i = 0; i < n_dofs; ++i)
              cell_rhs[i] += flux * fe_values.shape_grad(i, q);
          }

        cell->get_dof_indices(dof_indices);
        constraints.distribute_local_to_global(cell_rhs,
                                               dof_indices,
                                               rhs,
                                               cell_matrices[cell->active_cell_index()]);
      }

    rhs.compress(dealii::VectorOperation::add);
  }

  template <int dim, typename number>
  typename NarrowBandContinuousReinitialization<dim, number>::Statistics
  NarrowBandContinuousReinitialization<dim, number>::reinitialize(
    const SurfaceQuadratureFunction &surface_quadrature_function,
    const bool                       keep_prescribed_dofs)
  {
    const MPI_Comm     mpi_comm = dof_handler.get_mpi_communicator();
    const unsigned int n_dofs   = fe.n_dofs_per_cell();
    const double       penalty  = reinit_data.elliptic.penalty_parameter;
    const auto        &nlsolve  = reinit_data.elliptic.nlsolve;

    Statistics statistics;
    statistics.n_dofs = dof_handler.n_dofs();

    // constraints: prescribed DoFs (subcell distance) keep their values
    constraints.clear();
    constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    dealii::types::global_dof_index n_prescribed = 0;
    if (keep_prescribed_dofs)
      for (const auto i : locally_relevant_dofs)
        if (prescribed_dofs(i) > 0. and not constraints.is_constrained(i))
          {
            constraints.add_line(i);
            constraints.set_inhomogeneity(i, level_set(i));
            if (locally_owned_dofs.is_element(i))
              ++n_prescribed;
          }
    constraints.close();
    statistics.n_prescribed_dofs = dealii::Utilities::MPI::sum(n_prescribed, mpi_comm);

    dealii::DynamicSparsityPattern dsp(locally_relevant_dofs);
    dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    dealii::SparsityTools::distribute_sparsity_pattern(dsp,
                                                       locally_owned_dofs,
                                                       mpi_comm,
                                                       locally_relevant_dofs);
    system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_comm);

    // matrix (grad phi, grad v) + gamma (phi, v)_Gamma; Gamma: given quadrature (points) or the
    // zero level set of the level set before the reinitialization
    const dealii::hp::QCollection<1> q_collection{dealii::QGauss<1>(fe_degree + 1)};
    dealii::NonMatching::DiscreteQuadratureGenerator<dim> quadrature_generator(q_collection,
                                                                               dof_handler,
                                                                               level_set);
    dealii::FEValues<dim>                                 fe_values(mapping,
                                    fe,
                                    dealii::QGauss<dim>(fe_degree + 1),
                                    dealii::update_gradients | dealii::update_JxW_values);
    std::vector<dealii::types::global_dof_index>          dof_indices(n_dofs);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
          continue;

        auto &cell_matrix = cell_matrices[cell->active_cell_index()];
        cell_matrix.reinit(n_dofs, n_dofs);

        fe_values.reinit(cell);
        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
          for (unsigned int i = 0; i < n_dofs; ++i)
            for (unsigned int j = 0; j < n_dofs; ++j)
              cell_matrix(i, j) +=
                fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) * fe_values.JxW(q);

        dealii::NonMatching::ImmersedSurfaceQuadrature<dim> surface_quadrature;
        if (surface_quadrature_function)
          surface_quadrature = surface_quadrature_function(cell);
        if (surface_quadrature.size() == 0)
          {
            quadrature_generator.generate(cell);
            surface_quadrature = quadrature_generator.get_surface_quadrature();
          }

        if (penalty != 0. and surface_quadrature.size() > 0)
          {
            dealii::NonMatching::FEImmersedSurfaceValues<dim> surface_values(
              mapping, fe, surface_quadrature, dealii::update_values | dealii::update_JxW_values);
            surface_values.reinit(cell);
            for (unsigned int q = 0; q < surface_quadrature.size(); ++q)
              for (unsigned int i = 0; i < n_dofs; ++i)
                for (unsigned int j = 0; j < n_dofs; ++j)
                  cell_matrix(i, j) += penalty * surface_values.shape_value(i, q) *
                                       surface_values.shape_value(j, q) * surface_values.JxW(q);
          }

        cell->get_dof_indices(dof_indices);
        constraints.distribute_local_to_global(cell_matrix, dof_indices, system_matrix);
      }
    system_matrix.compress(dealii::VectorOperation::add);

    dealii::TrilinosWrappers::SolverDirect solver;
    solver.initialize(system_matrix);

    // fixed-point iteration with the convergence criteria of FixedPointSolver
    VectorType level_set_initial(level_set);
    VectorType phi(level_set);
    phi.zero_out_ghost_values();
    constraints.distribute(phi);
    phi.update_ghost_values();

    VectorType rhs(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
    VectorType rhs_new(rhs), phi_new(rhs), difference(rhs);
    assemble_rhs(phi, rhs);

    const unsigned int max_iterations =
      nlsolve.max_nonlinear_iterations + nlsolve.max_nonlinear_iterations_alt;
    statistics.converged = false;
    for (unsigned int iteration = 0; iteration < max_iterations; ++iteration)
      {
        phi_new.zero_out_ghost_values();
        solver.solve(phi_new, rhs);
        constraints.distribute(phi_new);
        phi_new.update_ghost_values();

        assemble_rhs(phi_new, rhs_new);

        difference.zero_out_ghost_values();
        difference.copy_locally_owned_data_from(phi_new);
        difference.add(-1., phi);
        const double update_norm = difference.l2_norm();

        difference.copy_locally_owned_data_from(rhs);
        difference.add(-1., rhs_new);
        constraints.set_zero(difference);
        const double residual_norm = difference.l2_norm();

        phi = phi_new;
        phi.update_ghost_values();
        rhs                     = rhs_new;
        statistics.n_iterations = iteration + 1;

        const bool alternative =
          iteration >= static_cast<unsigned int>(nlsolve.max_nonlinear_iterations);
        if (residual_norm <
              (alternative ? nlsolve.residual_tolerance_alt : nlsolve.residual_tolerance) and
            update_norm < (alternative ? nlsolve.field_correction_tolerance_alt :
                                         nlsolve.field_correction_tolerance))
          {
            statistics.converged = true;
            break;
          }
      }

    level_set.zero_out_ghost_values();
    level_set.copy_locally_owned_data_from(statistics.converged ? phi : level_set_initial);
    level_set.update_ghost_values();

    return statistics;
  }

  template <int dim, typename number>
  typename NarrowBandContinuousReinitialization<dim, number>::CorrectionStatistics
  NarrowBandContinuousReinitialization<dim, number>::close_interface(
    const std::vector<int> &cell_signs,
    const double            offset)
  {
    const auto &triangulation = dof_handler.get_triangulation();
    AssertDimension(cell_signs.size(), triangulation.n_active_cells());
    AssertThrow(not triangulation.has_hanging_nodes(),
                dealii::ExcMessage("The closed-interface correction does not support hanging "
                                   "nodes."));

    const MPI_Comm     mpi_comm  = dof_handler.get_mpi_communicator();
    const unsigned int n_dofs    = fe.n_dofs_per_cell();
    const auto         face_data = create_face_data<dim>(fe);

    CorrectionStatistics statistics;

    // faces between the locally owned cut cells and the cells, which are not cut (with a sign);
    // faces at the boundary of the domain are excluded (the interface may end there)
    struct BoundaryFace
    {
      std::vector<dealii::types::global_dof_index> dof_indices; // cell DoFs
      unsigned int                                 face_no;
      int                                          sign;
      double                                       cell_size;
      bool                                         is_linearized;
    };
    std::vector<BoundaryFace> faces;
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
          continue;

        for (const unsigned int f : cell->face_indices())
          {
            if (cell->at_boundary(f) and not cell->has_periodic_neighbor(f))
              continue;

            const auto neighbor =
              cell->at_boundary(f) ? cell->periodic_neighbor(f) : cell->neighbor(f);
            if (neighbor->get_fe().n_dofs_per_cell() > 0)
              continue;

            const int sign = cell_signs[neighbor->active_cell_index()];
            if (sign == 0)
              continue;

            BoundaryFace face{std::vector<dealii::types::global_dof_index>(n_dofs),
                              f,
                              sign,
                              cell->minimum_vertex_distance(),
                              false};
            cell->get_dof_indices(face.dof_indices);
            faces.push_back(std::move(face));
          }
      }

    // 1) clamping of the nodal values on the faces: s phi_i >= epsilon h (nodes on faces of
    // opposite signs are not changed)
    VectorType positive(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
    VectorType negative(positive), cell_sizes(positive);
    for (const auto &face : faces)
      for (const unsigned int i : face_data[face.face_no].dofs)
        {
          const auto index = face.dof_indices[i];
          (face.sign > 0 ? positive : negative)(index) += 1.;
          cell_sizes(index) += face.cell_size;
        }
    positive.compress(dealii::VectorOperation::add);
    negative.compress(dealii::VectorOperation::add);
    cell_sizes.compress(dealii::VectorOperation::add);

    double max_correction = 0.;
    level_set.zero_out_ghost_values();
    for (const auto i : locally_owned_dofs)
      {
        const double n_positive = positive(i), n_negative = negative(i);
        if (n_positive == 0. and n_negative == 0.)
          continue;

        if (n_positive > 0. and n_negative > 0.)
          {
            ++statistics.n_conflicting_dofs;
            continue;
          }

        const double sign      = n_positive > 0. ? 1. : -1.;
        const double cell_size = cell_sizes(i) / (n_positive + n_negative);
        if (sign * level_set(i) < offset * cell_size)
          {
            max_correction =
              std::max(max_correction,
                       std::abs(sign * offset * cell_size - level_set(i)) / cell_size);
            level_set(i) = sign * offset * cell_size;
            ++statistics.n_clamped_dofs;
          }
      }
    level_set.update_ghost_values();

    // 2) Bernstein test of a face: all Bernstein coefficients of the restriction of the level set
    // to the face have the sign of the neighbor cell (sufficient for s phi > 0 on the face)
    dealii::Vector<double> face_values, coefficients;
    const auto             passes_test = [&](const BoundaryFace &face) {
      const FaceData &data = face_data[face.face_no];
      face_values.reinit(data.dofs.size());
      coefficients.reinit(data.dofs.size());
      for (unsigned int j = 0; j < data.dofs.size(); ++j)
        face_values[j] = level_set(face.dof_indices[data.dofs[j]]);
      data.bernstein_coefficients.vmult(coefficients, face_values);
      for (const double coefficient : coefficients)
        if (face.sign * coefficient <= 0.)
          return false;
      return true;
    };

    // 3) faces, which fail the test: (multi)linear interpolant of the vertex values at the other
    // nodes of the face; repeated, since the nodes of an edge (3D) are shared by several faces
    // (each face is linearized at most once; the vertex values are not changed)
    VectorType             sum(positive), count(positive);
    dealii::Vector<double> vertex_values, linear_values;
    while (true)
      {
        sum   = 0.;
        count = 0.;

        unsigned int n_linearized_faces = 0;
        for (auto &face : faces)
          {
            if (face.is_linearized or passes_test(face))
              continue;

            const FaceData &data = face_data[face.face_no];
            vertex_values.reinit(data.linear_interpolation.n());
            linear_values.reinit(data.dofs.size());
            for (unsigned int j = 0, v = 0; j < data.dofs.size(); ++j)
              if (data.is_vertex[j])
                vertex_values[v++] = level_set(face.dof_indices[data.dofs[j]]);
            data.linear_interpolation.vmult(linear_values, vertex_values);

            for (unsigned int j = 0; j < data.dofs.size(); ++j)
              if (not data.is_vertex[j])
                {
                  const auto index = face.dof_indices[data.dofs[j]];
                  sum(index) += linear_values[j];
                  count(index) += 1.;
                  max_correction =
                    std::max(max_correction,
                             std::abs(linear_values[j] - level_set(index)) / face.cell_size);
                }

            face.is_linearized = true;
            ++n_linearized_faces;
          }

        n_linearized_faces = dealii::Utilities::MPI::sum(n_linearized_faces, mpi_comm);
        if (n_linearized_faces == 0)
          break;
        statistics.n_linearized_faces += n_linearized_faces;

        sum.compress(dealii::VectorOperation::add);
        count.compress(dealii::VectorOperation::add);
        level_set.zero_out_ghost_values();
        for (const auto i : locally_owned_dofs)
          if (count(i) > 0.)
            level_set(i) = sum(i) / count(i);
        level_set.update_ghost_values();
      }

    // faces, which still fail the test (conflicting nodes)
    for (const auto &face : faces)
      if (not passes_test(face))
        ++statistics.n_open_faces;

    statistics.n_faces =
      dealii::Utilities::MPI::sum(static_cast<unsigned int>(faces.size()), mpi_comm);
    statistics.n_clamped_dofs = dealii::Utilities::MPI::sum(statistics.n_clamped_dofs, mpi_comm);
    statistics.n_conflicting_dofs =
      dealii::Utilities::MPI::sum(statistics.n_conflicting_dofs, mpi_comm);
    statistics.n_open_faces   = dealii::Utilities::MPI::sum(statistics.n_open_faces, mpi_comm);
    statistics.max_correction = dealii::Utilities::MPI::max(max_correction, mpi_comm);

    return statistics;
  }

  template <int dim, typename number>
  const typename NarrowBandContinuousReinitialization<dim, number>::VectorType &
  NarrowBandContinuousReinitialization<dim, number>::get_level_set() const
  {
    return level_set;
  }

  template class NarrowBandContinuousReinitialization<1, double>;
  template class NarrowBandContinuousReinitialization<2, double>;
  template class NarrowBandContinuousReinitialization<3, double>;
} // namespace MeltPoolDG::LevelSet
