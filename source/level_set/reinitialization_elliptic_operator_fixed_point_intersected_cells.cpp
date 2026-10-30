#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/tools.h>

#include <meltpooldg/level_set/reinitialization_elliptic_operator_fixed_point_intersected_cells.hpp>

#include <algorithm>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim, typename number>
  ReinitializationEllipticOperatorFixedPointIntersectedCells<dim, number>::
    ReinitializationEllipticOperatorFixedPointIntersectedCells(
      const ScratchData<dim, dim, number> &scratch_data_in,
      const ReinitializationData<number>  &reinit_data_in,
      const unsigned int                   ls_dof_idx_in,
      const unsigned int                   reinit_quad_idx_in,
      const MappingInfoType               &mapping_info_surface_in)
    : scratch_data(scratch_data_in)
    , reinit_data(reinit_data_in)
    , reinit_quad_idx(reinit_quad_idx_in)
    , mapping_info_surface(mapping_info_surface_in)
  {
    this->reset_dof_index(ls_dof_idx_in);
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperatorFixedPointIntersectedCells<dim, number>::reinit()
  {
    const auto &dof_handler   = scratch_data.get_dof_handler(this->dof_idx);
    const auto &fe_collection = dof_handler.get_fe_collection();

    AssertThrow(dof_handler.has_hp_capabilities() and
                  fe_collection.size() > CutUtil::CellCategory::intersected,
                ExcMessage("The elliptic reinitialization on intersected cells requires a "
                           "DoFHandler with hp-capabilities, where the FE index "
                           "CutUtil::CellCategory::intersected denotes the intersected cells."));

    for (unsigned int i = 0; i < fe_collection.size(); ++i)
      {
        if (i == CutUtil::CellCategory::intersected)
          AssertThrow(dynamic_cast<const FE_DGQ<dim> *>(&fe_collection[i]) != nullptr,
                      ExcMessage("The elliptic reinitialization on intersected cells requires "
                                 "an FE_DGQ element on the intersected cells."));
        else
          AssertThrow(fe_collection[i].n_dofs_per_cell() == 0,
                      ExcMessage("The elliptic reinitialization on intersected cells requires "
                                 "FE_Nothing elements on the non-intersected cells."));
      }

    const auto &fe = fe_collection[CutUtil::CellCategory::intersected];

    // the FE might have changed, e.g., after a change of the parameters
    if (fe_point == nullptr or fe_point->degree != fe.degree)
      fe_point = std::make_unique<FE_DGQ<dim>>(fe.degree);

    fe_degree       = fe.degree;
    n_dofs_per_cell = fe.n_dofs_per_cell();

    // Interior penalty parameter: same scaling as in the elliptic DG reinitialization on the
    // full domain (ReinitializationEllipticOperatorFixedPoint), with the cell size divided by
    // the polynomial degree.
    const number p   = std::max<number>(1., fe_degree);
    interior_penalty = reinit_data.elliptic.interior_penalty_scaling_factor * p * p /
                       (scratch_data.get_min_cell_size() / p);
  }

  template <int dim, typename number>
  unsigned int
  ReinitializationEllipticOperatorFixedPointIntersectedCells<dim, number>::get_fe_degree() const
  {
    return fe_degree;
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperatorFixedPointIntersectedCells<dim, number>::
    set_face_terms_on_interface_faces_only(const bool flag)
  {
    face_terms_on_interface_faces_only = flag;
  }

  template <int dim, typename number>
  std::pair<unsigned int, unsigned int>
  ReinitializationEllipticOperatorFixedPointIntersectedCells<dim, number>::update_face_mask(
    const VectorType &level_set)
  {
    Assert(fe_point != nullptr, ExcMessage("Call reinit() before using the operator."));

    const auto &matrix_free = scratch_data.get_matrix_free();
    const auto &fe          = scratch_data.get_dof_handler(this->dof_idx)
                       .get_fe_collection()[CutUtil::CellCategory::intersected];

    const bool update_ghosts = not level_set.has_ghost_elements();
    if (update_ghosts)
      level_set.update_ghost_values();

    // sample points on the faces and cells: Gauss-Lobatto points including the vertices
    const Quadrature<dim - 1> face_quadrature = [&]() -> Quadrature<dim - 1> {
      if constexpr (dim == 1)
        return QGauss<0>(1);
      else
        return QGaussLobatto<dim - 1>(fe.degree + 2);
    }();
    const QGaussLobatto<dim> cell_quadrature(fe.degree + 2);

    FEFaceValues<dim> fe_face_values(scratch_data.get_mapping(),
                                     fe,
                                     face_quadrature,
                                     update_values);
    FEValues<dim>     fe_values(scratch_data.get_mapping(), fe, cell_quadrature, update_values);

    std::vector<number> face_values(face_quadrature.size());
    std::vector<number> cell_values(cell_quadrature.size());

    // the level set attains values of both signs (including zero)
    const auto has_sign_change = [](const std::vector<number> &values) {
      const auto [min, max] = std::minmax_element(values.begin(), values.end());
      return *min <= 0.0 and *max >= 0.0;
    };

    unsigned int n_faces           = 0;
    unsigned int n_interface_faces = 0;

    face_mask.resize_fast(matrix_free.n_inner_face_batches());
    for (unsigned int face = 0; face < matrix_free.n_inner_face_batches(); ++face)
      {
        face_mask[face] = VectorizedArrayType(0.0);

        if (not is_intersected_face_range(std::make_pair(face, face + 1)))
          continue;

        for (unsigned int lane = 0; lane < matrix_free.n_active_entries_per_face_batch(face);
             ++lane)
          {
            bool is_interface_face          = false;
            bool has_cell_without_interface = false;

            for (const bool is_interior : {true, false})
              {
                const auto [cell, face_no] =
                  matrix_free.get_face_iterator(face, lane, is_interior, this->dof_idx);

                fe_face_values.reinit(cell, face_no);
                fe_face_values.get_function_values(level_set, face_values);
                if (has_sign_change(face_values))
                  is_interface_face = true;

                fe_values.reinit(cell);
                fe_values.get_function_values(level_set, cell_values);
                if (not has_sign_change(cell_values))
                  has_cell_without_interface = true;
              }

            // keep the faces of intersected cells, which are not cut by the interface
            if (has_cell_without_interface)
              is_interface_face = true;

            face_mask[face][lane] = is_interface_face ? 1.0 : 0.0;

            ++n_faces;
            if (is_interface_face)
              ++n_interface_faces;
          }
      }

    if (update_ghosts)
      level_set.zero_out_ghost_values();

    const auto comm = level_set.get_mpi_communicator();
    return {Utilities::MPI::sum(n_interface_faces, comm), Utilities::MPI::sum(n_faces, comm)};
  }

  template <int dim, typename number>
  typename ReinitializationEllipticOperatorFixedPointIntersectedCells<dim,
                                                                      number>::VectorizedArrayType
  ReinitializationEllipticOperatorFixedPointIntersectedCells<dim, number>::get_face_mask(
    const unsigned int face_batch) const
  {
    if (not face_terms_on_interface_faces_only)
      return VectorizedArrayType(1.0);

    AssertIndexRange(face_batch, face_mask.size());
    return face_mask[face_batch];
  }

  template <int dim, typename number>
  bool
  ReinitializationEllipticOperatorFixedPointIntersectedCells<dim, number>::
    is_intersected_cell_range(const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    return scratch_data.get_matrix_free().get_cell_range_category(cell_range, this->dof_idx) ==
           CutUtil::CellCategory::intersected;
  }

  template <int dim, typename number>
  bool
  ReinitializationEllipticOperatorFixedPointIntersectedCells<dim, number>::
    is_intersected_face_range(const std::pair<unsigned int, unsigned int> &face_range) const
  {
    const auto face_category =
      scratch_data.get_matrix_free().get_face_range_category(face_range, this->dof_idx);

    return face_category.first == CutUtil::CellCategory::intersected and
           face_category.second == CutUtil::CellCategory::intersected;
  }

  template <int dim, typename number>
  FECellIntegrator<dim, 1, number>
  ReinitializationEllipticOperatorFixedPointIntersectedCells<dim, number>::create_cell_integrator()
    const
  {
    return FECellIntegrator<dim, 1, number>(scratch_data.get_matrix_free(),
                                            this->dof_idx,
                                            reinit_quad_idx,
                                            0 /*first selected component*/,
                                            CutUtil::CellCategory::intersected);
  }

  template <int dim, typename number>
  FEFaceIntegrator<dim, 1, number>
  ReinitializationEllipticOperatorFixedPointIntersectedCells<dim, number>::create_face_integrator(
    const bool is_interior_face) const
  {
    return FEFaceIntegrator<dim, 1, number>(scratch_data.get_matrix_free(),
                                            is_interior_face,
                                            this->dof_idx,
                                            reinit_quad_idx,
                                            0 /*first selected component*/,
                                            CutUtil::CellCategory::intersected);
  }

  template <int dim, typename number>
  typename ReinitializationEllipticOperatorFixedPointIntersectedCells<dim,
                                                                      number>::PointEvaluationType
  ReinitializationEllipticOperatorFixedPointIntersectedCells<dim, number>::create_point_evaluator()
    const
  {
    Assert(fe_point != nullptr, ExcMessage("Call reinit() before using the operator."));

    // The DoF values of the FEEvaluation objects are stored in lexicographic numbering.
    return PointEvaluationType(mapping_info_surface,
                               *fe_point,
                               0 /*first selected component*/,
                               true /*force lexicographic numbering*/);
  }

  template <int dim, typename number>
  typename ReinitializationEllipticOperatorFixedPointIntersectedCells<dim,
                                                                      number>::VectorizedArrayType
  ReinitializationEllipticOperatorFixedPointIntersectedCells<dim, number>::rhs_coefficient(
    const VectorizedArrayType &gradient_norm)
  {
    // g(s) = 1/s for s > 1 and g(s) = 2 - s for s <= 1 (Basting & Kuzmin, 2013)
    const VectorizedArrayType one(1.0);
    const VectorizedArrayType two(2.0);

    // avoid a division by zero in the branch which is not selected
    const VectorizedArrayType safe_norm = std::max(gradient_norm, one);

    return compare_and_apply_mask<SIMDComparison::greater_than>(gradient_norm,
                                                                one,
                                                                one / safe_norm,
                                                                two - gradient_norm);
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperatorFixedPointIntersectedCells<dim, number>::lhs_cell_operation(
    FECellIntegrator<dim, 1, number>           &phi,
    PointEvaluationType                        &point_eval,
    dealii::AlignedVector<VectorizedArrayType> &penalty_buffer) const
  {
    constexpr unsigned int n_lanes             = VectorizedArrayType::size();
    const unsigned int     cell_batch          = phi.get_current_cell_index();
    const number           penalty_coefficient = reinit_data.elliptic.penalty_parameter;
    const auto            &matrix_free         = scratch_data.get_matrix_free();

    AssertDimension(phi.dofs_per_cell, n_dofs_per_cell);

    // (1) interface penalty term: gamma (phi, v)_Gamma
    penalty_buffer.resize_fast(n_dofs_per_cell);
    for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
      penalty_buffer[i] = VectorizedArrayType(0.0);

    if (penalty_coefficient != 0.0)
      for (unsigned int lane = 0; lane < matrix_free.n_active_entries_per_cell_batch(cell_batch);
           ++lane)
        {
          point_eval.reinit(cell_batch * n_lanes + lane);

          // cells which are not cut by the zero level-set of the initial level-set field (but
          // still belong to the intersected cells) do not contribute
          const auto q_indices = point_eval.quadrature_point_indices();
          if (q_indices.begin() == q_indices.end())
            continue;

          point_eval.evaluate(StridedArrayView<const number, n_lanes>(
                                &phi.begin_dof_values()[0][lane], n_dofs_per_cell),
                              EvaluationFlags::values);

          for (const unsigned int q : q_indices)
            point_eval.submit_value(penalty_coefficient * point_eval.get_value(q), q);

          point_eval.integrate(StridedArrayView<number, n_lanes>(&penalty_buffer[0][lane],
                                                                 n_dofs_per_cell),
                               EvaluationFlags::values);
        }

    // (2) Laplace operator: (grad phi, grad v)
    phi.evaluate(EvaluationFlags::gradients);
    for (const unsigned int q : phi.quadrature_point_indices())
      phi.submit_gradient(phi.get_gradient(q), q);
    phi.integrate(EvaluationFlags::gradients);

    for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
      phi.begin_dof_values()[i] += penalty_buffer[i];
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperatorFixedPointIntersectedCells<dim, number>::lhs_face_operation(
    FEFaceIntegrator<dim, 1, number> &phi_m,
    FEFaceIntegrator<dim, 1, number> &phi_p) const
  {
    // faces between two intersected cells, which are not crossed by the interface, do not
    // contribute (if the face terms are restricted to the faces crossed by the interface)
    const VectorizedArrayType mask = get_face_mask(phi_m.get_current_cell_index());

    for (const unsigned int q : phi_m.quadrature_point_indices())
      {
        // normal vector pointing from the interior (minus) to the exterior (plus) cell
        const auto normal = phi_m.normal_vector(q);

        // jump [phi] = phi^- - phi^+ and normal derivative of the average {grad phi} . n
        const auto jump = phi_m.get_value(q) - phi_p.get_value(q);
        const auto average_normal_der =
          0.5 * (phi_m.get_normal_derivative(q) + phi_p.get_normal_derivative(q));

        //   - ({grad phi}.n, [v]) + sigma ([phi], [v])
        const auto value_flux = mask * (interior_penalty * jump - average_normal_der);
        phi_m.submit_value(value_flux, q);
        phi_p.submit_value(-value_flux, q);

        //   - ([phi], {grad v}.n)
        const auto gradient_flux = (-0.5 * mask * jump) * normal;
        phi_m.submit_gradient(gradient_flux, q);
        phi_p.submit_gradient(gradient_flux, q);
      }
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperatorFixedPointIntersectedCells<dim, number>::vmult(
    VectorType       &dst,
    const VectorType &src) const
  {
    scratch_data.get_matrix_free().template loop<VectorType, VectorType>(
      [&](const auto &, auto &dst, const auto &src, const auto cell_range) {
        if (not is_intersected_cell_range(cell_range))
          return;

        auto                                       phi        = create_cell_integrator();
        auto                                       point_eval = create_point_evaluator();
        dealii::AlignedVector<VectorizedArrayType> penalty_buffer(n_dofs_per_cell);

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            phi.reinit(cell);
            phi.read_dof_values(src);

            lhs_cell_operation(phi, point_eval, penalty_buffer);

            phi.distribute_local_to_global(dst);
          }
      },
      [&](const auto &, auto &dst, const auto &src, const auto face_range) {
        // faces between an intersected and a non-intersected cell belong to the boundary of the
        // domain of the level-set field: natural boundary condition
        if (not is_intersected_face_range(face_range))
          return;

        auto phi_m = create_face_integrator(true);
        auto phi_p = create_face_integrator(false);

        for (unsigned int face = face_range.first; face < face_range.second; ++face)
          {
            if (get_face_mask(face).sum() == 0.0)
              continue;

            phi_m.reinit(face);
            phi_p.reinit(face);

            phi_m.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);
            phi_p.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

            lhs_face_operation(phi_m, phi_p);

            phi_m.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
            phi_p.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
          }
      },
      [&](const auto &, auto &, const auto &, const auto) {
        // domain boundary: natural boundary condition
      },
      dst,
      src,
      true /*zero dst vector*/);
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperatorFixedPointIntersectedCells<dim, number>::create_rhs(
    VectorType       &dst,
    const VectorType &level_set_old) const
  {
    scratch_data.get_matrix_free().template loop<VectorType, VectorType>(
      [&](const auto &, auto &dst, const auto &src, const auto cell_range) {
        if (not is_intersected_cell_range(cell_range))
          return;

        auto phi = create_cell_integrator();

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            phi.reinit(cell);
            phi.gather_evaluate(src, EvaluationFlags::gradients);

            // (g(|grad phi^k|) grad phi^k, grad v)
            for (const unsigned int q : phi.quadrature_point_indices())
              {
                const auto gradient = phi.get_gradient(q);
                phi.submit_gradient(rhs_coefficient(gradient.norm()) * gradient, q);
              }

            phi.integrate_scatter(EvaluationFlags::gradients, dst);
          }
      },
      [&](const auto &, auto &dst, const auto &src, const auto face_range) {
        if (not is_intersected_face_range(face_range))
          return;

        auto phi_m = create_face_integrator(true);
        auto phi_p = create_face_integrator(false);

        for (unsigned int face = face_range.first; face < face_range.second; ++face)
          {
            const VectorizedArrayType mask = get_face_mask(face);
            if (mask.sum() == 0.0)
              continue;

            phi_m.reinit(face);
            phi_p.reinit(face);

            phi_m.gather_evaluate(src, EvaluationFlags::gradients);
            phi_p.gather_evaluate(src, EvaluationFlags::gradients);

            // - ({g(|grad phi^k|) grad phi^k} . n, [v])
            for (const unsigned int q : phi_m.quadrature_point_indices())
              {
                const auto gradient_m = phi_m.get_gradient(q);
                const auto gradient_p = phi_p.get_gradient(q);

                const auto average_normal_flux =
                  (0.5 * mask) *
                  (rhs_coefficient(gradient_m.norm()) * phi_m.get_normal_derivative(q) +
                   rhs_coefficient(gradient_p.norm()) * phi_p.get_normal_derivative(q));

                phi_m.submit_value(-average_normal_flux, q);
                phi_p.submit_value(average_normal_flux, q);
              }

            phi_m.integrate_scatter(EvaluationFlags::values, dst);
            phi_p.integrate_scatter(EvaluationFlags::values, dst);
          }
      },
      [&](const auto &, auto &, const auto &, const auto) {
        // domain boundary: natural boundary condition
      },
      dst,
      level_set_old,
      true /*zero dst vector*/);
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperatorFixedPointIntersectedCells<dim, number>::
    compute_system_matrix_from_matrixfree(TrilinosWrappers::SparseMatrix &system_matrix) const
  {
    system_matrix = 0.0;

    // no constraints for the DG discretization
    AffineConstraints<number> empty_constraints;
    empty_constraints.close();

    auto                                       point_eval = create_point_evaluator();
    dealii::AlignedVector<VectorizedArrayType> penalty_buffer(n_dofs_per_cell);

    // Note: cells and faces of cells with FE_Nothing are skipped by MatrixFreeTools.
    MatrixFreeTools::template compute_matrix<dim, -1, 0, 1, number, VectorizedArrayType>(
      scratch_data.get_matrix_free(),
      empty_constraints,
      system_matrix,
      [&](FECellIntegrator<dim, 1, number> &phi) {
        lhs_cell_operation(phi, point_eval, penalty_buffer);
      },
      [&](FEFaceIntegrator<dim, 1, number> &phi_m, FEFaceIntegrator<dim, 1, number> &phi_p) {
        phi_m.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
        phi_p.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

        lhs_face_operation(phi_m, phi_p);

        phi_m.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        phi_p.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
      },
      nullptr /*no boundary integrals: natural boundary condition*/,
      this->dof_idx,
      reinit_quad_idx,
      0 /* first selected component */);
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperatorFixedPointIntersectedCells<dim, number>::
    compute_inverse_diagonal_from_matrixfree(VectorType &diagonal) const
  {
    scratch_data.initialize_dof_vector(diagonal, this->dof_idx);

    auto                                       point_eval = create_point_evaluator();
    dealii::AlignedVector<VectorizedArrayType> penalty_buffer(n_dofs_per_cell);

    // Note: cells and faces of cells with FE_Nothing are skipped by MatrixFreeTools.
    MatrixFreeTools::template compute_diagonal<dim, -1, 0, 1, number, VectorizedArrayType>(
      scratch_data.get_matrix_free(),
      diagonal,
      [&](FECellIntegrator<dim, 1, number> &phi) {
        lhs_cell_operation(phi, point_eval, penalty_buffer);
      },
      [&](FEFaceIntegrator<dim, 1, number> &phi_m, FEFaceIntegrator<dim, 1, number> &phi_p) {
        phi_m.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
        phi_p.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

        lhs_face_operation(phi_m, phi_p);

        phi_m.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        phi_p.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
      },
      nullptr /*no boundary integrals: natural boundary condition*/,
      this->dof_idx,
      reinit_quad_idx,
      0 /* first selected component */);

    // ... and invert it
    const number linfty_norm = std::max<number>(1.0, diagonal.linfty_norm());
    for (auto &entry : diagonal)
      entry = (std::abs(entry) > 1.0e-14 * linfty_norm) ? (1.0 / entry) : 1.0;
  }

  template class ReinitializationEllipticOperatorFixedPointIntersectedCells<1, double>;
  template class ReinitializationEllipticOperatorFixedPointIntersectedCells<2, double>;
  template class ReinitializationEllipticOperatorFixedPointIntersectedCells<3, double>;
} // namespace MeltPoolDG::LevelSet
