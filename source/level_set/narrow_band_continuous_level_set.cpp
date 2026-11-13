#include <deal.II/base/exceptions.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/hp/q_collection.h>

#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/non_matching/fe_immersed_values.h>
#include <deal.II/non_matching/quadrature_generator.h>

#include <deal.II/numerics/data_component_interpretation.h>

#include <meltpooldg/level_set/narrow_band_continuous_level_set.hpp>
#include <meltpooldg/level_set/normal_vector_operator.hpp>
#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/linear_algebra/preconditioner_factory.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  namespace
  {
    /**
     * FE collection with the same structure as @p fe_collection_cut: an FE_Q element (vector
     * valued with @p n_components components) for the FE indices with DoFs and a non-dominating
     * FE_Nothing element otherwise. The default FE_Nothing element does not dominate, i.e., it
     * imposes no constraints on the FE_Q elements of the adjacent cells.
     */
    template <int dim>
    hp::FECollection<dim>
    create_continuous_fe_collection(const hp::FECollection<dim> &fe_collection_cut,
                                    const unsigned int           fe_degree,
                                    const unsigned int           n_components)
    {
      hp::FECollection<dim> fe_collection;
      for (unsigned int i = 0; i < fe_collection_cut.size(); ++i)
        {
          // Note: FESystem::compare_for_domination() is only implemented for other FESystem
          // elements, i.e., the vector-valued FE_Nothing has to be an FESystem as well.
          if (n_components == 1)
            {
              if (fe_collection_cut[i].n_dofs_per_cell() == 0)
                fe_collection.push_back(FE_Nothing<dim>());
              else
                fe_collection.push_back(FE_Q<dim>(fe_degree));
            }
          else
            {
              if (fe_collection_cut[i].n_dofs_per_cell() == 0)
                fe_collection.push_back(FESystem<dim>(FE_Nothing<dim>(), n_components));
              else
                fe_collection.push_back(FESystem<dim>(FE_Q<dim>(fe_degree), n_components));
            }
        }
      return fe_collection;
    }
  } // namespace

  /*--------------------------- NarrowBandContinuousMassOperator ---------------------------*/

  template <int dim, typename number>
  NarrowBandContinuousMassOperator<dim, number>::NarrowBandContinuousMassOperator(
    const ScratchData<dim, dim, number> &scratch_data_in,
    const unsigned int                   dof_idx_in,
    const unsigned int                   quad_idx_in,
    const number                         damping_factor_in,
    const number                         penalty_factor_in,
    const MappingInfoType               *mapping_info_surface_in)
    : scratch_data(scratch_data_in)
    , quad_idx(quad_idx_in)
    , damping_factor(damping_factor_in)
    , penalty_factor(penalty_factor_in)
    , mapping_info_surface(mapping_info_surface_in)
  {
    this->reset_dof_index(dof_idx_in);

    AssertThrow(damping_factor >= 0. and penalty_factor >= 0.,
                ExcMessage("The damping and the penalty factor must not be negative."));
    AssertThrow(penalty_factor == 0. or mapping_info_surface != nullptr,
                ExcMessage("The interface penalty requires a surface quadrature."));
  }

  template <int dim, typename number>
  void
  NarrowBandContinuousMassOperator<dim, number>::reinit()
  {
    const auto &dof_handler   = scratch_data.get_dof_handler(this->dof_idx);
    const auto &fe_collection = dof_handler.get_fe_collection();

    AssertThrow(dof_handler.has_hp_capabilities() and
                  fe_collection.size() > CutUtil::CellCategory::intersected,
                ExcMessage("The continuous field on the cut cells requires a DoFHandler with "
                           "hp-capabilities, where the FE index CutUtil::CellCategory::intersected "
                           "denotes the cut cells."));

    for (unsigned int i = 0; i < fe_collection.size(); ++i)
      {
        if (i == CutUtil::CellCategory::intersected)
          AssertThrow(dynamic_cast<const FE_Q<dim> *>(&fe_collection[i]) != nullptr,
                      ExcMessage("The continuous field on the cut cells requires an FE_Q "
                                 "element on the cut cells."));
        else
          AssertThrow(fe_collection[i].n_dofs_per_cell() == 0,
                      ExcMessage("The continuous field on the cut cells requires FE_Nothing "
                                 "elements on all other cells."));
      }

    const auto &fe = fe_collection[CutUtil::CellCategory::intersected];
    if (fe_point == nullptr or fe_point->degree != fe.degree)
      fe_point = std::make_unique<FE_Q<dim>>(fe.degree);
    n_dofs_per_cell = fe.n_dofs_per_cell();

    // the constrained DoFs refer to the previous DoF layout
    constrained_dof_mask = nullptr;
    matrix_constraints   = nullptr;
    scratch_data.initialize_dof_vector(src_free, this->dof_idx);

    // damping eta_K = c_eta h_K^2 (as in NormalVectorOperator and CurvatureOperator) and interface
    // penalty gamma_K = c_gamma h_K per cell batch
    const auto        &matrix_free    = scratch_data.get_matrix_free();
    const unsigned int n_cell_batches = matrix_free.n_cell_batches();
    damping.resize_fast(n_cell_batches);
    penalty.resize_fast(n_cell_batches);
    for (unsigned int cell = 0; cell < n_cell_batches; ++cell)
      {
        damping[cell] = compute_cell_size_dependent_filter_parameter_mf<dim, number>(
          scratch_data, this->dof_idx, cell, damping_factor);
        penalty[cell] = penalty_factor * scratch_data.get_cell_sizes()[cell];
      }
  }

  template <int dim, typename number>
  void
  NarrowBandContinuousMassOperator<dim, number>::set_constrained_dofs(
    const VectorType                        *mask,
    const dealii::AffineConstraints<number> *matrix_constraints_in)
  {
    AssertThrow((mask == nullptr) == (matrix_constraints_in == nullptr),
                ExcMessage("The mask of the constrained DoFs and the constraints of the system "
                           "matrix have to be given together."));
    constrained_dof_mask = mask;
    matrix_constraints   = matrix_constraints_in;
  }

  template <int dim, typename number>
  bool
  NarrowBandContinuousMassOperator<dim, number>::is_cut_cell_range(
    const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    return scratch_data.get_matrix_free().get_cell_range_category(cell_range, this->dof_idx) ==
           CutUtil::CellCategory::intersected;
  }

  template <int dim, typename number>
  FECellIntegrator<dim, 1, number>
  NarrowBandContinuousMassOperator<dim, number>::create_cell_integrator() const
  {
    return FECellIntegrator<dim, 1, number>(scratch_data.get_matrix_free(),
                                            this->dof_idx,
                                            quad_idx,
                                            0 /*first selected component*/,
                                            CutUtil::CellCategory::intersected);
  }

  template <int dim, typename number>
  std::unique_ptr<typename NarrowBandContinuousMassOperator<dim, number>::PointEvaluationType>
  NarrowBandContinuousMassOperator<dim, number>::create_point_evaluator() const
  {
    if (penalty_factor == 0.)
      return nullptr;

    Assert(fe_point != nullptr, ExcMessage("Call reinit() before using the operator."));

    // The DoF values of the FEEvaluation objects are stored in lexicographic numbering.
    return std::make_unique<PointEvaluationType>(*mapping_info_surface,
                                                 *fe_point,
                                                 0 /*first selected component*/,
                                                 true /*force lexicographic numbering*/);
  }

  template <int dim, typename number>
  void
  NarrowBandContinuousMassOperator<dim, number>::cell_operation(
    FECellIntegrator<dim, 1, number>           &phi,
    PointEvaluationType                        *point_eval,
    dealii::AlignedVector<VectorizedArrayType> &penalty_buffer) const
  {
    constexpr unsigned int n_lanes     = VectorizedArrayType::size();
    const unsigned int     cell_batch  = phi.get_current_cell_index();
    const auto            &matrix_free = scratch_data.get_matrix_free();

    AssertDimension(phi.dofs_per_cell, n_dofs_per_cell);

    // (1) interface penalty: gamma_K (u, w)_Gamma
    penalty_buffer.resize_fast(n_dofs_per_cell);
    for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
      penalty_buffer[i] = VectorizedArrayType(0.0);

    if (point_eval != nullptr)
      for (unsigned int lane = 0; lane < matrix_free.n_active_entries_per_cell_batch(cell_batch);
           ++lane)
        {
          point_eval->reinit(cell_batch * n_lanes + lane);

          // cut cells, which are not crossed by the interface, do not contribute
          const auto q_indices = point_eval->quadrature_point_indices();
          if (q_indices.begin() == q_indices.end())
            continue;

          point_eval->evaluate(StridedArrayView<const number, n_lanes>(
                                 &phi.begin_dof_values()[0][lane], n_dofs_per_cell),
                               EvaluationFlags::values);

          const number gamma = penalty[cell_batch][lane];
          for (const unsigned int q : q_indices)
            point_eval->submit_value(gamma * point_eval->get_value(q), q);

          point_eval->integrate(StridedArrayView<number, n_lanes>(&penalty_buffer[0][lane],
                                                                  n_dofs_per_cell),
                                EvaluationFlags::values);
        }

    // (2) mass matrix and damping: (u, w) + eta_K (grad u, grad w)
    const bool has_damping = damping_factor > 0.;
    const auto flags       = has_damping ? (EvaluationFlags::values | EvaluationFlags::gradients) :
                                           EvaluationFlags::values;

    phi.evaluate(flags);
    const VectorizedArrayType eta = has_damping ? damping[cell_batch] : VectorizedArrayType(0.);
    for (const unsigned int q : phi.quadrature_point_indices())
      {
        phi.submit_value(phi.get_value(q), q);
        if (has_damping)
          phi.submit_gradient(eta * phi.get_gradient(q), q);
      }
    phi.integrate(flags);

    for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
      phi.begin_dof_values()[i] += penalty_buffer[i];
  }

  template <int dim, typename number>
  void
  NarrowBandContinuousMassOperator<dim, number>::vmult(VectorType &dst, const VectorType &src) const
  {
    if (constrained_dof_mask == nullptr)
      {
        vmult_unconstrained(dst, src);
        return;
      }

    AssertDimension(constrained_dof_mask->locally_owned_size(), src.locally_owned_size());

    // operator on the free DoFs (the constrained DoFs do not contribute) and identity on the
    // constrained DoFs
    src_free.copy_locally_owned_data_from(src);
    for (unsigned int i = 0; i < src_free.locally_owned_size(); ++i)
      if (constrained_dof_mask->local_element(i) != 0.)
        src_free.local_element(i) = 0.;

    vmult_unconstrained(dst, src_free);

    for (unsigned int i = 0; i < dst.locally_owned_size(); ++i)
      if (constrained_dof_mask->local_element(i) != 0.)
        dst.local_element(i) = src.local_element(i);
  }

  template <int dim, typename number>
  void
  NarrowBandContinuousMassOperator<dim, number>::vmult_unconstrained(VectorType       &dst,
                                                                     const VectorType &src) const
  {
    scratch_data.get_matrix_free().template cell_loop<VectorType, VectorType>(
      [&](const auto &, auto &dst, const auto &src, const auto cell_range) {
        if (not is_cut_cell_range(cell_range))
          return;

        auto                                       phi        = create_cell_integrator();
        auto                                       point_eval = create_point_evaluator();
        dealii::AlignedVector<VectorizedArrayType> penalty_buffer(n_dofs_per_cell);

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            phi.reinit(cell);
            phi.read_dof_values(src);
            cell_operation(phi, point_eval.get(), penalty_buffer);
            phi.distribute_local_to_global(dst);
          }
      },
      dst,
      src,
      true /*zero dst vector*/);
  }

  template <int dim, typename number>
  void
  NarrowBandContinuousMassOperator<dim, number>::vmult(BlockVectorType       &dst,
                                                       const BlockVectorType &src) const
  {
    AssertDimension(dst.n_blocks(), src.n_blocks());
    for (unsigned int b = 0; b < dst.n_blocks(); ++b)
      vmult(dst.block(b), src.block(b));
  }

  template <int dim, typename number>
  void
  NarrowBandContinuousMassOperator<dim, number>::compute_system_matrix_from_matrixfree(
    TrilinosWrappers::SparseMatrix &system_matrix) const
  {
    system_matrix = 0.0;

    auto                                       point_eval = create_point_evaluator();
    dealii::AlignedVector<VectorizedArrayType> penalty_buffer(n_dofs_per_cell);

    // Note: cells with FE_Nothing are skipped by MatrixFreeTools.
    // With constrained DoFs, their rows and columns are eliminated (diagonal entries only).
    MatrixFreeTools::template compute_matrix<dim, -1, 0, 1, number, VectorizedArrayType>(
      scratch_data.get_matrix_free(),
      matrix_constraints != nullptr ? *matrix_constraints :
                                      scratch_data.get_constraint(this->dof_idx),
      system_matrix,
      [&](FECellIntegrator<dim, 1, number> &phi) {
        cell_operation(phi, point_eval.get(), penalty_buffer);
      },
      this->dof_idx,
      quad_idx,
      0 /*first selected component*/);

    // identity on the constrained DoFs (as in vmult())
    if (constrained_dof_mask != nullptr)
      {
        const auto &locally_owned_dofs = scratch_data.get_locally_owned_dofs(this->dof_idx);
        for (unsigned int i = 0; i < constrained_dof_mask->locally_owned_size(); ++i)
          if (constrained_dof_mask->local_element(i) != 0.)
            {
              const auto dof = locally_owned_dofs.nth_index_in_set(i);
              system_matrix.set(dof, dof, 1.0);
            }
        system_matrix.compress(VectorOperation::insert);
      }
  }

  template <int dim, typename number>
  void
  NarrowBandContinuousMassOperator<dim, number>::compute_inverse_diagonal_from_matrixfree(
    VectorType &diagonal) const
  {
    scratch_data.initialize_dof_vector(diagonal, this->dof_idx);

    auto                                       point_eval = create_point_evaluator();
    dealii::AlignedVector<VectorizedArrayType> penalty_buffer(n_dofs_per_cell);

    // Note: cells with FE_Nothing are skipped by MatrixFreeTools.
    MatrixFreeTools::template compute_diagonal<dim, -1, 0, 1, number, VectorizedArrayType>(
      scratch_data.get_matrix_free(),
      diagonal,
      [&](FECellIntegrator<dim, 1, number> &phi) {
        cell_operation(phi, point_eval.get(), penalty_buffer);
      },
      this->dof_idx,
      quad_idx,
      0 /*first selected component*/);

    const number linfty_norm = std::max<number>(1.0, diagonal.linfty_norm());
    for (auto &entry : diagonal)
      entry = (std::abs(entry) > 1.0e-14 * linfty_norm) ? (1.0 / entry) : 1.0;

    // identity on the constrained DoFs
    if (constrained_dof_mask != nullptr)
      for (unsigned int i = 0; i < diagonal.locally_owned_size(); ++i)
        if (constrained_dof_mask->local_element(i) != 0.)
          diagonal.local_element(i) = 1.0;
  }

  template <int dim, typename number>
  void
  NarrowBandContinuousMassOperator<dim, number>::compute_inverse_diagonal_from_matrixfree(
    BlockVectorType &diagonal) const
  {
    VectorType scalar_diagonal;
    compute_inverse_diagonal_from_matrixfree(scalar_diagonal);

    scratch_data.initialize_dof_vector(diagonal, this->dof_idx);
    for (unsigned int b = 0; b < diagonal.n_blocks(); ++b)
      diagonal.block(b) = scalar_diagonal;
  }

  /*----------------------------- NarrowBandContinuousLevelSet -----------------------------*/

  template <int dim, typename number>
  NarrowBandContinuousLevelSet<dim, number>::NarrowBandContinuousLevelSet(
    ScratchData<dim, dim, number>                             &scratch_data_in,
    const typename NarrowBandData<number>::ContinuousLevelSet &data_in,
    const NormalVectorData<number>                            &normal_vector_data_in,
    const CurvatureData<number>                               &curvature_data_in,
    const hp::FECollection<dim>                               &fe_collection_cut,
    const unsigned int                                         fe_degree_in,
    const unsigned int                                         cut_dof_idx_in,
    const unsigned int                                         quad_idx_in)
    : scratch_data(scratch_data_in)
    , data(data_in)
    , normal_vector_data(normal_vector_data_in)
    , curvature_data(curvature_data_in)
    , fe_degree(fe_degree_in)
    , cut_dof_idx(cut_dof_idx_in)
    , quad_idx(quad_idx_in)
    , fe_collection(create_continuous_fe_collection<dim>(fe_collection_cut, fe_degree_in, 1))
    , fe_collection_vector(
        create_continuous_fe_collection<dim>(fe_collection_cut, fe_degree_in, dim))
    , dof_handler(scratch_data_in.get_triangulation(cut_dof_idx_in))
    , dof_idx(scratch_data_in.attach_dof_handler_and_constraint(dof_handler, constraints))
    , fe_point(fe_degree_in)
    , mapping_info_surface_dg(scratch_data_in.get_mapping(), update_values | update_JxW_values)
    , projection_operator(scratch_data_in,
                          dof_idx,
                          quad_idx_in,
                          0.0 /*no damping*/,
                          data_in.interface_penalty,
                          &mapping_info_surface_dg)
    , normal_operator(scratch_data_in,
                      dof_idx,
                      quad_idx_in,
                      normal_vector_data_in.filter_parameter,
                      0.0 /*no interface penalty*/)
    , curvature_operator(scratch_data_in,
                         dof_idx,
                         quad_idx_in,
                         curvature_data_in.filter_parameter,
                         0.0 /*no interface penalty*/)
    , dof_handler_vector(scratch_data_in.get_triangulation(cut_dof_idx_in))
  {
    AssertThrow(fe_collection_cut.size() > CutUtil::CellCategory::intersected and
                  fe_collection_cut[CutUtil::CellCategory::intersected].n_dofs_per_cell() > 0,
                ExcMessage("The cut DoFHandler requires the level-set element at the FE index "
                           "CutUtil::CellCategory::intersected."));

    for (const auto *linear_solver :
         {&data.linear_solver, &normal_vector_data.linear_solver, &curvature_data.linear_solver})
      AssertThrow(linear_solver->do_matrix_free,
                  ExcMessage("The continuous level set, the normal vector and the curvature of the "
                             "narrow band are only implemented matrix-free. Set 'do matrix free' "
                             "to true."));

    // sign constraints: sample points on the boundary of the unit cell (lattice with 4p
    // subdivisions per direction, which contains the support points) and shape values
    if (data.sign_constraints)
      {
        const unsigned int n_subdivisions = 4 * fe_degree;
        const unsigned int n_points_1d    = n_subdivisions + 1;
        unsigned int       n_points       = 1;
        for (unsigned int d = 0; d < dim; ++d)
          n_points *= n_points_1d;

        for (unsigned int index = 0; index < n_points; ++index)
          {
            Point<dim>   point;
            bool         on_boundary = false;
            unsigned int rest        = index;
            for (unsigned int d = 0; d < dim; ++d)
              {
                const unsigned int i = rest % n_points_1d;
                rest /= n_points_1d;
                point[d] = static_cast<double>(i) / n_subdivisions;
                if (i == 0 or i == n_subdivisions)
                  on_boundary = true;
              }
            if (on_boundary)
              sign_sample_points.push_back(point);
          }

        sign_sample_shape_values.resize(sign_sample_points.size());
        for (unsigned int q = 0; q < sign_sample_points.size(); ++q)
          {
            sign_sample_shape_values[q].resize(fe_point.n_dofs_per_cell());
            for (unsigned int i = 0; i < fe_point.n_dofs_per_cell(); ++i)
              sign_sample_shape_values[q][i] = fe_point.shape_value(i, sign_sample_points[q]);
          }
      }

    projection_preconditioner =
      make_preconditioner<dim, number, NarrowBandContinuousMassOperator<dim, number>, VectorType>(
        data.linear_solver.preconditioner_type,
        &projection_operator,
        scratch_data,
        dof_idx,
        true /*matrix free*/);
    normal_preconditioner =
      make_preconditioner<dim,
                          number,
                          NarrowBandContinuousMassOperator<dim, number>,
                          BlockVectorType>(normal_vector_data.linear_solver.preconditioner_type,
                                           &normal_operator,
                                           scratch_data,
                                           dof_idx,
                                           true /*matrix free*/);
    curvature_preconditioner =
      make_preconditioner<dim, number, NarrowBandContinuousMassOperator<dim, number>, VectorType>(
        curvature_data.linear_solver.preconditioner_type,
        &curvature_operator,
        scratch_data,
        dof_idx,
        true /*matrix free*/);
  }

  template <int dim, typename number>
  void
  NarrowBandContinuousLevelSet<dim, number>::distribute_dofs(
    const dealii::DoFHandler<dim> &dof_handler_cut)
  {
    // same active FE indices as the cut DoFHandler
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        cell->set_active_fe_index(
          cell->as_dof_handler_iterator(dof_handler_cut)->active_fe_index());

    dof_handler.distribute_dofs(fe_collection);

    // Without hanging nodes, the non-dominating FE_Nothing elements impose no constraints on the
    // FE_Q elements, i.e., the field is continuous across the faces between two cut cells and
    // unconstrained at the faces to all other cells.
    constraints.clear();
    constraints.reinit(dof_handler.locally_owned_dofs(),
                       DoFTools::extract_locally_relevant_dofs(dof_handler));
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();
  }

  template <int dim, typename number>
  void
  NarrowBandContinuousLevelSet<dim, number>::reinit()
  {
    scratch_data.initialize_dof_vector(level_set, dof_idx);
    scratch_data.initialize_dof_vector(rhs, dof_idx);
    scratch_data.initialize_dof_vector(curvature, dof_idx);
    scratch_data.initialize_dof_vector(normal_vector, dof_idx);
    scratch_data.initialize_dof_vector(rhs_normal, dof_idx);
    if (data.sign_constraints)
      {
        scratch_data.initialize_dof_vector(sign_mask, dof_idx);
        scratch_data.initialize_dof_vector(sign_values, dof_idx);
        scratch_data.initialize_dof_vector(rhs_constrained, dof_idx);
      }

    projection_operator.reinit();
    normal_operator.reinit();
    curvature_operator.reinit();

    projection_preconditioner.reinit();
    normal_preconditioner.reinit();
    curvature_preconditioner.reinit();

    // The operators of the normal vector and the curvature only change with the DoF layout.
    normal_preconditioner.set_do_update_preconditioner(true);
    curvature_preconditioner.set_do_update_preconditioner(true);
  }

  template <int dim, typename number>
  void
  NarrowBandContinuousLevelSet<dim, number>::compute(const VectorType       &level_set_dg,
                                                     const std::vector<int> *cell_signs)
  {
    ScopedName         sc("narrow_band::continuous_level_set");
    TimerOutput::Scope scope(scratch_data.get_timer(), sc);

    statistics = Statistics();

    project(level_set_dg);

    if (data.sign_constraints)
      {
        AssertThrow(cell_signs != nullptr,
                    ExcMessage("The sign constraints of the continuous level set require the "
                               "signs of the cells, which are not cut."));
        enforce_sign_constraints(*cell_signs);
      }

    compute_interface_shift(level_set_dg);

    compute_normal_vector();

    if (curvature_data.enable)
      {
        compute_curvature();
        compute_curvature_statistics();
      }

    print_statistics();
  }

  template <int dim, typename number>
  void
  NarrowBandContinuousLevelSet<dim, number>::project(const VectorType &level_set_dg)
  {
    const auto &matrix_free = scratch_data.get_matrix_free();

    // surface quadrature of the zero level set of the DG level set (interface penalty)
    CutUtil::compute_immersed_surface_quadrature_intersected_cells(mapping_info_surface_dg,
                                                                   scratch_data.get_dof_handler(
                                                                     cut_dof_idx),
                                                                   level_set_dg,
                                                                   matrix_free,
                                                                   fe_degree);

    // right-hand side (phi_h, w)
    matrix_free.template cell_loop<VectorType, VectorType>(
      [&](const auto &, auto &dst, const auto &src, const auto cell_range) {
        if (not projection_operator.is_cut_cell_range(cell_range))
          return;

        FECellIntegrator<dim, 1, number> phi_dg(matrix_free,
                                                cut_dof_idx,
                                                quad_idx,
                                                0 /*first selected component*/,
                                                CutUtil::CellCategory::intersected);
        auto                             phi = projection_operator.create_cell_integrator();

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            phi_dg.reinit(cell);
            phi_dg.gather_evaluate(src, EvaluationFlags::values);

            phi.reinit(cell);
            for (const unsigned int q : phi.quadrature_point_indices())
              phi.submit_value(phi_dg.get_value(q), q);
            phi.integrate_scatter(EvaluationFlags::values, dst);
          }
      },
      rhs,
      level_set_dg,
      true /*zero dst vector*/);

    // The interface penalty changes with the interface of the DG level set.
    projection_preconditioner.set_do_update_preconditioner(true);
    projection_preconditioner.update();

    level_set.zero_out_ghost_values();
    statistics.n_iterations_projection = LinearSolver::solve<VectorType>(projection_operator,
                                                                         level_set,
                                                                         rhs,
                                                                         data.linear_solver,
                                                                         projection_preconditioner,
                                                                         "narrow_band_projection");
    constraints.distribute(level_set);
    level_set.update_ghost_values();
  }

  template <int dim, typename number>
  void
  NarrowBandContinuousLevelSet<dim, number>::enforce_sign_constraints(
    const std::vector<int> &cell_signs)
  {
    const auto &triangulation = dof_handler.get_triangulation();
    AssertThrow(cell_signs.size() == triangulation.n_active_cells(),
                ExcMessage("The cell signs have to be given for all active cells."));

    // 1) regions: vertices, edges and faces of the locally owned cut cells, which are shared with
    // a cell, which is not cut and has a known sign
    std::vector<std::vector<typename DoFHandler<dim>::active_cell_iterator>> vertex_to_cells(
      triangulation.n_vertices());
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (not cell->is_artificial())
        for (const unsigned int v : cell->vertex_indices())
          vertex_to_cells[cell->vertex_index(v)].push_back(cell);

    const auto &unit_support_points = fe_point.get_unit_support_points();
    const auto  is_in_region = [](const Point<dim> &point, const std::array<int, dim> &fixed) {
      for (unsigned int d = 0; d < dim; ++d)
        if (fixed[d] >= 0 and std::abs(point[d] - fixed[d]) > 1e-12)
          return false;
      return true;
    };

    sign_cells.clear();
    std::vector<typename DoFHandler<dim>::active_cell_iterator> neighbors;
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or
            cell->active_fe_index() != CutUtil::CellCategory::intersected)
          continue;

        neighbors.clear();
        for (const unsigned int v : cell->vertex_indices())
          for (const auto &other : vertex_to_cells[cell->vertex_index(v)])
            if (other->active_fe_index() != CutUtil::CellCategory::intersected and
                cell_signs[other->active_cell_index()] != 0 and
                std::find(neighbors.begin(), neighbors.end(), other) == neighbors.end())
              neighbors.push_back(other);

        if (neighbors.empty())
          continue;

        SignCell sign_cell;
        sign_cell.dof_indices.resize(fe_point.n_dofs_per_cell());
        cell->get_dof_indices(sign_cell.dof_indices);
        sign_cell.offset =
          data.sign_constraint_offset * cell->diameter() / std::sqrt(static_cast<number>(dim));

        for (const auto &neighbor : neighbors)
          {
            // shared region: the coordinates, in which all shared vertices agree, are fixed
            std::array<int, dim> fixed;
            fixed.fill(-2);
            for (const unsigned int v : cell->vertex_indices())
              {
                bool is_shared = false;
                for (const unsigned int w : neighbor->vertex_indices())
                  if (neighbor->vertex_index(w) == cell->vertex_index(v))
                    is_shared = true;
                if (not is_shared)
                  continue;

                const Point<dim> vertex = GeometryInfo<dim>::unit_cell_vertex(v);
                for (unsigned int d = 0; d < dim; ++d)
                  {
                    const int coordinate = static_cast<int>(std::round(vertex[d]));
                    if (fixed[d] == -2)
                      fixed[d] = coordinate;
                    else if (fixed[d] != coordinate)
                      fixed[d] = -1; // free coordinate
                  }
              }

            SignRegion region;
            region.sign = cell_signs[neighbor->active_cell_index()];
            for (unsigned int i = 0; i < unit_support_points.size(); ++i)
              if (is_in_region(unit_support_points[i], fixed))
                region.dofs.push_back(i);
            for (unsigned int q = 0; q < sign_sample_points.size(); ++q)
              if (is_in_region(sign_sample_points[q], fixed))
                region.samples.push_back(q);

            if (not region.dofs.empty())
              sign_cell.regions.push_back(std::move(region));
          }

        sign_cells.push_back(std::move(sign_cell));
      }

    // 2) check, constrain and project again
    constexpr unsigned int max_iterations = 5;

    VectorType required_positive, required_negative;
    scratch_data.initialize_dof_vector(required_positive, dof_idx);
    scratch_data.initialize_dof_vector(required_negative, dof_idx);

    sign_mask           = 0.;
    sign_values         = 0.;
    const MPI_Comm comm = scratch_data.get_mpi_comm();
    for (unsigned int iteration = 0;; ++iteration)
      {
        statistics.n_sign_violations = check_sign_constraints(required_positive, required_negative);
        if (statistics.n_sign_violations == 0 or iteration == max_iterations)
          break;

        // new or raised constraints; conflicting signs (a vertex shared with cells of both signs)
        // are constrained to zero
        unsigned int n_changed = 0;
        for (unsigned int i = 0; i < sign_mask.locally_owned_size(); ++i)
          {
            const number positive = required_positive.local_element(i);
            const number negative = required_negative.local_element(i);
            if (positive == 0. and negative == 0.)
              continue;

            const number value     = (positive > 0. and negative > 0.) ? 0. :
                                     (positive > 0.)                   ? positive :
                                                                         -negative;
            number      &old_value = sign_values.local_element(i);
            if (sign_mask.local_element(i) == 0.)
              {
                sign_mask.local_element(i) = 1.;
                old_value                  = value;
                ++n_changed;
              }
            else if (old_value * value < 0.)
              {
                old_value = 0.;
                ++n_changed;
              }
            else if (std::abs(value) > std::abs(old_value) and old_value != 0.)
              {
                old_value = value;
                ++n_changed;
              }
          }

        if (Utilities::MPI::sum(n_changed, comm) == 0)
          break;

        project_with_sign_constraints();
        ++statistics.n_sign_constraint_solves;
      }

    unsigned int n_constrained = 0;
    for (unsigned int i = 0; i < sign_mask.locally_owned_size(); ++i)
      if (sign_mask.local_element(i) != 0.)
        ++n_constrained;
    statistics.n_sign_constrained_dofs = Utilities::MPI::sum(n_constrained, comm);
  }

  template <int dim, typename number>
  unsigned int
  NarrowBandContinuousLevelSet<dim, number>::check_sign_constraints(
    VectorType &required_positive,
    VectorType &required_negative) const
  {
    required_positive = 0.;
    required_negative = 0.;

    const bool update_ghosts = not level_set.has_ghost_elements();
    if (update_ghosts)
      level_set.update_ghost_values();
    sign_mask.update_ghost_values();

    const unsigned int n_vertex_dofs =
      GeometryInfo<dim>::vertices_per_cell * fe_point.n_dofs_per_vertex();

    const auto require =
      [&](const dealii::types::global_dof_index dof, const int sign, const number magnitude) {
        number &entry = sign > 0 ? required_positive(dof) : required_negative(dof);
        entry         = std::max(entry, magnitude);
      };

    std::vector<number> values(fe_point.n_dofs_per_cell());
    std::vector<number> modified_values(fe_point.n_dofs_per_cell());
    const auto evaluate = [&](const std::vector<number> &dof_values, const unsigned int q) {
      number value = 0.;
      for (unsigned int i = 0; i < dof_values.size(); ++i)
        value += dof_values[i] * sign_sample_shape_values[q][i];
      return value;
    };

    unsigned int n_violations = 0;
    for (const auto &sign_cell : sign_cells)
      {
        for (unsigned int i = 0; i < values.size(); ++i)
          values[i] = level_set(sign_cell.dof_indices[i]);

        for (const auto &region : sign_cell.regions)
          {
            const int  s               = region.sign;
            const auto has_sign_change = [&](const std::vector<number> &dof_values) {
              for (const unsigned int q : region.samples)
                if (s * evaluate(dof_values, q) < 0.)
                  return true;
              return false;
            };

            // violation: nodal values with the wrong sign (or zero), which are not constrained,
            // or sign changes between the nodes of the region
            bool has_violation = false;
            for (const unsigned int i : region.dofs)
              if (s * values[i] <= 0. and sign_mask(sign_cell.dof_indices[i]) == 0.)
                has_violation = true;
            if (not has_violation)
              has_violation = has_sign_change(values);
            if (not has_violation)
              continue;
            ++n_violations;

            // all nodal values of the region are constrained (frozen), such that the next
            // projection cannot change the region: nodal values with s psi_j < c h are set to
            // s c h; if the sign still changes between the nodes, the nodal values in the interior
            // of the region are raised to the smallest s M with M = c h 2^k, which removes the
            // sign change, but at most to the maximum vertex value
            modified_values = values;
            for (const unsigned int i : region.dofs)
              modified_values[i] = s * std::max(s * values[i], sign_cell.offset);

            if (has_sign_change(modified_values))
              {
                number max_vertex_value = sign_cell.offset;
                for (const unsigned int i : region.dofs)
                  if (i < n_vertex_dofs)
                    max_vertex_value = std::max(max_vertex_value, s * modified_values[i]);

                const std::vector<number> nodal_values = modified_values;
                for (number magnitude = 2. * sign_cell.offset;; magnitude *= 2.)
                  {
                    magnitude = std::min(magnitude, max_vertex_value);
                    for (const unsigned int i : region.dofs)
                      if (i >= n_vertex_dofs)
                        modified_values[i] = s * std::max(s * nodal_values[i], magnitude);
                    if (not has_sign_change(modified_values) or magnitude >= max_vertex_value)
                      break;
                  }
              }

            for (const unsigned int i : region.dofs)
              require(sign_cell.dof_indices[i], s, s * modified_values[i]);
          }
      }

    if (update_ghosts)
      level_set.zero_out_ghost_values();
    sign_mask.zero_out_ghost_values();

    required_positive.compress(VectorOperation::max);
    required_negative.compress(VectorOperation::max);

    return Utilities::MPI::sum(n_violations, scratch_data.get_mpi_comm());
  }

  template <int dim, typename number>
  void
  NarrowBandContinuousLevelSet<dim, number>::project_with_sign_constraints()
  {
    // homogeneous constraints of the constrained DoFs for the system matrix of the preconditioner
    // (the DoFs of the locally owned cut cells: locally owned and ghost entries of the vector)
    sign_mask.update_ghost_values();
    sign_matrix_constraints.clear();
    sign_matrix_constraints.reinit(dof_handler.locally_owned_dofs(),
                                   DoFTools::extract_locally_relevant_dofs(dof_handler));
    DoFTools::make_hanging_node_constraints(dof_handler, sign_matrix_constraints);
    for (const IndexSet &dofs :
         {sign_mask.locally_owned_elements(), sign_mask.get_partitioner()->ghost_indices()})
      for (const auto dof : dofs)
        if (sign_mask(dof) != 0. and not sign_matrix_constraints.is_constrained(dof))
          sign_matrix_constraints.constrain_dof_to_zero(dof);
    sign_matrix_constraints.close();
    sign_mask.zero_out_ghost_values();

    // right-hand side: b - A g on the free DoFs and the prescribed values g on the constrained
    // DoFs, where rhs still contains b of project()
    projection_operator.set_constrained_dofs(nullptr, nullptr);
    sign_values.zero_out_ghost_values();
    projection_operator.vmult(rhs_constrained, sign_values);
    rhs_constrained.sadd(-1., 1., rhs);
    for (unsigned int i = 0; i < rhs_constrained.locally_owned_size(); ++i)
      if (sign_mask.local_element(i) != 0.)
        rhs_constrained.local_element(i) = sign_values.local_element(i);

    projection_operator.set_constrained_dofs(&sign_mask, &sign_matrix_constraints);
    projection_preconditioner.set_do_update_preconditioner(true);
    projection_preconditioner.update();

    level_set.zero_out_ghost_values();
    level_set = 0.;
    statistics.n_iterations_projection +=
      LinearSolver::solve<VectorType>(projection_operator,
                                      level_set,
                                      rhs_constrained,
                                      data.linear_solver,
                                      projection_preconditioner,
                                      "narrow_band_projection_sign_constraints");
    constraints.distribute(level_set);
    level_set.update_ghost_values();

    projection_operator.set_constrained_dofs(nullptr, nullptr);
  }

  template <int dim, typename number>
  void
  NarrowBandContinuousLevelSet<dim, number>::compute_interface_shift(const VectorType &level_set_dg)
  {
    const auto &dof_handler_cut = scratch_data.get_dof_handler(cut_dof_idx);

    const bool update_ghosts = not level_set_dg.has_ghost_elements();
    if (update_ghosts)
      level_set_dg.update_ghost_values();

    const hp::QCollection<1>                      q_collection_1d(QGauss<1>(fe_degree + 1));
    NonMatching::DiscreteQuadratureGenerator<dim> quadrature_generator(q_collection_1d,
                                                                       dof_handler_cut,
                                                                       level_set_dg);

    std::vector<number>                 values;
    std::vector<Tensor<1, dim, number>> gradients;

    number max_shift = 0., max_relative_shift = 0.;
    for (const auto &cell : dof_handler_cut.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or
            cell->active_fe_index() != CutUtil::CellCategory::intersected)
          continue;

        quadrature_generator.generate(cell);
        const auto &surface_quadrature = quadrature_generator.get_surface_quadrature();
        if (surface_quadrature.size() == 0)
          continue;

        // continuous level set at the quadrature points of the DG interface
        NonMatching::FEImmersedSurfaceValues<dim> fe_values(scratch_data.get_mapping(),
                                                            fe_point,
                                                            surface_quadrature,
                                                            update_values | update_gradients);
        fe_values.reinit(cell->as_dof_handler_iterator(dof_handler));

        values.resize(surface_quadrature.size());
        gradients.resize(surface_quadrature.size());
        fe_values.get_function_values(level_set, values);
        fe_values.get_function_gradients(level_set, gradients);

        const number h = cell->diameter() / std::sqrt(static_cast<number>(dim));
        for (unsigned int q = 0; q < values.size(); ++q)
          {
            const number shift =
              std::abs(values[q]) / std::max<number>(gradients[q].norm(), 1.0e-12);
            max_shift          = std::max(max_shift, shift);
            max_relative_shift = std::max(max_relative_shift, shift / h);
          }
      }

    if (update_ghosts)
      level_set_dg.zero_out_ghost_values();

    const MPI_Comm comm                     = scratch_data.get_mpi_comm();
    statistics.max_interface_shift          = Utilities::MPI::max(max_shift, comm);
    statistics.max_relative_interface_shift = Utilities::MPI::max(max_relative_shift, comm);
  }

  template <int dim, typename number>
  void
  NarrowBandContinuousLevelSet<dim, number>::compute_normal_vector()
  {
    const auto &matrix_free = scratch_data.get_matrix_free();

    // right-hand side (grad psi_h, w)
    matrix_free.template cell_loop<BlockVectorType, VectorType>(
      [&](const auto &, auto &dst, const auto &src, const auto cell_range) {
        if (not normal_operator.is_cut_cell_range(cell_range))
          return;

        auto                               phi = normal_operator.create_cell_integrator();
        FECellIntegrator<dim, dim, number> normal(matrix_free,
                                                  dof_idx,
                                                  quad_idx,
                                                  0 /*first selected component*/,
                                                  CutUtil::CellCategory::intersected);

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            phi.reinit(cell);
            phi.gather_evaluate(src, EvaluationFlags::gradients);

            normal.reinit(cell);
            for (const unsigned int q : normal.quadrature_point_indices())
              normal.submit_value(phi.get_gradient(q), q);
            normal.integrate_scatter(EvaluationFlags::values, dst);
          }
      },
      rhs_normal,
      level_set,
      true /*zero dst vector*/);

    normal_preconditioner.update();

    normal_vector.zero_out_ghost_values();
    statistics.n_iterations_normal =
      LinearSolver::solve<BlockVectorType>(normal_operator,
                                           normal_vector,
                                           rhs_normal,
                                           normal_vector_data.linear_solver,
                                           normal_preconditioner,
                                           "narrow_band_normal_vector");

    for (unsigned int d = 0; d < dim; ++d)
      constraints.distribute(normal_vector.block(d));

    // normalization of the nodal values
    for (unsigned int i = 0; i < normal_vector.block(0).locally_owned_size(); ++i)
      {
        number norm_square = 0.;
        for (unsigned int d = 0; d < dim; ++d)
          norm_square += Utilities::fixed_power<2>(normal_vector.block(d).local_element(i));

        const number norm = std::sqrt(norm_square);
        for (unsigned int d = 0; d < dim; ++d)
          normal_vector.block(d).local_element(i) =
            norm > 1.0e-12 ? normal_vector.block(d).local_element(i) / norm : 0.0;
      }

    normal_vector.update_ghost_values();
  }

  template <int dim, typename number>
  void
  NarrowBandContinuousLevelSet<dim, number>::compute_curvature()
  {
    const auto &matrix_free = scratch_data.get_matrix_free();

    // right-hand side -(div n_h, w)
    matrix_free.template cell_loop<VectorType, BlockVectorType>(
      [&](const auto &, auto &dst, const auto &src, const auto cell_range) {
        if (not curvature_operator.is_cut_cell_range(cell_range))
          return;

        auto                               phi = curvature_operator.create_cell_integrator();
        FECellIntegrator<dim, dim, number> normal(matrix_free,
                                                  dof_idx,
                                                  quad_idx,
                                                  0 /*first selected component*/,
                                                  CutUtil::CellCategory::intersected);

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            normal.reinit(cell);
            normal.gather_evaluate(src, EvaluationFlags::gradients);

            phi.reinit(cell);
            for (const unsigned int q : phi.quadrature_point_indices())
              phi.submit_value(-normal.get_divergence(q), q);
            phi.integrate_scatter(EvaluationFlags::values, dst);
          }
      },
      rhs,
      normal_vector,
      true /*zero dst vector*/);

    curvature_preconditioner.update();

    curvature.zero_out_ghost_values();
    statistics.n_iterations_curvature =
      LinearSolver::solve<VectorType>(curvature_operator,
                                      curvature,
                                      rhs,
                                      curvature_data.linear_solver,
                                      curvature_preconditioner,
                                      "narrow_band_curvature");
    constraints.distribute(curvature);
    curvature.update_ghost_values();
  }

  template <int dim, typename number>
  void
  NarrowBandContinuousLevelSet<dim, number>::compute_curvature_statistics()
  {
    const hp::QCollection<1>                      q_collection_1d(QGauss<1>(fe_degree + 1));
    NonMatching::DiscreteQuadratureGenerator<dim> quadrature_generator(q_collection_1d,
                                                                       dof_handler,
                                                                       level_set);

    std::vector<number> values;

    number sum = 0., measure = 0.;
    number min = std::numeric_limits<number>::max();
    number max = std::numeric_limits<number>::lowest();
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or
            cell->active_fe_index() != CutUtil::CellCategory::intersected)
          continue;

        quadrature_generator.generate(cell);
        const auto &surface_quadrature = quadrature_generator.get_surface_quadrature();
        if (surface_quadrature.size() == 0)
          continue;

        NonMatching::FEImmersedSurfaceValues<dim> fe_values(scratch_data.get_mapping(),
                                                            fe_point,
                                                            surface_quadrature,
                                                            update_values | update_JxW_values);
        fe_values.reinit(cell);

        values.resize(surface_quadrature.size());
        fe_values.get_function_values(curvature, values);

        for (unsigned int q = 0; q < values.size(); ++q)
          {
            sum += values[q] * fe_values.JxW(q);
            measure += fe_values.JxW(q);
            min = std::min(min, values[q]);
            max = std::max(max, values[q]);
          }
      }

    const MPI_Comm comm          = scratch_data.get_mpi_comm();
    statistics.interface_measure = Utilities::MPI::sum(measure, comm);
    sum                          = Utilities::MPI::sum(sum, comm);
    statistics.min_curvature     = Utilities::MPI::min(min, comm);
    statistics.max_curvature     = Utilities::MPI::max(max, comm);
    statistics.mean_curvature =
      statistics.interface_measure > 0. ? sum / statistics.interface_measure : 0.;
    if (statistics.interface_measure == 0.)
      statistics.min_curvature = statistics.max_curvature = 0.;
  }

  template <int dim, typename number>
  void
  NarrowBandContinuousLevelSet<dim, number>::print_statistics() const
  {
    const auto &pcout = scratch_data.get_pcout(1);
    {
      std::ostringstream str;
      // linear iterations of the projection, the normal vector and the curvature
      str << "continuous level set: interface shift " << std::scientific << std::setprecision(2)
          << statistics.max_interface_shift << " (" << statistics.max_relative_interface_shift
          << " h), iterations " << statistics.n_iterations_projection << "/"
          << statistics.n_iterations_normal;
      if (curvature_data.enable)
        str << "/" << statistics.n_iterations_curvature;
      Journal::print_line(pcout, str.str(), "narrow_band");
    }

    if (data.sign_constraints)
      {
        std::ostringstream str;
        str << "sign constraints: " << statistics.n_sign_constrained_dofs << " nodal values, "
            << statistics.n_sign_constraint_solves << " projections";
        Journal::print_line(pcout, str.str(), "narrow_band");
      }

    if (data.sign_constraints and statistics.n_sign_violations > 0)
      Journal::print_line(pcout,
                          "Warning: " + std::to_string(statistics.n_sign_violations) +
                            " sign violations of the continuous level set remain.",
                          "narrow_band");

    if (curvature_data.enable)
      {
        std::ostringstream str;
        str << "curvature on interface: mean " << std::scientific << std::setprecision(4)
            << statistics.mean_curvature << ", min " << statistics.min_curvature << ", max "
            << statistics.max_curvature;
        Journal::print_line(pcout, str.str(), "narrow_band");
      }
  }

  template <int dim, typename number>
  const typename NarrowBandContinuousLevelSet<dim, number>::VectorType &
  NarrowBandContinuousLevelSet<dim, number>::get_level_set() const
  {
    return level_set;
  }

  template <int dim, typename number>
  const typename NarrowBandContinuousLevelSet<dim, number>::BlockVectorType &
  NarrowBandContinuousLevelSet<dim, number>::get_normal_vector() const
  {
    return normal_vector;
  }

  template <int dim, typename number>
  const typename NarrowBandContinuousLevelSet<dim, number>::VectorType &
  NarrowBandContinuousLevelSet<dim, number>::get_curvature() const
  {
    return curvature;
  }

  template <int dim, typename number>
  const dealii::DoFHandler<dim> &
  NarrowBandContinuousLevelSet<dim, number>::get_dof_handler() const
  {
    return dof_handler;
  }

  template <int dim, typename number>
  unsigned int
  NarrowBandContinuousLevelSet<dim, number>::get_dof_index() const
  {
    return dof_idx;
  }

  template <int dim, typename number>
  const typename NarrowBandContinuousLevelSet<dim, number>::Statistics &
  NarrowBandContinuousLevelSet<dim, number>::get_statistics() const
  {
    return statistics;
  }

  template <int dim, typename number>
  void
  NarrowBandContinuousLevelSet<dim, number>::attach_output_vectors(
    GenericDataOut<dim, number> &data_out) const
  {
    level_set.update_ghost_values();
    data_out.add_data_vector(dof_handler, level_set, "level_set_cg");

    if (curvature_data.enable)
      {
        curvature.update_ghost_values();
        data_out.add_data_vector(dof_handler, curvature, "curvature");
      }

    // Vector-valued copy of the normal vector for the output: same active FE indices, FE_Q^dim on
    // the cut cells.
    for (const auto &cell : dof_handler_vector.active_cell_iterators())
      if (cell->is_locally_owned())
        cell->set_active_fe_index(cell->as_dof_handler_iterator(dof_handler)->active_fe_index());
    dof_handler_vector.distribute_dofs(fe_collection_vector);

    normal_vector_output.reinit(dof_handler_vector.locally_owned_dofs(),
                                DoFTools::extract_locally_relevant_dofs(dof_handler_vector),
                                scratch_data.get_mpi_comm());

    normal_vector.update_ghost_values();

    std::vector<Vector<number>> component_values(dim);
    Vector<number>              vector_values;
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
          continue;

        for (unsigned int d = 0; d < dim; ++d)
          {
            component_values[d].reinit(cell->get_fe().n_dofs_per_cell());
            cell->get_dof_values(normal_vector.block(d), component_values[d]);
          }

        const auto  cell_vector = cell->as_dof_handler_iterator(dof_handler_vector);
        const auto &fe_vector   = cell_vector->get_fe();
        vector_values.reinit(fe_vector.n_dofs_per_cell());
        for (unsigned int i = 0; i < fe_vector.n_dofs_per_cell(); ++i)
          {
            const auto [component, index] = fe_vector.system_to_component_index(i);
            vector_values[i]              = component_values[component][index];
          }
        cell_vector->set_dof_values(vector_values, normal_vector_output);
      }

    // the values of DoFs shared by several processes are identical
    normal_vector_output.compress(VectorOperation::insert);
    normal_vector_output.update_ghost_values();

    data_out.add_data_vector(dof_handler_vector,
                             normal_vector_output,
                             std::vector<std::string>(dim, "normal"),
                             std::vector<DataComponentInterpretation::DataComponentInterpretation>(
                               dim, DataComponentInterpretation::component_is_part_of_vector));
  }

  template class NarrowBandContinuousMassOperator<1, double>;
  template class NarrowBandContinuousMassOperator<2, double>;
  template class NarrowBandContinuousMassOperator<3, double>;

  template class NarrowBandContinuousLevelSet<1, double>;
  template class NarrowBandContinuousLevelSet<2, double>;
  template class NarrowBandContinuousLevelSet<3, double>;
} // namespace MeltPoolDG::LevelSet
