#include <gtest/gtest.h>

#include <deal.II/base/function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/non_matching/quadrature_generator.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/level_set/narrow_band_advection_DG_operator.hpp>
#include <meltpooldg/level_set/narrow_band_bound_preserving_limiter.hpp>
#include <meltpooldg/level_set/narrow_band_characteristic_map.hpp>
#include <meltpooldg/level_set/narrow_band_closest_point_extension.hpp>
#include <meltpooldg/level_set/narrow_band_closest_point_tools.hpp>
#include <meltpooldg/level_set/narrow_band_continuous_reinitialization.hpp>
#include <meltpooldg/level_set/narrow_band_data.hpp>
#include <meltpooldg/level_set/narrow_band_ghost_penalty_extension.hpp>
#include <meltpooldg/level_set/narrow_band_interface_points.hpp>
#include <meltpooldg/level_set/narrow_band_level_set_operation.hpp>
#include <meltpooldg/level_set/narrow_band_semi_lagrangian_advection.hpp>
#include <meltpooldg/level_set/narrow_band_utilities.hpp>
#include <meltpooldg/level_set/reinitialization_data.hpp>
#include <meltpooldg/level_set/reinitialization_elliptic_operation.hpp>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <array>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace MeltPoolDG::LevelSet
{
  namespace
  {
    using number     = double;
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    /**
     * Polynomial level-set function of degree p, whose zero iso-surface intersects the domain
     * [-1,1]^dim: phi(x) = x_0^p + 0.4 x_1 - 0.3 x_0 x_1 + 0.1 (+ 0.2 x_2 in 3D).
     */
    template <int dim>
    class PolynomialLevelSet : public dealii::Function<dim>
    {
    public:
      explicit PolynomialLevelSet(const unsigned int degree)
        : degree(degree)
      {}

      double
      value(const dealii::Point<dim> &p, const unsigned int) const override
      {
        double value = std::pow(p[0], degree) + 0.4 * p[1] + 0.1;
        if (degree >= 2)
          value -= 0.3 * p[0] * p[1];
        if (dim == 3)
          value += 0.2 * p[dim - 1];
        return value;
      }

    private:
      const unsigned int degree;
    };

    /**
     * Rigid-body rotation about the z-axis.
     */
    template <int dim>
    class Rotation : public dealii::Function<dim>
    {
    public:
      Rotation()
        : dealii::Function<dim>(dim)
      {}

      double
      value(const dealii::Point<dim> &p, const unsigned int component) const override
      {
        if (component == 0)
          return -p[1];
        if (component == 1)
          return p[0];
        return 0.;
      }
    };

    /**
     * Narrow band of the zero level set of a polynomial level-set function on [-1,1]^dim.
     */
    template <int dim>
    struct NarrowBandFixture
    {
      dealii::parallel::distributed::Triangulation<dim> triangulation{MPI_COMM_WORLD};
      const unsigned int                                fe_degree;
      const dealii::hp::FECollection<dim>               fe_collection_band;
      const dealii::hp::FECollection<dim>               fe_collection_cut;
      dealii::DoFHandler<dim>                           dof_handler_band{triangulation};
      dealii::DoFHandler<dim>                           dof_handler_cut{triangulation};
      dealii::MappingQ<dim>                             mapping;
      const PolynomialLevelSet<dim>                     level_set_function;

      /**
       * Constructor. The cut cells are determined from the polynomial level set of degree
       * @p level_set_degree (default: @p fe_degree) or from @p function, if given.
       */
      explicit NarrowBandFixture(
        const unsigned int           fe_degree,
        const unsigned int           n_refinements,
        const unsigned int           level_set_degree = dealii::numbers::invalid_unsigned_int,
        const dealii::Function<dim> *function         = nullptr)
        : fe_degree(fe_degree)
        , fe_collection_band(NarrowBand::create_band_fe_collection<dim>(fe_degree))
        , fe_collection_cut(NarrowBand::create_cut_fe_collection<dim>(fe_degree))
        , mapping(fe_degree)
        , level_set_function(level_set_degree == dealii::numbers::invalid_unsigned_int ?
                               fe_degree :
                               level_set_degree)
      {
        dealii::GridGenerator::hyper_cube(triangulation, -1., 1.);
        triangulation.refine_global(n_refinements);

        // determine the cut cells from a DG level-set field on all cells
        dealii::DoFHandler<dim> dof_handler_full(triangulation);
        dof_handler_full.distribute_dofs(dealii::FE_DGQ<dim>(fe_degree));
        VectorType level_set_full(dof_handler_full.locally_owned_dofs(),
                                  dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_full),
                                  MPI_COMM_WORLD);
        dealii::VectorTools::interpolate(mapping,
                                         dof_handler_full,
                                         function != nullptr ? *function : level_set_function,
                                         level_set_full);

        const std::vector<bool> is_cut_cell =
          NarrowBand::compute_cut_cells(dof_handler_full, level_set_full, fe_degree);

        NarrowBand::distribute_dofs(
          is_cut_cell, dof_handler_cut, fe_collection_cut, dof_handler_band, fe_collection_band);
      }

      void
      initialize_vector(const dealii::DoFHandler<dim> &dof_handler, VectorType &vector) const
      {
        vector.reinit(dof_handler.locally_owned_dofs(),
                      dealii::DoFTools::extract_locally_relevant_dofs(dof_handler),
                      MPI_COMM_WORLD);
      }
    };

    /**
     * Matrix-free data structures for the band and the cut DoFHandler of a narrow band.
     */
    template <int dim>
    struct MatrixFreeFixture
    {
      dealii::AffineConstraints<number> constraints_band;
      dealii::AffineConstraints<number> constraints_cut;
      ScratchData<dim, dim, number>     scratch_data{MPI_COMM_WORLD, 0, true};
      unsigned int                      band_dof_idx = 0;
      unsigned int                      quad_idx     = 0;

      explicit MatrixFreeFixture(NarrowBandFixture<dim> &narrow_band)
      {
        constraints_band.close();
        constraints_cut.close();

        scratch_data.set_mapping(std::make_shared<dealii::MappingQ<dim>>(narrow_band.fe_degree));
        band_dof_idx = scratch_data.attach_dof_handler_and_constraint(narrow_band.dof_handler_band,
                                                                      constraints_band);
        scratch_data.attach_dof_handler_and_constraint(narrow_band.dof_handler_cut,
                                                       constraints_cut);
        quad_idx = scratch_data.attach_quadrature(dealii::QGauss<dim>(narrow_band.fe_degree + 1));
        scratch_data.create_partitioning();
        scratch_data.build(false, true, true);
      }
    };

    /**
     * The ghost-penalty extension of a polynomial level-set field of degree p reproduces the
     * polynomial on the neighbor cells.
     */
    template <int dim>
    void
    test_extension_reproduces_polynomials(const unsigned int                fe_degree,
                                          const NarrowBandGhostPenaltyFaces faces,
                                          const double minimum_source_volume_fraction = 0.,
                                          const double tolerance                      = 1e-9)
    {
      NarrowBandFixture<dim> fixture(fe_degree, dim == 2 ? 3 : 2);

      EXPECT_GT(NarrowBand::n_cells_with_active_fe_index(fixture.dof_handler_band,
                                                         NarrowBand::ActiveFEIndex::cut),
                0U);
      EXPECT_GT(NarrowBand::n_cells_with_active_fe_index(fixture.dof_handler_band,
                                                         NarrowBand::ActiveFEIndex::neighbor),
                0U);

      // level set on the cut cells
      VectorType level_set_cut;
      fixture.initialize_vector(fixture.dof_handler_cut, level_set_cut);
      dealii::VectorTools::interpolate(fixture.mapping,
                                       fixture.dof_handler_cut,
                                       fixture.level_set_function,
                                       level_set_cut);

      // copy to the band and extend
      VectorType level_set_band;
      fixture.initialize_vector(fixture.dof_handler_band, level_set_band);
      NarrowBand::copy_cut_cell_values(fixture.dof_handler_cut,
                                       level_set_cut,
                                       fixture.dof_handler_band,
                                       level_set_band);

      NarrowBandData<number> data;
      data.extension.ghost_penalty_faces            = faces;
      data.extension.minimum_source_volume_fraction = minimum_source_volume_fraction;
      data.extension.linear_solver.rel_tolerance    = 1e-14;
      data.extension.linear_solver.abs_tolerance    = 1e-16;
      data.extension.linear_solver.max_iterations   = 10000;

      const dealii::ConditionalOStream pcout(std::cout, false);

      NarrowBandGhostPenaltyExtension<dim, number> extension(data.extension,
                                                             fixture.mapping,
                                                             pcout);
      extension.extend(fixture.dof_handler_band, level_set_band);

      // compare with the interpolated polynomial
      VectorType level_set_band_exact;
      fixture.initialize_vector(fixture.dof_handler_band, level_set_band_exact);
      dealii::VectorTools::interpolate(fixture.mapping,
                                       fixture.dof_handler_band,
                                       fixture.level_set_function,
                                       level_set_band_exact);

      level_set_band -= level_set_band_exact;
      EXPECT_LT(level_set_band.linfty_norm(), tolerance * level_set_band_exact.linfty_norm());
    }

    /**
     * The sign limiter removes spurious sign changes of the extended level set on the neighbor
     * cells. On the cut cells of the zero iso-surface of the linear function
     * l(x) = x_0 + 0.4 x_1 + 0.1, the level set phi = l + c l^2 with c = 3 / h is prescribed. Its
     * polynomial extension changes the sign within the neighbor cells on the side l < 0, since
     * phi > 0 for l < -1/c. With the limiter, the extended level set does not change its sign
     * within any neighbor cell, while the cell means are preserved.
     */
    template <int dim>
    void
    test_sign_limiter()
    {
      constexpr unsigned int fe_degree     = 2;
      constexpr unsigned int n_refinements = 3;
      NarrowBandFixture<dim> fixture(fe_degree, n_refinements, 1 /*linear level set*/);

      const double h = 2. / dealii::Utilities::pow(2, n_refinements);
      const double c = 3. / h;

      const dealii::ScalarFunctionFromFunctionObject<dim> curved_level_set(
        [&](const dealii::Point<dim> &p) {
          const double l = fixture.level_set_function.value(p, 0);
          return l + c * l * l;
        });

      const dealii::ConditionalOStream pcout(std::cout, false);

      // extend the level set from the cut cells for the given limiter tolerance
      const auto extend = [&](const double sign_limiter_tolerance) {
        VectorType level_set_cut, level_set_band;
        fixture.initialize_vector(fixture.dof_handler_cut, level_set_cut);
        fixture.initialize_vector(fixture.dof_handler_band, level_set_band);
        dealii::VectorTools::interpolate(fixture.mapping,
                                         fixture.dof_handler_cut,
                                         curved_level_set,
                                         level_set_cut);
        NarrowBand::copy_cut_cell_values(fixture.dof_handler_cut,
                                         level_set_cut,
                                         fixture.dof_handler_band,
                                         level_set_band);

        NarrowBandData<number> data;
        data.extension.sign_limiter_tolerance       = sign_limiter_tolerance;
        data.extension.linear_solver.rel_tolerance  = 1e-14;
        data.extension.linear_solver.abs_tolerance  = 1e-16;
        data.extension.linear_solver.max_iterations = 10000;

        NarrowBandGhostPenaltyExtension<dim, number> extension(data.extension,
                                                               fixture.mapping,
                                                               pcout);
        extension.extend(fixture.dof_handler_band, level_set_band);
        return level_set_band;
      };

      const VectorType level_set_unlimited = extend(-1.);
      const VectorType level_set_limited   = extend(0.05);

      // sample points and quadrature rule of the cell mean on the reference cell
      const auto &fe = fixture.fe_collection_band[NarrowBand::ActiveFEIndex::neighbor];
      const dealii::QGaussLobatto<dim> sample_points(fe_degree + 2);
      const dealii::QGauss<dim>        quadrature(fe_degree + 1);

      const auto evaluate = [&](const dealii::Vector<number> &values, const dealii::Point<dim> &p) {
        number value = 0.;
        for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
          value += values[i] * fe.shape_value(i, p);
        return value;
      };

      // sign change at the sample points (values below 1e-10 are considered as zero)
      const auto has_sign_change = [&](const dealii::Vector<number> &values) {
        number min_value = std::numeric_limits<number>::max();
        number max_value = std::numeric_limits<number>::lowest();
        for (const auto &p : sample_points.get_points())
          {
            min_value = std::min(min_value, evaluate(values, p));
            max_value = std::max(max_value, evaluate(values, p));
          }
        return min_value < -1e-10 and max_value > 1e-10;
      };

      const auto mean = [&](const dealii::Vector<number> &values) {
        number value = 0.;
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          value += evaluate(values, quadrature.point(q)) * quadrature.weight(q);
        return value;
      };

      unsigned int           n_sign_changes_unlimited = 0, n_sign_changes_limited = 0;
      dealii::Vector<number> values_unlimited(fe.n_dofs_per_cell());
      dealii::Vector<number> values_limited(fe.n_dofs_per_cell());

      for (const auto &cell : fixture.dof_handler_band.active_cell_iterators())
        {
          if (not cell->is_locally_owned() or
              cell->active_fe_index() != NarrowBand::ActiveFEIndex::neighbor)
            continue;

          cell->get_dof_values(level_set_unlimited, values_unlimited);
          cell->get_dof_values(level_set_limited, values_limited);

          n_sign_changes_unlimited += has_sign_change(values_unlimited);
          n_sign_changes_limited += has_sign_change(values_limited);

          EXPECT_NEAR(mean(values_limited), mean(values_unlimited), 1e-10);
        }

      EXPECT_GT(dealii::Utilities::MPI::sum(n_sign_changes_unlimited, MPI_COMM_WORLD), 0U);
      EXPECT_EQ(dealii::Utilities::MPI::sum(n_sign_changes_limited, MPI_COMM_WORLD), 0U);
    }

    /**
     * The upwind DG advection operator evaluates -u.grad(phi) exactly for a linear level-set
     * field and the rigid-body rotation, since the jumps across the faces vanish and u.grad(phi)
     * is linear.
     */
    template <int dim>
    void
    test_advection_operator_linear_field(const unsigned int fe_degree)
    {
      NarrowBandFixture<dim> fixture(fe_degree, dim == 2 ? 3 : 2);

      // linear level set phi(x) = 0.2 + x_0 - 0.5 x_1 and the exact result
      // -u.grad(phi) = x_1 + 0.5 x_0 of the rotation u = (-x_1, x_0, 0)
      const dealii::ScalarFunctionFromFunctionObject<dim> linear_level_set(
        [](const dealii::Point<dim> &p) { return 0.2 + p[0] - 0.5 * p[1]; });
      const dealii::ScalarFunctionFromFunctionObject<dim> exact_result(
        [](const dealii::Point<dim> &p) { return p[1] + 0.5 * p[0]; });

      MatrixFreeFixture<dim> matrix_free_fixture(fixture);
      const auto            &scratch_data = matrix_free_fixture.scratch_data;
      const unsigned int     band_dof_idx = matrix_free_fixture.band_dof_idx;

      NarrowBandAdvectionDGOperator<dim, number> advection_operator(scratch_data,
                                                                    band_dof_idx,
                                                                    matrix_free_fixture.quad_idx);
      advection_operator.set_velocity_function(std::make_shared<Rotation<dim>>());

      VectorType level_set, result, result_exact;
      scratch_data.initialize_dof_vector(level_set, band_dof_idx);
      scratch_data.initialize_dof_vector(result, band_dof_idx);
      scratch_data.initialize_dof_vector(result_exact, band_dof_idx);

      dealii::VectorTools::interpolate(scratch_data.get_mapping(),
                                       fixture.dof_handler_band,
                                       linear_level_set,
                                       level_set);
      dealii::VectorTools::interpolate(scratch_data.get_mapping(),
                                       fixture.dof_handler_band,
                                       exact_result,
                                       result_exact);

      advection_operator.apply_operator(0.0, result, level_set);

      result -= result_exact;
      EXPECT_LT(result.linfty_norm(), 1e-10 * result_exact.linfty_norm());
    }

    /**
     * Artificial viscosity of the DG advection operator: (a) the modal smoothness indicator flags
     * the cells with a kink of the level set and no cells of a linear level set; (b) with a
     * constant viscosity epsilon, the operator evaluates -u.grad(phi) + epsilon laplace(phi)
     * exactly for a quadratic level set on the cells, whose faces are all shared with other band
     * cells (SIPG consistency; natural boundary condition at the outer boundary of the band).
     */
    template <int dim>
    void
    test_advection_operator_artificial_viscosity(const unsigned int fe_degree)
    {
      NarrowBandFixture<dim> fixture(fe_degree, 3);
      MatrixFreeFixture<dim> matrix_free_fixture(fixture);
      const auto            &scratch_data = matrix_free_fixture.scratch_data;
      const auto            &dof_handler  = fixture.dof_handler_band;
      const unsigned int     band_dof_idx = matrix_free_fixture.band_dof_idx;

      NarrowBandAdvectionDGOperator<dim, number> advection_operator(scratch_data,
                                                                    band_dof_idx,
                                                                    matrix_free_fixture.quad_idx);
      advection_operator.set_velocity_function(std::make_shared<Rotation<dim>>());

      VectorType level_set, viscosity, result, result_exact;
      scratch_data.initialize_dof_vector(level_set, band_dof_idx);
      scratch_data.initialize_dof_vector(viscosity, band_dof_idx);
      scratch_data.initialize_dof_vector(result, band_dof_idx);
      scratch_data.initialize_dof_vector(result_exact, band_dof_idx);

      const number threshold         = -4. * std::log10(static_cast<number>(fe_degree));
      const auto   compute_viscosity = [&](const dealii::Function<dim> &function) {
        dealii::VectorTools::interpolate(scratch_data.get_mapping(),
                                         dof_handler,
                                         function,
                                         level_set);
        return advection_operator.compute_artificial_viscosity(
          level_set, viscosity, 0., 1e-3, 0.5, threshold, 0.5, false);
      };

      // (a1) linear level set: no viscosity
      {
        const dealii::ScalarFunctionFromFunctionObject<dim> linear(
          [](const dealii::Point<dim> &p) { return 0.2 + p[0] - 0.5 * p[1]; });
        EXPECT_EQ(compute_viscosity(linear).n_cells, 0U);
      }

      // (a2) kink at x_0 = 0.1, i.e., within the cells with 0 < x_0 < 0.25
      {
        const dealii::ScalarFunctionFromFunctionObject<dim> kink(
          [](const dealii::Point<dim> &p) { return std::abs(p[0] - 0.1); });
        const auto statistics = compute_viscosity(kink);

        unsigned int           n_kink_cells = 0;
        dealii::Vector<number> values(fixture.fe_collection_band[1].n_dofs_per_cell());
        for (const auto &cell : dof_handler.active_cell_iterators())
          if (cell->is_locally_owned() and cell->get_fe().n_dofs_per_cell() > 0)
            {
              const bool is_kink_cell = cell->center()[0] > 0. and cell->center()[0] < 0.25;
              cell->get_dof_values(viscosity, values);
              EXPECT_EQ(values[0] > 0., is_kink_cell);
              n_kink_cells += is_kink_cell;
            }

        EXPECT_GT(statistics.n_cells, 0U);
        EXPECT_EQ(statistics.n_cells, dealii::Utilities::MPI::sum(n_kink_cells, MPI_COMM_WORLD));
      }

      // (b) constant viscosity and quadratic level set phi = x_0^2: -u.grad(phi) = 2 x_0 x_1 for
      // the rotation u = (-x_1, x_0, 0) and laplace(phi) = 2
      const number epsilon = 0.1;
      viscosity            = epsilon;
      viscosity.update_ghost_values();
      advection_operator.set_artificial_viscosity(&viscosity);

      const dealii::ScalarFunctionFromFunctionObject<dim> quadratic(
        [](const dealii::Point<dim> &p) { return p[0] * p[0]; });
      const dealii::ScalarFunctionFromFunctionObject<dim> exact_result(
        [&](const dealii::Point<dim> &p) { return 2. * p[0] * p[1] + 2. * epsilon; });

      dealii::VectorTools::interpolate(scratch_data.get_mapping(),
                                       dof_handler,
                                       quadratic,
                                       level_set);
      dealii::VectorTools::interpolate(scratch_data.get_mapping(),
                                       dof_handler,
                                       exact_result,
                                       result_exact);

      advection_operator.apply_operator(0.0, result, level_set);
      result.update_ghost_values();
      result_exact.update_ghost_values();

      unsigned int           n_interior_cells = 0;
      number                 max_error        = 0.;
      dealii::Vector<number> values(fixture.fe_collection_band[1].n_dofs_per_cell());
      dealii::Vector<number> values_exact(values.size());
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
            continue;

          bool is_interior = true;
          for (const unsigned int f : cell->face_indices())
            if (cell->at_boundary(f) or cell->neighbor(f)->get_fe().n_dofs_per_cell() == 0)
              is_interior = false;
          if (not is_interior)
            continue;

          ++n_interior_cells;
          cell->get_dof_values(result, values);
          cell->get_dof_values(result_exact, values_exact);
          values -= values_exact;
          max_error = std::max(max_error, values.linfty_norm());
        }

      EXPECT_GT(dealii::Utilities::MPI::sum(n_interior_cells, MPI_COMM_WORLD), 0U);
      EXPECT_LT(dealii::Utilities::MPI::max(max_error, MPI_COMM_WORLD), 1e-10);

      advection_operator.set_artificial_viscosity(nullptr);
    }

    /**
     * Test the upwind flux of the DG advection operator for a piecewise constant level-set field
     * and the constant velocity u = e_x: within each cell, the mean value of the result is
     * -(phi_K - phi_left) / h, where phi_left is the value of the left neighbor (inflow face),
     * if the left neighbor belongs to the band. At the outer boundary of the band and at the
     * domain boundary, phi_ext = phi_int is used, i.e., the inflow face does not contribute.
     */
    template <int dim>
    void
    test_advection_operator_upwind_flux(const unsigned int fe_degree)
    {
      NarrowBandFixture<dim> fixture(fe_degree, dim == 2 ? 3 : 2);
      MatrixFreeFixture<dim> matrix_free_fixture(fixture);
      const auto            &scratch_data = matrix_free_fixture.scratch_data;
      const auto            &dof_handler  = fixture.dof_handler_band;
      const unsigned int     band_dof_idx = matrix_free_fixture.band_dof_idx;

      std::vector<double> velocity(dim, 0.);
      velocity[0] = 1.;

      NarrowBandAdvectionDGOperator<dim, number> advection_operator(scratch_data,
                                                                    band_dof_idx,
                                                                    matrix_free_fixture.quad_idx);
      advection_operator.set_velocity_function(
        std::make_shared<dealii::Functions::ConstantFunction<dim>>(velocity));

      // piecewise constant level set
      const auto cell_value = [](const auto &cell) {
        const auto x = cell->center();
        return std::sin(3. * x[0]) + x[1] * x[1];
      };

      VectorType level_set, result;
      scratch_data.initialize_dof_vector(level_set, band_dof_idx);
      scratch_data.initialize_dof_vector(result, band_dof_idx);

      for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned() and cell->get_fe().n_dofs_per_cell() > 0)
          {
            dealii::Vector<number> values(cell->get_fe().n_dofs_per_cell());
            values = cell_value(cell);
            cell->set_dof_values(values, level_set);
          }

      advection_operator.apply_operator(0.0, result, level_set);
      result.update_ghost_values();

      // cell means of the result
      const dealii::hp::MappingCollection<dim> mapping_collection(scratch_data.get_mapping());
      dealii::hp::QCollection<dim>             q_collection;
      for (unsigned int i = 0; i < dof_handler.get_fe_collection().size(); ++i)
        q_collection.push_back(dealii::QGauss<dim>(fe_degree + 1));
      dealii::hp::FEValues<dim> hp_fe_values(mapping_collection,
                                             dof_handler.get_fe_collection(),
                                             q_collection,
                                             dealii::update_values | dealii::update_JxW_values);

      unsigned int n_cells_with_inflow = 0;
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
            continue;

          hp_fe_values.reinit(cell);
          const auto         &fe_values = hp_fe_values.get_present_fe_values();
          std::vector<number> values(fe_values.n_quadrature_points);
          fe_values.get_function_values(result, values);

          number mean = 0., volume = 0.;
          for (const unsigned int q : fe_values.quadrature_point_indices())
            {
              mean += values[q] * fe_values.JxW(q);
              volume += fe_values.JxW(q);
            }
          mean /= volume;

          // face 0 is the face with the outer normal -e_x
          number expected = 0.;
          if (not cell->at_boundary(0) and cell->neighbor(0)->get_fe().n_dofs_per_cell() > 0)
            {
              expected =
                -(cell_value(cell) - cell_value(cell->neighbor(0))) / cell->extent_in_direction(0);
              ++n_cells_with_inflow;
            }

          EXPECT_NEAR(mean, expected, 1e-10 * (1. + std::abs(expected)));
        }

      EXPECT_GT(dealii::Utilities::MPI::sum(n_cells_with_inflow, MPI_COMM_WORLD), 0U);
    }

    /**
     * Interpolate a function on the cut cells and copy it to the band.
     */
    template <int dim>
    VectorType
    create_band_vector_from_cut_cells(const NarrowBandFixture<dim> &fixture,
                                      const dealii::Function<dim>  &function)
    {
      VectorType level_set_cut, level_set_band;
      fixture.initialize_vector(fixture.dof_handler_cut, level_set_cut);
      fixture.initialize_vector(fixture.dof_handler_band, level_set_band);
      dealii::VectorTools::interpolate(fixture.mapping,
                                       fixture.dof_handler_cut,
                                       function,
                                       level_set_cut);
      NarrowBand::copy_cut_cell_values(fixture.dof_handler_cut,
                                       level_set_cut,
                                       fixture.dof_handler_band,
                                       level_set_band);
      return level_set_band;
    }

    /**
     * Maximum error of the level set on the band with respect to a function at the nodes of the
     * neighbor cells (summed over all processes).
     */
    template <int dim>
    number
    compute_neighbor_cell_error(const NarrowBandFixture<dim> &fixture,
                                const VectorType             &level_set_band,
                                const dealii::Function<dim>  &function)
    {
      VectorType level_set_band_exact;
      fixture.initialize_vector(fixture.dof_handler_band, level_set_band_exact);
      dealii::VectorTools::interpolate(fixture.mapping,
                                       fixture.dof_handler_band,
                                       function,
                                       level_set_band_exact);

      const auto            &fe = fixture.fe_collection_band[NarrowBand::ActiveFEIndex::neighbor];
      dealii::Vector<number> values(fe.n_dofs_per_cell()), values_exact(fe.n_dofs_per_cell());

      number error = 0.;
      for (const auto &cell : fixture.dof_handler_band.active_cell_iterators())
        if (cell->is_locally_owned() and
            cell->active_fe_index() == NarrowBand::ActiveFEIndex::neighbor)
          {
            cell->get_dof_values(level_set_band, values);
            cell->get_dof_values(level_set_band_exact, values_exact);
            values -= values_exact;
            error = std::max(error, values.linfty_norm());
          }

      return dealii::Utilities::MPI::max(error, MPI_COMM_WORLD);
    }

    /**
     * The closest-point extension of a linear level-set field reproduces the linear field on the
     * neighbor cells, since the first-order Taylor expansion from any point is exact.
     */
    template <int dim>
    void
    test_closest_point_extension_linear_field(const unsigned int fe_degree)
    {
      NarrowBandFixture<dim> fixture(fe_degree, dim == 2 ? 3 : 2, 1 /*linear level set*/);

      VectorType level_set_band =
        create_band_vector_from_cut_cells(fixture, fixture.level_set_function);

      NarrowBandData<number> data;
      data.extension.type = NarrowBandExtensionType::closest_point;

      const dealii::ConditionalOStream             pcout(std::cout, false);
      NarrowBandClosestPointExtension<dim, number> extension(data.extension,
                                                             fixture.mapping,
                                                             pcout);
      extension.extend(fixture.dof_handler_band, level_set_band);

      EXPECT_LT(compute_neighbor_cell_error(fixture, level_set_band, fixture.level_set_function),
                1e-12);
    }

    /**
     * The closest-point extension of the interpolated signed distance function of a circle
     * approximates the signed distance function on the neighbor cells.
     */
    template <int dim>
    number
    compute_closest_point_extension_error_circle(const unsigned int fe_degree,
                                                 const unsigned int n_refinements)
    {
      dealii::Point<dim> center;
      center[0] = 0.1;
      center[1] = -0.05;
      const dealii::Functions::SignedDistance::Sphere<dim> sphere(center, 0.55);

      NarrowBandFixture<dim> fixture(fe_degree, n_refinements, fe_degree, &sphere);

      VectorType level_set_band = create_band_vector_from_cut_cells(fixture, sphere);

      NarrowBandData<number> data;
      data.extension.type = NarrowBandExtensionType::closest_point;

      const dealii::ConditionalOStream             pcout(std::cout, false);
      NarrowBandClosestPointExtension<dim, number> extension(data.extension,
                                                             fixture.mapping,
                                                             pcout);
      extension.extend(fixture.dof_handler_band, level_set_band);

      return compute_neighbor_cell_error(fixture, level_set_band, sphere);
    }

    /**
     * Elliptic reinitialization on the cut cells of the narrow band of two circles separated by a
     * gap smaller than the cell size with the SIPG face terms on all faces between two cut cells or
     * only on the faces crossed by the interface. The cut cells of the two circles share faces,
     * which are not crossed by the interface. The input is the signed distance function scaled by
     * a smooth positive factor (same zero level set, |grad phi| != 1). Returns the maximum nodal
     * error with respect to the signed distance function before and after the reinitialization
     * and the number of faces crossed by the interface and between two cut cells.
     */
    template <int dim>
    std::tuple<number, number, std::pair<unsigned int, unsigned int>>
    test_reinitialization_faces(const unsigned int fe_degree,
                                const unsigned int n_refinements,
                                const bool         interface_faces_only)
    {
      const double h      = 2. / dealii::Utilities::pow(2, n_refinements);
      const double radius = 0.3;
      const double gap    = 0.4 * h;

      dealii::Point<dim> center_0, center_1;
      center_0[0] = -radius - 0.5 * gap;
      center_1[0] = radius + 0.5 * gap;
      center_0[1] = center_1[1] = 0.02;

      const dealii::Functions::SignedDistance::Sphere<dim> sphere_0(center_0, radius);
      const dealii::Functions::SignedDistance::Sphere<dim> sphere_1(center_1, radius);

      // signed distance function of the union of the two circles and scaled level set
      const dealii::ScalarFunctionFromFunctionObject<dim> signed_distance(
        [&](const dealii::Point<dim> &p) {
          return std::min(sphere_0.value(p), sphere_1.value(p));
        });
      const dealii::ScalarFunctionFromFunctionObject<dim> scaled_level_set(
        [&](const dealii::Point<dim> &p) {
          return (1. + 0.4 * p[0] + 0.2 * p[1]) * signed_distance.value(p);
        });

      NarrowBandFixture<dim> fixture(fe_degree, n_refinements, fe_degree, &signed_distance);
      MatrixFreeFixture<dim> matrix_free_fixture(fixture);
      auto                  &scratch_data = matrix_free_fixture.scratch_data;
      const unsigned int     cut_dof_idx  = matrix_free_fixture.band_dof_idx + 1;

      ReinitializationData<number> reinit_data;
      reinit_data.modeltype                                   = ModelType::elliptic;
      reinit_data.fe.type                                     = FiniteElementType::FE_DGQ;
      reinit_data.fe.degree                                   = fe_degree;
      reinit_data.elliptic.intersected_cells_only             = true;
      reinit_data.elliptic.penalty_parameter                  = 1e4;
      reinit_data.elliptic.nlsolve.max_nonlinear_iterations   = 1000;
      reinit_data.elliptic.nlsolve.residual_tolerance         = 1e-8;
      reinit_data.elliptic.nlsolve.field_correction_tolerance = 1e-8;
      reinit_data.elliptic.nlsolve.verbosity_level            = 0;
      reinit_data.linear_solver.solver_type                   = LinearSolverType::CG;
      reinit_data.linear_solver.preconditioner_type           = PreconditionerType::ILU;
      reinit_data.linear_solver.do_matrix_free                = true;
      reinit_data.linear_solver.rel_tolerance                 = 1e-12;
      reinit_data.linear_solver.abs_tolerance                 = 1e-20;
      reinit_data.linear_solver.max_iterations                = 1000;

      ReinitializationEllipticOperation<dim, number> reinit_operation(
        scratch_data, reinit_data, cut_dof_idx, matrix_free_fixture.quad_idx, cut_dof_idx);
      reinit_operation.set_face_terms_on_interface_faces_only(interface_faces_only);
      reinit_operation.reinit();

      VectorType level_set, level_set_exact;
      scratch_data.initialize_dof_vector(level_set, cut_dof_idx);
      scratch_data.initialize_dof_vector(level_set_exact, cut_dof_idx);
      dealii::VectorTools::interpolate(scratch_data.get_mapping(),
                                       fixture.dof_handler_cut,
                                       scaled_level_set,
                                       level_set);
      dealii::VectorTools::interpolate(scratch_data.get_mapping(),
                                       fixture.dof_handler_cut,
                                       signed_distance,
                                       level_set_exact);

      VectorType error = level_set;
      error -= level_set_exact;
      const number error_initial = error.linfty_norm();

      reinit_operation.set_initial_condition(level_set);
      reinit_operation.solve();

      error = reinit_operation.get_level_set();
      error -= level_set_exact;

      return {error_initial, error.linfty_norm(), reinit_operation.get_n_interface_faces()};
    }

    /**
     * Reinitialization of a scaled signed distance of a circle (radius 0.5, p = 2, cut cells of
     * 16^2 cells on [-1,1]^2, penalty term at the zero level set) in the continuous level set space
     * (NarrowBandContinuousReinitialization). Returns the maximum nodal error with respect to the
     * signed distance on the cut cells, the maximum jump of the result between adjacent cut cells
     * (continuity of the discontinuous storage) and the number of fixed-point iterations.
     */
    std::tuple<number, number, unsigned int>
    test_continuous_reinitialization()
    {
      constexpr int      dim       = 2;
      const unsigned int fe_degree = 2;

      dealii::Point<dim> center;
      center[0] = 0.031;
      center[1] = -0.027;
      const dealii::Functions::SignedDistance::Sphere<dim> sphere(center, 0.5);
      const dealii::ScalarFunctionFromFunctionObject<dim>  signed_distance(
        [&](const dealii::Point<dim> &p) { return -sphere.value(p); });
      const dealii::ScalarFunctionFromFunctionObject<dim> scaled_level_set(
        [&](const dealii::Point<dim> &p) {
          return (1. + 0.4 * p[0] + 0.2 * p[1]) * signed_distance.value(p);
        });

      NarrowBandFixture<dim> fixture(fe_degree, 4, fe_degree, &signed_distance);

      ReinitializationData<number> reinit_data;
      reinit_data.elliptic.penalty_parameter                  = 1e4;
      reinit_data.elliptic.nlsolve.max_nonlinear_iterations   = 1000;
      reinit_data.elliptic.nlsolve.residual_tolerance         = 1e-8;
      reinit_data.elliptic.nlsolve.field_correction_tolerance = 1e-8;

      VectorType level_set, level_set_exact;
      fixture.initialize_vector(fixture.dof_handler_cut, level_set);
      fixture.initialize_vector(fixture.dof_handler_cut, level_set_exact);
      dealii::VectorTools::interpolate(fixture.mapping,
                                       fixture.dof_handler_cut,
                                       scaled_level_set,
                                       level_set);
      dealii::VectorTools::interpolate(fixture.mapping,
                                       fixture.dof_handler_cut,
                                       signed_distance,
                                       level_set_exact);

      const dealii::ConditionalOStream                  pcout(std::cout, false);
      NarrowBandContinuousReinitialization<dim, number> continuous_reinitialization(
        fixture.mapping,
        fixture.triangulation,
        reinit_data,
        fe_degree,
        fixture.fe_collection_cut,
        pcout);
      continuous_reinitialization.distribute_dofs(fixture.dof_handler_cut);
      continuous_reinitialization.interpolate_from_discontinuous(fixture.dof_handler_cut,
                                                                 level_set);
      const auto statistics = continuous_reinitialization.reinitialize({}, false);
      continuous_reinitialization.interpolate_to_discontinuous(fixture.dof_handler_cut, level_set);

      number error = 0.;
      for (unsigned int i = 0; i < level_set.locally_owned_size(); ++i)
        error =
          std::max<number>(error,
                           std::abs(level_set.local_element(i) - level_set_exact.local_element(i)));
      error = dealii::Utilities::MPI::max(error, MPI_COMM_WORLD);

      // jump at the midpoints of the faces between locally owned cut cells
      level_set.update_ghost_values();
      number                 jump = 0.;
      dealii::Vector<number> values, values_neighbor;
      for (const auto &cell : fixture.dof_handler_cut.active_cell_iterators())
        {
          if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
            continue;
          values.reinit(cell->get_fe().n_dofs_per_cell());
          cell->get_dof_values(level_set, values);
          for (const unsigned int f : cell->face_indices())
            {
              if (cell->at_boundary(f))
                continue;
              const auto neighbor = cell->neighbor(f);
              if (not neighbor->is_locally_owned() or neighbor->get_fe().n_dofs_per_cell() == 0)
                continue;
              values_neighbor.reinit(neighbor->get_fe().n_dofs_per_cell());
              neighbor->get_dof_values(level_set, values_neighbor);
              const dealii::Point<dim> x = cell->face(f)->center();
              const auto p_cell          = fixture.mapping.transform_real_to_unit_cell(cell, x);
              const auto p_neighbor      = fixture.mapping.transform_real_to_unit_cell(neighbor, x);
              number     value = 0., value_neighbor = 0.;
              for (unsigned int i = 0; i < values.size(); ++i)
                {
                  value += values[i] * cell->get_fe().shape_value(i, p_cell);
                  value_neighbor +=
                    values_neighbor[i] * neighbor->get_fe().shape_value(i, p_neighbor);
                }
              jump = std::max<number>(jump, std::abs(value - value_neighbor));
            }
        }
      jump = dealii::Utilities::MPI::max(jump, MPI_COMM_WORLD);

      return {error, jump, statistics.n_iterations};
    }

    /**
     * Result of test_closed_interface().
     */
    template <int dim>
    struct ClosedInterfaceResult
    {
      typename NarrowBandContinuousReinitialization<dim, number>::CorrectionStatistics statistics;

      /// Minimum of s phi / h on the faces between the cut cells and the cells, which are not cut
      /// (s: sign of the neighbor cell), at 21^(dim-1) points per face.
      number min_value = std::numeric_limits<number>::max();

      /// Maximum jump of the level set across the faces between cut cells.
      number jump = 0.;

      /// Maximum change of the level set at the nodes of the cut cells.
      number max_change = 0.;

      /// Maximum change of the level set at the nodes on the upper faces of the cut cells.
      number max_change_top = 0.;

      /// Number of cells behind faces crossed by the interface before the correction (see
      /// NarrowBandContinuousReinitialization::find_cells_behind_open_faces()).
      unsigned int n_cells_behind_open_faces = 0;
    };

    /**
     * Closed-interface correction (NarrowBandContinuousReinitialization::close_interface()) of a
     * perturbed plane: the cut cells are the layer 0 < x_{d-1} < 0.25 of 8^dim cells on
     * [-1,1]^dim (zero level set x_{d-1} = 0.1), the cells below (above) the layer have the sign
     * -1 (+1). The level set phi = x_{d-1} - 0.1 - a prod_{e < d-1} sin(2 pi x_e) of the cut
     * cells leaves them through the faces to the cells above and below for a = 0.2 (open
     * interface).
     */
    template <int dim>
    ClosedInterfaceResult<dim>
    test_closed_interface(const unsigned int                                       fe_degree,
                          const number                                             amplitude,
                          const std::function<number(const dealii::Point<dim> &)> &function = {})
    {
      const dealii::ScalarFunctionFromFunctionObject<dim> plane(
        [](const dealii::Point<dim> &p) { return p[dim - 1] - 0.1; });
      const dealii::ScalarFunctionFromFunctionObject<dim> perturbed_plane(
        [&](const dealii::Point<dim> &p) {
          number perturbation = amplitude;
          for (unsigned int e = 0; e + 1 < dim; ++e)
            perturbation *= std::sin(2. * dealii::numbers::PI * p[e]);
          return function ? function(p) : p[dim - 1] - 0.1 - perturbation;
        });

      NarrowBandFixture<dim> fixture(fe_degree, 3, fe_degree, &plane);
      const number           h = 0.25;

      std::vector<int> cell_signs(fixture.triangulation.n_active_cells(), 0);
      for (const auto &cell : fixture.triangulation.active_cell_iterators())
        if (not cell->is_artificial())
          cell_signs[cell->active_cell_index()] = plane.value(cell->center()) > 0. ? 1 : -1;

      VectorType level_set, level_set_initial;
      fixture.initialize_vector(fixture.dof_handler_cut, level_set);
      dealii::VectorTools::interpolate(fixture.mapping,
                                       fixture.dof_handler_cut,
                                       perturbed_plane,
                                       level_set);
      level_set_initial = level_set;

      const ReinitializationData<number>                reinit_data;
      const dealii::ConditionalOStream                  pcout(std::cout, false);
      NarrowBandContinuousReinitialization<dim, number> continuous_reinitialization(
        fixture.mapping,
        fixture.triangulation,
        reinit_data,
        fe_degree,
        fixture.fe_collection_cut,
        pcout);
      continuous_reinitialization.distribute_dofs(fixture.dof_handler_cut);
      continuous_reinitialization.interpolate_from_discontinuous(fixture.dof_handler_cut,
                                                                 level_set);

      ClosedInterfaceResult<dim> result;
      result.n_cells_behind_open_faces = dealii::Utilities::MPI::sum(
        static_cast<unsigned int>(
          continuous_reinitialization.find_cells_behind_open_faces(cell_signs).size()),
        MPI_COMM_WORLD);
      result.statistics = continuous_reinitialization.close_interface(cell_signs, 0.01);
      continuous_reinitialization.interpolate_to_discontinuous(fixture.dof_handler_cut, level_set);

      // change at the nodes on the upper faces (x_{d-1} = 1 in unit coordinates)
      {
        dealii::Vector<number> values, values_initial;
        for (const auto &cell : fixture.dof_handler_cut.active_cell_iterators())
          {
            if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
              continue;
            values.reinit(cell->get_fe().n_dofs_per_cell());
            values_initial.reinit(cell->get_fe().n_dofs_per_cell());
            cell->get_dof_values(level_set, values);
            cell->get_dof_values(level_set_initial, values_initial);
            for (unsigned int i = 0; i < values.size(); ++i)
              if (cell->get_fe().get_unit_support_points()[i][dim - 1] > 1. - 1e-12)
                result.max_change_top =
                  std::max<number>(result.max_change_top, std::abs(values[i] - values_initial[i]));
          }
        result.max_change_top = dealii::Utilities::MPI::max(result.max_change_top, MPI_COMM_WORLD);
      }

      for (unsigned int i = 0; i < level_set.locally_owned_size(); ++i)
        result.max_change = std::max<number>(result.max_change,
                                             std::abs(level_set.local_element(i) -
                                                      level_set_initial.local_element(i)));

      // s phi on the faces to the cells, which are not cut, and the jump across the faces between
      // cut cells (locally owned and ghost neighbors)
      level_set.update_ghost_values();
      const unsigned int     n_points_1d = 21;
      const unsigned int     n_points    = dealii::Utilities::pow(n_points_1d, dim - 1);
      dealii::Vector<number> values, values_neighbor;
      for (const auto &cell : fixture.dof_handler_cut.active_cell_iterators())
        {
          if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
            continue;

          const auto &fe = cell->get_fe();
          values.reinit(fe.n_dofs_per_cell());
          cell->get_dof_values(level_set, values);
          for (const unsigned int f : cell->face_indices())
            {
              if (cell->at_boundary(f))
                continue;

              const auto neighbor = cell->neighbor(f);
              const bool is_cut   = neighbor->get_fe().n_dofs_per_cell() > 0;
              if (is_cut)
                {
                  values_neighbor.reinit(neighbor->get_fe().n_dofs_per_cell());
                  neighbor->get_dof_values(level_set, values_neighbor);
                }

              for (unsigned int q = 0; q < n_points; ++q)
                {
                  // point on the face in unit coordinates of the cell
                  dealii::Point<dim> unit_point;
                  for (unsigned int d = 0, index = q; d < dim; ++d)
                    if (d == f / 2)
                      unit_point[d] = f % 2;
                    else
                      {
                        unit_point[d] =
                          static_cast<number>(index % n_points_1d) / (n_points_1d - 1);
                        index /= n_points_1d;
                      }

                  number value = 0.;
                  for (unsigned int i = 0; i < values.size(); ++i)
                    value += values[i] * fe.shape_value(i, unit_point);

                  if (is_cut)
                    {
                      const auto x = fixture.mapping.transform_unit_to_real_cell(cell, unit_point);
                      const auto unit_point_neighbor =
                        fixture.mapping.transform_real_to_unit_cell(neighbor, x);
                      number value_neighbor = 0.;
                      for (unsigned int i = 0; i < values_neighbor.size(); ++i)
                        value_neighbor +=
                          values_neighbor[i] * fe.shape_value(i, unit_point_neighbor);
                      result.jump = std::max<number>(result.jump, std::abs(value - value_neighbor));
                    }
                  else
                    result.min_value =
                      std::min<number>(result.min_value,
                                       cell_signs[neighbor->active_cell_index()] * value / h);
                }
            }
        }

      result.min_value  = dealii::Utilities::MPI::min(result.min_value, MPI_COMM_WORLD);
      result.jump       = dealii::Utilities::MPI::max(result.jump, MPI_COMM_WORLD);
      result.max_change = dealii::Utilities::MPI::max(result.max_change, MPI_COMM_WORLD);
      return result;
    }

    /**
     * Elliptic reinitialization on the cut cells of a circle with prescribed DoFs: the DoFs of
     * the cut cells right of the center (x_0 > 0) are prescribed by the signed distance function,
     * while the other cut cells start from the scaled signed distance function (same zero level
     * set, |grad phi| != 1). The prescribed cells are coupled to the other cells by the SIPG face
     * terms or decoupled (@p couple). Returns the maximum change of the prescribed DoFs, the
     * maximum nodal error of the other DoFs with respect to the signed distance function with and
     * without the prescription, and the maximum jump of the level set across the faces between
     * prescribed and free cells (at the Gauss-Lobatto points).
     */
    template <int dim>
    std::array<number, 4>
    test_reinitialization_prescribed_dofs(const unsigned int fe_degree,
                                          const unsigned int n_refinements,
                                          const bool         couple)
    {
      dealii::Point<dim> center;
      center[0] = 0.031;
      center[1] = -0.027;
      const dealii::Functions::SignedDistance::Sphere<dim> sphere(center, 0.5);
      const dealii::ScalarFunctionFromFunctionObject<dim>  signed_distance(
        [&](const dealii::Point<dim> &p) { return -sphere.value(p); });
      const dealii::ScalarFunctionFromFunctionObject<dim> scaled_level_set(
        [&](const dealii::Point<dim> &p) {
          return (1. + 0.4 * p[0] + 0.2 * p[1]) * signed_distance.value(p);
        });

      NarrowBandFixture<dim> fixture(fe_degree, n_refinements, fe_degree, &signed_distance);
      MatrixFreeFixture<dim> matrix_free_fixture(fixture);
      auto                  &scratch_data = matrix_free_fixture.scratch_data;
      const unsigned int     cut_dof_idx  = matrix_free_fixture.band_dof_idx + 1;

      ReinitializationData<number> reinit_data;
      reinit_data.modeltype                                   = ModelType::elliptic;
      reinit_data.fe.type                                     = FiniteElementType::FE_DGQ;
      reinit_data.fe.degree                                   = fe_degree;
      reinit_data.elliptic.intersected_cells_only             = true;
      reinit_data.elliptic.penalty_parameter                  = 1e4;
      reinit_data.elliptic.nlsolve.max_nonlinear_iterations   = 1000;
      reinit_data.elliptic.nlsolve.residual_tolerance         = 1e-8;
      reinit_data.elliptic.nlsolve.field_correction_tolerance = 1e-8;
      reinit_data.elliptic.nlsolve.verbosity_level            = 0;
      reinit_data.linear_solver.solver_type                   = LinearSolverType::CG;
      reinit_data.linear_solver.preconditioner_type           = PreconditionerType::ILU;
      reinit_data.linear_solver.do_matrix_free                = true;
      reinit_data.linear_solver.rel_tolerance                 = 1e-12;
      reinit_data.linear_solver.abs_tolerance                 = 1e-20;
      reinit_data.linear_solver.max_iterations                = 1000;

      ReinitializationEllipticOperation<dim, number> reinit_operation(
        scratch_data, reinit_data, cut_dof_idx, matrix_free_fixture.quad_idx, cut_dof_idx);
      reinit_operation.reinit();

      VectorType level_set, level_set_exact, mask;
      scratch_data.initialize_dof_vector(level_set, cut_dof_idx);
      scratch_data.initialize_dof_vector(level_set_exact, cut_dof_idx);
      scratch_data.initialize_dof_vector(mask, cut_dof_idx);
      dealii::VectorTools::interpolate(scratch_data.get_mapping(),
                                       fixture.dof_handler_cut,
                                       scaled_level_set,
                                       level_set);
      dealii::VectorTools::interpolate(scratch_data.get_mapping(),
                                       fixture.dof_handler_cut,
                                       signed_distance,
                                       level_set_exact);

      // without prescribed DoFs
      reinit_operation.set_initial_condition(level_set);
      reinit_operation.solve();
      VectorType error_reference = reinit_operation.get_level_set();
      error_reference -= level_set_exact;

      // prescribed DoFs: cut cells right of the center
      dealii::Vector<number> ones;
      for (const auto &cell : fixture.dof_handler_cut.active_cell_iterators())
        if (cell->is_locally_owned() and cell->get_fe().n_dofs_per_cell() > 0 and
            cell->center()[0] > center[0])
          {
            ones.reinit(cell->get_fe().n_dofs_per_cell());
            ones = 1.;
            cell->set_dof_values(ones, mask);
          }
      for (unsigned int i = 0; i < level_set.locally_owned_size(); ++i)
        if (mask.local_element(i) != 0.)
          level_set.local_element(i) = level_set_exact.local_element(i);

      reinit_operation.set_prescribed_dofs(&mask, &level_set, couple);
      reinit_operation.set_initial_condition(level_set);
      reinit_operation.solve();
      reinit_operation.set_prescribed_dofs(nullptr, nullptr);

      const VectorType &result            = reinit_operation.get_level_set();
      number            change_prescribed = 0., error_free = 0., error_free_reference = 0.;
      for (unsigned int i = 0; i < result.locally_owned_size(); ++i)
        if (mask.local_element(i) != 0.)
          change_prescribed =
            std::max<number>(change_prescribed,
                             std::abs(result.local_element(i) - level_set_exact.local_element(i)));
        else
          {
            error_free = std::max<number>(error_free,
                                          std::abs(result.local_element(i) -
                                                   level_set_exact.local_element(i)));
            error_free_reference =
              std::max<number>(error_free_reference, std::abs(error_reference.local_element(i)));
          }

      // jump across the faces between prescribed and free cells (vectors with all locally
      // relevant DoFs)
      const dealii::IndexSet locally_relevant_dofs =
        dealii::DoFTools::extract_locally_relevant_dofs(fixture.dof_handler_cut);
      VectorType result_relevant(fixture.dof_handler_cut.locally_owned_dofs(),
                                 locally_relevant_dofs,
                                 MPI_COMM_WORLD);
      VectorType mask_relevant(result_relevant);
      result_relevant.copy_locally_owned_data_from(result);
      mask_relevant.copy_locally_owned_data_from(mask);
      result_relevant.update_ghost_values();
      mask_relevant.update_ghost_values();
      const dealii::FE_DGQ<dim>            fe(fe_degree);
      const dealii::QGaussLobatto<dim - 1> face_quadrature(fe_degree + 2);
      dealii::FEFaceValues<dim>            fe_face_values(scratch_data.get_mapping(),
                                               fe,
                                               face_quadrature,
                                               dealii::update_values |
                                                 dealii::update_quadrature_points);
      dealii::FEFaceValues<dim>            fe_face_values_neighbor(scratch_data.get_mapping(),
                                                        fe,
                                                        face_quadrature,
                                                        dealii::update_values |
                                                          dealii::update_quadrature_points);
      std::vector<number>                  values(face_quadrature.size()), values_neighbor(values);
      dealii::Vector<number>               mask_values;
      number                               max_jump = 0.;
      for (const auto &cell : fixture.dof_handler_cut.active_cell_iterators())
        {
          if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
            continue;
          mask_values.reinit(cell->get_fe().n_dofs_per_cell());
          cell->get_dof_values(mask_relevant, mask_values);
          if (mask_values[0] == 0.)
            continue;

          for (const unsigned int face : cell->face_indices())
            {
              if (cell->at_boundary(face))
                continue;
              const auto neighbor = cell->neighbor(face);
              if (neighbor->get_fe().n_dofs_per_cell() == 0)
                continue;
              mask_values.reinit(neighbor->get_fe().n_dofs_per_cell());
              neighbor->get_dof_values(mask_relevant, mask_values);
              if (mask_values[0] != 0.)
                continue;

              fe_face_values.reinit(cell, face);
              fe_face_values_neighbor.reinit(neighbor, cell->neighbor_of_neighbor(face));
              fe_face_values.get_function_values(result_relevant, values);
              fe_face_values_neighbor.get_function_values(result_relevant, values_neighbor);
              for (unsigned int q = 0; q < values.size(); ++q)
                for (unsigned int r = 0; r < values.size(); ++r)
                  if (fe_face_values.quadrature_point(q).distance(
                        fe_face_values_neighbor.quadrature_point(r)) < 1e-12)
                    max_jump = std::max<number>(max_jump, std::abs(values[q] - values_neighbor[r]));
            }
        }

      return {{dealii::Utilities::MPI::max(change_prescribed, MPI_COMM_WORLD),
               dealii::Utilities::MPI::max(error_free, MPI_COMM_WORLD),
               dealii::Utilities::MPI::max(error_free_reference, MPI_COMM_WORLD),
               dealii::Utilities::MPI::max(max_jump, MPI_COMM_WORLD)}};
    }

    /**
     * Activation of the neighbor cells depending on the volume fraction of the positive phase of
     * the cut cells. The level set phi(x) = (x_0 - a_0) (a_1 - x_0) on [-1,1]^dim with 8 cells
     * per direction is positive for a_0 < x_0 < a_1. The zero level set is located in the
     * columns of cells 3 (-0.25 < x_0 < 0) and 5 (0.25 < x_0 < 0.5), which have the volume
     * fractions theta_0 = -a_0 / h and theta_1 = (a_1 - 0.25) / h of the positive phase. The face
     * neighbors of the cut cells are the columns 2, 4 and 6, where column 4 is a face neighbor of
     * both columns of cut cells. Returns the column indices of the neighbor cells (summed over
     * all processes).
     */
    template <int dim>
    std::vector<unsigned int>
    compute_neighbor_columns(const double               theta_0,
                             const double               theta_1,
                             const double               threshold,
                             const std::vector<double> &velocity = {})
    {
      constexpr unsigned int fe_degree     = 2;
      constexpr unsigned int n_refinements = 3;
      const double           h             = 0.25;

      dealii::parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
      dealii::GridGenerator::hyper_cube(triangulation, -1., 1.);
      triangulation.refine_global(n_refinements);

      const double                                        a_0 = -theta_0 * h;
      const double                                        a_1 = 0.25 + theta_1 * h;
      const dealii::ScalarFunctionFromFunctionObject<dim> level_set_function(
        [&](const dealii::Point<dim> &p) { return (p[0] - a_0) * (a_1 - p[0]); });

      // cut cells and volume fractions from a DG level-set field on all cells
      dealii::DoFHandler<dim> dof_handler_full(triangulation);
      dof_handler_full.distribute_dofs(dealii::FE_DGQ<dim>(fe_degree));
      VectorType                  level_set_full(dof_handler_full.locally_owned_dofs(),
                                dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_full),
                                MPI_COMM_WORLD);
      const dealii::MappingQ<dim> mapping(1);
      dealii::VectorTools::interpolate(mapping,
                                       dof_handler_full,
                                       level_set_function,
                                       level_set_full);

      std::vector<number>     volume_fractions;
      const std::vector<bool> is_cut_cell = NarrowBand::compute_cut_cells(dof_handler_full,
                                                                          level_set_full,
                                                                          fe_degree,
                                                                          &volume_fractions);

      const auto column = [&](const auto &cell) {
        return static_cast<unsigned int>(std::floor((cell->center()[0] + 1.) / h));
      };

      std::vector<bool> is_small_cut_cell(is_cut_cell.size(), false);
      for (const auto &cell : triangulation.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            const unsigned int index = cell->active_cell_index();
            if (column(cell) == 3 or column(cell) == 5)
              {
                EXPECT_TRUE(is_cut_cell[index]);
                EXPECT_NEAR(volume_fractions[index], column(cell) == 3 ? theta_0 : theta_1, 1e-10);
              }
            else
              EXPECT_FALSE(is_cut_cell[index]);

            is_small_cut_cell[index] = is_cut_cell[index] and volume_fractions[index] < threshold;
          }

      const dealii::hp::FECollection<dim> fe_collection_band =
        NarrowBand::create_band_fe_collection<dim>(fe_degree);
      const dealii::hp::FECollection<dim> fe_collection_cut =
        NarrowBand::create_cut_fe_collection<dim>(fe_degree);
      dealii::DoFHandler<dim> dof_handler_band(triangulation);
      dealii::DoFHandler<dim> dof_handler_cut(triangulation);

      // optionally, only the face neighbors downstream of the cut cells are activated
      dealii::Functions::ConstantFunction<dim> velocity_function(
        velocity.empty() ? std::vector<double>(dim, 0.) : velocity);
      std::function<bool(const typename dealii::Triangulation<dim>::active_cell_iterator &,
                         const unsigned int)>
        is_activating_face;
      if (not velocity.empty())
        is_activating_face =
          [&](const typename dealii::Triangulation<dim>::active_cell_iterator &cell,
              const unsigned int                                               face_no) {
            return NarrowBand::is_downstream_face<dim>(
              mapping, velocity_function, {0.}, cell, face_no, fe_degree + 2);
          };

      NarrowBand::distribute_dofs<dim>(is_cut_cell,
                                       dof_handler_cut,
                                       fe_collection_cut,
                                       dof_handler_band,
                                       fe_collection_band,
                                       is_small_cut_cell,
                                       is_activating_face);

      std::vector<unsigned int> n_neighbor_cells_per_column(8, 0);
      for (const auto &cell : dof_handler_band.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            if (cell->active_fe_index() == NarrowBand::ActiveFEIndex::neighbor)
              ++n_neighbor_cells_per_column[column(cell)];
            EXPECT_EQ(cell->active_fe_index() == NarrowBand::ActiveFEIndex::cut,
                      is_cut_cell[cell->active_cell_index()]);
          }

      // columns, which consist of neighbor cells only
      const unsigned int        n_cells_per_column = dealii::Utilities::pow(8, dim - 1);
      std::vector<unsigned int> neighbor_columns;
      for (unsigned int c = 0; c < 8; ++c)
        {
          const unsigned int n_cells =
            dealii::Utilities::MPI::sum(n_neighbor_cells_per_column[c], MPI_COMM_WORLD);
          EXPECT_TRUE(n_cells == 0 or n_cells == n_cells_per_column);
          if (n_cells == n_cells_per_column)
            neighbor_columns.push_back(c);
        }

      return neighbor_columns;
    }

    /**
     * Exclusion of small cut cells as sources of the extension. The level set
     * phi(x) = (x_0 - a_0 - s x_1) (a_1 - x_0) on [-1,1]^2 with 8 x 8 cells is positive for
     * a_0 + s x_1 < x_0 < a_1. The zero level set is located in the columns of cells 3 and 5. The
     * volume fraction of the positive phase of the cut cells of column 3 decreases from 0.85
     * (bottom row) to 0.15 (top row), the one of column 5 is 0.5. With the minimum source volume
     * fraction 0.5, the cut cells in the upper half of column 3 are small cut cells. The level set
     * of the small cut cells is scaled by a factor (same zero level set and volume fraction) and
     * the extension is compared with the one of the unscaled level set. Returns the maximum
     * difference of the extended level sets on the neighbor cells for each column and half
     * (index 2 * column + (upper half ? 1 : 0)).
     */
    template <int dim>
    std::vector<double>
    compute_small_cut_cell_sensitivity(const NarrowBandExtensionType     type,
                                       const NarrowBandGhostPenaltyFaces faces,
                                       const double minimum_source_volume_fraction)
    {
      static_assert(dim == 2, "Only implemented for 2D.");

      constexpr unsigned int fe_degree     = 2;
      constexpr unsigned int n_refinements = 3;
      const double           h             = 0.25;

      dealii::parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
      dealii::GridGenerator::hyper_cube(triangulation, -1., 1.);
      triangulation.refine_global(n_refinements);

      const double a_0 = -0.125, s = 0.1, a_1 = 0.25 + 0.5 * h;
      const dealii::ScalarFunctionFromFunctionObject<dim> level_set_function(
        [&](const dealii::Point<dim> &p) { return (p[0] - a_0 - s * p[1]) * (a_1 - p[0]); });

      const dealii::MappingQ<dim> mapping(1);

      // cut cells from a DG level-set field on all cells
      dealii::DoFHandler<dim> dof_handler_full(triangulation);
      dof_handler_full.distribute_dofs(dealii::FE_DGQ<dim>(fe_degree));
      VectorType level_set_full(dof_handler_full.locally_owned_dofs(),
                                dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_full),
                                MPI_COMM_WORLD);
      dealii::VectorTools::interpolate(mapping,
                                       dof_handler_full,
                                       level_set_function,
                                       level_set_full);
      const std::vector<bool> is_cut_cell =
        NarrowBand::compute_cut_cells(dof_handler_full, level_set_full, fe_degree);

      const dealii::hp::FECollection<dim> fe_collection_band =
        NarrowBand::create_band_fe_collection<dim>(fe_degree);
      const dealii::hp::FECollection<dim> fe_collection_cut =
        NarrowBand::create_cut_fe_collection<dim>(fe_degree);
      dealii::DoFHandler<dim> dof_handler_band(triangulation);
      dealii::DoFHandler<dim> dof_handler_cut(triangulation);
      NarrowBand::distribute_dofs(
        is_cut_cell, dof_handler_cut, fe_collection_cut, dof_handler_band, fe_collection_band);

      const auto column = [&](const auto &cell) {
        return static_cast<unsigned int>(std::floor((cell->center()[0] + 1.) / h));
      };
      const auto is_upper_half = [](const auto &cell) { return cell->center()[1] > 0.; };

      NarrowBandData<number> data;
      data.extension.type                           = type;
      data.extension.ghost_penalty_faces            = faces;
      data.extension.minimum_source_volume_fraction = minimum_source_volume_fraction;
      data.extension.sign_limiter_tolerance         = -1.;
      data.extension.linear_solver.rel_tolerance    = 1e-14;
      data.extension.linear_solver.abs_tolerance    = 1e-16;
      data.extension.linear_solver.max_iterations   = 10000;

      const dealii::ConditionalOStream pcout(std::cout, false);

      // extend the level set, whose values on the small cut cells are scaled by the given factor
      const auto extend = [&](const double factor) {
        VectorType level_set_band(dof_handler_band.locally_owned_dofs(),
                                  dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_band),
                                  MPI_COMM_WORLD);
        dealii::VectorTools::interpolate(mapping,
                                         dof_handler_band,
                                         level_set_function,
                                         level_set_band);

        dealii::Vector<number> values(fe_collection_band[1].n_dofs_per_cell());
        for (const auto &cell : dof_handler_band.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              if (cell->active_fe_index() == NarrowBand::ActiveFEIndex::neighbor)
                {
                  values = 0.;
                  cell->set_dof_values(values, level_set_band);
                }
              else if (cell->active_fe_index() == NarrowBand::ActiveFEIndex::cut and
                       column(cell) == 3 and is_upper_half(cell))
                {
                  cell->get_dof_values(level_set_band, values);
                  values *= factor;
                  cell->set_dof_values(values, level_set_band);
                }
            }

        if (type == NarrowBandExtensionType::ghost_penalty)
          NarrowBandGhostPenaltyExtension<dim, number>(data.extension, mapping, pcout)
            .extend(dof_handler_band, level_set_band);
        else
          NarrowBandClosestPointExtension<dim, number>(data.extension, mapping, pcout)
            .extend(dof_handler_band, level_set_band);

        return level_set_band;
      };

      const VectorType level_set_unscaled = extend(1.0);
      const VectorType level_set_scaled   = extend(1.5);

      std::vector<double>    differences(16, 0.);
      dealii::Vector<number> values_unscaled(fe_collection_band[1].n_dofs_per_cell());
      dealii::Vector<number> values_scaled(fe_collection_band[1].n_dofs_per_cell());
      for (const auto &cell : dof_handler_band.active_cell_iterators())
        if (cell->is_locally_owned() and
            cell->active_fe_index() == NarrowBand::ActiveFEIndex::neighbor)
          {
            cell->get_dof_values(level_set_unscaled, values_unscaled);
            cell->get_dof_values(level_set_scaled, values_scaled);
            values_scaled -= values_unscaled;
            auto &difference = differences[2 * column(cell) + (is_upper_half(cell) ? 1 : 0)];
            difference       = std::max(difference, values_scaled.linfty_norm());
          }

      for (auto &difference : differences)
        difference = dealii::Utilities::MPI::max(difference, MPI_COMM_WORLD);

      return differences;
    }

    /**
     * Layers of the advected band around the cut cells of the zero level set of
     * phi(x) = |x - c|^2 - r^2 on [-1,1]^2 with 8 x 8 cells, where the circle is contained in the
     * cell with the center c. Returns the number of cells with the FE indices cut, neighbor and
     * outer (summed over all processes).
     */
    std::array<unsigned int, 3>
    count_layered_band_cells(const unsigned int n_layers)
    {
      constexpr int          dim       = 2;
      constexpr unsigned int fe_degree = 2;

      dealii::parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
      dealii::GridGenerator::hyper_cube(triangulation, -1., 1.);
      triangulation.refine_global(3);

      const dealii::Point<dim>                            center(0.125, 0.125);
      const dealii::ScalarFunctionFromFunctionObject<dim> level_set_function(
        [&](const dealii::Point<dim> &p) { return p.distance_square(center) - 0.1 * 0.1; });

      dealii::DoFHandler<dim> dof_handler_full(triangulation);
      dof_handler_full.distribute_dofs(dealii::FE_DGQ<dim>(fe_degree));
      VectorType level_set_full(dof_handler_full.locally_owned_dofs(),
                                dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_full),
                                MPI_COMM_WORLD);
      dealii::VectorTools::interpolate(dealii::MappingQ<dim>(1),
                                       dof_handler_full,
                                       level_set_function,
                                       level_set_full);
      const std::vector<bool> is_cut_cell =
        NarrowBand::compute_cut_cells(dof_handler_full, level_set_full, fe_degree);

      const auto fe_collection_band = NarrowBand::create_layered_band_fe_collection<dim>(fe_degree);
      const auto fe_collection_cut  = NarrowBand::create_layered_cut_fe_collection<dim>(fe_degree);
      dealii::DoFHandler<dim> dof_handler_band(triangulation);
      dealii::DoFHandler<dim> dof_handler_cut(triangulation);

      NarrowBand::distribute_dofs_layered(is_cut_cell,
                                          n_layers,
                                          dof_handler_cut,
                                          fe_collection_cut,
                                          dof_handler_band,
                                          fe_collection_band);

      // the cut DoFHandler only holds DoFs on the cut cells
      for (const auto &cell : dof_handler_cut.active_cell_iterators())
        if (cell->is_locally_owned())
          EXPECT_EQ(cell->get_fe().n_dofs_per_cell() > 0,
                    cell->active_fe_index() == NarrowBand::ActiveFEIndex::cut);

      return {NarrowBand::n_cells_with_active_fe_index(dof_handler_band, NarrowBand::cut),
              NarrowBand::n_cells_with_active_fe_index(dof_handler_band, NarrowBand::neighbor),
              NarrowBand::n_cells_with_active_fe_index(dof_handler_band, NarrowBand::outer)};
    }

    /**
     * Geometric redistancing of the layers of the advected band: the signed distance function of
     * a circle is interpolated on the cut cells, the layers are redistanced. Returns the maximum
     * nodal error with respect to the signed distance function on the layers.
     */
    template <int dim>
    number
    compute_redistancing_error_circle(const unsigned int fe_degree,
                                      const unsigned int n_refinements,
                                      const unsigned int n_layers,
                                      const double       sign_clamp_epsilon = 0.)
    {
      dealii::Point<dim> center;
      center[0] = 0.1;
      center[1] = -0.05;
      const dealii::Functions::SignedDistance::Sphere<dim> sphere(center, 0.55);

      NarrowBandFixture<dim> fixture(fe_degree, n_refinements, fe_degree, &sphere);

      // layered band from the cut cells of the fixture
      std::vector<bool> is_cut_cell(fixture.triangulation.n_active_cells(), false);
      for (const auto &cell : fixture.dof_handler_band.active_cell_iterators())
        if (cell->is_locally_owned())
          is_cut_cell[cell->active_cell_index()] =
            cell->active_fe_index() == NarrowBand::ActiveFEIndex::cut;

      const auto fe_collection_band = NarrowBand::create_layered_band_fe_collection<dim>(fe_degree);
      const auto fe_collection_cut  = NarrowBand::create_layered_cut_fe_collection<dim>(fe_degree);
      dealii::DoFHandler<dim> dof_handler_band(fixture.triangulation);
      dealii::DoFHandler<dim> dof_handler_cut(fixture.triangulation);
      NarrowBand::distribute_dofs_layered(is_cut_cell,
                                          n_layers,
                                          dof_handler_cut,
                                          fe_collection_cut,
                                          dof_handler_band,
                                          fe_collection_band);

      VectorType level_set_band, level_set_exact;
      fixture.initialize_vector(dof_handler_band, level_set_band);
      fixture.initialize_vector(dof_handler_band, level_set_exact);
      dealii::VectorTools::interpolate(fixture.mapping, dof_handler_band, sphere, level_set_exact);

      // signed distance function on the cut cells, zero on the layers
      std::vector<bool>      is_layer_cell(fixture.triangulation.n_active_cells(), false);
      dealii::Vector<number> values(fe_collection_band[1].n_dofs_per_cell());
      for (const auto &cell : dof_handler_band.active_cell_iterators())
        if (cell->is_locally_owned() and cell->get_fe().n_dofs_per_cell() > 0)
          {
            cell->get_dof_values(level_set_exact, values);
            if (cell->active_fe_index() != NarrowBand::ActiveFEIndex::cut)
              {
                values                                   = 0.;
                is_layer_cell[cell->active_cell_index()] = true;
              }
            cell->set_dof_values(values, level_set_band);
          }

      NarrowBandData<number> data;
      data.extension.sign_clamp_epsilon = sign_clamp_epsilon;
      const dealii::ConditionalOStream             pcout(std::cout, false);
      NarrowBandClosestPointExtension<dim, number> extension(data.extension,
                                                             fixture.mapping,
                                                             pcout);
      const unsigned int                           n_redistanced_cells =
        extension.redistance(dof_handler_band, level_set_band, is_layer_cell, "layers");

      EXPECT_EQ(n_redistanced_cells,
                NarrowBand::n_cells_with_active_fe_index(dof_handler_band, NarrowBand::neighbor) +
                  NarrowBand::n_cells_with_active_fe_index(dof_handler_band, NarrowBand::outer));

      level_set_band -= level_set_exact;
      return level_set_band.linfty_norm();
    }

    /**
     * Semi-Lagrangian advection with a vanishing velocity of a level set that jumps across the
     * faces between the cut cells (planar level set plus a cell-wise constant). The foot points
     * coincide with the nodes, and the polynomial of the cell of the node is evaluated also at the
     * nodes on the faces. Returns the maximum difference between the advected and the old values
     * on the cut cells.
     */
    template <int dim>
    number
    compute_semi_lagrangian_zero_velocity_error(const unsigned int fe_degree)
    {
      dealii::Tensor<1, dim> normal;
      normal[0] = 0.8;
      normal[1] = 0.6;
      const dealii::ScalarFunctionFromFunctionObject<dim> planar(
        [&](const dealii::Point<dim> &p) { return normal * p - 0.1037; });

      NarrowBandFixture<dim> fixture(fe_degree, 3, 1, &planar);

      std::vector<bool> is_cut_cell(fixture.triangulation.n_active_cells(), false);
      for (const auto &cell : fixture.dof_handler_band.active_cell_iterators())
        if (cell->is_locally_owned())
          is_cut_cell[cell->active_cell_index()] =
            cell->active_fe_index() == NarrowBand::ActiveFEIndex::cut;

      const auto fe_collection_band = NarrowBand::create_layered_band_fe_collection<dim>(fe_degree);
      const auto fe_collection_cut  = NarrowBand::create_layered_cut_fe_collection<dim>(fe_degree);
      dealii::DoFHandler<dim> dof_handler_band(fixture.triangulation);
      dealii::DoFHandler<dim> dof_handler_cut(fixture.triangulation);
      NarrowBand::distribute_dofs_layered(
        is_cut_cell, 1, dof_handler_cut, fe_collection_cut, dof_handler_band, fe_collection_band);

      VectorType level_set_cut, level_set_band;
      fixture.initialize_vector(dof_handler_cut, level_set_cut);
      fixture.initialize_vector(dof_handler_band, level_set_band);
      dealii::VectorTools::interpolate(fixture.mapping, dof_handler_cut, planar, level_set_cut);

      // cell-wise constant: the level set jumps across the faces between the cut cells
      dealii::Vector<number> values;
      for (const auto &cell : dof_handler_cut.active_cell_iterators())
        {
          if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
            continue;
          values.reinit(cell->get_fe().n_dofs_per_cell());
          cell->get_dof_values(level_set_cut, values);
          for (auto &value : values)
            value += 0.01 * (cell->active_cell_index() % 3);
          cell->set_dof_values(values, level_set_cut);
        }

      std::vector<int> cell_signs(fixture.triangulation.n_active_cells(), 0);
      for (const auto &cell : fixture.triangulation.active_cell_iterators())
        if (not cell->is_artificial())
          cell_signs[cell->active_cell_index()] = planar.value(cell->center()) > 0. ? 1 : -1;

      dealii::Functions::ZeroFunction<dim>           velocity_function(dim);
      const dealii::ConditionalOStream               pcout(std::cout, false);
      NarrowBandSemiLagrangianAdvection<dim, number> advection(pcout);
      advection.advect(dof_handler_cut,
                       level_set_cut,
                       dof_handler_band,
                       level_set_band,
                       cell_signs,
                       velocity_function,
                       0.,
                       0.1);

      number                 error = 0.;
      dealii::Vector<number> values_band;
      for (const auto &cell : dof_handler_cut.active_cell_iterators())
        {
          if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
            continue;
          values.reinit(cell->get_fe().n_dofs_per_cell());
          values_band.reinit(values.size());
          cell->get_dof_values(level_set_cut, values);
          cell->as_dof_handler_iterator(dof_handler_band)
            ->get_dof_values(level_set_band, values_band);
          for (unsigned int i = 0; i < values.size(); ++i)
            error = std::max<number>(error, std::abs(values[i] - values_band[i]));
        }

      return dealii::Utilities::MPI::max(error, MPI_COMM_WORLD);
    }

    /**
     * Rigid rotation u = omega (-x_1, x_0) about the origin (dim components).
     */
    template <int dim>
    class RotationVelocity : public dealii::Function<dim>
    {
    public:
      explicit RotationVelocity(const number omega)
        : dealii::Function<dim>(dim)
        , omega(omega)
      {}

      number
      value(const dealii::Point<dim> &p, const unsigned int component) const override
      {
        if (component == 0)
          return -omega * p[1];
        if (component == 1)
          return omega * p[0];
        return 0.;
      }

    private:
      const number omega;
    };

    /**
     * Uniform contraction u = -a (x - c) towards a center c (dim components).
     */
    template <int dim>
    class ContractionVelocity : public dealii::Function<dim>
    {
    public:
      ContractionVelocity(const dealii::Point<dim> &center, const number rate)
        : dealii::Function<dim>(dim)
        , center(center)
        , rate(rate)
      {}

      number
      value(const dealii::Point<dim> &p, const unsigned int component) const override
      {
        return -rate * (p[component] - center[component]);
      }

    private:
      const dealii::Point<dim> center;
      const number             rate;
    };

    /**
     * Time-reversed single-vortex velocity on the unit square (2 components), u = cos(pi t / T)
     * (-2 sin^2(pi x) sin(pi y) cos(pi y), 2 sin(pi x) cos(pi x) sin^2(pi y)).
     */
    class SingleVortexVelocity : public dealii::Function<2>
    {
    public:
      explicit SingleVortexVelocity(const number period)
        : dealii::Function<2>(2)
        , period(period)
      {}

      number
      value(const dealii::Point<2> &p, const unsigned int component) const override
      {
        const number pi     = dealii::numbers::PI;
        const number factor = std::cos(pi * this->get_time() / period);
        if (component == 0)
          return -2. * factor * std::pow(std::sin(pi * p[0]), 2) * std::sin(pi * p[1]) *
                 std::cos(pi * p[1]);
        return 2. * factor * std::sin(pi * p[0]) * std::cos(pi * p[0]) *
               std::pow(std::sin(pi * p[1]), 2);
      }

    private:
      const number period;
    };

    /**
     * Errors of the foot points at t = 0 of the nodes (FE_DGQ(2) support points) of all cells of
     * the unit square (32^2 cells) at the time @p end_time in the single-vortex flow (T = 4),
     * computed with the characteristic map (composition of the backward characteristics of the
     * time steps of size @p time_step) and by the backward integration over all time steps (one
     * Runge-Kutta step per time step), with respect to the backward integration with 4000
     * Runge-Kutta steps.
     */
    struct CharacteristicMapResult
    {
      /// Maximum error of the foot points of the characteristic map.
      number error_map = 0.;

      /// Maximum error of the foot points of the backward integration.
      number error_integration = 0.;

      /// Maximum difference of the foot points of the vertices computed from different cells.
      number max_vertex_mismatch = 0.;

      /// Number of extrapolated nodes in the last time step.
      unsigned int n_nodes_extrapolated = 0;
    };

    CharacteristicMapResult
    compute_characteristic_map_errors(const number time_step, const number end_time)
    {
      constexpr int                                     dim = 2;
      dealii::parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
      dealii::GridGenerator::hyper_cube(triangulation, 0., 1.);
      triangulation.refine_global(5);

      // band: all cells
      const dealii::FE_DGQ<dim> fe(2);
      dealii::DoFHandler<dim>   dof_handler(triangulation);
      dof_handler.distribute_dofs(fe);

      SingleVortexVelocity velocity(4.);

      NarrowBandCharacteristicMap<dim, number> map(triangulation,
                                                   2,
                                                   dealii::ConditionalOStream(std::cout, false));
      map.reset(dof_handler);

      const unsigned int      n_steps = static_cast<unsigned int>(std::round(end_time / time_step));
      CharacteristicMapResult result;
      for (unsigned int n = 0; n < n_steps; ++n)
        result.n_nodes_extrapolated =
          map.advance(velocity, n * time_step, time_step).n_nodes_extrapolated;

      const auto backward = [&](const dealii::Point<dim> &x, const unsigned int n_substeps) {
        const number       h = -end_time / n_substeps;
        dealii::Point<dim> y = x;
        const auto         u = [&](const dealii::Point<dim> &p, const number t) {
          velocity.set_time(t);
          return dealii::Tensor<1, dim>({velocity.value(p, 0), velocity.value(p, 1)});
        };
        for (unsigned int s = 0; s < n_substeps; ++s)
          {
            const number t1 = end_time + s * h;
            const auto   k1 = u(y, t1);
            const auto   k2 = u(y + 0.5 * h * k1, t1 + 0.5 * h);
            const auto   k3 = u(y + 0.5 * h * k2, t1 + 0.5 * h);
            const auto   k4 = u(y + h * k3, t1 + h);
            y += (h / 6.) * (k1 + 2. * k2 + 2. * k3 + k4);
          }
        return y;
      };

      std::map<unsigned int, dealii::Point<dim>> vertex_foot_points;
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (not cell->is_locally_owned())
            continue;
          for (const auto &unit_point : fe.get_unit_support_points())
            {
              const dealii::Point<dim> x =
                NarrowBand::internal::AffineCellMap<dim>(cell).to_real(unit_point);
              const dealii::Point<dim> exact = backward(x, 4000);
              result.error_map =
                std::max(result.error_map, map.foot_point(cell, unit_point).distance(exact));
              result.error_integration =
                std::max(result.error_integration, backward(x, n_steps).distance(exact));
            }
          for (const unsigned int v : cell->vertex_indices())
            {
              dealii::Point<dim> unit_vertex;
              for (unsigned int d = 0; d < dim; ++d)
                unit_vertex[d] = (v >> d) & 1U;
              const dealii::Point<dim> foot = map.foot_point(cell, unit_vertex);
              const auto [entry, inserted] =
                vertex_foot_points.emplace(cell->vertex_index(v), foot);
              if (not inserted)
                result.max_vertex_mismatch =
                  std::max(result.max_vertex_mismatch, entry->second.distance(foot));
            }
        }
      result.error_map = dealii::Utilities::MPI::max(result.error_map, MPI_COMM_WORLD);
      result.error_integration =
        dealii::Utilities::MPI::max(result.error_integration, MPI_COMM_WORLD);
      result.max_vertex_mismatch =
        dealii::Utilities::MPI::max(result.max_vertex_mismatch, MPI_COMM_WORLD);
      return result;
    }

    /**
     * Statistics of the interface points of a circle (radius 0.3, signed distance function, p = 1,
     * 16^2 cells, narrow band of the circle with the extension band).
     */
    template <int dim>
    struct InterfacePointsResult
    {
      /// Maximum error of the positions of the surface quadratures of the cells.
      number error_points = 0.;

      /// Error of the sum of the weights of the surface quadratures of the cells.
      number error_weights = 0.;

      /// Length of the discrete interface (sum of the weights).
      number length = 0.;

      /// Maximum deviation of the points, their weights and their normal vectors from the initial
      /// ones after a full rigid rotation about the center of the domain.
      number error_rotation = 0.;

      /// Maximum deviation of the distances of the advected points to the center from 0.3 / 3
      /// after the contraction, relative deviation of the sum of their weights from one third of
      /// the initial one (contracted area elements) and maximum number of points per cell.
      number       error_contraction         = 0.;
      number       error_weights_contraction = 0.;
      unsigned int max_n_points_per_cell     = 0;

      /// Statistics of the assignment: initial, after the rotation, for the level set of a larger
      /// circle, after the contraction.
      typename NarrowBandInterfacePoints<dim, number>::Statistics initial, rotated, larger,
        contracted;
    };

    /**
     * Interface points of a circle: the points seeded on the zero level set lie in the cut cells,
     * their weights sum up to the length of the discrete interface, and the surface quadratures of
     * the cells reproduce the positions and the weights. A full rigid rotation (100 time steps)
     * returns the points and their weights to their initial values. For the level set of a larger
     * circle (radius increased by 0.4 h), the points are not supported (|phi| > 0.2 h) and are
     * removed, and new points are seeded on its zero level set. A uniform contraction of the
     * circle by a factor of three (located in the narrow band of the circle of radius 0.1)
     * contracts the weights (area elements) by the same factor and leads to merged points, while
     * the sum of the weights of the advected points is preserved by the merging.
     */
    template <int dim>
    InterfacePointsResult<dim>
    test_interface_points_circle()
    {
      InterfacePointsResult<dim> result;

      const number       radius = 0.3;
      dealii::Point<dim> center;
      center[0] = 0.4;
      const dealii::Functions::SignedDistance::Sphere<dim> sphere(center, radius);

      NarrowBandFixture<dim> fixture(1, 4, 1, &sphere);
      VectorType             level_set, level_set_band;
      fixture.initialize_vector(fixture.dof_handler_cut, level_set);
      fixture.initialize_vector(fixture.dof_handler_band, level_set_band);
      dealii::VectorTools::interpolate(fixture.mapping, fixture.dof_handler_cut, sphere, level_set);
      dealii::VectorTools::interpolate(fixture.mapping,
                                       fixture.dof_handler_band,
                                       sphere,
                                       level_set_band);

      const auto locate = [&](NarrowBandInterfacePoints<dim, number> &interface_points,
                              const VectorType                       &level_set_band) {
        const std::vector<bool> is_cut_cell =
          NarrowBand::compute_cut_cells(fixture.dof_handler_band, level_set_band, 1);
        return interface_points.locate(fixture.dof_handler_band, level_set_band, 2, is_cut_cell);
      };

      const dealii::ConditionalOStream       pcout(std::cout, false);
      NarrowBandInterfacePoints<dim, number> interface_points(pcout);
      interface_points.seed(fixture.dof_handler_cut, level_set, 2);
      result.initial = locate(interface_points, level_set_band);

      // positions and weights of the surface quadratures of the cells (Cartesian cells of size h:
      // x = x_0 + h xi, JxW = w h^{dim-1} for unit normal vectors)
      const auto &points = interface_points.get_points();
      for (const auto &cell : fixture.triangulation.active_cell_iterators())
        {
          if (not cell->is_locally_owned())
            continue;

          const auto   quadrature = interface_points.get_surface_quadrature(cell);
          const number h          = cell->vertex(1)[0] - cell->vertex(0)[0];
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            {
              const dealii::Point<dim> point =
                cell->vertex(0) + h * dealii::Tensor<1, dim>(quadrature.point(q));
              result.length += quadrature.weight(q) * std::pow(h, dim - 1);

              // closest seeded point
              number distance = std::numeric_limits<number>::max();
              for (const auto &x : points)
                distance = std::min(distance, x.distance(point));
              result.error_points = std::max(result.error_points, distance);
            }
        }
      result.length       = dealii::Utilities::MPI::sum(result.length, MPI_COMM_WORLD);
      result.error_points = dealii::Utilities::MPI::max(result.error_points, MPI_COMM_WORLD);

      number length_seeded = 0.;
      for (const auto weight : interface_points.get_weights())
        length_seeded += weight;
      result.error_weights = std::abs(result.length - length_seeded);

      // full rigid rotation about the center of the domain
      const std::vector<dealii::Point<dim>> points_initial  = points;
      const std::vector<number>             weights_initial = interface_points.get_weights();
      {
        RotationVelocity<dim> velocity(2. * dealii::numbers::PI);
        const number          time_step = 0.01;
        for (unsigned int n = 0; n < 100; ++n)
          interface_points.advect(velocity, n * time_step, time_step);
      }
      for (unsigned int k = 0; k < points.size(); ++k)
        result.error_rotation =
          std::max({result.error_rotation,
                    points[k].distance(points_initial[k]),
                    std::abs(interface_points.get_weights()[k] - weights_initial[k])});
      result.rotated = locate(interface_points, level_set_band);

      // level set of a larger circle on the band
      {
        const number h = 2. / 16.;
        VectorType   level_set_larger;
        fixture.initialize_vector(fixture.dof_handler_band, level_set_larger);
        dealii::VectorTools::interpolate(
          fixture.mapping,
          fixture.dof_handler_band,
          dealii::Functions::SignedDistance::Sphere<dim>(center, radius + 0.4 * h),
          level_set_larger);
        NarrowBandInterfacePoints<dim, number> interface_points_larger(pcout);
        interface_points_larger.seed(fixture.dof_handler_cut, level_set, 2);
        result.larger = locate(interface_points_larger, level_set_larger);
      }

      // uniform contraction by a factor of three: x(t) = c + (x_0 - c) exp(-a t), a t = ln 3
      number weight_initial = 0.;
      for (const auto weight : interface_points.get_weights())
        weight_initial += weight;
      {
        ContractionVelocity<dim> velocity(center, std::log(3.));
        const number             time_step = 0.05;
        for (unsigned int n = 0; n < 20; ++n)
          interface_points.advect(velocity, n * time_step, time_step);
      }

      // narrow band of the contracted circle
      const dealii::Functions::SignedDistance::Sphere<dim> sphere_contracted(center, radius / 3.);
      NarrowBandFixture<dim> fixture_contracted(1, 4, 1, &sphere_contracted);
      VectorType             level_set_contracted;
      fixture_contracted.initialize_vector(fixture_contracted.dof_handler_band,
                                           level_set_contracted);
      dealii::VectorTools::interpolate(fixture_contracted.mapping,
                                       fixture_contracted.dof_handler_band,
                                       sphere_contracted,
                                       level_set_contracted);
      {
        std::vector<bool> is_cut_cell =
          NarrowBand::compute_cut_cells(fixture_contracted.dof_handler_band,
                                        level_set_contracted,
                                        1);
        result.contracted = interface_points.locate(fixture_contracted.dof_handler_band,
                                                    level_set_contracted,
                                                    2,
                                                    is_cut_cell);
      }

      // the advected points precede the points seeded on the zero level set of the contracted
      // level set
      const unsigned int n_advected_points = points.size() - result.contracted.n_inserted_points;
      number             weight_contracted = 0.;
      for (unsigned int k = 0; k < n_advected_points; ++k)
        {
          weight_contracted += interface_points.get_weights()[k];
          result.error_contraction =
            std::max(result.error_contraction, std::abs(points[k].distance(center) - radius / 3.));
        }
      result.error_weights_contraction = std::abs(3. * weight_contracted / weight_initial - 1.);

      for (const auto &cell : fixture_contracted.triangulation.active_cell_iterators())
        if (cell->is_locally_owned())
          result.max_n_points_per_cell =
            std::max<unsigned int>(result.max_n_points_per_cell,
                                   interface_points.get_surface_quadrature(cell).size());
      result.max_n_points_per_cell =
        dealii::Utilities::MPI::max(result.max_n_points_per_cell, MPI_COMM_WORLD);

      return result;
    }

    /**
     * Signed distance function of a strip y_0 - w/2 < x_1 < y_0 + w/2 (positive inside).
     */
    template <int dim>
    class StripLevelSet : public dealii::Function<dim>
    {
    public:
      StripLevelSet(const number center, const number width)
        : center(center)
        , width(width)
      {}

      double
      value(const dealii::Point<dim> &p, const unsigned int /*component*/ = 0) const override
      {
        return 0.5 * width - std::abs(p[1] - center);
      }

    private:
      const number center;
      const number width;
    };

    struct SubcellDistanceResult
    {
      /// Maximum distance of the zeros of the level set on the vertical lines through the cell
      /// centers from the exact zeros, and minimum number of zeros per cell: interpolated level
      /// set and projected subcell distance (of a reinitialized level set, which has lost the
      /// strip).
      number       error_zeros_interpolated = 0.;
      number       error_zeros_projected    = 0.;
      unsigned int min_n_zeros_interpolated = std::numeric_limits<unsigned int>::max();
      unsigned int min_n_zeros_projected    = std::numeric_limits<unsigned int>::max();

      /// Maximum error of the signed distance to the points in the middle of the strip and at a
      /// distance of 0.7 h from it.
      number error_distance = 0.;

      typename NarrowBandInterfacePoints<2, number>::ProjectionStatistics statistics;
    };

    /**
     * Subcell distance of a thin strip (p = 2, 8^2 cells on [-1,1]^2, h = 0.25), by default of
     * width 0.4 h in the middle of a row of cells: the points are seeded on the exact zero level
     * set (interpolated on 64^2 cells), and both sides of the strip are unresolved. The
     * reinitialized level set of the cut cells is replaced by -0.1 (strip lost), and the
     * projection of the subcell distance restores both zeros in each cut cell. The center and the
     * width of the strip are given relative to h, and the projection optionally uses the
     * orientation filter or the pinning penalty, or is computed in the continuous level set space
     * (NarrowBandContinuousReinitialization, coupled projection on the projected cells).
     */
    SubcellDistanceResult
    test_subcell_distance_strip(const number relative_center    = 0.5,
                                const number relative_width     = 0.4,
                                const bool   orientation_filter = false,
                                const number pinning_penalty    = 0.,
                                const bool   continuous         = false)
    {
      constexpr int            dim = 2;
      const number             h = 0.25, width = relative_width * h, center = relative_center * h;
      const StripLevelSet<dim> strip(center, width);

      SubcellDistanceResult result;

      // points on the exact zero level set: the kink of the strip lies on a face of the fine mesh
      const dealii::ConditionalOStream       pcout(std::cout, false);
      NarrowBandInterfacePoints<dim, number> interface_points(pcout);
      {
        NarrowBandFixture<dim> fine(2, 6, 2, &strip);
        VectorType             level_set_fine;
        fine.initialize_vector(fine.dof_handler_cut, level_set_fine);
        dealii::VectorTools::interpolate(fine.mapping, fine.dof_handler_cut, strip, level_set_fine);
        interface_points.seed(fine.dof_handler_cut, level_set_fine, 3);
      }

      NarrowBandFixture<dim> coarse(2, 3, 2, &strip);
      VectorType             level_set, level_set_advected, level_set_band;
      coarse.initialize_vector(coarse.dof_handler_cut, level_set);
      coarse.initialize_vector(coarse.dof_handler_band, level_set_band);
      dealii::VectorTools::interpolate(coarse.mapping, coarse.dof_handler_cut, strip, level_set);
      dealii::VectorTools::interpolate(coarse.mapping,
                                       coarse.dof_handler_band,
                                       strip,
                                       level_set_band);
      const std::vector<bool> is_cut_cell =
        NarrowBand::compute_cut_cells(coarse.dof_handler_band, level_set_band, 2);
      interface_points.locate(coarse.dof_handler_band, level_set_band, 3, is_cut_cell);

      // zeros on the vertical lines through the cell centers
      const auto check_zeros = [&](const VectorType &vector, number &error, unsigned int &n_min) {
        dealii::Vector<number> values;
        for (const auto &cell : coarse.dof_handler_cut.active_cell_iterators())
          {
            if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
              continue;
            values.reinit(cell->get_fe().n_dofs_per_cell());
            cell->get_dof_values(vector, values);

            const unsigned int n_samples = 2000;
            unsigned int       n_zeros   = 0;
            number             previous  = 0.;
            for (unsigned int k = 0; k <= n_samples; ++k)
              {
                const dealii::Point<dim> unit_point(0.5, static_cast<number>(k) / n_samples);
                number                   value = 0.;
                for (unsigned int i = 0; i < values.size(); ++i)
                  value += values[i] * cell->get_fe().shape_value(i, unit_point);
                if (k > 0 and value * previous < 0.)
                  {
                    const number y = cell->vertex(0)[1] + h * (k - 0.5) / n_samples;
                    error          = std::max(error,
                                     std::min(std::abs(y - center + 0.5 * width),
                                              std::abs(y - center - 0.5 * width)));
                    ++n_zeros;
                  }
                previous = value;
              }
            n_min = std::min(n_min, n_zeros);
          }
        error = dealii::Utilities::MPI::max(error, MPI_COMM_WORLD);
        n_min = dealii::Utilities::MPI::min(n_min, MPI_COMM_WORLD);
      };

      check_zeros(level_set, result.error_zeros_interpolated, result.min_n_zeros_interpolated);

      // reinitialized level set, which has lost the strip
      level_set_advected = level_set;
      level_set          = -0.1;
      if (continuous)
        {
          const ReinitializationData<number>                reinit_data;
          NarrowBandContinuousReinitialization<dim, number> continuous_reinitialization(
            coarse.mapping,
            coarse.triangulation,
            reinit_data,
            coarse.fe_degree,
            coarse.fe_collection_cut,
            pcout);
          continuous_reinitialization.distribute_dofs(coarse.dof_handler_cut);
          continuous_reinitialization.interpolate_from_discontinuous(coarse.dof_handler_cut,
                                                                     level_set);
          result.statistics = continuous_reinitialization.project_subcell_distance(
            interface_points, coarse.dof_handler_cut, level_set_advected, 1., 4);
          continuous_reinitialization.interpolate_to_discontinuous(coarse.dof_handler_cut,
                                                                   level_set);
        }
      else
        result.statistics = interface_points.project_signed_distance(coarse.mapping,
                                                                     coarse.dof_handler_cut,
                                                                     level_set_advected,
                                                                     level_set,
                                                                     1.,
                                                                     4,
                                                                     nullptr,
                                                                     orientation_filter,
                                                                     pinning_penalty);
      check_zeros(level_set, result.error_zeros_projected, result.min_n_zeros_projected);

      result.error_distance =
        std::max(std::abs(interface_points.signed_distance(dealii::Point<dim>(0.3, center), h) -
                          0.5 * width),
                 std::abs(interface_points.signed_distance(
                            dealii::Point<dim>(0.3, center + 0.5 * width + 0.7 * h), h) +
                          0.7 * h));
      return result;
    }

    /**
     * Rotating circle (radius 0.3, center (0.4, 0), rigid rotation by a quarter turn in 25 time
     * steps, semi-Lagrangian band, p = 1, 32^2 cells on [-1,1]^2, reinitialization in every time
     * step, or sub-cycling of the level-set update) with the interface penalty term of the
     * reinitialization at the zero level set or at
     * the advected interface points (new points from the level set given by @p point_insertion),
     * in the level set space @p level_set_space (with the closed-interface correction for the
     * continuous space). Returns the mean signed radial deviation int_Gamma_h (|x - c| - R) ds /
     * |Gamma_h| of the discrete interface from the exact circle at the end.
     */
    template <int dim>
    number
    compute_rotating_circle_radial_deviation(
      const NarrowBandReinitializationInterface reinitialization_interface,
      const NarrowBandLevelSetSpace  level_set_space = NarrowBandLevelSetSpace::discontinuous,
      const NarrowBandPointInsertion point_insertion = NarrowBandPointInsertion::final_level_set,
      const number                   time_step       = 0.01,
      const number                   level_set_update_cfl = 0.,
      const NarrowBandSubcyclingCharacteristics characteristics =
        NarrowBandSubcyclingCharacteristics::integration)
    {
      const unsigned int fe_degree = 1;
      const number       radius    = 0.3;
      const number       omega     = 2. * dealii::numbers::PI;

      dealii::parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
      dealii::GridGenerator::hyper_cube(triangulation, -1., 1.);
      triangulation.refine_global(5);

      ScratchData<dim, dim, number> scratch_data(MPI_COMM_WORLD, 0, true);
      scratch_data.set_mapping(std::make_shared<dealii::MappingQ<dim>>(1));

      NarrowBandData<number> narrow_band_data;
      narrow_band_data.type                             = NarrowBandType::semi_lagrangian;
      narrow_band_data.reinitialization_interface       = reinitialization_interface;
      narrow_band_data.level_set_space                  = level_set_space;
      narrow_band_data.advected_points.point_insertion  = point_insertion;
      narrow_band_data.level_set_update_cfl             = level_set_update_cfl;
      narrow_band_data.level_set_update_characteristics = characteristics;

      ReinitializationData<number> reinit_data;
      reinit_data.enable                                      = true;
      reinit_data.modeltype                                   = ModelType::elliptic;
      reinit_data.fe.type                                     = FiniteElementType::FE_DGQ;
      reinit_data.fe.degree                                   = fe_degree;
      reinit_data.elliptic.intersected_cells_only             = true;
      reinit_data.elliptic.penalty_parameter                  = 1e4;
      reinit_data.elliptic.nlsolve.max_nonlinear_iterations   = 1000;
      reinit_data.elliptic.nlsolve.residual_tolerance         = 1e-6;
      reinit_data.elliptic.nlsolve.field_correction_tolerance = 1e-6;
      reinit_data.elliptic.nlsolve.verbosity_level            = 0;
      reinit_data.linear_solver.solver_type                   = LinearSolverType::CG;
      reinit_data.linear_solver.preconditioner_type           = PreconditionerType::ILU;
      reinit_data.linear_solver.rel_tolerance                 = 1e-10;
      reinit_data.linear_solver.abs_tolerance                 = 1e-20;
      reinit_data.linear_solver.max_iterations                = 1000;

      auto velocity = std::make_shared<RotationVelocity<dim>>(omega);

      NarrowBandLevelSetOperation<dim, number> operation(
        scratch_data, triangulation, narrow_band_data, reinit_data, fe_degree, velocity);

      dealii::Point<dim> center;
      center[0] = 0.4;
      const dealii::Functions::SignedDistance::Sphere<dim> sphere(center, radius);
      operation.set_initial_condition(sphere);

      const unsigned int n_time_steps = static_cast<unsigned int>(std::round(0.25 / time_step));
      for (unsigned int n = 0; n < n_time_steps; ++n)
        {
          velocity->set_time(n * time_step);
          operation.advance_time_step(n * time_step, time_step, n + 1);
        }

      // exact center after the rotation
      const number       angle = omega * n_time_steps * time_step;
      dealii::Point<dim> center_end;
      center_end[0] = std::cos(angle) * center[0];
      center_end[1] = std::sin(angle) * center[0];

      const auto &dof_handler = operation.get_dof_handler_cut();
      const auto &level_set   = operation.get_level_set();
      level_set.update_ghost_values();
      // Note: the generator keeps a pointer to the quadrature collection
      const dealii::hp::QCollection<1> q_collection{dealii::QGauss<1>(fe_degree + 2)};
      dealii::NonMatching::DiscreteQuadratureGenerator<dim> generator(q_collection,
                                                                      dof_handler,
                                                                      level_set);

      number deviation = 0., length = 0.;
      for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned() and cell->get_fe().n_dofs_per_cell() > 0)
          {
            generator.generate(cell);
            const auto  &quadrature = generator.get_surface_quadrature();
            const number h          = cell->vertex(1)[0] - cell->vertex(0)[0];
            for (unsigned int q = 0; q < quadrature.size(); ++q)
              {
                const dealii::Point<dim> point =
                  cell->vertex(0) + h * dealii::Tensor<1, dim>(quadrature.point(q));
                const number weight = quadrature.weight(q) * std::pow(h, dim - 1);
                deviation += weight * (point.distance(center_end) - radius);
                length += weight;
              }
          }

      return dealii::Utilities::MPI::sum(deviation, MPI_COMM_WORLD) /
             dealii::Utilities::MPI::sum(length, MPI_COMM_WORLD);
    }

    /**
     * Velocity u = -U sign(x) e_x (sign(0) = 1): both halves of the domain move rigidly towards
     * the plane x = 0 (dim components).
     */
    template <int dim>
    class ConvergingVelocity : public dealii::Function<dim>
    {
    public:
      explicit ConvergingVelocity(const number velocity)
        : dealii::Function<dim>(dim)
        , velocity(velocity)
      {}

      number
      value(const dealii::Point<dim> &p, const unsigned int component) const override
      {
        if (component != 0)
          return 0.;
        return p[0] >= 0. ? -velocity : velocity;
      }

    private:
      const number velocity;
    };

    /**
     * Velocity u = -U sign(y) e_y (sign(0) = 1): both halves of the domain move rigidly towards
     * the axis y = 0 (dim components).
     */
    template <int dim>
    class SqueezingVelocity : public dealii::Function<dim>
    {
    public:
      explicit SqueezingVelocity(const number velocity)
        : dealii::Function<dim>(dim)
        , velocity(velocity)
      {}

      number
      value(const dealii::Point<dim> &p, const unsigned int component) const override
      {
        if (component != 1)
          return 0.;
        return p[1] >= 0. ? -velocity : velocity;
      }

    private:
      const number velocity;
    };

    /**
     * Break-up of two overlapping circles (radius r = 0.24, distance D = 0.46 of the centers on
     * the axis y = 0), which are squeezed towards the axis with u = -U sign(y) e_y (U = 0.25): the
     * neck pinches off at t = sqrt(r^2 - D^2/4) / U = 0.274, and the two parts are lenses with
     * the length 2 * 2 r (pi - 2 asin(U t / r)) of their boundaries. Returns the relative error of
     * the length of the zero level set at t = 0.6 (advected points, subcell distance, 32^2 cells).
     */
    template <int dim>
    number
    compute_breakup_circles_length_error(const number merge_distance)
    {
      const unsigned int fe_degree = 2;
      const number       radius = 0.24, distance = 0.46, velocity = 0.25, time_step = 0.01;
      const unsigned int n_time_steps = 60;

      dealii::parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
      dealii::GridGenerator::hyper_cube(triangulation, -1., 1.);
      triangulation.refine_global(5);

      ScratchData<dim, dim, number> scratch_data(MPI_COMM_WORLD, 0, true);
      scratch_data.set_mapping(std::make_shared<dealii::MappingQ<dim>>(1));

      NarrowBandData<number> narrow_band_data;
      narrow_band_data.type                   = NarrowBandType::semi_lagrangian;
      narrow_band_data.level_set_space        = NarrowBandLevelSetSpace::continuous;
      narrow_band_data.reinitialization_faces = NarrowBandReinitializationFaces::interface_faces;
      narrow_band_data.reinitialization_interface =
        NarrowBandReinitializationInterface::advected_points;
      narrow_band_data.advected_points.unresolved_cells =
        NarrowBandUnresolvedCells::subcell_distance;
      narrow_band_data.advected_points.subcell_coupling     = NarrowBandSubcellCoupling::prescribed;
      narrow_band_data.advected_points.points_per_direction = 3;
      narrow_band_data.advected_points.merge_distance       = merge_distance;

      ReinitializationData<number> reinit_data;
      reinit_data.enable                                      = true;
      reinit_data.modeltype                                   = ModelType::elliptic;
      reinit_data.fe.type                                     = FiniteElementType::FE_DGQ;
      reinit_data.fe.degree                                   = fe_degree;
      reinit_data.elliptic.intersected_cells_only             = true;
      reinit_data.elliptic.penalty_parameter                  = 1e4;
      reinit_data.elliptic.nlsolve.max_nonlinear_iterations   = 1000;
      reinit_data.elliptic.nlsolve.residual_tolerance         = 1e-6;
      reinit_data.elliptic.nlsolve.field_correction_tolerance = 1e-6;
      reinit_data.elliptic.nlsolve.verbosity_level            = 0;
      reinit_data.linear_solver.solver_type                   = LinearSolverType::CG;
      reinit_data.linear_solver.preconditioner_type           = PreconditionerType::ILU;
      reinit_data.linear_solver.rel_tolerance                 = 1e-10;
      reinit_data.linear_solver.abs_tolerance                 = 1e-20;
      reinit_data.linear_solver.max_iterations                = 1000;

      auto velocity_function = std::make_shared<SqueezingVelocity<dim>>(velocity);

      NarrowBandLevelSetOperation<dim, number> operation(
        scratch_data, triangulation, narrow_band_data, reinit_data, fe_degree, velocity_function);

      // signed distance of the union of the two overlapping disks (outside and away from the neck)
      dealii::Point<dim> center_minus, center_plus;
      center_minus[0] = 0.013 - 0.5 * distance;
      center_plus[0]  = 0.013 + 0.5 * distance;
      const dealii::ScalarFunctionFromFunctionObject<dim> initial_condition(
        [&](const dealii::Point<dim> &p) {
          return radius - std::min(p.distance(center_minus), p.distance(center_plus));
        });
      operation.set_initial_condition(initial_condition);

      for (unsigned int n = 0; n < n_time_steps; ++n)
        {
          velocity_function->set_time(n * time_step);
          operation.advance_time_step(n * time_step, time_step, n + 1);
        }

      const auto &dof_handler = operation.get_dof_handler_cut();
      const auto &level_set   = operation.get_level_set();
      level_set.update_ghost_values();
      // Note: the generator keeps a pointer to the quadrature collection
      const dealii::hp::QCollection<1> q_collection{dealii::QGauss<1>(fe_degree + 2)};
      dealii::NonMatching::DiscreteQuadratureGenerator<dim> generator(q_collection,
                                                                      dof_handler,
                                                                      level_set);

      number length = 0.;
      for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned() and cell->get_fe().n_dofs_per_cell() > 0)
          {
            generator.generate(cell);
            const auto  &quadrature = generator.get_surface_quadrature();
            const number h          = cell->vertex(1)[0] - cell->vertex(0)[0];
            for (unsigned int q = 0; q < quadrature.size(); ++q)
              length += quadrature.weight(q) * std::pow(h, dim - 1);
          }
      length = dealii::Utilities::MPI::sum(length, MPI_COMM_WORLD);

      const number squeeze = velocity * n_time_steps * time_step;
      const number exact_length =
        4. * radius * (dealii::numbers::PI - 2. * std::asin(squeeze / radius));
      return (length - exact_length) / exact_length;
    }

    /**
     * Restart: rotating circle; the level set is advanced by @p n_steps_before time steps, the
     * state is saved (archive and triangulation files), and it is advanced by @p n_steps_after
     * further time steps. Then, the state is loaded into a new triangulation and a new operation
     * and advanced by @p n_steps_after time steps. Returns the maximum difference of the DoF
     * values of the level set on the cut cells and on the band between both runs (-1, if the cut
     * cells or the band differ).
     */
    template <int dim>
    number
    compute_restart_difference(const NarrowBandData<number> &narrow_band_data,
                               const number                  time_step,
                               const unsigned int            n_steps_before,
                               const unsigned int            n_steps_after,
                               const std::string            &basename)
    {
      const unsigned int fe_degree = 1;

      ReinitializationData<number> reinit_data;
      reinit_data.enable                                      = true;
      reinit_data.modeltype                                   = ModelType::elliptic;
      reinit_data.fe.type                                     = FiniteElementType::FE_DGQ;
      reinit_data.fe.degree                                   = fe_degree;
      reinit_data.elliptic.intersected_cells_only             = true;
      reinit_data.elliptic.penalty_parameter                  = 1e4;
      reinit_data.elliptic.nlsolve.max_nonlinear_iterations   = 1000;
      reinit_data.elliptic.nlsolve.residual_tolerance         = 1e-6;
      reinit_data.elliptic.nlsolve.field_correction_tolerance = 1e-6;
      reinit_data.elliptic.nlsolve.verbosity_level            = 0;
      reinit_data.linear_solver.solver_type                   = LinearSolverType::CG;
      reinit_data.linear_solver.preconditioner_type           = PreconditionerType::ILU;
      reinit_data.linear_solver.rel_tolerance                 = 1e-10;
      reinit_data.linear_solver.abs_tolerance                 = 1e-20;
      reinit_data.linear_solver.max_iterations                = 1000;

      auto velocity = std::make_shared<RotationVelocity<dim>>(2. * dealii::numbers::PI);

      dealii::Point<dim> center;
      center[0] = 0.4;
      const dealii::Functions::SignedDistance::Sphere<dim> sphere(center, 0.3);

      const auto advance = [&](NarrowBandLevelSetOperation<dim, number> &operation,
                               const unsigned int                        first,
                               const unsigned int                        n_steps) {
        for (unsigned int n = first; n < first + n_steps; ++n)
          {
            velocity->set_time(n * time_step);
            operation.advance_time_step(n * time_step, time_step, n + 1);
          }
      };

      // DoF values of the level set on the cut cells and on the band of the locally owned cells
      using CellValues =
        std::map<dealii::CellId, std::pair<std::vector<number>, std::vector<number>>>;
      const auto collect = [](const NarrowBandLevelSetOperation<dim, number> &operation) {
        CellValues             cell_values;
        dealii::Vector<number> values;
        const auto             get_values =
          [&](const auto &cell, const VectorType &vector, std::vector<number> &result) {
            values.reinit(cell->get_fe().n_dofs_per_cell());
            if (values.size() > 0)
              cell->get_dof_values(vector, values);
            result.assign(values.begin(), values.end());
          };
        for (const auto &cell : operation.get_dof_handler_cut().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              auto &[cut_values, band_values] = cell_values[cell->id()];
              get_values(cell, operation.get_level_set(), cut_values);
              get_values(cell->as_dof_handler_iterator(operation.get_dof_handler_band()),
                         operation.get_level_set_band(),
                         band_values);
            }
        return cell_values;
      };

      std::stringstream archive_stream;
      CellValues        values_continued, values_restarted;
      {
        dealii::parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
        dealii::GridGenerator::hyper_cube(triangulation, -1., 1.);
        triangulation.refine_global(5);

        ScratchData<dim, dim, number> scratch_data(MPI_COMM_WORLD, 0, true);
        scratch_data.set_mapping(std::make_shared<dealii::MappingQ<dim>>(1));
        NarrowBandLevelSetOperation<dim, number> operation(
          scratch_data, triangulation, narrow_band_data, reinit_data, fe_degree, velocity);

        operation.set_initial_condition(sphere);
        advance(operation, 0, n_steps_before);

        {
          boost::archive::binary_oarchive archive(archive_stream);
          archive << static_cast<const NarrowBandLevelSetOperation<dim, number> &>(operation);
        }
        operation.prepare_for_serialization();
        triangulation.save(basename);

        advance(operation, n_steps_before, n_steps_after);
        values_continued = collect(operation);
      }
      {
        dealii::parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
        dealii::GridGenerator::hyper_cube(triangulation, -1., 1.);
        triangulation.load(basename);

        ScratchData<dim, dim, number> scratch_data(MPI_COMM_WORLD, 0, true);
        scratch_data.set_mapping(std::make_shared<dealii::MappingQ<dim>>(1));
        NarrowBandLevelSetOperation<dim, number> operation(
          scratch_data, triangulation, narrow_band_data, reinit_data, fe_degree, velocity);

        {
          boost::archive::binary_iarchive archive(archive_stream);
          archive >> operation;
        }
        operation.deserialize();

        advance(operation, n_steps_before, n_steps_after);
        values_restarted = collect(operation);
      }

      MPI_Barrier(MPI_COMM_WORLD);
      if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        for (const std::string suffix : {"", ".info", "_fixed.data", "_variable.data"})
          std::filesystem::remove(basename + suffix);

      bool   same_cells     = values_continued.size() == values_restarted.size();
      number max_difference = 0.;
      for (const auto &[id, values] : values_continued)
        {
          const auto it = values_restarted.find(id);
          if (it == values_restarted.end() or it->second.first.size() != values.first.size() or
              it->second.second.size() != values.second.size())
            {
              same_cells = false;
              continue;
            }
          for (unsigned int i = 0; i < values.first.size(); ++i)
            max_difference =
              std::max(max_difference, std::abs(values.first[i] - it->second.first[i]));
          for (unsigned int i = 0; i < values.second.size(); ++i)
            max_difference =
              std::max(max_difference, std::abs(values.second[i] - it->second.second[i]));
        }

      if (dealii::Utilities::MPI::min(same_cells ? 1 : 0, MPI_COMM_WORLD) == 0)
        return -1.;
      return dealii::Utilities::MPI::max(max_difference, MPI_COMM_WORLD);
    }

    /**
     * Two circles of the radius r = 0.25 with the initial distance D_0 = 0.6 of their centers
     * (0 -+ D_0 / 2, 0.013) move towards each other with the velocity U = 0.25 (ConvergingVelocity)
     * and merge at t = 0.2 (continuous level set space with the closed-interface correction, p = 2,
     * advected points with the subcell distance, 32^2 cells, U dt = 0.04 h). Returns the relative
     * error of the length of the zero level set at t = 0.4 (exact: the boundary of the union of the
     * two disks with D = 0.4) for the merge distance @p merge_distance.
     */
    template <int dim>
    number
    compute_merging_circles_length_error(const number merge_distance)
    {
      const unsigned int fe_degree = 2;
      const number       radius = 0.25, distance = 0.6, velocity = 0.25, time_step = 0.01;
      const unsigned int n_time_steps = 40;

      dealii::parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
      dealii::GridGenerator::hyper_cube(triangulation, -1., 1.);
      triangulation.refine_global(5);

      ScratchData<dim, dim, number> scratch_data(MPI_COMM_WORLD, 0, true);
      scratch_data.set_mapping(std::make_shared<dealii::MappingQ<dim>>(1));

      NarrowBandData<number> narrow_band_data;
      narrow_band_data.type                   = NarrowBandType::semi_lagrangian;
      narrow_band_data.level_set_space        = NarrowBandLevelSetSpace::continuous;
      narrow_band_data.reinitialization_faces = NarrowBandReinitializationFaces::interface_faces;
      narrow_band_data.reinitialization_interface =
        NarrowBandReinitializationInterface::advected_points;
      narrow_band_data.advected_points.unresolved_cells =
        NarrowBandUnresolvedCells::subcell_distance;
      narrow_band_data.advected_points.subcell_coupling     = NarrowBandSubcellCoupling::prescribed;
      narrow_band_data.advected_points.points_per_direction = 3;
      narrow_band_data.advected_points.merge_distance       = merge_distance;

      ReinitializationData<number> reinit_data;
      reinit_data.enable                                      = true;
      reinit_data.modeltype                                   = ModelType::elliptic;
      reinit_data.fe.type                                     = FiniteElementType::FE_DGQ;
      reinit_data.fe.degree                                   = fe_degree;
      reinit_data.elliptic.intersected_cells_only             = true;
      reinit_data.elliptic.penalty_parameter                  = 1e4;
      reinit_data.elliptic.nlsolve.max_nonlinear_iterations   = 1000;
      reinit_data.elliptic.nlsolve.residual_tolerance         = 1e-6;
      reinit_data.elliptic.nlsolve.field_correction_tolerance = 1e-6;
      reinit_data.elliptic.nlsolve.verbosity_level            = 0;
      reinit_data.linear_solver.solver_type                   = LinearSolverType::CG;
      reinit_data.linear_solver.preconditioner_type           = PreconditionerType::ILU;
      reinit_data.linear_solver.rel_tolerance                 = 1e-10;
      reinit_data.linear_solver.abs_tolerance                 = 1e-20;
      reinit_data.linear_solver.max_iterations                = 1000;

      auto velocity_function = std::make_shared<ConvergingVelocity<dim>>(velocity);

      NarrowBandLevelSetOperation<dim, number> operation(
        scratch_data, triangulation, narrow_band_data, reinit_data, fe_degree, velocity_function);

      // signed distance of the two disjoint disks
      dealii::Point<dim> center_minus, center_plus;
      center_minus[0] = -0.5 * distance;
      center_plus[0]  = 0.5 * distance;
      center_minus[1] = center_plus[1] = 0.013;
      const dealii::ScalarFunctionFromFunctionObject<dim> initial_condition(
        [&](const dealii::Point<dim> &p) {
          return radius - std::min(p.distance(center_minus), p.distance(center_plus));
        });
      operation.set_initial_condition(initial_condition);

      for (unsigned int n = 0; n < n_time_steps; ++n)
        {
          velocity_function->set_time(n * time_step);
          operation.advance_time_step(n * time_step, time_step, n + 1);
        }

      const auto &dof_handler = operation.get_dof_handler_cut();
      const auto &level_set   = operation.get_level_set();
      level_set.update_ghost_values();
      // Note: the generator keeps a pointer to the quadrature collection
      const dealii::hp::QCollection<1> q_collection{dealii::QGauss<1>(fe_degree + 2)};
      dealii::NonMatching::DiscreteQuadratureGenerator<dim> generator(q_collection,
                                                                      dof_handler,
                                                                      level_set);

      number length = 0.;
      for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned() and cell->get_fe().n_dofs_per_cell() > 0)
          {
            generator.generate(cell);
            const auto  &quadrature = generator.get_surface_quadrature();
            const number h          = cell->vertex(1)[0] - cell->vertex(0)[0];
            for (unsigned int q = 0; q < quadrature.size(); ++q)
              length += quadrature.weight(q) * std::pow(h, dim - 1);
          }
      length = dealii::Utilities::MPI::sum(length, MPI_COMM_WORLD);

      const number D = distance - 2. * velocity * n_time_steps * time_step;
      const number exact_length =
        4. * dealii::numbers::PI * radius - 4. * radius * std::acos(0.5 * D / radius);
      return (length - exact_length) / exact_length;
    }

    /**
     * Semi-Lagrangian advection of a planar signed distance function phi = n.x - c (|n| = 1) with
     * a constant velocity u: the back-traced level set phi(x - u dt) is reproduced exactly on all
     * candidate cells (cut cells and their vertex neighbors), since the polynomials of the cut
     * cells are exact and the distance to the planar interface with the sign of the cell is exact
     * for the foot points outside of the cut cells. If the signs of all cells are flipped, the
     * values with foot points outside of the cut cells change their sign, while the values with
     * foot points in the cut cells remain unchanged. The values are checked on the candidate cells
     * in the interior of the domain (the closest points of the nodes near the domain boundary may
     * be located outside of the cut cells). Returns the numbers of nodes with unchanged and with
     * flipped values for the flipped signs (summed over all processes).
     */
    template <int dim>
    std::pair<unsigned int, unsigned int>
    test_semi_lagrangian_planar(const unsigned int fe_degree)
    {
      dealii::Tensor<1, dim> normal;
      normal[0] = 0.8;
      normal[1] = 0.6;
      const dealii::ScalarFunctionFromFunctionObject<dim> planar(
        [&](const dealii::Point<dim> &p) { return normal * p - 0.1037; });

      NarrowBandFixture<dim> fixture(fe_degree, 3, 1, &planar);

      std::vector<bool> is_cut_cell(fixture.triangulation.n_active_cells(), false);
      for (const auto &cell : fixture.dof_handler_band.active_cell_iterators())
        if (cell->is_locally_owned())
          is_cut_cell[cell->active_cell_index()] =
            cell->active_fe_index() == NarrowBand::ActiveFEIndex::cut;

      // band of the candidate cells: cut cells and one layer of vertex neighbors
      const auto fe_collection_band = NarrowBand::create_layered_band_fe_collection<dim>(fe_degree);
      const auto fe_collection_cut  = NarrowBand::create_layered_cut_fe_collection<dim>(fe_degree);
      dealii::DoFHandler<dim> dof_handler_band(fixture.triangulation);
      dealii::DoFHandler<dim> dof_handler_cut(fixture.triangulation);
      NarrowBand::distribute_dofs_layered(
        is_cut_cell, 1, dof_handler_cut, fe_collection_cut, dof_handler_band, fe_collection_band);

      VectorType level_set_cut, level_set_band, level_set_exact;
      fixture.initialize_vector(dof_handler_cut, level_set_cut);
      fixture.initialize_vector(dof_handler_band, level_set_band);
      fixture.initialize_vector(dof_handler_band, level_set_exact);
      dealii::VectorTools::interpolate(fixture.mapping, dof_handler_cut, planar, level_set_cut);

      // constant velocity (displacement of 0.4 h within the time step)
      std::vector<double> velocity(dim, 0.);
      velocity[0] = 0.5;
      velocity[1] = -0.3;
      dealii::Functions::ConstantFunction<dim> velocity_function(velocity);
      const double                             time_step =
        0.1 / std::sqrt(velocity[0] * velocity[0] + velocity[1] * velocity[1]);

      dealii::Tensor<1, dim> displacement;
      for (unsigned int d = 0; d < dim; ++d)
        displacement[d] = time_step * velocity[d];
      const dealii::ScalarFunctionFromFunctionObject<dim> exact(
        [&](const dealii::Point<dim> &p) { return planar.value(p - displacement); });
      dealii::VectorTools::interpolate(fixture.mapping, dof_handler_band, exact, level_set_exact);

      // signs of the locally owned and ghost cells
      std::vector<int> cell_signs(fixture.triangulation.n_active_cells(), 0);
      for (const auto &cell : fixture.triangulation.active_cell_iterators())
        if (not cell->is_artificial())
          cell_signs[cell->active_cell_index()] = planar.value(cell->center()) > 0. ? 1 : -1;

      const dealii::ConditionalOStream               pcout(std::cout, false);
      NarrowBandSemiLagrangianAdvection<dim, number> advection(pcout);

      // (a) exact reproduction
      const auto statistics = advection.advect(dof_handler_cut,
                                               level_set_cut,
                                               dof_handler_band,
                                               level_set_band,
                                               cell_signs,
                                               velocity_function,
                                               0.,
                                               time_step);
      EXPECT_GT(statistics.n_nodes_outside, 0U);
      EXPECT_LT(statistics.n_nodes_outside, statistics.n_nodes);
      EXPECT_NEAR(statistics.max_displacement, 0.4, 1e-10);

      // nodal values of the locally owned candidate cells in the interior of the domain
      const auto for_each_interior_value = [&](const auto &function) {
        dealii::Vector<number> values(fe_collection_band[1].n_dofs_per_cell());
        dealii::Vector<number> values_exact(values.size());
        for (const auto &cell : dof_handler_band.active_cell_iterators())
          {
            if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
              continue;

            bool is_interior = true;
            for (unsigned int d = 0; d < dim; ++d)
              is_interior = is_interior and std::abs(cell->center()[d]) < 0.5;
            if (not is_interior)
              continue;

            cell->get_dof_values(level_set_band, values);
            cell->get_dof_values(level_set_exact, values_exact);
            for (unsigned int i = 0; i < values.size(); ++i)
              function(values[i], values_exact[i]);
          }
      };

      unsigned int n_interior_values = 0;
      for_each_interior_value([&](const double value, const double value_exact) {
        EXPECT_NEAR(value, value_exact, 1e-10);
        ++n_interior_values;
      });
      EXPECT_GT(dealii::Utilities::MPI::sum(n_interior_values, MPI_COMM_WORLD), 0U);

      // (b) flipped signs of all cells
      for (auto &sign : cell_signs)
        sign = -sign;
      advection.advect(dof_handler_cut,
                       level_set_cut,
                       dof_handler_band,
                       level_set_band,
                       cell_signs,
                       velocity_function,
                       0.,
                       time_step);

      unsigned int n_unchanged = 0, n_flipped = 0;
      for_each_interior_value([&](const double value, const double value_exact) {
        if (std::abs(value - value_exact) < 1e-10)
          ++n_unchanged;
        else if (std::abs(value + value_exact) < 1e-10)
          ++n_flipped;
        else
          ADD_FAILURE() << "unexpected value " << value << " (exact " << value_exact << ")";
      });

      return {dealii::Utilities::MPI::sum(n_unchanged, MPI_COMM_WORLD),
              dealii::Utilities::MPI::sum(n_flipped, MPI_COMM_WORLD)};
    }

    /**
     * Sign clamp of the extended level set: as in test_sign_limiter(), the curved level set
     * phi = l + c l^2 is prescribed on the cut cells of the zero iso-surface of a linear function.
     * The extension (without sign limiter) is computed with and without sign clamp. With the sign
     * clamp, the nodal values of each neighbor cell have one sign; all modified values equal
     * +-epsilon h, and only nodal values of the opposite sign of the cell side are modified.
     * Returns the number of nodal sign changes without sign clamp (summed over all processes).
     */
    template <int dim>
    unsigned int
    test_sign_clamp(const NarrowBandExtensionType type)
    {
      constexpr unsigned int fe_degree     = 2;
      constexpr unsigned int n_refinements = 3;
      constexpr double       epsilon       = 0.01;
      NarrowBandFixture<dim> fixture(fe_degree, n_refinements, 1 /*linear level set*/);

      const double h = 2. / dealii::Utilities::pow(2, n_refinements);
      const double c = 3. / h;

      const dealii::ScalarFunctionFromFunctionObject<dim> curved_level_set(
        [&](const dealii::Point<dim> &p) {
          const double l = fixture.level_set_function.value(p, 0);
          return l + c * l * l;
        });

      const dealii::ConditionalOStream pcout(std::cout, false);

      const auto extend = [&](const double sign_clamp_epsilon) {
        VectorType level_set_band = create_band_vector_from_cut_cells(fixture, curved_level_set);

        NarrowBandData<number> data;
        data.extension.type                         = type;
        data.extension.sign_limiter_tolerance       = -1.;
        data.extension.sign_clamp_epsilon           = sign_clamp_epsilon;
        data.extension.linear_solver.rel_tolerance  = 1e-14;
        data.extension.linear_solver.abs_tolerance  = 1e-16;
        data.extension.linear_solver.max_iterations = 10000;

        if (type == NarrowBandExtensionType::ghost_penalty)
          NarrowBandGhostPenaltyExtension<dim, number>(data.extension, fixture.mapping, pcout)
            .extend(fixture.dof_handler_band, level_set_band);
        else
          NarrowBandClosestPointExtension<dim, number>(data.extension, fixture.mapping, pcout)
            .extend(fixture.dof_handler_band, level_set_band);
        return level_set_band;
      };

      const VectorType level_set_unclamped = extend(0.);
      const VectorType level_set_clamped   = extend(epsilon);

      const auto            &fe = fixture.fe_collection_band[NarrowBand::ActiveFEIndex::neighbor];
      dealii::Vector<number> values_unclamped(fe.n_dofs_per_cell());
      dealii::Vector<number> values_clamped(fe.n_dofs_per_cell());

      const auto has_sign_change = [](const dealii::Vector<number> &values) {
        return *std::min_element(values.begin(), values.end()) <= 0. and
               *std::max_element(values.begin(), values.end()) >= 0.;
      };

      unsigned int n_sign_changes_unclamped = 0, n_modified_cells = 0;
      for (const auto &cell : fixture.dof_handler_band.active_cell_iterators())
        {
          if (not cell->is_locally_owned() or
              cell->active_fe_index() != NarrowBand::ActiveFEIndex::neighbor)
            continue;

          cell->get_dof_values(level_set_unclamped, values_unclamped);
          cell->get_dof_values(level_set_clamped, values_clamped);

          n_sign_changes_unclamped += has_sign_change(values_unclamped);
          EXPECT_FALSE(has_sign_change(values_clamped));

          // side of the cell: sign of the clamped values
          const double side          = values_clamped[0] > 0. ? 1. : -1.;
          const double clamped_value = side * epsilon * cell->minimum_vertex_distance();

          bool is_modified = false;
          for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
            if (values_clamped[i] != values_unclamped[i])
              {
                EXPECT_LE(side * values_unclamped[i], 0.);
                EXPECT_NEAR(values_clamped[i], clamped_value, 1e-14);
                is_modified = true;
              }
          n_modified_cells += is_modified;
        }

      n_sign_changes_unclamped =
        dealii::Utilities::MPI::sum(n_sign_changes_unclamped, MPI_COMM_WORLD);
      n_modified_cells = dealii::Utilities::MPI::sum(n_modified_cells, MPI_COMM_WORLD);

      // cells with sign changes are modified
      EXPECT_GE(n_modified_cells, n_sign_changes_unclamped);

      return n_sign_changes_unclamped;
    }

    /**
     * Bound-preserving limiter: the linear level set phi^n = x_0 on the band defines the bounds.
     * On each band cell, whose left and right neighbors belong to the band, the nodal value at the
     * lower right vertex is increased by h/2 (overshoot). With the bounds from all face neighbors,
     * the overshoot is within the range of the right neighbor, i.e., no cell is limited. With the
     * upwind bounds for the velocity e_x, the bounds of a cell are given by the cell and its left
     * neighbor, i.e., all perturbed cells are limited: afterwards, their values at the sample
     * points satisfy the bounds and the cell means are preserved. Returns the number of limited
     * cells and the number of perturbed cells (summed over all processes).
     */
    template <int dim>
    std::pair<unsigned int, unsigned int>
    test_bound_preserving_limiter(
      const NarrowBandLimiterBounds bounds_type,
      const NarrowBandLimiterType   limiter_type = NarrowBandLimiterType::maximum_principle)
    {
      constexpr unsigned int fe_degree = 2;
      NarrowBandFixture<dim> fixture(fe_degree, 3);

      const auto &dof_handler = fixture.dof_handler_band;
      const auto &fe          = fixture.fe_collection_band[NarrowBand::ActiveFEIndex::cut];

      const bool   is_sign_limiter = limiter_type == NarrowBandLimiterType::sign;
      const number offset          = is_sign_limiter ? 3. : 0.;

      // linear level set, which is positive on the whole domain for the sign limiter
      const dealii::ScalarFunctionFromFunctionObject<dim> linear_function(
        [&](const dealii::Point<dim> &p) { return p[0] + offset; });

      VectorType level_set;
      fixture.initialize_vector(dof_handler, level_set);
      dealii::VectorTools::interpolate(fixture.mapping, dof_handler, linear_function, level_set);

      typename NarrowBandData<number>::AdvectionLimiter data;
      data.cells  = NarrowBandLimiterCells::all_band_cells;
      data.type   = limiter_type;
      data.bounds = bounds_type;

      std::vector<double> velocity(dim, 0.);
      velocity[0] = 1.;
      dealii::Functions::ConstantFunction<dim> velocity_function(velocity);

      // time step: strips of a quarter of the cell size
      const number h         = 2. / dealii::Utilities::pow(2, 3);
      const number time_step = 0.25 * h;

      NarrowBandBoundPreservingLimiter<dim, number> limiter(data, fixture.mapping);
      limiter.compute_bounds(dof_handler, level_set, velocity_function, 0., time_step);

      // nodal index of the lower right vertex (lexicographic numbering)
      const unsigned int lower_right_node = fe_degree;

      const auto is_band_cell = [](const auto &cell) {
        return cell->get_fe().n_dofs_per_cell() > 0;
      };

      // perturbation of the cells with an upwind neighbor in the band (face 0: -x, face 1: +x):
      // overshoot for the maximum principle, negative value for the sign limiter
      std::vector<bool>      is_perturbed(fixture.triangulation.n_active_cells(), false);
      dealii::Vector<number> values(fe.n_dofs_per_cell());
      unsigned int           n_perturbed_cells = 0;
      for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned() and is_band_cell(cell) and not cell->at_boundary(0) and
            not cell->at_boundary(1) and is_band_cell(cell->neighbor(0)) and
            is_band_cell(cell->neighbor(1)))
          {
            cell->get_dof_values(level_set, values);
            if (is_sign_limiter)
              values[lower_right_node] = -0.5 * h;
            else
              values[lower_right_node] += 0.5 * h;
            cell->set_dof_values(values, level_set);
            is_perturbed[cell->active_cell_index()] = true;
            ++n_perturbed_cells;
          }

      // cell means before the limiter
      dealii::FEValues<dim> fe_values(fixture.mapping,
                                      fe,
                                      dealii::QGauss<dim>(fe_degree + 1),
                                      dealii::update_values | dealii::update_JxW_values |
                                        dealii::update_quadrature_points);
      const auto            cell_mean = [&](const auto &cell, const VectorType &vector) {
        fe_values.reinit(cell);
        std::vector<number> quadrature_values(fe_values.n_quadrature_points);
        fe_values.get_function_values(vector, quadrature_values);
        number mean = 0., volume = 0.;
        for (const unsigned int q : fe_values.quadrature_point_indices())
          {
            mean += quadrature_values[q] * fe_values.JxW(q);
            volume += fe_values.JxW(q);
          }
        return mean / volume;
      };

      VectorType level_set_unlimited = level_set;
      level_set_unlimited.update_ghost_values();

      const auto statistics = limiter.limit(dof_handler, level_set);
      EXPECT_EQ(statistics.n_mean_violations, 0U);
      level_set.update_ghost_values();

      // values at the sample points after the limiter
      const dealii::QGaussLobatto<dim> sample_points(fe_degree + 2);
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (not cell->is_locally_owned() or not is_perturbed[cell->active_cell_index()])
            continue;

          EXPECT_NEAR(cell_mean(cell, level_set), cell_mean(cell, level_set_unlimited), 1e-12);

          // bounds: range of the cell and its left neighbor (or the strip of the left neighbor)
          const number x_left  = cell->center()[0] - 0.5 * h;
          const number x_right = cell->center()[0] + 0.5 * h;

          number lower_bound = -std::numeric_limits<number>::max();
          number upper_bound = std::numeric_limits<number>::max();
          if (is_sign_limiter)
            lower_bound = data.sign_epsilon * h;
          else if (bounds_type == NarrowBandLimiterBounds::upwind_face_neighbors)
            {
              lower_bound = x_left - h;
              upper_bound = x_right;
            }
          else if (bounds_type == NarrowBandLimiterBounds::upwind_strips)
            {
              lower_bound = x_left - time_step;
              upper_bound = x_right;
            }

          cell->get_dof_values(level_set, values);
          for (const auto &point : sample_points.get_points())
            {
              number value = 0.;
              for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
                value += values[i] * fe.shape_value(i, point);
              EXPECT_LE(value, upper_bound + 1e-12);
              EXPECT_GE(value, lower_bound - 1e-12);
            }

          // the limited bound is attained
          if (statistics.n_limited_cells > 0 and
              bounds_type != NarrowBandLimiterBounds::face_neighbors)
            {
              number min_value = std::numeric_limits<number>::max();
              number max_value = -std::numeric_limits<number>::max();
              for (const auto &point : sample_points.get_points())
                {
                  number value = 0.;
                  for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
                    value += values[i] * fe.shape_value(i, point);
                  min_value = std::min(min_value, value);
                  max_value = std::max(max_value, value);
                }
              if (is_sign_limiter)
                EXPECT_NEAR(min_value, lower_bound, 1e-12);
              else
                EXPECT_NEAR(max_value, upper_bound, 1e-12);
            }
        }

      return {statistics.n_limited_cells,
              dealii::Utilities::MPI::sum(n_perturbed_cells, MPI_COMM_WORLD)};
    }

    /**
     * Continuous level set, normal vector and curvature of the semi-Lagrangian narrow band on
     * [-1,1]^dim for the level-set function @p level_set_function, which is interpolated onto the
     * cut cells (without reinitialization). Returns the statistics of the continuous level set and
     * the maximum nodal errors of the continuous level set (with respect to
     * @p level_set_function), of the normal vector (with respect to @p exact_normal) and of the
     * curvature (with respect to @p exact_curvature, i.e., the curvature of the iso-surface
     * through the node) at the locally owned DoFs of the continuous fields.
     */
    template <int dim>
    std::pair<typename NarrowBandContinuousLevelSet<dim, number>::Statistics, std::array<number, 3>>
    compute_continuous_level_set(
      const dealii::Function<dim>                                             &level_set_function,
      const std::function<dealii::Tensor<1, dim>(const dealii::Point<dim> &)> &exact_normal,
      const std::function<number(const dealii::Point<dim> &)>                 &exact_curvature,
      const unsigned int                                                       fe_degree,
      const unsigned int                                                       n_refinements,
      const bool   do_reinitialize   = false,
      const number interface_penalty = 100.)
    {
      dealii::parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
      dealii::GridGenerator::hyper_cube(triangulation, -1., 1.);
      triangulation.refine_global(n_refinements);

      ScratchData<dim, dim, number> scratch_data(MPI_COMM_WORLD, 0, true);
      scratch_data.set_mapping(std::make_shared<dealii::MappingQ<dim>>(1));

      NarrowBandData<number> narrow_band_data;
      narrow_band_data.type                                   = NarrowBandType::semi_lagrangian;
      narrow_band_data.continuous_level_set.enable            = true;
      narrow_band_data.continuous_level_set.interface_penalty = interface_penalty;
      narrow_band_data.continuous_level_set.linear_solver.rel_tolerance = 1e-13;

      ReinitializationData<number> reinit_data;
      reinit_data.modeltype                                   = ModelType::elliptic;
      reinit_data.fe.type                                     = FiniteElementType::FE_DGQ;
      reinit_data.fe.degree                                   = fe_degree;
      reinit_data.elliptic.intersected_cells_only             = true;
      reinit_data.elliptic.penalty_parameter                  = 1e4;
      reinit_data.elliptic.nlsolve.max_nonlinear_iterations   = 1000;
      reinit_data.elliptic.nlsolve.residual_tolerance         = 1e-8;
      reinit_data.elliptic.nlsolve.field_correction_tolerance = 1e-8;
      reinit_data.elliptic.nlsolve.verbosity_level            = 0;
      reinit_data.linear_solver.solver_type                   = LinearSolverType::CG;
      reinit_data.linear_solver.preconditioner_type           = PreconditionerType::ILU;
      reinit_data.linear_solver.rel_tolerance                 = 1e-12;
      reinit_data.linear_solver.abs_tolerance                 = 1e-20;
      reinit_data.linear_solver.max_iterations                = 1000;

      NormalVectorData<number> normal_vector_data;
      normal_vector_data.filter_parameter            = 0.;
      normal_vector_data.linear_solver.rel_tolerance = 1e-13;
      CurvatureData<number> curvature_data;
      curvature_data.filter_parameter            = 0.;
      curvature_data.linear_solver.rel_tolerance = 1e-13;

      NarrowBandLevelSetOperation<dim, number> operation(
        scratch_data,
        triangulation,
        narrow_band_data,
        reinit_data,
        fe_degree,
        std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim),
        &normal_vector_data,
        &curvature_data);
      operation.set_initial_condition(level_set_function, do_reinitialize);

      const auto *continuous_level_set = operation.get_continuous_level_set();
      EXPECT_NE(continuous_level_set, nullptr);
      const auto        &dof_handler = continuous_level_set->get_dof_handler();
      const unsigned int dof_idx     = continuous_level_set->get_dof_index();

      // exact nodal values
      const auto interpolate = [&](const std::function<number(const dealii::Point<dim> &)> &f) {
        VectorType vector;
        scratch_data.initialize_dof_vector(vector, dof_idx);
        dealii::VectorTools::interpolate(scratch_data.get_mapping(),
                                         dof_handler,
                                         dealii::ScalarFunctionFromFunctionObject<dim>(f),
                                         vector);
        return vector;
      };

      const VectorType level_set_exact =
        interpolate([&](const dealii::Point<dim> &p) { return level_set_function.value(p); });
      const VectorType        curvature_exact = interpolate(exact_curvature);
      std::vector<VectorType> normal_exact;
      for (unsigned int d = 0; d < dim; ++d)
        normal_exact.push_back(
          interpolate([&](const dealii::Point<dim> &p) { return exact_normal(p)[d]; }));

      const auto &level_set = continuous_level_set->get_level_set();
      const auto &normal    = continuous_level_set->get_normal_vector();
      const auto &curvature = continuous_level_set->get_curvature();

      std::array<number, 3> errors = {{0., 0., 0.}};
      for (unsigned int i = 0; i < level_set.locally_owned_size(); ++i)
        {
          errors[0] =
            std::max(errors[0],
                     std::abs(level_set.local_element(i) - level_set_exact.local_element(i)));

          number error_normal = 0.;
          for (unsigned int d = 0; d < dim; ++d)
            error_normal += dealii::Utilities::fixed_power<2>(normal.block(d).local_element(i) -
                                                              normal_exact[d].local_element(i));
          errors[1] = std::max(errors[1], std::sqrt(error_normal));

          errors[2] =
            std::max(errors[2],
                     std::abs(curvature.local_element(i) - curvature_exact.local_element(i)));
        }

      for (auto &error : errors)
        error = dealii::Utilities::MPI::max(error, MPI_COMM_WORLD);

      return {continuous_level_set->get_statistics(), errors};
    }
  } // namespace

  TEST(NarrowBandLevelSetTest, ExtensionReproducesPolynomials2D)
  {
    for (unsigned int degree = 1; degree <= 3; ++degree)
      {
        test_extension_reproduces_polynomials<2>(degree,
                                                 NarrowBandGhostPenaltyFaces::all_band_faces);
        test_extension_reproduces_polynomials<2>(degree,
                                                 NarrowBandGhostPenaltyFaces::cut_cell_faces);
      }
  }

  TEST(NarrowBandLevelSetTest, ExtensionReproducesPolynomialsWithoutSmallCutCells2D)
  {
    // The extension of a polynomial does not depend on the set of ghost-penalty faces. The
    // tolerance accounts for the worse conditioning of the system, if neighbor cells are only
    // connected to cut cells via other neighbor cells.
    for (unsigned int degree = 1; degree <= 3; ++degree)
      for (const auto faces : {NarrowBandGhostPenaltyFaces::all_band_faces,
                               NarrowBandGhostPenaltyFaces::cut_cell_faces})
        test_extension_reproduces_polynomials<2>(degree, faces, 0.3, 1e-7);
  }

  TEST(NarrowBandLevelSetTest, ExtensionReproducesPolynomials3D)
  {
    test_extension_reproduces_polynomials<3>(2, NarrowBandGhostPenaltyFaces::all_band_faces);
  }

  TEST(NarrowBandLevelSetTest, ExtensionSignLimiter2D)
  {
    test_sign_limiter<2>();
  }

  TEST(NarrowBandLevelSetTest, AdvectionOperatorLinearField2D)
  {
    for (unsigned int degree = 1; degree <= 3; ++degree)
      test_advection_operator_linear_field<2>(degree);
  }

  TEST(NarrowBandLevelSetTest, AdvectionOperatorLinearField3D)
  {
    test_advection_operator_linear_field<3>(2);
  }

  TEST(NarrowBandLevelSetTest, ModalSmoothnessIndicator2D)
  {
    constexpr unsigned int                        fe_degree = 2;
    const dealii::FE_DGQ<2>                       fe(fe_degree);
    const auto                                   &points = fe.get_unit_support_points();
    std::vector<number>                           values(fe.n_dofs_per_cell());
    const NarrowBand::ModalSmoothnessIndicator<2> indicator(fe);

    // Legendre polynomials of degree 1 and 2 on the unit interval with the squared norms 1/3 and
    // 1/5
    const auto legendre_1 = [](const double x) { return 2. * x - 1.; };
    const auto legendre_2 = [](const double x) { return 6. * x * x - 6. * x + 1.; };

    const auto ratio = [&](const auto &function) {
      for (unsigned int i = 0; i < points.size(); ++i)
        values[i] = function(points[i]);
      return indicator.compute_energy_ratio(values.data());
    };

    // constant and bilinear polynomials: no highest modes
    EXPECT_LT(ratio([](const auto &) { return 0.3; }), 1e-6);
    EXPECT_LT(ratio([](const auto &x) { return 0.2 + x[0] - 0.5 * x[1] + x[0] * x[1]; }), 1e-24);

    // highest mode only
    EXPECT_NEAR(ratio([&](const auto &x) { return 0.7 + legendre_2(x[1]); }), 1., 1e-12);

    // a P_1(x_0) + b P_2(x_0): ratio b^2/5 / (a^2/3 + b^2/5)
    const double a = 0.4, b = 0.1;
    EXPECT_NEAR(ratio([&](const auto &x) { return a * legendre_1(x[0]) + b * legendre_2(x[0]); }),
                (b * b / 5.) / (a * a / 3. + b * b / 5.),
                1e-12);
  }

  TEST(NarrowBandLevelSetTest, AdvectionOperatorArtificialViscosity2D)
  {
    for (unsigned int degree = 2; degree <= 3; ++degree)
      test_advection_operator_artificial_viscosity<2>(degree);
  }

  TEST(NarrowBandLevelSetTest, AdvectionOperatorUpwindFlux2D)
  {
    for (unsigned int degree = 1; degree <= 3; ++degree)
      test_advection_operator_upwind_flux<2>(degree);
  }

  TEST(NarrowBandLevelSetTest, ClosestPointExtensionLinearField2D)
  {
    for (unsigned int degree = 1; degree <= 3; ++degree)
      test_closest_point_extension_linear_field<2>(degree);
  }

  TEST(NarrowBandLevelSetTest, ClosestPointExtensionLinearField3D)
  {
    test_closest_point_extension_linear_field<3>(2);
  }

  TEST(NarrowBandLevelSetTest, ClosestPointExtensionCircle2D)
  {
    // maximum nodal error of the signed distance function on the neighbor cells (the error of the
    // ghost-penalty extension is about 0.025, 0.011 and 0.0039 for p = 1, 2, 3)
    const std::vector<double> tolerances = {0.02, 2e-3, 2e-4};
    for (unsigned int degree = 1; degree <= 3; ++degree)
      EXPECT_LT(compute_closest_point_extension_error_circle<2>(degree, 4), tolerances[degree - 1]);
  }

  TEST(NarrowBandLevelSetTest, ReinitializationInterfaceFaces2D)
  {
    std::vector<std::pair<unsigned int, unsigned int>> n_faces;
    for (const bool interface_faces_only : {false, true})
      {
        const auto [error_initial, error, n_interface_faces] =
          test_reinitialization_faces<2>(2, 4, interface_faces_only);

        EXPECT_GT(error_initial, 0.03);
        EXPECT_LT(error, 1e-3);

        n_faces.push_back(n_interface_faces);
      }

    // all faces between two cut cells
    EXPECT_EQ(n_faces[0].first, 0U);
    EXPECT_EQ(n_faces[0].second, 0U);

    // faces crossed by the interface
    EXPECT_GT(n_faces[1].first, 0U);
    EXPECT_LT(n_faces[1].first, n_faces[1].second);
  }

  TEST(NarrowBandLevelSetTest, NeighborActivationVolumeFraction2D)
  {
    using Columns = std::vector<unsigned int>;

    // all face neighbors of the cut cells are activated
    EXPECT_EQ(compute_neighbor_columns<2>(0.05, 0.5, 0.0), (Columns{2, 4, 6}));

    // the cut cells of column 3 do not activate their face neighbors; column 4 is activated by
    // the cut cells of column 5
    EXPECT_EQ(compute_neighbor_columns<2>(0.05, 0.5, 0.1), (Columns{4, 6}));

    // no cut cell activates its face neighbors
    EXPECT_EQ(compute_neighbor_columns<2>(0.05, 0.08, 0.1), (Columns{}));

    // volume fraction above the threshold
    EXPECT_EQ(compute_neighbor_columns<2>(0.3, 0.08, 0.1), (Columns{2, 4}));
  }

  TEST(NarrowBandLevelSetTest, DownstreamNeighborActivation2D)
  {
    using Columns = std::vector<unsigned int>;

    // interfaces in the columns 3 and 5: only the face neighbors, into which the velocity points
    // from the cut cells, are activated
    EXPECT_EQ(compute_neighbor_columns<2>(0.3, 0.5, 0.0, {1., 0.}), (Columns{4, 6}));
    EXPECT_EQ(compute_neighbor_columns<2>(0.3, 0.5, 0.0, {-1., 0.}), (Columns{2, 4}));
    EXPECT_EQ(compute_neighbor_columns<2>(0.3, 0.5, 0.0, {1., 1.}), (Columns{4, 6}));

    // velocity tangential to the interfaces: no neighbor cells
    EXPECT_EQ(compute_neighbor_columns<2>(0.3, 0.5, 0.0, {0., 1.}), (Columns{}));

    // combination with the neighbor activation volume fraction: the cut cells of column 5 do not
    // activate their face neighbors
    EXPECT_EQ(compute_neighbor_columns<2>(0.3, 0.08, 0.1, {1., 0.}), (Columns{4}));
  }

  TEST(NarrowBandLevelSetTest, ExtensionWithoutSmallCutCells2D)
  {
    constexpr double tolerance = 1e-10;

    // index of the neighbor cells of column c in the lower and upper half
    const auto lower = [](const unsigned int c) { return 2 * c; };
    const auto upper = [](const unsigned int c) { return 2 * c + 1; };

    const auto gp = NarrowBandExtensionType::ghost_penalty;
    const auto cp = NarrowBandExtensionType::closest_point;

    // all cut cells are sources: the extension to the neighbor cells next to the small cut cells
    // depends on their values
    {
      const auto difference =
        compute_small_cut_cell_sensitivity<2>(gp, NarrowBandGhostPenaltyFaces::all_band_faces, 0.);
      EXPECT_GT(difference[upper(2)], 1e-3);
      EXPECT_GT(difference[upper(4)], 1e-3);
    }

    // ghost-penalty extension, all band faces: all neighbor cells are connected to cut cells with
    // a sufficient volume fraction (column 2 via the faces between the neighbor cells), i.e., the
    // extension does not depend on the small cut cells
    {
      const auto difference =
        compute_small_cut_cell_sensitivity<2>(gp, NarrowBandGhostPenaltyFaces::all_band_faces, 0.5);
      for (const double d : difference)
        EXPECT_LT(d, tolerance);
    }

    // ghost-penalty extension, cut cell faces: the neighbor cells of column 4 are extended from
    // the cut cells of column 5, while the neighbor cells in the upper half of column 2 are only
    // connected to small cut cells (fallback: their faces to the small cut cells are kept)
    {
      const auto difference =
        compute_small_cut_cell_sensitivity<2>(gp, NarrowBandGhostPenaltyFaces::cut_cell_faces, 0.5);
      EXPECT_LT(difference[lower(4)], tolerance);
      EXPECT_LT(difference[upper(4)], tolerance);
      EXPECT_GT(difference[upper(2)], 1e-3);
    }

    // closest-point extension: the neighbor cells of column 4 share a vertex with the cut cells
    // of column 5, i.e., the small cut cells are not considered
    {
      const auto difference_all =
        compute_small_cut_cell_sensitivity<2>(cp, NarrowBandGhostPenaltyFaces::all_band_faces, 0.);
      EXPECT_GT(difference_all[upper(4)], 1e-3);

      const auto difference =
        compute_small_cut_cell_sensitivity<2>(cp, NarrowBandGhostPenaltyFaces::all_band_faces, 0.5);
      EXPECT_LT(difference[lower(4)], tolerance);
      EXPECT_LT(difference[upper(4)], tolerance);
    }
  }

  TEST(NarrowBandLevelSetTest, SemiLagrangianPlanar2D)
  {
    for (unsigned int degree = 1; degree <= 3; ++degree)
      {
        const auto [n_unchanged, n_flipped] = test_semi_lagrangian_planar<2>(degree);
        EXPECT_GT(n_unchanged, 0U);
        EXPECT_GT(n_flipped, 0U);
      }
  }

  TEST(NarrowBandLevelSetTest, SemiLagrangianZeroVelocityDiscontinuous2D)
  {
    // with a vanishing velocity, the advected level set equals the old one also at the nodes on
    // the faces, where the level set jumps
    for (unsigned int degree = 1; degree <= 3; ++degree)
      EXPECT_LT(compute_semi_lagrangian_zero_velocity_error<2>(degree), 1e-12);
  }

  TEST(NarrowBandLevelSetTest, InterfacePointsCircle2D)
  {
    const auto result = test_interface_points_circle<2>();

    // the surface quadratures of the cells reproduce the seeded points and their weights
    EXPECT_LT(result.error_points, 1e-12);
    EXPECT_LT(result.error_weights, 1e-12);
    EXPECT_NEAR(result.length, 2. * dealii::numbers::PI * 0.3, 0.03);

    // seeded points: all points in cut cells, every cut cell contains points
    EXPECT_GT(result.initial.n_points, 0U);
    EXPECT_EQ(result.initial.n_points_outside, 0U);
    EXPECT_EQ(result.initial.n_inserted_points, 0U);
    EXPECT_EQ(result.initial.n_merged_points, 0U);
    EXPECT_EQ(result.initial.n_points_in_non_cut_cells, 0U);
    EXPECT_EQ(result.initial.n_cut_cells_without_points, 0U);

    // full rigid rotation: the points return to their initial positions (RK4)
    EXPECT_LT(result.error_rotation, 1e-6);
    EXPECT_EQ(result.rotated.n_points, result.initial.n_points);
    EXPECT_EQ(result.rotated.n_inserted_points, 0U);
    EXPECT_EQ(result.rotated.n_points_in_non_cut_cells, 0U);

    // larger circle: the initial points are not supported by its level set (|phi| = 0.4 h) and
    // removed, the points are seeded anew on its zero level set
    EXPECT_EQ(result.larger.n_unsupported_points, result.initial.n_points);
    EXPECT_GT(result.larger.n_inserted_points, 0U);
    EXPECT_EQ(result.larger.n_points, result.larger.n_inserted_points);
    EXPECT_EQ(result.larger.n_points_in_non_cut_cells, 0U);
    EXPECT_EQ(result.larger.n_cut_cells_without_points, 0U);

    // contraction: merged points, at most 2 (p+1) points per cell, the weights of the advected
    // points are contracted by the factor of three, and their sum is preserved by the merging
    EXPECT_GT(result.contracted.n_merged_points, 0U);
    EXPECT_EQ(result.contracted.n_points_outside, 0U);
    EXPECT_LE(result.max_n_points_per_cell, 4U);
    EXPECT_LT(result.error_weights_contraction, 1e-6);
    EXPECT_LT(result.error_contraction, 0.01);
  }

  TEST(NarrowBandLevelSetTest, InterfacePointsPerDirection2D)
  {
    // Seeding with n = 2 (p + 1 for p = 1) and n = 4 Gauss points per direction on the circle of
    // radius 0.3 (16^2 cells): every piece of the interface in a cut cell gets n points, the sum
    // of the weights is the length of the discrete interface in both cases (up to the accuracy of
    // the surface quadratures, observed difference 5e-5), no point is inserted and no point is
    // merged (at most 2 n points per cell in 2D), and every cut cell contains points.
    constexpr int      dim    = 2;
    const number       radius = 0.3;
    dealii::Point<dim> center;
    center[0] = 0.4;
    const dealii::Functions::SignedDistance::Sphere<dim> sphere(center, radius);

    NarrowBandFixture<dim> fixture(1, 4, 1, &sphere);
    VectorType             level_set, level_set_band;
    fixture.initialize_vector(fixture.dof_handler_cut, level_set);
    fixture.initialize_vector(fixture.dof_handler_band, level_set_band);
    dealii::VectorTools::interpolate(fixture.mapping, fixture.dof_handler_cut, sphere, level_set);
    dealii::VectorTools::interpolate(fixture.mapping,
                                     fixture.dof_handler_band,
                                     sphere,
                                     level_set_band);
    const std::vector<bool> is_cut_cell =
      NarrowBand::compute_cut_cells(fixture.dof_handler_band, level_set_band, 1);

    NarrowBandData<number> narrow_band_data;
    EXPECT_EQ(narrow_band_data.advected_points.get_points_per_direction(1), 2U);
    narrow_band_data.advected_points.points_per_direction = 4;
    EXPECT_EQ(narrow_band_data.advected_points.get_points_per_direction(1), 4U);

    const dealii::ConditionalOStream pcout(std::cout, false);
    std::array<unsigned int, 2>      n_points{};
    std::array<number, 2>            length{};
    for (unsigned int k = 0; k < 2; ++k)
      {
        const unsigned int                     n = k == 0 ? 2 : 4;
        NarrowBandInterfacePoints<dim, number> interface_points(pcout);
        interface_points.seed(fixture.dof_handler_cut, level_set, n);
        const auto statistics =
          interface_points.locate(fixture.dof_handler_band, level_set_band, n, is_cut_cell);

        EXPECT_EQ(statistics.n_inserted_points, 0U);
        EXPECT_EQ(statistics.n_merged_points, 0U);
        EXPECT_EQ(statistics.n_points_outside, 0U);
        EXPECT_EQ(statistics.n_cut_cells_without_points, 0U);

        n_points[k] = statistics.n_points;
        for (const auto weight : interface_points.get_weights())
          length[k] += weight;
      }

    EXPECT_EQ(n_points[1], 2 * n_points[0]);
    EXPECT_NEAR(length[0], length[1], 1e-4);
    EXPECT_NEAR(length[0], 2. * dealii::numbers::PI * radius, 0.03);
  }

  TEST(NarrowBandLevelSetTest, ReinitializationAdvectedPoints2D)
  {
    const number deviation_zero_level_set = compute_rotating_circle_radial_deviation<2>(
      NarrowBandReinitializationInterface::zero_level_set);
    const number deviation_advected_points = compute_rotating_circle_radial_deviation<2>(
      NarrowBandReinitializationInterface::advected_points);

    // insertion of new points from the advected level set (instead of the reinitialized level
    // set), continuous level set space with the closed-interface correction
    const number deviation_advected_insertion = compute_rotating_circle_radial_deviation<2>(
      NarrowBandReinitializationInterface::advected_points,
      NarrowBandLevelSetSpace::discontinuous,
      NarrowBandPointInsertion::advected_level_set);
    const number deviation_continuous = compute_rotating_circle_radial_deviation<2>(
      NarrowBandReinitializationInterface::advected_points, NarrowBandLevelSetSpace::continuous);

    // p = 1: the interpolation of the advected signed distance function shrinks the circle; the
    // advected interface points remove this drift
    EXPECT_LT(deviation_zero_level_set, -1e-3);
    EXPECT_LT(std::abs(deviation_advected_points), 0.2 * std::abs(deviation_zero_level_set));
    EXPECT_LT(std::abs(deviation_advected_insertion), 0.2 * std::abs(deviation_zero_level_set));
    EXPECT_LT(std::abs(deviation_continuous), 0.2 * std::abs(deviation_zero_level_set));
  }

  TEST(NarrowBandLevelSetTest, SemiLagrangianSubcycling2D)
  {
    // Rotating circle (quarter turn, continuous level set with the closed-interface correction,
    // advected points, p = 1): the mean radial deviation does not depend on the time step, with
    // the update in every time step (the circle is resolved; observed: -1.0917e-3 for dt = 0.01
    // and 0.001) and with sub-cycling (update, once the displacement since the reference exceeds
    // 0.5 h; observed: -1.8064e-3 for dt = 0.001, where the last level set is evaluated from the
    // reference without reinitialization, and -1.0917e-3 for dt = 0.01, where every time step
    // is an update).
    //
    // With the characteristic map instead of the backward integration, the foot points are the
    // same up to round-off errors: the displacement of the rotation is linear and represented
    // exactly by the continuous field, and the composition of the Runge-Kutta steps of the time
    // steps is the Runge-Kutta integration over all time steps.
    const auto deviation = [](const number                              time_step,
                              const number                              cfl,
                              const NarrowBandSubcyclingCharacteristics characteristics =
                                NarrowBandSubcyclingCharacteristics::integration) {
      return compute_rotating_circle_radial_deviation<2>(
        NarrowBandReinitializationInterface::advected_points,
        NarrowBandLevelSetSpace::continuous,
        NarrowBandPointInsertion::final_level_set,
        time_step,
        cfl,
        characteristics);
    };
    const number reference      = deviation(0.01, 0.);
    const number small_steps    = deviation(0.001, 0.);
    const number subcycling     = deviation(0.001, 0.5);
    const number large_steps    = deviation(0.01, 0.5);
    const number subcycling_map = deviation(0.001, 0.5, NarrowBandSubcyclingCharacteristics::map);

    EXPECT_LT(std::abs(reference), 2e-3);
    EXPECT_NEAR(small_steps, reference, 1e-6);
    EXPECT_LT(std::abs(subcycling), 2. * std::abs(reference));
    EXPECT_NEAR(large_steps, reference, 1e-12);
    EXPECT_NEAR(subcycling_map, subcycling, 1e-10);
  }

  TEST(NarrowBandLevelSetTest, CharacteristicMap2D)
  {
    // Foot points at t = 0 of the nodes at t = 0.02 in the single-vortex flow (displacement up
    // to 0.64 h, 32^2 cells, p = 2). The error of the characteristic map (interpolation of the
    // displacement) does not grow with the number of time steps (observed: 1.39e-6 = 4.5e-5 h
    // for 20 steps and 1.49e-6 for 200 steps; the backward integration over all time steps is
    // exact up to 1e-13), the foot points of the vertices are the same for all cells (continuous
    // field), and no node is extrapolated.
    const CharacteristicMapResult coarse = compute_characteristic_map_errors(1e-3, 0.02);
    const CharacteristicMapResult fine   = compute_characteristic_map_errors(1e-4, 0.02);

    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "characteristic map: error " << coarse.error_map << " (20 steps), "
                << fine.error_map << " (200 steps); integration: " << coarse.error_integration
                << ", " << fine.error_integration << "; vertex mismatch "
                << std::max(coarse.max_vertex_mismatch, fine.max_vertex_mismatch) << std::endl;

    EXPECT_LT(coarse.error_map, 1e-5);
    EXPECT_LT(fine.error_map, 1e-5);
    EXPECT_LT(fine.error_map, 2. * coarse.error_map);
    EXPECT_LT(coarse.max_vertex_mismatch, 1e-14);
    EXPECT_LT(fine.max_vertex_mismatch, 1e-14);
    EXPECT_EQ(coarse.n_nodes_extrapolated, 0U);
    EXPECT_EQ(fine.n_nodes_extrapolated, 0U);
  }

  TEST(NarrowBandLevelSetTest, MergingCircles2D)
  {
    // Two circles merge (topology change): the advected interface points of the colliding sides
    // stop at the plane of the collision inside of the merged phase. Without the merge rule, they
    // pin spurious parts of the interface inside of the merged phase (observed: relative length
    // error 8.0e-2 at t = 0.4); the merge rule removes them, and the merged interface is accurate
    // (observed: -3.3e-4).
    const number error_without_rule = compute_merging_circles_length_error<2>(0.);
    const number error_with_rule    = compute_merging_circles_length_error<2>(0.05);

    EXPECT_GT(std::abs(error_without_rule), 2e-2);
    EXPECT_LT(std::abs(error_with_rule), 2e-3);
  }

  TEST(NarrowBandLevelSetTest, BreakupCircles2D)
  {
    // Two overlapping circles are squeezed towards their axis, and the neck pinches off
    // (topology change). Without the merge and break-up rules, the advected interface points of
    // the two sides of the neck and of the tips, which are pushed against each other at the axis,
    // remain there and keep a thin film between the two parts by the subcell distance (observed:
    // relative length error 1.7e-1 at t = 0.6); the break-up rule removes them (observed:
    // 2.0e-2).
    const number error_without_rule = compute_breakup_circles_length_error<2>(0.);
    const number error_with_rule    = compute_breakup_circles_length_error<2>(0.05);

    EXPECT_GT(std::abs(error_without_rule), 8e-2);
    EXPECT_LT(std::abs(error_with_rule), 4e-2);
  }

  TEST(NarrowBandLevelSetTest, ReinitializationPrescribedDoFs2D)
  {
    // The prescribed DoFs keep their values, and the other cut cells are reinitialized with the
    // same accuracy as without the prescription. The SIPG face terms couple them to the prescribed
    // cells (small jumps across their common faces); without the coupling, the faces to the
    // prescribed cells are a natural boundary of the other cells.
    for (const bool couple : {true, false})
      {
        const auto [change_prescribed, error_free, error_free_reference, max_jump] =
          test_reinitialization_prescribed_dofs<2>(2, 4, couple);

        EXPECT_LT(change_prescribed, 1e-13);
        EXPECT_LT(error_free, 1e-3);
        EXPECT_LT(error_free, 2. * error_free_reference + 1e-6);
        if (couple)
          EXPECT_LT(max_jump, 1e-4);
      }
  }

  TEST(NarrowBandLevelSetTest, SubcellDistanceStrip2D)
  {
    const auto result = test_subcell_distance_strip();

    // the cut cells of the strip (one row of eight cells) are unresolved and projected
    EXPECT_EQ(result.statistics.n_cells, 8U);
    EXPECT_EQ(result.statistics.n_rejected_cells, 0U);

    // exact signed distance to the flat patches of the points
    EXPECT_LT(result.error_distance, 1e-3);

    // the projection restores both zeros in every cut cell (L2-projection of the distance with
    // the kink on the medial axis onto Q2: zeros within 0.05 h)
    EXPECT_EQ(result.min_n_zeros_interpolated, 2U);
    EXPECT_EQ(result.min_n_zeros_projected, 2U);
    EXPECT_LT(result.error_zeros_projected, 0.05 * 0.25);
  }

  TEST(NarrowBandLevelSetTest, SubcellDistanceOrientationFilter2D)
  {
    // Strip of width 0.75 h, which contains the face y = 0 in the ratio 1:3: the sides of the strip
    // lie in two rows of cells, and the medial axis lies inside the cells of the upper row. With
    // all points, the kink of the distance on the medial axis shifts the zero of the upper cells;
    // with the orientation filter, every cell only uses the points of its side (observed: 0.034 h
    // and 0.0003 h).
    const auto all_points = test_subcell_distance_strip(0.1875, 0.75, false);
    const auto filtered   = test_subcell_distance_strip(0.1875, 0.75, true);

    EXPECT_EQ(all_points.statistics.n_filtered_cells, 0U);
    EXPECT_EQ(filtered.statistics.n_filtered_cells, filtered.statistics.n_cells);
    EXPECT_EQ(filtered.min_n_zeros_projected, 1U);
    EXPECT_GT(all_points.error_zeros_projected, 0.01 * 0.25);
    EXPECT_LT(filtered.error_zeros_projected, 1e-3 * 0.25);
  }

  TEST(NarrowBandLevelSetTest, SubcellDistancePinned2D)
  {
    // Strip of width 0.75 h, whose sides lie in two rows of cells (see
    // SubcellDistanceOrientationFilter2D): the distance to all points has its kink on the medial
    // axis inside the cells of the upper row, and the pinning penalty keeps the zero of these
    // cells on their points (observed: 0.034 h without and 0.00025 h with pinning).
    const auto plain  = test_subcell_distance_strip(0.1875, 0.75, false, 0.);
    const auto pinned = test_subcell_distance_strip(0.1875, 0.75, false, 1000.);

    EXPECT_EQ(plain.statistics.n_pinned_cells, 0U);
    EXPECT_EQ(pinned.statistics.n_pinned_cells, pinned.statistics.n_cells);
    EXPECT_EQ(pinned.statistics.n_filtered_cells, 0U);
    EXPECT_EQ(pinned.min_n_zeros_projected, 1U);
    EXPECT_GT(plain.error_zeros_projected, 0.01 * 0.25);
    EXPECT_LT(pinned.error_zeros_projected, 2e-3 * 0.25);

    // strip of width 0.4 h inside one row of cells: every cut cell contains points of both
    // orientations and is not pinned
    const auto both_sides = test_subcell_distance_strip(0.5, 0.4, false, 1000.);
    EXPECT_EQ(both_sides.statistics.n_cells, 8U);
    EXPECT_EQ(both_sides.statistics.n_pinned_cells, 0U);
    EXPECT_EQ(both_sides.min_n_zeros_projected, 2U);
  }

  TEST(NarrowBandLevelSetTest, SubcellDistanceContinuous2D)
  {
    // Subcell distance in the continuous level set space (coupled projection on the projected
    // cells): the strip of width 0.4 h inside one row of cells keeps both zeros in every cut cell,
    // and the strip of width 0.75 h, whose sides lie in two rows of cells, keeps one zero per cell
    // (observed: 0.050 h and 0.047 h; discontinuous: 0.049 h and 0.034 h).
    const auto one_row  = test_subcell_distance_strip(0.5, 0.4, false, 0., true);
    const auto two_rows = test_subcell_distance_strip(0.1875, 0.75, false, 0., true);

    EXPECT_EQ(one_row.statistics.n_cells, 8U);
    EXPECT_EQ(one_row.min_n_zeros_projected, 2U);
    EXPECT_LT(one_row.error_zeros_projected, 0.06 * 0.25);
    EXPECT_EQ(two_rows.statistics.n_cells, 16U);
    EXPECT_EQ(two_rows.min_n_zeros_projected, 1U);
    EXPECT_LT(two_rows.error_zeros_projected, 0.06 * 0.25);
  }

  TEST(NarrowBandLevelSetTest, ContinuousReinitializationCircle2D)
  {
    // the continuous reinitialization recovers the signed distance from a scaled signed distance
    // (observed: 2.9e-4) and the result is continuous across the faces between cut cells
    const auto [error, jump, n_iterations] = test_continuous_reinitialization();
    EXPECT_LT(error, 6e-4);
    EXPECT_LT(jump, 1e-12);
    EXPECT_GT(n_iterations, 1U);
  }

  TEST(NarrowBandLevelSetTest, ClosedInterface)
  {
    // closed plane (a = 0): no correction, the level set on the faces to the cells above and below
    // the layer of cut cells is at least 0.1 / h = 0.4 away from zero
    for (const unsigned int fe_degree : {1U, 2U, 3U})
      {
        const auto result = test_closed_interface<2>(fe_degree, 0.);
        EXPECT_EQ(result.statistics.n_faces, 16U);
        EXPECT_EQ(result.statistics.n_lifted_faces, 0U);
        EXPECT_EQ(result.n_cells_behind_open_faces, 0U);
        EXPECT_EQ(result.statistics.max_correction, 0.);
        EXPECT_LT(result.max_change, 1e-14);
        EXPECT_NEAR(result.min_value, 0.4, 1e-12);
      }

    // perturbed plane (a = 0.2): the interface leaves the cut cells through faces to the cells
    // above and below; these faces are lifted, after which the level set has the sign of the
    // neighbor cell on the whole faces (minimum at least the offset 0.01 h) and is continuous;
    // there are no conflicting nodes. Maximum correction: 0.41 h at the bottom faces (minimum
    // -0.4 h, lifted to 0.01 h).
    for (const unsigned int fe_degree : {1U, 2U, 3U})
      {
        const auto result = test_closed_interface<2>(fe_degree, 0.2);
        EXPECT_EQ(result.statistics.n_faces, 16U);
        EXPECT_GT(result.statistics.n_lifted_faces, 0U);
        EXPECT_EQ(result.n_cells_behind_open_faces, result.statistics.n_lifted_faces);
        EXPECT_EQ(result.statistics.n_conflicting_dofs, 0U);
        EXPECT_EQ(result.statistics.n_open_faces, 0U);
        EXPECT_NEAR(result.statistics.max_correction, 0.41, 1e-3);
        EXPECT_GT(result.min_value, 0.01 - 1e-6);
        EXPECT_LT(result.jump, 1e-12);
      }

    // parabola phi = y - 0.25 + 0.5 (x - 0.125)^2 (p = 2, represented exactly): it touches the
    // upper face of the cell [0, 0.25] at its midpoint and crosses the lower faces for
    // |x - 0.125| > 1/sqrt(2), i.e., on the three faces [-1, -0.75], [-0.75, -0.5] and [0.75, 1].
    // Only these are lifted; the touching face and all upper faces keep their values.
    {
      const auto result = test_closed_interface<2>(2, 0., [](const dealii::Point<2> &p) {
        return p[1] - 0.25 + 0.5 * (p[0] - 0.125) * (p[0] - 0.125);
      });
      EXPECT_EQ(result.statistics.n_lifted_faces, 3U);
      EXPECT_EQ(result.n_cells_behind_open_faces, 3U);
      EXPECT_EQ(result.statistics.n_open_faces, 0U);
      EXPECT_EQ(result.max_change_top, 0.);
      EXPECT_GE(result.min_value, -1e-14);
      EXPECT_LT(result.jump, 1e-12);
    }

    // 3D (a = 0.2, p = 2): the nodes of the edges are shared by the faces of adjacent cells
    {
      const auto result = test_closed_interface<3>(2, 0.2);
      EXPECT_EQ(result.statistics.n_faces, 128U);
      EXPECT_GT(result.statistics.n_lifted_faces, 0U);
      EXPECT_EQ(result.statistics.n_conflicting_dofs, 0U);
      EXPECT_EQ(result.statistics.n_open_faces, 0U);
      EXPECT_GT(result.min_value, 0.01 - 1e-6);
      EXPECT_LT(result.jump, 1e-12);
    }
  }

  TEST(NarrowBandLevelSetTest, SignedDistanceToPatchesShell3D)
  {
    // Points seeded on a thin spherical shell (radii 0.45 and 0.75, width 1.2 h, p = 2, 8^3
    // cells on [-1,1]^3), i.e., two close branches of opposite orientation: the signed distance to
    // the patches of the nearest points (24 in 3D) equals the minimum distance to the patches of
    // all points (brute force) at points between and around the branches, and its sign is the side
    // of the closest patch. With the 8 nearest points (as in 2D), the distance was overestimated by
    // up to 8.2e-4 (0.003 h): the closest patch is not always among them in 3D.
    constexpr int                                       dim       = 3;
    const unsigned int                                  fe_degree = 2;
    const number                                        h         = 0.25;
    const number                                        radius = 0.6, half_width = 0.15;
    const dealii::ScalarFunctionFromFunctionObject<dim> shell(
      [&](const dealii::Point<dim> &p) { return half_width - std::abs(p.norm() - radius); });

    NarrowBandFixture<dim> fixture(fe_degree, 3, fe_degree, &shell);
    VectorType             level_set;
    fixture.initialize_vector(fixture.dof_handler_cut, level_set);
    dealii::VectorTools::interpolate(fixture.mapping, fixture.dof_handler_cut, shell, level_set);

    const dealii::ConditionalOStream       pcout(std::cout, false);
    NarrowBandInterfacePoints<dim, number> interface_points(pcout);
    interface_points.seed(fixture.dof_handler_cut, level_set, fe_degree + 1);

    const auto &points  = interface_points.get_points();
    const auto &weights = interface_points.get_weights();
    const auto &normals = interface_points.get_normals();
    ASSERT_GT(points.size(), 1000U);

    // brute force: patch of each point (disk of the area of its weight, at most of radius h)
    const auto brute_force = [&](const dealii::Point<dim> &x) {
      number min_distance = std::numeric_limits<number>::max(), sign = 0.;
      for (unsigned int k = 0; k < points.size(); ++k)
        {
          const number patch_radius      = std::min(std::sqrt(weights[k] / dealii::numbers::PI), h);
          const dealii::Tensor<1, dim> d = x - points[k];
          const number                 normal_distance  = normals[k] * d;
          const number                 tangent_distance = (d - normal_distance * normals[k]).norm();
          const number                 distance         = std::sqrt(
            normal_distance * normal_distance +
            dealii::Utilities::fixed_power<2>(std::max(0., tangent_distance - patch_radius)));
          if (distance < min_distance)
            {
              min_distance = distance;
              sign         = normal_distance > 0. ? 1. : -1.;
            }
        }
      return sign * min_distance;
    };

    // points in the octant x, y, z > 0 with radii between 0.3 and 0.9 (deterministic sequence)
    number       max_error     = 0.;
    unsigned int n_wrong_signs = 0;
    for (unsigned int i = 0; i < 400; ++i)
      {
        const number r     = 0.3 + 0.6 * std::fmod(0.618033988749895 * (i + 1), 1.);
        const number theta = 0.5 * dealii::numbers::PI * std::fmod(0.754877666246693 * (i + 1), 1.);
        const number phi   = 0.5 * dealii::numbers::PI * std::fmod(0.569840290998053 * (i + 1), 1.);
        const dealii::Point<dim> x(r * std::sin(theta) * std::cos(phi),
                                   r * std::sin(theta) * std::sin(phi),
                                   r * std::cos(theta));
        const number             distance           = interface_points.signed_distance(x, h);
        const number             distance_reference = brute_force(x);
        max_error =
          std::max<number>(max_error, std::abs(std::abs(distance) - std::abs(distance_reference)));
        if (std::abs(distance_reference) > 1e-3 and distance * distance_reference < 0.)
          ++n_wrong_signs;
      }

    EXPECT_LT(max_error, 1e-12);
    EXPECT_EQ(n_wrong_signs, 0U);
  }

  TEST(NarrowBandLevelSetTest, InterfacePointsInsertionFinalLevelSet2D)
  {
    // The points seeded on a circle (radius 0.3) are not supported by the advected level set of a
    // larger circle (radius 0.3 + 0.4 h) and removed. Without insertion in locate(), the gaps are
    // filled by insert_points() from the final level set of the cut cells (here: the initial
    // circle, e.g., after the reinitialization with other points), i.e., the new points are the
    // seeded points again, and not the points of the zero level set of the advected level set
    // (insertion in locate()).
    constexpr int      dim    = 2;
    const number       radius = 0.3;
    const number       h      = 2. / 16.;
    dealii::Point<dim> center;
    center[0] = 0.4;
    const dealii::Functions::SignedDistance::Sphere<dim> sphere(center, radius);
    const dealii::Functions::SignedDistance::Sphere<dim> sphere_larger(center, radius + 0.4 * h);

    NarrowBandFixture<dim> fixture(1, 4, 1, &sphere);
    VectorType             level_set, level_set_larger;
    fixture.initialize_vector(fixture.dof_handler_cut, level_set);
    fixture.initialize_vector(fixture.dof_handler_band, level_set_larger);
    dealii::VectorTools::interpolate(fixture.mapping, fixture.dof_handler_cut, sphere, level_set);
    dealii::VectorTools::interpolate(fixture.mapping,
                                     fixture.dof_handler_band,
                                     sphere_larger,
                                     level_set_larger);
    const std::vector<bool> is_cut_cell =
      NarrowBand::compute_cut_cells(fixture.dof_handler_band, level_set_larger, 1);

    const dealii::ConditionalOStream pcout(std::cout, false);

    // insertion in locate(): points on the zero level set of the advected level set (larger
    // circle, interpolation error of p = 1 observed: 0.035 h)
    NarrowBandInterfacePoints<dim, number> points_advected(pcout);
    points_advected.seed(fixture.dof_handler_cut, level_set, 2);
    const unsigned int n_seeded = points_advected.get_points().size();
    const auto         statistics =
      points_advected.locate(fixture.dof_handler_band, level_set_larger, 2, is_cut_cell, true);
    EXPECT_EQ(statistics.n_unsupported_points, n_seeded);
    EXPECT_EQ(statistics.n_inserted_points, statistics.n_points);
    for (const auto &point : points_advected.get_points())
      EXPECT_NEAR(point.distance(center), radius + 0.4 * h, 0.1 * h);

    // insertion after the reinitialization: points on the final level set
    NarrowBandInterfacePoints<dim, number> points_final(pcout);
    points_final.seed(fixture.dof_handler_cut, level_set, 2);
    const auto seeded_points = points_final.get_points();
    const auto statistics_located =
      points_final.locate(fixture.dof_handler_band, level_set_larger, 2, is_cut_cell, false);
    EXPECT_EQ(statistics_located.n_unsupported_points, n_seeded);
    EXPECT_EQ(statistics_located.n_inserted_points, 0U);
    EXPECT_EQ(statistics_located.n_points, 0U);
    EXPECT_TRUE(points_final.is_located());

    const auto statistics_final =
      points_final.insert_points(fixture.dof_handler_cut, level_set, fixture.dof_handler_band, 2);
    EXPECT_EQ(statistics_final.n_unsupported_points, n_seeded);
    EXPECT_EQ(statistics_final.n_inserted_points, n_seeded);
    EXPECT_EQ(statistics_final.n_points, n_seeded);
    EXPECT_EQ(statistics_final.n_cut_cells_without_points, 0U);
    EXPECT_EQ(statistics_final.n_points_in_non_cut_cells, 0U);
    ASSERT_EQ(points_final.get_points().size(), seeded_points.size());
    number error = 0.;
    for (unsigned int k = 0; k < seeded_points.size(); ++k)
      error = std::max<number>(error, points_final.get_points()[k].distance(seeded_points[k]));
    EXPECT_LT(error, 1e-14);

    // a second insertion does not add points (no gaps)
    const auto statistics_again =
      points_final.insert_points(fixture.dof_handler_cut, level_set, fixture.dof_handler_band, 2);
    EXPECT_EQ(statistics_again.n_points, n_seeded);
  }

  TEST(NarrowBandLevelSetTest, LayeredBand2D)
  {
    // one cut cell, 8 cells sharing a vertex with it (first layer) and 16 cells in the second
    // layer (and 24 cells in the third layer)
    EXPECT_EQ(count_layered_band_cells(2), (std::array<unsigned int, 3>{1, 8, 16}));
    EXPECT_EQ(count_layered_band_cells(3), (std::array<unsigned int, 3>{1, 8, 40}));
  }

  TEST(NarrowBandLevelSetTest, RedistancingCircle2D)
  {
    // maximum nodal error of the signed distance function on two layers for p = 1, 2, 3 (the
    // error converges with the order p + 1)
    const std::vector<double> tolerances = {0.03, 4e-3, 2e-4};
    for (unsigned int degree = 1; degree <= 3; ++degree)
      EXPECT_LT(compute_redistancing_error_circle<2>(degree, 4, 2), tolerances[degree - 1]);

    EXPECT_LT(compute_redistancing_error_circle<2>(2, 5, 2), 5e-4);
    EXPECT_LT(compute_redistancing_error_circle<2>(2, 4, 3), 4e-3);

    // the redistanced signed distance function of a circle does not change its sign within a
    // cell, i.e., the sign clamp does not modify it
    EXPECT_EQ(compute_redistancing_error_circle<2>(2, 4, 2, 0.01),
              compute_redistancing_error_circle<2>(2, 4, 2));
  }

  TEST(NarrowBandLevelSetTest, ExtensionSignClamp2D)
  {
    // the polynomial extension changes its sign within neighbor cells, which is removed by the
    // sign clamp
    EXPECT_GT(test_sign_clamp<2>(NarrowBandExtensionType::ghost_penalty), 0U);
    EXPECT_GT(test_sign_clamp<2>(NarrowBandExtensionType::closest_point), 0U);
  }

  TEST(NarrowBandLevelSetTest, ContinuousLevelSetPlanar2D)
  {
    // A planar level set is reproduced exactly by the constrained projection (it belongs to the
    // continuous space and vanishes on the interface), its normal vector is constant and its
    // curvature vanishes.
    dealii::Tensor<1, 2> normal;
    normal[0] = 0.8;
    normal[1] = -0.6;
    const dealii::ScalarFunctionFromFunctionObject<2> planar(
      [&](const dealii::Point<2> &p) { return normal * p - 0.1037; });

    for (unsigned int degree = 1; degree <= 3; ++degree)
      {
        const auto [statistics, errors] = compute_continuous_level_set<2>(
          planar,
          [&](const dealii::Point<2> &) { return normal; },
          [](const dealii::Point<2> &) { return 0.; },
          degree,
          3);

        EXPECT_LT(errors[0], 1e-11);
        EXPECT_LT(errors[1], 1e-11);
        EXPECT_LT(errors[2], 1e-9);
        EXPECT_LT(statistics.max_interface_shift, 1e-11);
        EXPECT_NEAR(statistics.interface_measure, 2. / 0.8, 1e-10);
        EXPECT_LT(std::abs(statistics.min_curvature), 1e-9);
        EXPECT_LT(std::abs(statistics.max_curvature), 1e-9);
      }
  }

  TEST(NarrowBandLevelSetTest, ContinuousLevelSetCircle2D)
  {
    // reinitialized signed distance function of a circle (positive inside), i.e., the DG level set
    // is discontinuous across the faces of the cut cells; exact curvature 1/r of the iso-lines and
    // inner unit normal
    const dealii::Point<2>                            center(0.031, -0.027);
    const double                                      radius = 0.5;
    const dealii::ScalarFunctionFromFunctionObject<2> circle(
      [&](const dealii::Point<2> &p) { return radius - p.distance(center); });
    const auto exact_normal = [&](const dealii::Point<2> &p) {
      const dealii::Tensor<1, 2> r = center - p;
      return r / r.norm();
    };
    const auto exact_curvature = [&](const dealii::Point<2> &p) { return 1. / p.distance(center); };

    // nodal errors of the level set and the normal vector, maximum and mean relative error of the
    // curvature on the interface
    std::vector<std::array<double, 4>> errors;
    for (const unsigned int n_refinements : {4U, 5U})
      {
        const auto [statistics, nodal_errors] = compute_continuous_level_set<2>(
          circle, exact_normal, exact_curvature, 2, n_refinements, true /*reinitialize*/);

        const double h = 2. / dealii::Utilities::pow(2, n_refinements);
        EXPECT_GT(statistics.max_relative_interface_shift, 0.);
        EXPECT_LT(statistics.max_relative_interface_shift, 2e-5);
        EXPECT_NEAR(statistics.interface_measure, 2. * dealii::numbers::PI * radius, 1e-3 * h);

        errors.push_back({{nodal_errors[0],
                           nodal_errors[1],
                           std::max(std::abs(statistics.min_curvature * radius - 1.),
                                    std::abs(statistics.max_curvature * radius - 1.)),
                           std::abs(statistics.mean_curvature * radius - 1.)}});
      }

    // (observed for 16^2 cells: 9.0e-5, 9.4e-3, 4.9e-2, 1.4e-3; the maximum curvature error
    // converges with the order p - 1)
    const std::array<double, 4> tolerances = {{1.5e-4, 1.5e-2, 7e-2, 2.5e-3}};
    for (unsigned int i = 0; i < 4; ++i)
      {
        EXPECT_LT(errors[0][i], tolerances[i]);
        EXPECT_LT(errors[1][i], errors[0][i]);
      }

    // the interface penalty reduces the shift of the interface
    const auto shift = [&](const double interface_penalty) {
      return compute_continuous_level_set<2>(
               circle, exact_normal, exact_curvature, 2, 4, true, interface_penalty)
        .first.max_relative_interface_shift;
    };
    EXPECT_LT(shift(100.), 0.5 * shift(0.));
  }

  TEST(NarrowBandLevelSetTest, ContinuousLevelSetSignConstraints2D)
  {
    // Horizontal interface y = y_0 in the row of cut cells below the grid line y = 0 (h = 1/8). In
    // the cut cell [0,h] x [-h,0], the DG level set is perturbed by a bilinear bump with the
    // maximum 0.8h at the vertex V = (0,0), such that its zero level set reaches the face y = 0 to
    // the cell above, which is not cut (negative). Without the sign constraints, the continuous
    // level set is positive on a part of this face, i.e., its zero level set leaves the cut cells.
    // With the sign constraints, it is negative on the faces to the cells above and positive on
    // the faces to the cells below, and the continuous level set only changes near V.
    constexpr unsigned int dim = 2;
    const double           h   = 2. / 16.;
    const double           y_0 = -0.3 * h;

    const dealii::ScalarFunctionFromFunctionObject<dim> level_set_function(
      [&](const dealii::Point<dim> &p) { return y_0 - p[1]; });
    const auto cellwise_function = [&](const dealii::CellAccessor<dim, dim> &cell,
                                       const dealii::Point<dim>             &p) {
      const auto center = cell.center();
      double     value  = y_0 - p[1];
      if (center[0] > 0. and center[0] < h and center[1] > -h and center[1] < 0.)
        value += 0.8 * h * (1. - p[0] / h) * (1. + p[1] / h);
      return value;
    };

    struct Result
    {
      NarrowBandContinuousLevelSet<dim, number>::Statistics statistics;
      unsigned int                                          n_violations = 0;
      std::map<unsigned int, double>                        center_values;
    };

    const auto run = [&](const bool sign_constraints) {
      dealii::parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
      dealii::GridGenerator::hyper_cube(triangulation, -1., 1.);
      triangulation.refine_global(4);

      ScratchData<dim, dim, number> scratch_data(MPI_COMM_WORLD, 0, true);
      scratch_data.set_mapping(std::make_shared<dealii::MappingQ<dim>>(1));

      NarrowBandData<number> narrow_band_data;
      narrow_band_data.type                                   = NarrowBandType::semi_lagrangian;
      narrow_band_data.continuous_level_set.enable            = true;
      narrow_band_data.continuous_level_set.sign_constraints  = sign_constraints;
      narrow_band_data.continuous_level_set.interface_penalty = 100.;
      narrow_band_data.continuous_level_set.linear_solver.rel_tolerance = 1e-13;
      narrow_band_data.check_input_parameters(2);

      ReinitializationData<number> reinit_data;
      reinit_data.modeltype                       = ModelType::elliptic;
      reinit_data.fe.type                         = FiniteElementType::FE_DGQ;
      reinit_data.fe.degree                       = 2;
      reinit_data.elliptic.intersected_cells_only = true;

      NormalVectorData<number> normal_vector_data;
      normal_vector_data.filter_parameter = 0.;
      CurvatureData<number> curvature_data;
      curvature_data.filter_parameter = 0.;

      NarrowBandLevelSetOperation<dim, number> operation(
        scratch_data,
        triangulation,
        narrow_band_data,
        reinit_data,
        2,
        std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim),
        &normal_vector_data,
        &curvature_data);
      operation.set_initial_condition(level_set_function, false, cellwise_function);

      const auto *continuous_level_set = operation.get_continuous_level_set();
      const auto &dof_handler          = continuous_level_set->get_dof_handler();
      const auto &level_set            = continuous_level_set->get_level_set();
      level_set.update_ghost_values();

      Result result;
      result.statistics = continuous_level_set->get_statistics();

      // sign of the continuous level set on the faces to the cells, which are not cut
      const dealii::FE_Q<dim>   fe(2);
      dealii::FEFaceValues<dim> fe_face_values(scratch_data.get_mapping(),
                                               fe,
                                               dealii::QIterated<dim - 1>(dealii::QTrapezoid<1>(),
                                                                          16),
                                               dealii::update_values);
      dealii::FEValues<dim>     fe_values(scratch_data.get_mapping(),
                                      fe,
                                      dealii::QMidpoint<dim>(),
                                      dealii::update_values);
      std::vector<double>       values(fe_face_values.n_quadrature_points);
      std::vector<double>       center_value(1);
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
            continue;

          for (const unsigned int face : cell->face_indices())
            {
              if (cell->at_boundary(face) or cell->neighbor(face)->get_fe().n_dofs_per_cell() > 0)
                continue;

              const double sign =
                level_set_function.value(cell->neighbor(face)->center()) > 0. ? 1. : -1.;
              fe_face_values.reinit(cell, face);
              fe_face_values.get_function_values(level_set, values);
              for (const double value : values)
                if (sign * value < 0.)
                  ++result.n_violations;
            }

          fe_values.reinit(cell);
          fe_values.get_function_values(level_set, center_value);
          if (cell->center().distance(dealii::Point<dim>()) > 3. * h)
            result.center_values[cell->active_cell_index()] = center_value[0];
        }
      result.n_violations = dealii::Utilities::MPI::sum(result.n_violations, MPI_COMM_WORLD);
      return result;
    };

    const Result without_constraints = run(false);
    const Result with_constraints    = run(true);

    EXPECT_GT(without_constraints.n_violations, 0U);
    EXPECT_EQ(without_constraints.statistics.n_sign_constrained_dofs, 0U);

    EXPECT_EQ(with_constraints.n_violations, 0U);
    EXPECT_EQ(with_constraints.statistics.n_sign_violations, 0U);
    EXPECT_GT(with_constraints.statistics.n_sign_constrained_dofs, 0U);
    EXPECT_LE(with_constraints.statistics.n_sign_constrained_dofs, 6U);
    EXPECT_GE(with_constraints.statistics.n_sign_constraint_solves, 1U);

    // local change: cut cells farther than 3h from V
    double max_change = 0.;
    for (const auto &[index, value] : with_constraints.center_values)
      max_change =
        std::max(max_change, std::abs(value - without_constraints.center_values.at(index)));
    max_change = dealii::Utilities::MPI::max(max_change, MPI_COMM_WORLD);
    EXPECT_LT(max_change, 1e-3 * h);
  }

  TEST(NarrowBandLevelSetTest, BoundPreservingLimiter2D)
  {
    // all face neighbors: the overshoot is within the range of the right neighbor
    {
      const auto [n_limited_cells, n_perturbed_cells] =
        test_bound_preserving_limiter<2>(NarrowBandLimiterBounds::face_neighbors);
      EXPECT_GT(n_perturbed_cells, 0U);
      EXPECT_EQ(n_limited_cells, 0U);
    }

    // upwind face neighbors and upwind strips (velocity e_x): all perturbed cells are limited
    for (const auto bounds_type :
         {NarrowBandLimiterBounds::upwind_face_neighbors, NarrowBandLimiterBounds::upwind_strips})
      {
        const auto [n_limited_cells, n_perturbed_cells] =
          test_bound_preserving_limiter<2>(bounds_type);
        EXPECT_GT(n_perturbed_cells, 0U);
        EXPECT_EQ(n_limited_cells, n_perturbed_cells);
      }

    // sign preservation: the negative values of the perturbed cells are removed
    {
      const auto [n_limited_cells, n_perturbed_cells] =
        test_bound_preserving_limiter<2>(NarrowBandLimiterBounds::upwind_strips,
                                         NarrowBandLimiterType::sign);
      EXPECT_GT(n_perturbed_cells, 0U);
      EXPECT_EQ(n_limited_cells, n_perturbed_cells);
    }
  }

  TEST(NarrowBandLevelSetTest, RestartReproducesContinuedRun2D)
  {
    // The restarted simulation continues exactly as the simulation, which saved the state, for
    // the semi-Lagrangian band (advected interface points, continuous level set space, with and
    // without sub-cycling by the backward integration or the characteristic map; the restart
    // between two updates of the reference), the extension
    // band (with the downstream neighbor activation and small cut cells) and the advected band.
    {
      NarrowBandData<number> data;
      data.type                       = NarrowBandType::semi_lagrangian;
      data.reinitialization_interface = NarrowBandReinitializationInterface::advected_points;
      data.level_set_space            = NarrowBandLevelSetSpace::continuous;
      EXPECT_EQ(compute_restart_difference<2>(data, 0.01, 6, 6, "restart_test_semi_lagrangian"),
                0.);
    }
    {
      NarrowBandData<number> data;
      data.type                       = NarrowBandType::semi_lagrangian;
      data.reinitialization_interface = NarrowBandReinitializationInterface::advected_points;
      data.level_set_space            = NarrowBandLevelSetSpace::continuous;
      data.level_set_update_cfl       = 0.5;
      EXPECT_EQ(compute_restart_difference<2>(data, 0.002, 7, 9, "restart_test_subcycling"), 0.);
    }
    {
      NarrowBandData<number> data;
      data.type                             = NarrowBandType::semi_lagrangian;
      data.reinitialization_interface       = NarrowBandReinitializationInterface::advected_points;
      data.level_set_space                  = NarrowBandLevelSetSpace::continuous;
      data.level_set_update_cfl             = 0.5;
      data.level_set_update_characteristics = NarrowBandSubcyclingCharacteristics::map;
      EXPECT_EQ(compute_restart_difference<2>(data, 0.002, 7, 9, "restart_test_subcycling_map"),
                0.);
    }
    {
      NarrowBandData<number> data;
      data.type                = NarrowBandType::extension;
      data.neighbor_activation = NarrowBandNeighborActivation::downstream_face_neighbors;
      data.neighbor_activation_volume_fraction = 0.1;
      EXPECT_EQ(compute_restart_difference<2>(data, 0.002, 5, 5, "restart_test_extension"), 0.);
    }
    {
      NarrowBandData<number> data;
      data.type = NarrowBandType::advected_layers;
      EXPECT_EQ(compute_restart_difference<2>(data, 0.002, 5, 5, "restart_test_layers"), 0.);
    }
  }
} // namespace MeltPoolDG::LevelSet
