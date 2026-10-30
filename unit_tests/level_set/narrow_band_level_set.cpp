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
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/level_set/narrow_band_advection_DG_operator.hpp>
#include <meltpooldg/level_set/narrow_band_closest_point_extension.hpp>
#include <meltpooldg/level_set/narrow_band_data.hpp>
#include <meltpooldg/level_set/narrow_band_ghost_penalty_extension.hpp>
#include <meltpooldg/level_set/narrow_band_utilities.hpp>
#include <meltpooldg/level_set/reinitialization_data.hpp>
#include <meltpooldg/level_set/reinitialization_elliptic_operation.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
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
    compute_neighbor_columns(const double theta_0, const double theta_1, const double threshold)
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

      NarrowBand::distribute_dofs(is_cut_cell,
                                  dof_handler_cut,
                                  fe_collection_cut,
                                  dof_handler_band,
                                  fe_collection_band,
                                  is_small_cut_cell);

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
                                      const unsigned int n_layers)
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

      NarrowBandData<number>                       data;
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
  }
} // namespace MeltPoolDG::LevelSet
