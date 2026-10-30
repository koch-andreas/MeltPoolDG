#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/vector.h>

#include <deal.II/non_matching/quadrature_generator.h>

#include <deal.II/numerics/rtree.h>

#include <meltpooldg/level_set/narrow_band_closest_point_extension.hpp>
#include <meltpooldg/level_set/narrow_band_utilities.hpp>
#include <meltpooldg/utilities/journal.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <limits>
#include <sstream>
#include <vector>

namespace MeltPoolDG::LevelSet
{
  namespace
  {
    /**
     * Affine map x = origin + jacobian * xi of a parallelepiped cell.
     */
    template <int dim>
    struct AffineCellMap
    {
      dealii::Point<dim>     origin;
      dealii::Tensor<2, dim> jacobian;
      dealii::Tensor<2, dim> inverse_jacobian;
      dealii::Tensor<2, dim> inverse_jacobian_transposed;
      double                 cell_size = 0.;

      template <typename CellIteratorType>
      explicit AffineCellMap(const CellIteratorType &cell)
        : origin(cell->vertex(0))
        , cell_size(cell->minimum_vertex_distance())
      {
        for (unsigned int d = 0; d < dim; ++d)
          {
            const dealii::Tensor<1, dim> edge = cell->vertex(1U << d) - origin;
            for (unsigned int i = 0; i < dim; ++i)
              jacobian[i][d] = edge[i];
          }

        // check that the cell is a parallelepiped
        for (unsigned int v = 0; v < cell->n_vertices(); ++v)
          {
            dealii::Point<dim> unit_vertex;
            for (unsigned int d = 0; d < dim; ++d)
              unit_vertex[d] = (v >> d) & 1U;
            AssertThrow((to_real(unit_vertex) - cell->vertex(v)).norm() <= 1e-10 * cell_size,
                        dealii::ExcMessage("The closest-point extension of the narrow-band level "
                                           "set requires parallelepiped cells."));
          }

        inverse_jacobian            = dealii::invert(jacobian);
        inverse_jacobian_transposed = dealii::transpose(inverse_jacobian);
      }

      /**
       * Constructor from the origin, the Jacobian and the cell size.
       */
      AffineCellMap(const dealii::Point<dim>     &origin,
                    const dealii::Tensor<2, dim> &jacobian,
                    const double                  cell_size)
        : origin(origin)
        , jacobian(jacobian)
        , inverse_jacobian(dealii::invert(jacobian))
        , inverse_jacobian_transposed(dealii::transpose(inverse_jacobian))
        , cell_size(cell_size)
      {}

      dealii::Point<dim>
      to_real(const dealii::Point<dim> &unit_point) const
      {
        return origin + jacobian * dealii::Tensor<1, dim>(unit_point);
      }

      dealii::Point<dim>
      to_unit(const dealii::Point<dim> &real_point) const
      {
        return dealii::Point<dim>(inverse_jacobian * (real_point - origin));
      }

      /**
       * Project a real point onto the cell (in unit coordinates onto [0,1]^dim).
       */
      dealii::Point<dim>
      clamp(const dealii::Point<dim> &real_point) const
      {
        dealii::Point<dim> unit_point = to_unit(real_point);
        for (unsigned int d = 0; d < dim; ++d)
          unit_point[d] = std::clamp(unit_point[d], 0., 1.);
        return to_real(unit_point);
      }
    };

    /**
     * Data of a cut cell for the closest-point search.
     */
    template <int dim, typename number>
    struct CutCellData
    {
      AffineCellMap<dim>              map;
      dealii::Vector<number>          values;
      std::vector<dealii::Point<dim>> interface_points;
      bool                            is_small = false;
    };

    /**
     * Tensor-product Lagrange basis of an FE_DGQ element (lexicographic numbering) for the
     * evaluation of the polynomial of a cell, its gradient and its Hessian at arbitrary points.
     */
    template <int dim>
    class TensorProductBasis
    {
    public:
      static constexpr unsigned int max_n_points_1d = 16;

      explicit TensorProductBasis(const dealii::FiniteElement<dim> &fe)
        : n_points_1d(fe.degree + 1)
      {
        AssertThrow(n_points_1d <= max_n_points_1d and
                      fe.n_dofs_per_cell() == dealii::Utilities::pow(n_points_1d, dim) and
                      fe.has_support_points(),
                    dealii::ExcNotImplemented());

        // 1d support points: the first n_points_1d support points (lexicographic numbering)
        std::vector<dealii::Point<1>> points_1d(n_points_1d);
        for (unsigned int i = 0; i < n_points_1d; ++i)
          points_1d[i][0] = fe.get_unit_support_points()[i][0];
        basis = dealii::Polynomials::generate_complete_Lagrange_basis(points_1d);

        // check the numbering of the shape functions
        dealii::Point<dim> point;
        for (unsigned int d = 0; d < dim; ++d)
          point[d] = 0.31 + 0.17 * d;
        dealii::Vector<double> unit_vector(fe.n_dofs_per_cell());
        for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
          {
            unit_vector                  = 0.;
            unit_vector[i]               = 1.;
            double                 value = 0.;
            dealii::Tensor<1, dim> gradient;
            evaluate(unit_vector, point, value, gradient, nullptr);
            AssertThrow(std::abs(value - fe.shape_value(i, point)) < 1e-10,
                        dealii::ExcMessage("The closest-point extension requires an element "
                                           "with a tensor-product Lagrange basis in "
                                           "lexicographic numbering (FE_DGQ)."));
          }
      }

      /**
       * Evaluate the polynomial with the coefficients @p values, its gradient and optionally its
       * Hessian (with respect to the unit coordinates) at @p unit_point.
       */
      void
      evaluate(const dealii::Vector<double> &values,
               const dealii::Point<dim>     &unit_point,
               double                       &value,
               dealii::Tensor<1, dim>       &gradient,
               dealii::Tensor<2, dim>       *hessian) const
      {
        using number = double;

        // values and first and second derivatives of the 1d basis functions
        std::array<std::array<std::array<double, 3>, max_n_points_1d>, dim> shape_1d;
        for (unsigned int d = 0; d < dim; ++d)
          for (unsigned int i = 0; i < n_points_1d; ++i)
            basis[i].value(unit_point[d], 2, shape_1d[d][i].data());

        value    = 0.;
        gradient = 0.;
        if (hessian != nullptr)
          *hessian = 0.;

        for (unsigned int index = 0; index < values.size(); ++index)
          {
            std::array<unsigned int, dim> i;
            for (unsigned int d = 0, remainder = index; d < dim; ++d, remainder /= n_points_1d)
              i[d] = remainder % n_points_1d;

            number shape_value = 1.;
            for (unsigned int d = 0; d < dim; ++d)
              shape_value *= shape_1d[d][i[d]][0];
            value += values[index] * shape_value;

            for (unsigned int k = 0; k < dim; ++k)
              {
                number shape_derivative = 1.;
                for (unsigned int d = 0; d < dim; ++d)
                  shape_derivative *= shape_1d[d][i[d]][d == k ? 1 : 0];
                gradient[k] += values[index] * shape_derivative;

                if (hessian != nullptr)
                  for (unsigned int l = 0; l < dim; ++l)
                    {
                      number shape_second_derivative = 1.;
                      for (unsigned int d = 0; d < dim; ++d)
                        shape_second_derivative *= shape_1d[d][i[d]][(d == k) + (d == l)];
                      (*hessian)[k][l] += values[index] * shape_second_derivative;
                    }
              }
          }
      }

    private:
      unsigned int                                         n_points_1d;
      std::vector<dealii::Polynomials::Polynomial<double>> basis;
    };

    /**
     * Evaluate the polynomial of a cut cell and its gradient at a real point.
     */
    template <int dim, typename number>
    std::pair<number, dealii::Tensor<1, dim, number>>
    evaluate(const TensorProductBasis<dim>  &basis,
             const CutCellData<dim, number> &cut_cell,
             const dealii::Point<dim>       &real_point)
    {
      number                         value = 0.;
      dealii::Tensor<1, dim, number> unit_gradient;
      basis.evaluate(
        cut_cell.values, cut_cell.map.to_unit(real_point), value, unit_gradient, nullptr);

      return {value, cut_cell.map.inverse_jacobian_transposed * unit_gradient};
    }

    /**
     * Evaluate the polynomial of a cut cell, its gradient and its Hessian at a real point.
     */
    template <int dim, typename number>
    std::tuple<number, dealii::Tensor<1, dim, number>, dealii::Tensor<2, dim, number>>
    evaluate_with_hessian(const TensorProductBasis<dim>  &basis,
                          const CutCellData<dim, number> &cut_cell,
                          const dealii::Point<dim>       &real_point)
    {
      number                         value = 0.;
      dealii::Tensor<1, dim, number> unit_gradient;
      dealii::Tensor<2, dim, number> unit_hessian;
      basis.evaluate(
        cut_cell.values, cut_cell.map.to_unit(real_point), value, unit_gradient, &unit_hessian);

      const auto &inverse_jacobian            = cut_cell.map.inverse_jacobian;
      const auto &inverse_jacobian_transposed = cut_cell.map.inverse_jacobian_transposed;

      return {value,
              inverse_jacobian_transposed * unit_gradient,
              inverse_jacobian_transposed * unit_hessian * inverse_jacobian};
    }

    /**
     * Solve the small dense linear system A x = b by Gaussian elimination with partial pivoting.
     * Returns false if the matrix is (numerically) singular.
     */
    template <int n, typename number>
    bool
    solve_dense(std::array<std::array<number, n>, n> A, std::array<number, n> &b)
    {
      number scale = 0.;
      for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j)
          scale = std::max(scale, std::abs(A[i][j]));
      if (scale == 0.)
        return false;

      for (unsigned int k = 0; k < n; ++k)
        {
          unsigned int pivot = k;
          for (unsigned int i = k + 1; i < n; ++i)
            if (std::abs(A[i][k]) > std::abs(A[pivot][k]))
              pivot = i;
          if (std::abs(A[pivot][k]) <= 1e-14 * scale)
            return false;
          std::swap(A[k], A[pivot]);
          std::swap(b[k], b[pivot]);

          for (unsigned int i = k + 1; i < n; ++i)
            {
              const number factor = A[i][k] / A[k][k];
              for (unsigned int j = k; j < n; ++j)
                A[i][j] -= factor * A[k][j];
              b[i] -= factor * b[k];
            }
        }

      for (int i = n - 1; i >= 0; --i)
        {
          for (unsigned int j = i + 1; j < n; ++j)
            b[i] -= A[i][j] * b[j];
          b[i] /= A[i][i];
        }

      return true;
    }
    /**
     * Search the closest point of @p x on the interface of a cut cell, starting from the nearest
     * quadrature point of the interface. Returns the closest point, the (estimated) distance to the
     * interface and the number of iterations.
     */
    template <int dim, typename number>
    std::tuple<dealii::Point<dim>, number, unsigned int>
    search_closest_point(const TensorProductBasis<dim>  &basis,
                         const CutCellData<dim, number> &cut_cell,
                         const dealii::Point<dim>       &x)
    {
      // maximum number of iterations and relative tolerance of the closest-point search
      constexpr unsigned int max_iterations = 20;
      constexpr number       tolerance      = 1e-10;

      dealii::Point<dim> y = cut_cell.map.clamp(x);
      if (not cut_cell.interface_points.empty())
        {
          y = cut_cell.interface_points.front();
          for (const auto &point : cut_cell.interface_points)
            if (x.distance_square(point) < x.distance_square(y))
              y = point;
        }

      const number step_tolerance = tolerance * cut_cell.map.cell_size;

      // (a) projection of the starting point onto the zero level set
      for (unsigned int k = 0; k < 3; ++k)
        {
          const auto [value, gradient] = evaluate(basis, cut_cell, y);
          if (gradient.norm_square() == 0.)
            break;
          y = cut_cell.map.clamp(y - (value / gradient.norm_square()) * gradient);
        }

      // Estimate of the distance of x to the interface via y: distance to y plus the distance of y
      // to the interface (first-order estimate). If the iteration does not converge (e.g., if the
      // closest point of the interface within the cell is located at the boundary of the cell or
      // near a ridge of the distance function), the iterate with the smallest estimate is used.
      const auto distance_estimate = [&](const dealii::Point<dim> &point) {
        const auto [value, gradient] = evaluate(basis, cut_cell, point);
        return x.distance(point) + (gradient.norm() > 0. ? std::abs(value) / gradient.norm() :
                                                           std::numeric_limits<number>::max());
      };

      dealii::Point<dim> y_best        = y;
      number             distance_best = distance_estimate(y);

      // (b) Newton's method for the first-order optimality conditions of the minimization of
      // |x - y|^2 / 2 subject to phi(y) = 0 (Saye, 2014): y - x + lambda grad phi(y) = 0 and
      // phi(y) = 0, starting with the Lagrange multiplier of the projected point.
      number lambda = 0.;
      {
        const auto [value, gradient] = evaluate(basis, cut_cell, y);
        (void)value;
        if (gradient.norm_square() > 0.)
          lambda = ((x - y) * gradient) / gradient.norm_square();
      }

      // number of consecutive iterations, in which the iterate has been clamped to the cell
      unsigned int n_clamped = 0;
      bool         converged = false;

      unsigned int iteration = 0;
      for (; iteration < max_iterations; ++iteration)
        {
          const auto [value, gradient, hessian] = evaluate_with_hessian(basis, cut_cell, y);
          const number gradient_norm_square     = gradient.norm_square();
          if (gradient_norm_square == 0.)
            break;

          // KKT system [I + lambda H, grad phi; grad phi^T, 0] [dy; dlambda] = -[r; phi]
          std::array<std::array<number, dim + 1>, dim + 1> matrix{};
          std::array<number, dim + 1>                      rhs{};
          for (unsigned int i = 0; i < dim; ++i)
            {
              for (unsigned int j = 0; j < dim; ++j)
                matrix[i][j] = (i == j ? 1. : 0.) + lambda * hessian[i][j];
              matrix[i][dim] = gradient[i];
              matrix[dim][i] = gradient[i];
              rhs[i]         = -(y[i] - x[i] + lambda * gradient[i]);
            }
          rhs[dim] = -value;

          dealii::Point<dim> y_new;
          if (solve_dense<dim + 1>(matrix, rhs))
            {
              dealii::Tensor<1, dim, number> dy;
              for (unsigned int i = 0; i < dim; ++i)
                dy[i] = rhs[i];
              number dlambda = rhs[dim];

              // damping: limit the step to half the cell size
              const number max_step = 0.5 * cut_cell.map.cell_size;
              if (dy.norm() > max_step)
                {
                  const number factor = max_step / dy.norm();
                  dy *= factor;
                  dlambda *= factor;
                }

              y_new = cut_cell.map.clamp(y + dy);
              lambda += dlambda;

              // the iterate has been clamped to the cell: restart the multiplier
              if ((y_new - (y + dy)).norm() > step_tolerance)
                {
                  ++n_clamped;
                  const auto [value_new, gradient_new] = evaluate(basis, cut_cell, y_new);
                  (void)value_new;
                  if (gradient_new.norm_square() > 0.)
                    lambda = ((x - y_new) * gradient_new) / gradient_new.norm_square();
                }
              else
                n_clamped = 0;
            }
          else
            {
              // singular system (e.g., at a focal point): projection onto the zero level set and
              // tangential step towards the foot point of x
              const dealii::Point<dim> y_projected = y - (value / gradient_norm_square) * gradient;
              const dealii::Tensor<1, dim, number> normal =
                gradient / std::sqrt(gradient_norm_square);
              const dealii::Tensor<1, dim, number> direction = x - y_projected;
              y_new  = cut_cell.map.clamp(y_projected + direction - (direction * normal) * normal);
              lambda = ((x - y_new) * gradient) / gradient_norm_square;
            }

          const number step = y_new.distance(y);
          y                 = y_new;

          if (const number distance = distance_estimate(y); distance < distance_best)
            {
              y_best        = y;
              distance_best = distance;
            }

          if (step <= step_tolerance)
            {
              converged = true;
              break;
            }

          // The closest point of the interface within the cell is located at the boundary of the
          // cell, i.e., the closest point is found in a neighboring cut cell. The iterate is
          // accepted as an approximation.
          if (n_clamped >= 3)
            break;
        }

      // final projection onto the zero level set
      {
        const auto [value, gradient] = evaluate(basis, cut_cell, y);
        if (gradient.norm_square() > 0.)
          y = cut_cell.map.clamp(y - (value / gradient.norm_square()) * gradient);

        const number distance = distance_estimate(y);
        if (converged or distance < distance_best)
          {
            y_best        = y;
            distance_best = distance;
          }
      }

      return std::make_tuple(y_best, distance_best, iteration);
    }
  } // namespace

  template <int dim, typename number>
  NarrowBandClosestPointExtension<dim, number>::NarrowBandClosestPointExtension(
    const typename NarrowBandData<number>::Extension &data,
    const dealii::Mapping<dim>                       &mapping,
    const dealii::ConditionalOStream                 &pcout)
    : data(data)
    , mapping(mapping)
    , pcout(pcout)
  {}

  template <int dim, typename number>
  unsigned int
  NarrowBandClosestPointExtension<dim, number>::extend(const dealii::DoFHandler<dim> &dof_handler,
                                                       VectorType &level_set) const
  {
    using namespace NarrowBand;

    const auto &fe_collection = dof_handler.get_fe_collection();
    AssertThrow(dof_handler.has_hp_capabilities() and fe_collection.size() == 3 and
                  fe_collection[ActiveFEIndex::cut].n_dofs_per_cell() > 0 and
                  fe_collection[ActiveFEIndex::neighbor].n_dofs_per_cell() > 0,
                dealii::ExcMessage("The closest-point extension requires the band DoFHandler of "
                                   "the narrow-band level set."));
    AssertThrow(not dof_handler.get_triangulation().has_hanging_nodes(),
                dealii::ExcMessage("The narrow-band level set does not support hanging nodes."));

    const auto        &fe_cut      = fe_collection[ActiveFEIndex::cut];
    const auto        &fe_neighbor = fe_collection[ActiveFEIndex::neighbor];
    const unsigned int fe_degree   = fe_cut.degree;

    AssertThrow(fe_neighbor.has_support_points(),
                dealii::ExcMessage("The closest-point extension requires a nodal element."));

    const TensorProductBasis<dim> basis(fe_cut);

    const MPI_Comm mpi_comm = dof_handler.get_mpi_communicator();

    // The values of the cut cells of the ghost layer are required as well.
    VectorType level_set_relevant(dof_handler.locally_owned_dofs(),
                                  dealii::DoFTools::extract_locally_relevant_dofs(dof_handler),
                                  mpi_comm);
    level_set_relevant.copy_locally_owned_data_from(level_set);
    level_set_relevant.update_ghost_values();

    // 1) Collect the data of the locally relevant cut cells (locally owned and ghost cells): DoF
    // values, affine map and quadrature points of the discrete interface (starting points of the
    // closest-point search).
    std::vector<CutCellData<dim, number>>  cut_cells;
    std::vector<std::vector<unsigned int>> vertex_to_cut_cells(
      dof_handler.get_triangulation().n_vertices());
    {
      const dealii::hp::QCollection<1> q_collection(dealii::QGauss<1>(fe_degree + 1));
      dealii::NonMatching::DiscreteQuadratureGenerator<dim> quadrature_generator(
        q_collection, dof_handler, level_set_relevant);

      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (cell->is_artificial() or cell->active_fe_index() != ActiveFEIndex::cut)
            continue;

          CutCellData<dim, number> cut_cell{AffineCellMap<dim>(cell),
                                            dealii::Vector<number>(fe_cut.n_dofs_per_cell()),
                                            {}};
          cell->get_dof_values(level_set_relevant, cut_cell.values);

          quadrature_generator.generate(cell);
          for (const auto &unit_point : quadrature_generator.get_surface_quadrature().get_points())
            cut_cell.interface_points.push_back(cut_cell.map.to_real(unit_point));

          // volume fraction of the positive phase (reference cell with unit volume)
          if (data.minimum_source_volume_fraction > 0.)
            {
              number volume_fraction = 0.;
              for (const auto weight : quadrature_generator.get_outside_quadrature().get_weights())
                volume_fraction += weight;
              cut_cell.is_small = volume_fraction < data.minimum_source_volume_fraction;
            }

          // Note: a cut cell might not contain quadrature points of the interface after the
          // reinitialization. Such cells are still considered, the search starts from the point
          // of the cell nearest to the node.
          for (const unsigned int v : cell->vertex_indices())
            vertex_to_cut_cells[cell->vertex_index(v)].push_back(cut_cells.size());

          cut_cells.emplace_back(std::move(cut_cell));
        }
    }

    // 2) Closest-point search for the nodes of the locally owned neighbor cells.
    const auto &unit_support_points = fe_neighbor.get_unit_support_points();

    unsigned int max_n_iterations = 0;

    dealii::Vector<number>    values(fe_neighbor.n_dofs_per_cell());
    std::vector<unsigned int> candidates;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or cell->active_fe_index() != ActiveFEIndex::neighbor)
          continue;

        // candidates: cut cells sharing a vertex with the neighbor cell
        candidates.clear();
        for (const unsigned int v : cell->vertex_indices())
          for (const unsigned int c : vertex_to_cut_cells[cell->vertex_index(v)])
            candidates.push_back(c);
        std::sort(candidates.begin(), candidates.end());
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

        AssertThrow(not candidates.empty(),
                    dealii::ExcMessage("A neighbor cell of the narrow band does not share a vertex "
                                       "with a cut cell."));

        // small cut cells are only considered, if no other cut cells are available
        if (std::any_of(candidates.begin(), candidates.end(), [&](const unsigned int c) {
              return not cut_cells[c].is_small;
            }))
          candidates.erase(std::remove_if(candidates.begin(),
                                          candidates.end(),
                                          [&](const unsigned int c) {
                                            return cut_cells[c].is_small;
                                          }),
                           candidates.end());

        const AffineCellMap<dim> map(cell);

        for (unsigned int i = 0; i < fe_neighbor.n_dofs_per_cell(); ++i)
          {
            const dealii::Point<dim> x = map.to_real(unit_support_points[i]);

            number min_distance = std::numeric_limits<number>::max();
            for (const unsigned int c : candidates)
              {
                const auto [y, distance, n_iterations] =
                  search_closest_point(basis, cut_cells[c], x);
                max_n_iterations = std::max(max_n_iterations, n_iterations);

                if (distance < min_distance)
                  {
                    min_distance = distance;

                    // first-order Taylor expansion from the closest point
                    const auto [value, gradient] = evaluate(basis, cut_cells[c], y);
                    values[i]                    = value + gradient * (x - y);
                  }
              }
          }

        cell->set_dof_values(values, level_set);
      }

    max_n_iterations = dealii::Utilities::MPI::max(max_n_iterations, mpi_comm);

    const bool         use_sign_limiter = data.sign_limiter_tolerance >= 0.;
    const unsigned int n_limited_cells =
      use_sign_limiter ?
        limit_sign_changes(mapping, dof_handler, level_set, data.sign_limiter_tolerance) :
        0;

    {
      std::ostringstream str;
      str << "closest-point extension to "
          << n_cells_with_active_fe_index(dof_handler, ActiveFEIndex::neighbor)
          << " neighbor cells: max. " << max_n_iterations << " iterations";
      if (use_sign_limiter)
        str << ", " << n_limited_cells << " limited cells";
      Journal::print_line(pcout, str.str(), "narrow_band");
    }

    return max_n_iterations;
  }

  template <int dim, typename number>
  unsigned int
  NarrowBandClosestPointExtension<dim, number>::redistance(
    const dealii::DoFHandler<dim> &dof_handler,
    VectorType                    &level_set,
    const std::vector<bool>       &cell_flags,
    const std::string             &label) const
  {
    using namespace NarrowBand;

    const auto &fe_collection = dof_handler.get_fe_collection();
    AssertThrow(dof_handler.has_hp_capabilities() and fe_collection.size() > ActiveFEIndex::cut and
                  fe_collection[ActiveFEIndex::cut].n_dofs_per_cell() > 0,
                dealii::ExcMessage("The redistancing requires the band DoFHandler of the "
                                   "narrow-band level set."));
    AssertDimension(cell_flags.size(), dof_handler.get_triangulation().n_active_cells());

    const auto        &fe_cut    = fe_collection[ActiveFEIndex::cut];
    const unsigned int fe_degree = fe_cut.degree;
    const unsigned int n_dofs    = fe_cut.n_dofs_per_cell();

    const TensorProductBasis<dim> basis(fe_cut);

    const MPI_Comm mpi_comm = dof_handler.get_mpi_communicator();

    // 1) Data of the locally owned cut cells (cell size, origin, Jacobian, DoF values and
    // quadrature points of the interface), packed into a vector.
    std::vector<double> local_data;
    {
      const bool update_ghosts = not level_set.has_ghost_elements();
      if (update_ghosts)
        level_set.update_ghost_values();

      const dealii::hp::QCollection<1> q_collection(dealii::QGauss<1>(fe_degree + 1));
      dealii::NonMatching::DiscreteQuadratureGenerator<dim> quadrature_generator(q_collection,
                                                                                 dof_handler,
                                                                                 level_set);
      dealii::Vector<number>                                values(n_dofs);

      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (not cell->is_locally_owned() or cell->active_fe_index() != ActiveFEIndex::cut)
            continue;

          quadrature_generator.generate(cell);
          const auto &points = quadrature_generator.get_surface_quadrature().get_points();
          if (points.empty())
            continue;

          const AffineCellMap<dim> map(cell);
          cell->get_dof_values(level_set, values);

          local_data.push_back(map.cell_size);
          for (unsigned int d = 0; d < dim; ++d)
            local_data.push_back(map.origin[d]);
          for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
              local_data.push_back(map.jacobian[i][j]);
          for (unsigned int i = 0; i < n_dofs; ++i)
            local_data.push_back(values[i]);
          local_data.push_back(points.size());
          for (const auto &unit_point : points)
            {
              const dealii::Point<dim> point = map.to_real(unit_point);
              for (unsigned int d = 0; d < dim; ++d)
                local_data.push_back(point[d]);
            }
        }

      if (update_ghosts)
        level_set.zero_out_ghost_values();
    }

    // 2) Data of all cut cells of all processes. The interface is small compared to the domain.
    std::vector<CutCellData<dim, number>>                    cut_cells;
    std::vector<std::pair<dealii::Point<dim>, unsigned int>> interface_points;
    for (const auto &data_of_process : dealii::Utilities::MPI::all_gather(mpi_comm, local_data))
      for (std::size_t k = 0; k < data_of_process.size();)
        {
          const double       cell_size = data_of_process[k++];
          dealii::Point<dim> origin;
          for (unsigned int d = 0; d < dim; ++d)
            origin[d] = data_of_process[k++];
          dealii::Tensor<2, dim> jacobian;
          for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
              jacobian[i][j] = data_of_process[k++];

          CutCellData<dim, number> cut_cell{AffineCellMap<dim>(origin, jacobian, cell_size),
                                            dealii::Vector<number>(n_dofs),
                                            {}};
          for (unsigned int i = 0; i < n_dofs; ++i)
            cut_cell.values[i] = data_of_process[k++];

          const unsigned int n_points = static_cast<unsigned int>(data_of_process[k++]);
          for (unsigned int q = 0; q < n_points; ++q)
            {
              dealii::Point<dim> point;
              for (unsigned int d = 0; d < dim; ++d)
                point[d] = data_of_process[k++];
              cut_cell.interface_points.push_back(point);
              interface_points.emplace_back(point, cut_cells.size());
            }

          cut_cells.emplace_back(std::move(cut_cell));
        }

    unsigned int n_redistanced_cells = 0;
    unsigned int max_n_iterations    = 0;

    if (not interface_points.empty())
      {
        const auto tree = dealii::pack_rtree(interface_points);

        // number of nearest quadrature points of the interface, whose cut cells are candidates of
        // the closest-point search
        const unsigned int n_nearest_points = dim == 1 ? 2 : (dim == 2 ? 8 : 24);

        std::vector<std::pair<dealii::Point<dim>, unsigned int>> nearest_points;
        std::vector<unsigned int>                                candidates;

        // 3) Closest-point search for the nodes of the flagged locally owned cells.
        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            if (not cell->is_locally_owned() or not cell_flags[cell->active_cell_index()] or
                cell->active_fe_index() == ActiveFEIndex::cut or
                cell->get_fe().n_dofs_per_cell() == 0)
              continue;

            AssertThrow(cell->get_fe().has_support_points(),
                        dealii::ExcMessage("The redistancing requires a nodal element."));

            const auto              &unit_support_points = cell->get_fe().get_unit_support_points();
            const AffineCellMap<dim> map(cell);
            dealii::Vector<number>   values(cell->get_fe().n_dofs_per_cell());

            for (unsigned int i = 0; i < values.size(); ++i)
              {
                const dealii::Point<dim> x = map.to_real(unit_support_points[i]);

                nearest_points.clear();
                tree.query(boost::geometry::index::nearest(x, n_nearest_points),
                           std::back_inserter(nearest_points));

                candidates.clear();
                for (const auto &point : nearest_points)
                  candidates.push_back(point.second);
                std::sort(candidates.begin(), candidates.end());
                candidates.erase(std::unique(candidates.begin(), candidates.end()),
                                 candidates.end());

                number min_distance = std::numeric_limits<number>::max();
                for (const unsigned int c : candidates)
                  {
                    const auto [y, distance, n_iterations] =
                      search_closest_point(basis, cut_cells[c], x);
                    max_n_iterations = std::max(max_n_iterations, n_iterations);

                    if (distance < min_distance)
                      {
                        min_distance = distance;

                        // first-order Taylor expansion from the closest point
                        const auto [value, gradient] = evaluate(basis, cut_cells[c], y);
                        values[i]                    = value + gradient * (x - y);
                      }
                  }
              }

            cell->set_dof_values(values, level_set);
            ++n_redistanced_cells;
          }
      }

    n_redistanced_cells = dealii::Utilities::MPI::sum(n_redistanced_cells, mpi_comm);
    max_n_iterations    = dealii::Utilities::MPI::max(max_n_iterations, mpi_comm);

    {
      std::ostringstream str;
      str << "redistancing (" << label << ") of " << n_redistanced_cells << " cells: max. "
          << max_n_iterations << " iterations";
      Journal::print_line(pcout, str.str(), "narrow_band");
    }

    return n_redistanced_cells;
  }

  template class NarrowBandClosestPointExtension<1, double>;
  template class NarrowBandClosestPointExtension<2, double>;
  template class NarrowBandClosestPointExtension<3, double>;
} // namespace MeltPoolDG::LevelSet
