#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>

#include <deal.II/fe/fe.h>

#include <deal.II/lac/vector.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

/**
 * Internal tools of the closest-point extension, the geometric redistancing and the
 * semi-Lagrangian advection of the narrow-band level set: affine maps of parallelepiped cells,
 * the evaluation of the polynomials of FE_DGQ elements at arbitrary points and the closest-point
 * search on the discrete interface of a cut cell.
 */
namespace MeltPoolDG::LevelSet::NarrowBand::internal
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
} // namespace MeltPoolDG::LevelSet::NarrowBand::internal
