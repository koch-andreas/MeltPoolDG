
#include <meltpooldg/flow/compressible_flow_boundary_conditions.hpp>
#include <meltpooldg/flow/compressible_flow_eos_utils_base.hpp>
#include <meltpooldg/flow/compressible_flow_ideal_gas_utils.hpp>
#include <meltpooldg/flow/compressible_flow_noble_abel_stiffened_gas_utils.hpp>
#include <meltpooldg/flow/compressible_flow_stiffened_gas_utils.hpp>
#include <meltpooldg/flow/compressible_flow_views.hpp>
#include <meltpooldg/utilities/dealii_tensor.hpp>
#include <meltpooldg/utilities/vector_tools.templates.hpp>

namespace MeltPoolDG::Flow
{
  using namespace dealii;

  template <int dim, typename number>
  void
  CompressibleFlowBoundaryConditions<dim, number>::update_boundary_conditions(const number time)
  {
    for (auto &i : this->inflow_boundaries)
      i.second->set_time(time);
    for (auto &i : this->subsonic_outflow_fixed_pressure)
      i.second->set_time(time);
    for (auto &i : this->subsonic_outflow_fixed_energy)
      i.second->set_time(time);
  }

  template <int dim, typename number>
  void
  CompressibleFlowBoundaryConditions<dim, number>::set_boundary_condition(
    const BoundaryType boundary_condition,
    const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      boundary_condition_function)
  {
    if (not boundary_condition_function.empty())
      {
        if (boundary_condition == BoundaryType::inflow)
          inflow_boundaries = boundary_condition_function;
        else if (boundary_condition == BoundaryType::subsonic_outflow_fixed_energy)
          subsonic_outflow_fixed_energy = boundary_condition_function;
        else if (boundary_condition == BoundaryType::subsonic_outflow_fixed_pressure)
          subsonic_outflow_fixed_pressure = boundary_condition_function;
        else if (boundary_condition == BoundaryType::slip_wall)
          for (const auto &boundary : boundary_condition_function)
            slip_wall_boundaries.insert(boundary.first);
        else if (boundary_condition == BoundaryType::no_slip_wall)
          for (const auto &boundary : boundary_condition_function)
            no_slip_adiabatic_wall_boundaries.insert(boundary.first);
        else
          AssertThrow(false,
                      dealii::ExcMessage("The provided boundary condition is not supported."));
      }
  }

  template <int dim, typename number>
  dealii::Tensor<1, dim + 2, dealii::VectorizedArray<number>>
  CompressibleFlowBoundaryConditions<dim, number>::get_boundary_value(
    const dealii::types::boundary_id                           boundary_id,
    const BoundaryType                                         boundary_condition,
    const dealii::Point<dim, dealii::VectorizedArray<number>> &location) const
  {
    if (boundary_condition == BoundaryType::inflow)
      return VectorTools::evaluate_function_at_vectorized_points<dim, number, dim + 2>(
        *inflow_boundaries.find(boundary_id)->second, location);
    if (boundary_condition == BoundaryType::subsonic_outflow_fixed_energy)
      return VectorTools::evaluate_function_at_vectorized_points<dim, number, dim + 2>(
        *subsonic_outflow_fixed_energy.find(boundary_id)->second, location);
    if (boundary_condition == BoundaryType::subsonic_outflow_fixed_pressure)
      return VectorTools::evaluate_function_at_vectorized_points<dim, number, dim + 2>(
        *subsonic_outflow_fixed_pressure.find(boundary_id)->second, location);

    AssertThrow(false,
                dealii::ExcMessage("The boundary condition " + std::to_string(boundary_condition) +
                                   " does not have a prescribed boundary value."));
  }

  template <int dim, typename number>
  dealii::VectorizedArray<number>
  CompressibleFlowBoundaryConditions<dim, number>::get_boundary_value(
    const dealii::types::boundary_id                           boundary_id,
    const BoundaryType                                         boundary_condition,
    const dealii::Point<dim, dealii::VectorizedArray<number>> &location,
    const unsigned                                             component) const
  {
    if (boundary_condition == BoundaryType::inflow)
      return VectorTools::evaluate_function_at_vectorized_points<dim, number>(
        *inflow_boundaries.find(boundary_id)->second, location, component);
    if (boundary_condition == BoundaryType::subsonic_outflow_fixed_energy)
      return VectorTools::evaluate_function_at_vectorized_points<dim, number>(
        *subsonic_outflow_fixed_energy.find(boundary_id)->second, location, component);
    if (boundary_condition == BoundaryType::subsonic_outflow_fixed_pressure)
      return VectorTools::evaluate_function_at_vectorized_points<dim, number>(
        *subsonic_outflow_fixed_pressure.find(boundary_id)->second, location, component);

    AssertThrow(false,
                dealii::ExcMessage("The boundary condition " + std::to_string(boundary_condition) +
                                   " does not have a prescribed boundary value."));
  }

  template <int dim, typename number>
  auto
  CompressibleFlowBoundaryConditions<dim, number>::get_boundary_face_value_and_gradient(
    const dealii::Point<dim, dealii::VectorizedArray<number>>     &q_point,
    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
    const dealii::types::boundary_id                               boundary_id,
    const ConservedVariablesType                                  &w_m,
    const ConservedVariablesGradType                              &grad_w_m,
    const CompressibleFlowMaterial<dim, number>                   &material,
    const bool is_gas_phase) const -> std::tuple<ConservedVariablesType, ConservedVariablesGradType>
  {
    using namespace dealii;

    ConservedVariablesType     w_p;
    ConservedVariablesGradType grad_w_p;
    if (const BoundaryType boundary_type = get_boundary_type(boundary_id);
        boundary_type == BoundaryType::slip_wall)
      {
        // homogeneous Neumann
        auto rho_u_dot_n = w_m[1] * normal[0];
        for (unsigned int d = 1; d < dim; ++d)
          rho_u_dot_n += w_m[1 + d] * normal[d];
        w_p[0]      = w_m[0];
        grad_w_p[0] = -(grad_w_m[0]);
        // symmetry
        for (unsigned int d = 0; d < dim; ++d)
          {
            w_p[d + 1] = w_m[d + 1] - 2. * rho_u_dot_n * normal[d];
            grad_w_p[d + 1] =
              grad_w_m[d + 1] - 2. * scalar_product(grad_w_m[d + 1], normal) * normal;
          }
        // homogeneous Neumann
        grad_w_p[dim + 1] = -(grad_w_m[dim + 1]);
        w_p[dim + 1]      = w_m[dim + 1];
      }
    else if (boundary_type == BoundaryType::no_slip_wall)
      {
        // homogeneous Neumann
        w_p[0]      = w_m[0];
        grad_w_p[0] = -(grad_w_m[0]);
        // Dirichlet
        for (unsigned int d = 0; d < dim; ++d)
          {
            w_p[d + 1]      = 0.;
            grad_w_p[d + 1] = grad_w_m[d + 1];
          }
        // homogeneous Neumann
        grad_w_p[dim + 1] = -(grad_w_m[dim + 1]);
        w_p[dim + 1]      = w_m[dim + 1];
      }
    else if (boundary_type == BoundaryType::inflow)
      {
        const unsigned int component_offset = is_gas_phase ? 0 : dim + 2;
        // Dirichlet
        for (unsigned int i = 0; i < dim + 2; ++i)
          w_p[i] =
            get_boundary_value(boundary_id, BoundaryType::inflow, q_point, i + component_offset);
        grad_w_p = grad_w_m;
      }
    else if (boundary_type == BoundaryType::subsonic_outflow_fixed_pressure)
      {
        // homogeneous Neumann
        w_p      = w_m;
        grad_w_p = -grad_w_m;

        // Dirichlet
        auto p_dyn = VectorizedArray<number>(0.);
        for (unsigned int i = 1; i < dim + 1; ++i)
          p_dyn += w_m[i] * w_m[i];

        p_dyn /= (w_m[0] * 2.);
        const unsigned int            component_offset = is_gas_phase ? 0 : 1;
        const VectorizedArray<number> pressure         = get_boundary_value(
          boundary_id, BoundaryType::subsonic_outflow_fixed_pressure, q_point, component_offset);

        // consider equation of state for computation of inner energy from given pressure
        const VectorizedArray<number> inner_energy =
          material.eos_utils->compute_inner_energy_from_pressure(pressure, w_p[0]);

        w_p[dim + 1]      = inner_energy + p_dyn;
        grad_w_p[dim + 1] = grad_w_m[dim + 1];
      }
    else if (boundary_type == BoundaryType::subsonic_outflow_fixed_energy)
      {
        // homogeneous Neumann
        w_p      = w_m;
        grad_w_p = -grad_w_m;
        // Dirichlet
        const unsigned int component_offset = is_gas_phase ? 0 : 1;
        w_p[dim + 1]                        = get_boundary_value(boundary_id,
                                          BoundaryType::subsonic_outflow_fixed_energy,
                                          q_point,
                                          component_offset);
        grad_w_p[dim + 1]                   = grad_w_m[dim + 1];
      }
    else
      {
        std::cout << "ID: " << boundary_id << std::endl;
        std::cout << "Condition:" << boundary_type << std::endl;
        AssertThrow(false,
                    dealii::ExcMessage("Unknown boundary id, did "
                                       "you set a boundary condition for "
                                       "this part of the domain boundary?"));
      }

    return {w_p, grad_w_p};
  }

  template <int dim, typename number>
  auto
  CompressibleFlowBoundaryConditions<dim, number>::get_boundary_face_value_and_gradient(
    const dealii::Point<dim, dealii::VectorizedArray<number>>     &q_point,
    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
    dealii::types::boundary_id                                     boundary_id,
    CompressibleFlow::DofValueAndGradientStateView<
      dim,
      number,
      const CompressibleFlow::ConservedVariablesType<dim, number>,
      const CompressibleFlow::ConservedVariablesGradientType<dim, number>> w_m_view) const
    -> std::tuple<ConservedVariablesType, ConservedVariablesGradType>
  {
    CompressibleFlow::ConservedVariablesType<dim, number>         w_p;
    CompressibleFlow::ConservedVariablesGradientType<dim, number> grad_w_p;

    CompressibleFlow::DofValueView<dim, CompressibleFlow::ConservedVariablesType<dim, number>>
      w_p_view(w_p);
    CompressibleFlow::DofGradientView<dim,
                                      CompressibleFlow::ConservedVariablesGradientType<dim, number>>
      grad_w_p_view(grad_w_p);

    if (const BoundaryType boundary_type = get_boundary_type(boundary_id);
        boundary_type == BoundaryType::slip_wall)
      {
        // homogeneous Neumann
        w_p_view.density()           = w_m_view.density();
        grad_w_p_view.grad_density() = -(w_m_view.grad_density());

        // homogeneous Neumann
        VectorizedArrayType rho_u_dot_n = 0;
        for (unsigned int d = 0; d < dim; ++d)
          rho_u_dot_n += w_m_view.momentum(d) * normal[d];
        // symmetry
        for (unsigned int d = 0; d < dim; ++d)
          {
            w_p_view.momentum(d) = w_m_view.momentum(d) - 2. * rho_u_dot_n * normal[d];
            grad_w_p_view.grad_momentum(d) =
              w_m_view.grad_momentum(d) -
              2. * scalar_product(w_m_view.grad_momentum(d), normal) * normal;
          }
        // homogeneous Neumann
        grad_w_p_view.grad_total_energy() = -(w_m_view.grad_total_energy());
        w_p_view.total_energy()           = w_m_view.total_energy();
      }
    else if (boundary_type == BoundaryType::no_slip_wall)
      {
        // homogeneous Neumann
        w_p_view.density()           = w_m_view.density();
        grad_w_p_view.grad_density() = -(w_m_view.grad_density());

        // Dirichlet
        for (unsigned int d = 0; d < dim; ++d)
          {
            w_p_view.momentum(d)           = 0.;
            grad_w_p_view.grad_momentum(d) = w_m_view.grad_momentum(d);
          }
        // homogeneous Neumann
        grad_w_p_view.grad_total_energy() = (w_m_view.grad_total_energy());
        w_p_view.total_energy()           = 70000.;
      }
    else if (boundary_type == BoundaryType::inflow)
      {
        w_p_view.density() = get_boundary_value(boundary_id, BoundaryType::inflow, q_point, 0);
        grad_w_p_view.grad_density() = w_m_view.grad_density();

        for (unsigned int d = 0; d < dim; ++d)
          {
            w_p_view.momentum(d) =
              get_boundary_value(boundary_id, BoundaryType::inflow, q_point, d + 1);
            grad_w_p_view.grad_momentum(d) = w_m_view.grad_momentum(d);
          }

        w_p_view.total_energy() =
          get_boundary_value(boundary_id, BoundaryType::inflow, q_point, dim + 1);
        grad_w_p_view.grad_total_energy() = w_m_view.grad_total_energy();
      }
    else if (boundary_type == BoundaryType::subsonic_outflow_fixed_pressure)
      {
        // homogeneous Neumann
        w_p_view.density()           = w_m_view.density();
        grad_w_p_view.grad_density() = -(w_m_view.grad_density());

        // Dirichlet
        auto p_dyn = VectorizedArray<number>(0.);
        for (unsigned int i = 0; i < dim; ++i)
          p_dyn += w_m_view.momentum(i) * w_m_view.momentum(i);

        p_dyn /= (w_m_view.density() * 2.);
        const VectorizedArray<number> pressure = get_boundary_value(
          boundary_id, BoundaryType::subsonic_outflow_fixed_pressure, q_point, 0);

        // consider equation of state for computation of inner energy from given pressure
        const VectorizedArray<number> inner_energy = w_m_view.inner_energy_from_pressure(pressure);

        w_p_view.total_energy()           = inner_energy + p_dyn;
        grad_w_p_view.grad_total_energy() = w_m_view.grad_total_energy();
      }
    else if (boundary_type == BoundaryType::subsonic_outflow_fixed_energy)
      {
        // homogeneous Neumann
        w_p_view.density()           = w_m_view.density();
        grad_w_p_view.grad_density() = -(w_m_view.grad_density());
        for (unsigned int i = 0; i < dim; ++i)
          {
            w_p_view.momentum(i)           = w_m_view.momentum(i);
            grad_w_p_view.grad_momentum(i) = -(w_m_view.grad_momentum(i));
          }

        // Dirichlet
        w_p_view.total_energy() =
          get_boundary_value(boundary_id, BoundaryType::subsonic_outflow_fixed_energy, q_point, 0);
        grad_w_p_view.grad_total_energy() = w_m_view.grad_total_energy();
      }
    else
      {
        std::cout << "ID: " << boundary_id << std::endl;
        std::cout << "Condition:" << boundary_type << std::endl;
        AssertThrow(false,
                    dealii::ExcMessage("Unknown boundary id, did "
                                       "you set a boundary condition for "
                                       "this part of the domain boundary?"));
      }

    return {w_p, grad_w_p};
  }

  template <int dim, typename number>
  auto
  CompressibleFlowBoundaryConditions<dim, number>::get_jacobian_boundary_face_value_and_gradient(
    const dealii::Point<dim, dealii::VectorizedArray<number>>     &q_point,
    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
    const dealii::types::boundary_id                               boundary_id,
    const ConservedVariablesType                                  &w_m,
    const ConservedVariablesType                                  &delta_w_m,
    const ConservedVariablesGradType                              &grad_w_m,
    const ConservedVariablesGradType                              &grad_delta_w_m,
    number gamma) const -> std::tuple<ConservedVariablesType,
                                      ConservedVariablesGradType,
                                      ConservedVariablesType,
                                      ConservedVariablesGradType>
  {
    ConservedVariablesType     w_p, delta_w_p;
    ConservedVariablesGradType grad_w_p, grad_delta_w_p;

    if (const BoundaryType boundary_type = get_boundary_type(boundary_id);
        boundary_type == BoundaryType::slip_wall)
      {
        // homogeneous Neumann
        auto rho_u_dot_n = w_m[1] * normal[0];
        for (unsigned int d = 1; d < dim; ++d)
          rho_u_dot_n += w_m[1 + d] * normal[d];
        w_p[0]            = w_m[0];
        grad_w_p[0]       = -(grad_w_m[0]);
        grad_delta_w_p[0] = -grad_delta_w_m[0];
        // symmetry
        for (unsigned int d = 0; d < dim; ++d)
          {
            w_p[d + 1] = w_m[d + 1] - 2. * rho_u_dot_n * normal[d];
            grad_w_p[d + 1] =
              grad_w_m[d + 1] - 2. * scalar_product(grad_w_m[d + 1], normal) * normal;
            grad_delta_w_p[d + 1] =
              grad_delta_w_m[d + 1] - 2. * scalar_product(grad_delta_w_m[d + 1], normal) * normal;
          }
        // homogeneous Neumann
        grad_w_p[dim + 1]       = -(grad_w_m[dim + 1]);
        grad_delta_w_p[dim + 1] = -grad_delta_w_m[dim + 1];
        w_p[dim + 1]            = w_m[dim + 1];

        delta_w_p[0] = delta_w_m[0];
        dealii::Tensor<1, dim, dealii::VectorizedArray<number>> delta_momentum;
        for (unsigned int i = 1; i < dim + 1; ++i)
          {
            delta_momentum[i - 1] = delta_w_m[i];
          }
        const auto matrix =
          -2.0 * dyadic_product<dim, dim, dealii::VectorizedArray<number>>(normal, normal);
        const auto helper =
          matrix_vector_product<dim, dim, dealii::VectorizedArray<number>>(matrix, delta_momentum);
        for (unsigned int i = 1; i < dim + 1; ++i)
          {
            delta_w_p[i] = helper[i - 1] + delta_w_m[i];
          }
        delta_w_p[dim + 1] = delta_w_m[dim + 1];
      }
    else if (boundary_type == BoundaryType::no_slip_wall)
      {
        // homogeneous Neumann
        w_p[0]            = w_m[0];
        grad_w_p[0]       = -(grad_w_m[0]);
        delta_w_p[0]      = delta_w_m[0];
        grad_delta_w_p[0] = -grad_delta_w_m[0];
        // Dirichlet
        for (unsigned int d = 0; d < dim; ++d)
          {
            w_p[d + 1]            = 0.;
            grad_w_p[d + 1]       = grad_w_m[d + 1];
            delta_w_p[d + 1]      = 0.0;
            grad_delta_w_p[d + 1] = grad_delta_w_m[d + 1];
          }
        // homogeneous Neumann
        grad_w_p[dim + 1]       = -(grad_w_m[dim + 1]);
        w_p[dim + 1]            = w_m[dim + 1];
        delta_w_p[dim + 1]      = delta_w_m[dim + 1];
        grad_delta_w_p[dim + 1] = -grad_delta_w_m[dim + 1];
      }
    else if (boundary_type == BoundaryType::inflow)
      {
        // Dirichlet
        w_p = get_boundary_value(boundary_id, BoundaryType::inflow, q_point);

        grad_w_p = grad_w_m;
        // delta_w_p is zero at Dirichlet boundaries. Hence, nothing needs to be done here.

        grad_delta_w_p = grad_delta_w_m;
      }
    else if (boundary_type == BoundaryType::subsonic_outflow_fixed_pressure)
      {
        // homogeneous Neumann
        w_p            = w_m;
        grad_w_p       = -grad_w_m;
        delta_w_p      = delta_w_m;
        grad_delta_w_p = -grad_delta_w_m;

        // Dirichlet
        auto p_dyn = dealii::VectorizedArray<number>(0.);
        for (unsigned int i = 1; i < dim + 1; ++i)
          p_dyn += w_m[i] * w_m[i];

        p_dyn /= (w_m[0] * 2.);
        w_p[dim + 1] = get_boundary_value(boundary_id,
                                          BoundaryType::subsonic_outflow_fixed_pressure,
                                          q_point,
                                          0) /
                         (gamma - 1.) +
                       p_dyn;
        grad_w_p[dim + 1]       = grad_w_m[dim + 1];
        grad_delta_w_p[dim + 1] = grad_delta_w_m[dim + 1];

        dealii::VectorizedArray<number>                         rho_inv = 1.0 / w_m[0];
        dealii::Tensor<1, dim, dealii::VectorizedArray<number>> momentum;
        dealii::Tensor<1, dim, dealii::VectorizedArray<number>> delta_momentum;
        for (unsigned int i = 1; i < dim + 1; ++i)
          {
            momentum[i - 1]       = w_m[i];
            delta_momentum[i - 1] = delta_w_m[i];
          }
        delta_w_p[dim + 1] = momentum.norm() * rho_inv * rho_inv * rho_inv * delta_w_m[0] +
                             rho_inv * momentum * delta_momentum;
      }
    else if (boundary_type == BoundaryType::subsonic_outflow_fixed_energy)
      {
        // homogeneous Neumann
        w_p            = w_m;
        grad_w_p       = -grad_w_m;
        delta_w_p      = delta_w_m;
        grad_delta_w_p = -grad_delta_w_m;

        // Dirichlet
        w_p[dim + 1] =
          get_boundary_value(boundary_id, BoundaryType::subsonic_outflow_fixed_energy, q_point, 0);

        grad_w_p[dim + 1]       = grad_w_m[dim + 1];
        delta_w_p[dim + 1]      = 0.0;
        grad_delta_w_p[dim + 1] = grad_delta_w_m[dim + 1];
      }
    else
      AssertThrow(false,
                  ExcMessage("Unknown boundary id, did "
                             "you set a boundary condition for "
                             "this part of the domain boundary?"));
    return {w_p, grad_w_p, delta_w_p, grad_delta_w_p};
  }

  template class CompressibleFlowBoundaryConditions<1, double>;
  template class CompressibleFlowBoundaryConditions<2, double>;
  template class CompressibleFlowBoundaryConditions<3, double>;
} // namespace MeltPoolDG::Flow
