/**
 * @brief Factory function for the preconditioner.
 */

#pragma once

#include <deal.II/base/exceptions.h>

#include <deal.II/lac/trilinos_precondition.h>

#include "meltpooldg/core/scratch_data.hpp"
#include <meltpooldg/linear_algebra/linear_solver_data.hpp>
#include <meltpooldg/linear_algebra/preconditioner.hpp>
#include <meltpooldg/linear_algebra/preconditioner_block_jacobi.hpp>
#include <meltpooldg/linear_algebra/preconditioner_jacobi.hpp>
#include <meltpooldg/linear_algebra/preconditioner_trilinos_wrapper.hpp>

namespace MeltPoolDG
{
  /**
   * A preconditioner factory function. This function creates a preconditioner based on the passed
   * preconditioner type and the flag indicating whether the preconditioner is applied in a
   * matrix-free or a matrix-based context.
   *
   * @param preconditioner_type Type of the desired preconditioner.
   * @param operator_in Operator that supports the required functions for the computations inside
   * the preconditioner (see specific preconditioner classes for additional information).
   * @param scratch_data Scratch data object to get relevant dof information for the preconditioner.
   * @param dof_idx Relevant dof index in the scratch data object.
   * @param do_matrix_free A flag indicating if the operator shall be used in a matrix-free or
   * matrix-based way.
   *
   * @return Preconditioner object using the passed preconditioner type.
   * @throws Exception if the preconditioner type is not supported.
   */
  template <int dim, typename number, typename OperatorType, typename VectorType>
  Preconditioner<dim, VectorType, number>
  make_preconditioner(const PreconditionerType            &preconditioner_type,
                      const OperatorType                  *operator_in,
                      const ScratchData<dim, dim, number> &scratch_data,
                      const unsigned                       dof_idx,
                      const bool                           do_matrix_free = true)
  {
    switch (preconditioner_type)
      {
          case PreconditionerType::Identity: {
            return Preconditioner<dim, VectorType, number>(
              IdentityPreconditioner<dim, VectorType, number>());
          }
          case PreconditionerType::AMG: {
            return Preconditioner<dim, VectorType, number>(
              DealiiPreconditionerWrapper<dim,
                                          number,
                                          VectorType,
                                          dealii::TrilinosWrappers::PreconditionAMG,
                                          OperatorType>(
                operator_in, scratch_data, dof_idx, do_matrix_free));
          }
          case PreconditionerType::ILU: {
            return Preconditioner<dim, VectorType, number>(
              DealiiPreconditionerWrapper<dim,
                                          number,
                                          VectorType,
                                          dealii::TrilinosWrappers::PreconditionILU,
                                          OperatorType>(
                operator_in, scratch_data, dof_idx, do_matrix_free));
          }
          case PreconditionerType::Diagonal: {
            if constexpr (JacobiPreconditionerOperatorType<OperatorType, VectorType>)
              {
                if (do_matrix_free)
                  {
                    return Preconditioner<dim, VectorType, number>(
                      JacobiPreconditioner<dim, number, VectorType, OperatorType>(*operator_in,
                                                                                  scratch_data,
                                                                                  dof_idx));
                  }
              }
            return Preconditioner<dim, VectorType, number>(
              DealiiPreconditionerWrapper<dim,
                                          number,
                                          VectorType,
                                          dealii::TrilinosWrappers::PreconditionJacobi,
                                          OperatorType>(operator_in, scratch_data, dof_idx, false));
          }
          case PreconditionerType::BlockJacobi: {
            if constexpr (BlockJacobiPreconditionerOperatorType<OperatorType, VectorType>)
              {
                AssertThrow(do_matrix_free,
                            dealii::ExcMessage(
                              "The block-Jacobi preconditioner is only available for matrix-free "
                              "operators."));
                return Preconditioner<dim, VectorType, number>(
                  BlockJacobiPreconditioner<dim, number, VectorType, OperatorType>(*operator_in));
              }
            else
              {
                AssertThrow(
                  false,
                  dealii::ExcMessage(
                    "The block-Jacobi preconditioner is not supported by this operator."));
                return Preconditioner<dim, VectorType, number>(
                  IdentityPreconditioner<dim, VectorType, number>());
              }
          }
          default: {
            AssertThrow(false,
                        dealii::ExcMessage("The requested preconditioner type is not"
                                           " implemented."));
          }
      }
  }
} // namespace MeltPoolDG
