/**
 * @brief Block-Jacobi preconditioner for matrix-free (cut)DG operators.
 */

#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/memory_consumption.h>
#include <deal.II/base/numbers.h>

#include <any>
#include <cstddef>
#include <vector>

namespace MeltPoolDG
{
  /**
   * Storage of dense inverse cell blocks for a block-Jacobi preconditioner.
   *
   * A block is identified by the matrix-free cell batch, the SIMD lane within that batch and a
   * block index within the cell, e.g., the phase in a two-phase cutDG discretization. Only cells
   * that require a dense block store one. For all other cells, get_block() returns a nullptr and
   * the operator is expected to apply a cheaper, e.g., analytical, inverse of the cell block.
   *
   * Each block is a square matrix of size block_size() x block_size(), stored in row-major order.
   *
   * @note The blocks are addressed via the matrix-free cell batch numbering. Thus, the data becomes
   * invalid whenever the MatrixFree object is reinitialized and has to be recomputed.
   */
  template <typename number>
  class CellwiseBlockInverses
  {
  public:
    /**
     * Initialize the internal data structures and remove all blocks.
     *
     * @param n_cell_batches_in Number of cell batches of the MatrixFree object.
     * @param n_lanes_in Number of SIMD lanes per cell batch.
     * @param n_blocks_per_cell_in Number of blocks per cell (e.g., number of phases).
     * @param block_size_in Number of rows (and columns) of a single block.
     */
    void
    reinit(const unsigned int n_cell_batches_in,
           const unsigned int n_lanes_in,
           const unsigned int n_blocks_per_cell_in,
           const unsigned int block_size_in)
    {
      n_cell_batches_    = n_cell_batches_in;
      n_lanes            = n_lanes_in;
      n_blocks_per_cell  = n_blocks_per_cell_in;
      block_size_        = block_size_in;
      block_size_squared = static_cast<std::size_t>(block_size_in) * block_size_in;

      block_indices.assign(static_cast<std::size_t>(n_cell_batches_) * n_lanes * n_blocks_per_cell,
                           dealii::numbers::invalid_unsigned_int);
      values.clear();
      batch_has_blocks.assign(n_cell_batches_, false);
      batches_with_blocks.clear();
      initialized = true;
    }

    /**
     * Remove all blocks and release the memory.
     */
    void
    clear()
    {
      block_indices.clear();
      block_indices.shrink_to_fit();
      values.clear();
      values.shrink_to_fit();
      batch_has_blocks.clear();
      batch_has_blocks.shrink_to_fit();
      batches_with_blocks.clear();
      batches_with_blocks.shrink_to_fit();
      n_cell_batches_ = 0;
      initialized     = false;
    }

    /**
     * Return whether reinit() has been called since the last call to clear().
     */
    bool
    is_initialized() const
    {
      return initialized;
    }

    /**
     * Add a block for the given cell and return a pointer to its storage (block_size()^2 entries,
     * row-major). The pointer is only valid until the next call to add_block().
     */
    number *
    add_block(const unsigned int cell_batch, const unsigned int lane, const unsigned int block)
    {
      const std::size_t idx = index(cell_batch, lane, block);
      Assert(block_indices[idx] == dealii::numbers::invalid_unsigned_int,
             dealii::ExcMessage("A block has already been added for this cell."));

      block_indices[idx] = n_blocks();
      values.resize(values.size() + block_size_squared);

      if (!batch_has_blocks[cell_batch])
        {
          batch_has_blocks[cell_batch] = true;
          batches_with_blocks.push_back(cell_batch);
        }

      return values.data() + values.size() - block_size_squared;
    }

    /**
     * Return a pointer to the block of the given cell (block_size()^2 entries, row-major) or a
     * nullptr if no block has been stored for this cell.
     */
    const number *
    get_block(const unsigned int cell_batch,
              const unsigned int lane,
              const unsigned int block) const
    {
      const unsigned int b = block_indices[index(cell_batch, lane, block)];
      return (b == dealii::numbers::invalid_unsigned_int) ?
               nullptr :
               values.data() + static_cast<std::size_t>(b) * block_size_squared;
    }

    /**
     * Return whether at least one block has been stored for a cell of the given cell batch.
     */
    bool
    cell_batch_has_blocks(const unsigned int cell_batch) const
    {
      AssertIndexRange(cell_batch, n_cell_batches_);
      return batch_has_blocks[cell_batch];
    }

    /**
     * Return the cell batches that contain at least one cell with a block, in the order in which
     * the first block of each batch was added.
     */
    const std::vector<unsigned int> &
    get_cell_batches_with_blocks() const
    {
      return batches_with_blocks;
    }

    /**
     * Number of rows (and columns) of a single block.
     */
    unsigned int
    block_size() const
    {
      return block_size_;
    }

    /**
     * Number of cell batches the data has been set up for.
     */
    unsigned int
    n_cell_batches() const
    {
      return n_cell_batches_;
    }

    /**
     * Number of stored blocks.
     */
    unsigned int
    n_blocks() const
    {
      return block_size_squared == 0 ?
               0 :
               static_cast<unsigned int>(values.size() / block_size_squared);
    }

    /**
     * Memory consumption in bytes.
     */
    std::size_t
    memory_consumption() const
    {
      return dealii::MemoryConsumption::memory_consumption(block_indices) +
             dealii::MemoryConsumption::memory_consumption(values) +
             dealii::MemoryConsumption::memory_consumption(batch_has_blocks) +
             dealii::MemoryConsumption::memory_consumption(batches_with_blocks);
    }

  private:
    std::size_t
    index(const unsigned int cell_batch, const unsigned int lane, const unsigned int block) const
    {
      AssertIndexRange(cell_batch, n_cell_batches_);
      AssertIndexRange(lane, n_lanes);
      AssertIndexRange(block, n_blocks_per_cell);
      return (static_cast<std::size_t>(cell_batch) * n_lanes + lane) * n_blocks_per_cell + block;
    }

    unsigned int n_cell_batches_    = 0;
    unsigned int n_lanes            = 0;
    unsigned int n_blocks_per_cell  = 0;
    unsigned int block_size_        = 0;
    std::size_t  block_size_squared = 0;
    bool         initialized        = false;

    //! Index into @p values for each (cell batch, lane, block), invalid_unsigned_int if no block
    std::vector<unsigned int> block_indices;

    //! Entries of all blocks
    std::vector<number> values;

    //! Flag for each cell batch whether it contains at least one block
    std::vector<bool> batch_has_blocks;

    //! Cell batches that contain at least one block
    std::vector<unsigned int> batches_with_blocks;
  };



  /**
   * A concept for matrix-free operators that are compatible with the block-Jacobi preconditioner.
   */
  template <typename OperatorType, typename VectorType>
  concept BlockJacobiPreconditionerOperatorType =
    requires(const OperatorType                           &op,
             typename OperatorType::BlockJacobiData       &data,
             const typename OperatorType::BlockJacobiData &const_data,
             VectorType                                   &dst,
             const VectorType                             &src) {
      /**
       * Compute the inverses of the diagonal blocks of the system matrix and store them in @p data.
       */
      op.compute_inverse_block_diagonal_from_matrixfree(data);

      /**
       * Apply the inverse block diagonal stored in @p const_data to @p src and write the result
       * into @p dst.
       */
      op.apply_inverse_block_diagonal(const_data, dst, src);
    };



  /**
   * Block-Jacobi preconditioner for matrix-free operators.
   *
   * The preconditioner applies the inverse of the block diagonal of the system matrix. Which DoFs
   * form a block, how the blocks are computed and how they are applied is defined by the operator
   * (see BlockJacobiPreconditionerOperatorType). For DG discretizations, a block typically contains
   * all DoFs of a cell (and phase), and the block diagonal is the exact inverse of the system
   * matrix wherever it does not couple neighboring cells (e.g., a DG mass matrix away from the
   * ghost-penalty faces of a cutDG discretization).
   */
  template <int dim,
            typename number,
            typename VectorType,
            BlockJacobiPreconditionerOperatorType<VectorType> OperatorType>
  class BlockJacobiPreconditioner
  {
  public:
    /**
     * Constructor.
     *
     * @param operator_in Operator object used to compute and apply the inverse block diagonal.
     */
    explicit BlockJacobiPreconditioner(const OperatorType &operator_in)
      : eq_operator(operator_in)
    {}

    /**
     * Apply the preconditioner to the given @p src vector and store the result in the @p dst vector.
     *
     * @param dst Vector in which the result is stored.
     * @param src Source vector to which the preconditioner is applied.
     */
    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      Assert(block_inverses.is_initialized(),
             dealii::ExcMessage("The block-Jacobi preconditioner has not been updated after its "
                                "last reinitialization. Call update() before vmult()."));
      eq_operator.apply_inverse_block_diagonal(block_inverses, dst, src);
    }

    /**
     * Update the preconditioner, i.e., compute the inverse diagonal blocks of the system matrix.
     *
     * @note The function takes a single value only to conform to the interface. It is not used.
     */
    void
    update(const std::any & = std::any())
    {
      eq_operator.compute_inverse_block_diagonal_from_matrixfree(block_inverses);
    }

    /**
     * Reset the internal data structures. They are rebuilt by the next call to update().
     */
    void
    reinit()
    {
      block_inverses.clear();
    }

  private:
    //! Operator providing the computation and application of the inverse block diagonal
    const OperatorType &eq_operator;

    //! Inverse diagonal blocks
    typename OperatorType::BlockJacobiData block_inverses;
  };
} // namespace MeltPoolDG
