#pragma once

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/level_set/narrow_band_data.hpp>

namespace MeltPoolDG::LevelSet
{
  /**
   * @brief Global face-based ghost-penalty extension of a DG level-set field from the cut cells to
   * the neighbor cells of the narrow band.
   *
   * The level-set field $\phi_h$ lives on the band DoFHandler (see NarrowBand::ActiveFEIndex),
   * i.e., on the cut cells $\mathcal{T}^\Gamma$ and their face neighbors $\mathcal{T}^N$. Its
   * values on the cut cells are given; its values on the neighbor cells are determined by the
   * extension problem: find $\phi_h \in V^p_\mathrm{DG}(\mathcal{T}^B)$ with
   * $\phi_h|_K = \phi^n_h|_K$ for all $K \in \mathcal{T}^\Gamma$ such that
   *
   * $g_h(\phi_h, v) = 0$ for all $v \in V^p_\mathrm{DG}(\mathcal{T}^B)$ with
   * $v|_{\mathcal{T}^\Gamma} = 0$,
   *
   * with the face-based ghost penalty
   *
   * $g_h(\phi, v) = \sum_{F \in \mathcal{F}^\mathrm{gp}} \sum_{k=0}^{p} \gamma_k h_F^{2k+1}
   *   \int_F [\partial_n^k \phi] [\partial_n^k v] \, \mathrm{d}s$.
   *
   * The ghost-penalty faces $\mathcal{F}^\mathrm{gp}$ are either all interior faces of the band
   * with at least one adjacent neighbor cell (global coupling of the neighbor cells) or only the
   * faces between a cut cell and a neighbor cell (decoupled, cell-local extension), see
   * NarrowBandGhostPenaltyFaces. The face size $h_F$ is the minimum vertex distance of the two
   * adjacent cells.
   *
   * Following the ghost-penalty extrapolation of CutUtil::SolutionTransferOperator, the system is
   * assembled matrix-based and the DoFs of the cut cells are prescribed by inhomogeneous
   * constraints. The system is symmetric positive definite, since every neighbor cell shares a
   * face with a cut cell and the jumps of all normal derivatives up to the polynomial degree are
   * penalized.
   *
   * Optionally, cut cells with a small volume fraction of the positive phase are excluded as
   * sources of the extension (see NarrowBandData::Extension::minimum_source_volume_fraction): the
   * faces between such small cut cells and neighbor cells are removed from
   * $\mathcal{F}^\mathrm{gp}$, if the neighbor cell is connected to a cut cell with a sufficient
   * volume fraction via the remaining ghost-penalty faces (directly or via other neighbor cells).
   * Otherwise, the faces are kept to obtain a regular system.
   *
   * Optionally, spurious sign changes of the extended field on the neighbor cells are removed
   * (see NarrowBand::limit_sign_changes()), since the neighbor cells do not contain the interface
   * by definition.
   *
   * @note The extension is implemented for polynomial degrees $1 \le p \le 3$ and triangulations
   * without hanging nodes.
   */
  template <int dim, typename number>
  class NarrowBandGhostPenaltyExtension
  {
  public:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    /**
     * @brief Constructor.
     *
     * @param data Parameters of the extension.
     * @param mapping Mapping.
     * @param pcout Output stream for the solver statistics.
     */
    NarrowBandGhostPenaltyExtension(const typename NarrowBandData<number>::Extension &data,
                                    const dealii::Mapping<dim>                       &mapping,
                                    const dealii::ConditionalOStream                 &pcout);

    /**
     * @brief Extend the level-set field from the cut cells to the neighbor cells.
     *
     * @param dof_handler Band DoFHandler.
     * @param level_set Level-set vector of the band DoFHandler. On input, the DoFs of the cut
     * cells hold the given values; on output, the DoFs of the neighbor cells hold the extended
     * values.
     *
     * @return Number of iterations of the linear solver.
     */
    unsigned int
    extend(const dealii::DoFHandler<dim> &dof_handler, VectorType &level_set) const;

  private:
    /// Parameters of the extension.
    const typename NarrowBandData<number>::Extension &data;

    /// Mapping.
    const dealii::Mapping<dim> &mapping;

    /// Output stream.
    const dealii::ConditionalOStream pcout;
  };
} // namespace MeltPoolDG::LevelSet
