#pragma once

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/level_set/narrow_band_data.hpp>

namespace MeltPoolDG::LevelSet
{
  /**
   * @brief Closest-point extension of a DG level-set field from the cut cells to the neighbor
   * cells of the narrow band.
   *
   * The level-set field $\phi_h$ lives on the band DoFHandler (see NarrowBand::ActiveFEIndex),
   * i.e., on the cut cells $\mathcal{T}^\Gamma$ and their face neighbors $\mathcal{T}^N$. Its
   * values on the cut cells are given. In contrast to the ghost-penalty extension
   * (NarrowBandGhostPenaltyExtension), which extrapolates the polynomials of the cut cells, the
   * level set on the neighbor cells is constructed from the geometry of the discrete interface
   * $\Gamma_h = \{\phi_h = 0\} \cap \mathcal{T}^\Gamma$: for each node $x_i$ of a neighbor cell,
   * the closest point $y_i \in \Gamma_h$ is determined and the nodal value is set to the
   * first-order Taylor expansion
   *
   * $\phi_h(x_i) = \phi_h(y_i) + \nabla \phi_h(y_i) \cdot (x_i - y_i)$,
   *
   * where $\phi_h$ and $\nabla \phi_h$ are evaluated with the polynomial of the cut cell
   * containing $y_i$. Since $\phi_h(y_i) = 0$ and $x_i - y_i \parallel \nabla \phi_h(y_i)$ at the
   * closest point, this is the signed distance to $\Gamma_h$ scaled by the gradient magnitude at
   * the closest point, i.e., the extension is consistent with the cut cells if $|\nabla \phi_h|
   * \neq 1$. Near thin structures (e.g., the tail of a stretched bubble), the closest point
   * selects the nearest interface branch, while the polynomial extrapolation of the ghost-penalty
   * extension mixes the information of both branches.
   *
   * The closest point is searched in the cut cells sharing at least one vertex with the neighbor
   * cell (i.e., within the ghost layer of a parallel triangulation): in each candidate cut cell,
   * the search starts from the nearest quadrature point of the interface
   * (NonMatching::DiscreteQuadratureGenerator), which is projected onto the zero level set,
   * $y \leftarrow y - \phi_h(y) \nabla\phi_h(y) / |\nabla\phi_h(y)|^2$. Then, the first-order
   * optimality conditions $y - x_i + \lambda \nabla\phi_h(y) = 0$, $\phi_h(y) = 0$ of the
   * constrained minimization of $|x_i - y|$ are solved by a damped Newton method (Saye, 2014,
   * https://doi.org/10.2140/camcos.2014.9.107), where the iterate is kept within the cut cell. The
   * candidate with the smallest distance to the interface is selected.
   *
   * Optionally, cut cells with a small volume fraction of the positive phase are only considered
   * in the search, if no other cut cell shares a vertex with the neighbor cell (see
   * NarrowBandData::Extension::minimum_source_volume_fraction).
   *
   * Optionally, spurious sign changes of the extended field on the neighbor cells are removed by
   * NarrowBand::limit_sign_changes().
   *
   * @note The extension requires parallelepiped cells (affine mapping) and triangulations without
   * hanging nodes.
   */
  template <int dim, typename number>
  class NarrowBandClosestPointExtension
  {
  public:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    /**
     * @brief Constructor.
     *
     * @param data Parameters of the extension.
     * @param mapping Mapping.
     * @param pcout Output stream for the statistics.
     */
    NarrowBandClosestPointExtension(const typename NarrowBandData<number>::Extension &data,
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
     * @return Maximum number of iterations of the closest-point search.
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
