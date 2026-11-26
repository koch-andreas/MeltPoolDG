#pragma once

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/level_set/narrow_band_data.hpp>

namespace MeltPoolDG::LevelSet
{
  /**
   * @brief Geometric reconstruction of the level set of the advected narrow band.
   *
   * The level set of all cells of the band (FE_DGQ, band DoFHandler) is replaced by
   * 1. the signed distance $d(x) = \operatorname{sign}(\phi_K(x))\,\operatorname{dist}(x,
   *    \Gamma_h)$ to the discrete interface $\Gamma_h$ at the quadrature points of the cells,
   *    where $\Gamma_h$ consists of the zero level sets of the cells and of the points of the
   *    faces, at which the traces of the two adjacent cells have opposite signs (no interface can
   *    hide in a face), and the sign is taken from the polynomial $\phi_K$ of the cell itself;
   * 2. the morphological opening of both phases with the radius $r = t h_K / 2$ ($t$ =
   *    NarrowBandData::GeometricReconstruction::thickness): erosion $E = \{d > r\}$ (zero level
   *    set of the cell-wise $L^2$ projection of $d - r$) and dilation $d \leftarrow
   *    \operatorname{dist}_\pm(x, \partial E) + r$, first for the positive, then for the
   *    negative phase. Features thinner than $t h$ disappear completely, resolved parts (radius of
   *    curvature larger than $r$) are unchanged;
   * 3. the cell-wise projection onto the polynomials, which minimizes $\|\phi - d\|_K^2 + \gamma
   *    h_K \|\phi\|_{\Gamma^*\cap K}^2$ (Gauss-Lobatto quadrature with $p+2$ points per direction),
   *    where $\Gamma^*$ are the quadrature points ($2(p+1)$ per direction) of the discrete zero
   *    level set of the cell, which survive the opening ($|d| < r/2$ after the opening). The
   *    opened distance is only used at the points, at which the opening has changed the signed
   *    distance by more than $r/2$ (removed thin features); smaller changes (rounded corners, the
   *    undershoot of the projection of the erosion at ridges of the distance) are ignored, such
   *    that repeated reconstructions do not erode the interface. The remaining interface stays in
   *    place. Cells without points of $\Gamma^*$ get the $L^2$ projection, which is shifted, if it
   *    changes its sign in a cell, in which the distance has one sign at all quadrature points
   *    (undershoot next to an interface, which touches the cell).
   *
   * The distances are computed by the closest-point search on the polynomials of the cells (see
   * NarrowBand::internal::search_closest_point()); the data of the cells containing the interface
   * is gathered from all processes.
   *
   * @note Parallelepiped cells without hanging nodes.
   */
  template <int dim, typename number>
  class NarrowBandGeometricReconstruction
  {
  public:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    NarrowBandGeometricReconstruction(
      const typename NarrowBandData<number>::GeometricReconstruction &data,
      const dealii::ConditionalOStream                               &pcout);

    /**
     * Reconstruct the level set @p level_set of all locally owned cells with DoFs of the band
     * DoFHandler @p dof_handler.
     */
    void
    reconstruct(const dealii::DoFHandler<dim> &dof_handler, VectorType &level_set) const;

  private:
    /// Parameters.
    const typename NarrowBandData<number>::GeometricReconstruction &data;

    /// Output stream.
    const dealii::ConditionalOStream pcout;
  };
} // namespace MeltPoolDG::LevelSet
