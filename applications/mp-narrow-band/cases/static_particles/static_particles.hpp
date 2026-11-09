#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <meltpooldg/level_set/level_set_type.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/utilities/powder_bed.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "../../narrow_band_case.hpp"
#include "../../narrow_band_postprocessing.hpp"

namespace MeltPoolDG::Simulation::NarrowBandStaticParticles
{
  /**
   * @brief Static particles: the signed distance function of a list of spherical particles
   * (circles in 2D), scaled by a distortion factor, at rest (zero velocity).
   *
   * The particles are read from a particle list file (see MeltPool::PowderBedLevelSet); in 2D,
   * the plane of the simulation is the slice y = slice location of the particle space, whose
   * z-axis corresponds to the y-axis of the simulation. With a vanishing velocity, the
   * semi-Lagrangian advection reproduces the level set, and each time step applies the
   * reinitialization on the cut cells and the projection onto the continuous level set. This case
   * tests the reinitialization of many close interfaces, e.g., with the SIPG face terms on all
   * faces between two cut cells or only on the faces crossed by the interface (parameter
   * 'narrow band/reinitialization faces').
   *
   * The initial level set on the cut cells is either the interpolated signed distance function of
   * the union of the particles (initial condition = union) or, cell by cell, the signed distance
   * function of the particle whose surface intersects the cell (initial condition = cellwise; in
   * cells intersected by the surfaces of several particles, the signed distance function of the
   * union). The latter jumps across the faces between cut cells of different particles and does
   * not contain the kink of the signed distance function of the union between close particles,
   * unless both particles intersect the same cell.
   *
   * If user-defined postprocessing is enabled, the size of the narrow band, the geometric
   * measures of the interface and the errors with respect to the signed distance function of the
   * union of the particles (exact for non-overlapping particles) are printed at every output time.
   */
  template <int dim, typename number>
  class SimulationStaticParticles : public LevelSet::NarrowBandCase<dim, number>
  {
  public:
    SimulationStaticParticles(std::string parameter_file, const MPI_Comm mpi_communicator)
      : LevelSet::NarrowBandCase<dim, number>(parameter_file, mpi_communicator)
    {}

    void
    create_spatial_discretization() override
    {
      AssertThrow(dim >= 2, dealii::ExcMessage("The case static_particles requires dim >= 2."));
      AssertThrow(domain_min.size() >= dim and domain_max.size() >= dim,
                  dealii::ExcMessage("Specify dim coordinates for the domain corners."));
      AssertThrow(cell_repetitions.size() == dim,
                  dealii::ExcMessage("Specify dim cell repetitions."));

      this->triangulation =
        std::make_shared<dealii::parallel::distributed::Triangulation<dim>>(this->mpi_communicator);

      dealii::Point<dim> lower, upper;
      for (unsigned int d = 0; d < dim; ++d)
        {
          lower[d] = domain_min[d];
          upper[d] = domain_max[d];
        }
      dealii::GridGenerator::subdivided_hyper_rectangle(*this->triangulation,
                                                        cell_repetitions,
                                                        lower,
                                                        upper);
      this->triangulation->refine_global(this->parameters.base.global_refinements);
    }

    void
    set_boundary_conditions() override
    {
      // nothing to do: the particles are at rest
    }

    void
    set_field_conditions() override
    {
      AssertThrow(distortion_factor > 0.,
                  dealii::ExcMessage("The distortion factor must be positive."));

      signed_distance = std::make_shared<MeltPool::PowderBedLevelSet<dim, number>>(
        powder_bed_data, LevelSet::LevelSetType::signed_distance);

      const auto   sd     = signed_distance;
      const number factor = distortion_factor;
      this->attach_initial_condition(
        std::make_shared<dealii::ScalarFunctionFromFunctionObject<dim, number>>(
          [sd, factor](const dealii::Point<dim> &p) { return factor * sd->value(p, 0); }),
        "level_set");
      this->attach_field_function(std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim),
                                  "prescribed_velocity",
                                  "level_set");
    }

    bool
    add_case_specific_parameters(dealii::ParameterHandler &prm) override
    {
      prm.add_parameter("domain min",
                        domain_min,
                        "Lower left corner of the domain. Specify comma-separated values for the "
                        "coordinates.");
      prm.add_parameter("domain max",
                        domain_max,
                        "Upper right corner of the domain. Specify comma-separated values for the "
                        "coordinates.");
      prm.add_parameter("cell repetitions",
                        cell_repetitions,
                        "Number of cells per direction before the global refinements.");
      prm.add_parameter("distortion factor",
                        distortion_factor,
                        "Scaling factor of the initial signed distance function.");
      prm.add_parameter("initial condition",
                        initial_condition,
                        "Initial level set on the cut cells: interpolated signed distance "
                        "function of the union of the particles (union) or, cell by cell, the "
                        "signed distance function of the particle whose surface intersects the "
                        "cell (cellwise; in cells intersected by the surfaces of several "
                        "particles, the signed distance function of the union).",
                        dealii::Patterns::Selection("union|cellwise"));
      powder_bed_data.add_parameters(prm);

      return this->parameters.base.do_print_parameters;
    }

    /**
     * @brief Cell by cell, the scaled signed distance function of the particle whose surface
     * intersects the cell; in cells intersected by the surfaces of several particles, the scaled
     * signed distance function of the union (only for initial condition = cellwise).
     */
    typename LevelSet::NarrowBandLevelSetOperation<dim, number>::CellwiseFunction
    get_cellwise_initial_condition() const override
    {
      if (initial_condition != "cellwise")
        return {};

      AssertThrow(signed_distance != nullptr,
                  dealii::ExcMessage("The field conditions are not set."));
      AssertThrow(powder_bed_data.substrate_level == std::numeric_limits<number>::lowest(),
                  dealii::ExcMessage("The initial condition cellwise does not support a "
                                     "substrate."));

      const auto   sd     = signed_distance;
      const number factor = distortion_factor;
      return [sd, factor](const dealii::CellAccessor<dim, dim> &cell, const dealii::Point<dim> &p) {
        const auto &particles = sd->get_particles();

        // particles whose surface intersects the bounding box of the cell: the minimum distance
        // of the particle center to the box is at most and the maximum distance to its vertices
        // at least the radius
        const auto   box            = cell.bounding_box();
        unsigned int n_intersecting = 0;
        std::size_t  k_intersecting = 0;
        for (std::size_t k = 0; k < particles.size(); ++k)
          {
            dealii::Point<dim> closest, farthest;
            for (unsigned int d = 0; d < dim; ++d)
              {
                const number lower = box.lower_bound(d), upper = box.upper_bound(d);
                const number c =
                  dim == 2 and d == 1 ? particles[k].first[2] : particles[k].first[d];
                closest[d]  = std::clamp(c, lower, upper);
                farthest[d] = std::abs(c - lower) > std::abs(c - upper) ? lower : upper;
              }
            const number d_min = sd->to_particle_space(closest).distance(particles[k].first);
            const number d_max = sd->to_particle_space(farthest).distance(particles[k].first);
            if (d_min <= particles[k].second and particles[k].second <= d_max)
              {
                ++n_intersecting;
                k_intersecting = k;
              }
          }

        if (n_intersecting != 1)
          return factor * sd->value(p, 0);

        return factor * (particles[k_intersecting].second -
                         sd->to_particle_space(p).distance(particles[k_intersecting].first));
      };
    }

    /**
     * @brief Print the size of the narrow band, the measures of the interface and the errors with
     * respect to the signed distance function of the union of the particles.
     */
    void
    do_postprocessing(const GenericDataOut<dim, number> &generic_data_out) const final
    {
      LevelSet::print_narrow_band_statistics<dim, number>(generic_data_out,
                                                          "level_set_band",
                                                          this->mpi_communicator,
                                                          "static_particles");
      LevelSet::print_narrow_band_errors<dim, number>(generic_data_out,
                                                      "level_set",
                                                      *signed_distance,
                                                      0. /*exact interface measure*/,
                                                      this->mpi_communicator,
                                                      "static_particles");

      if (this->parameters.narrow_band.continuous_level_set.enable)
        LevelSet::print_interface_measures<dim, number>(generic_data_out,
                                                        "level_set_cg",
                                                        this->mpi_communicator,
                                                        "level_set_cg");
    }

  private:
    /// Corners of the domain
    std::vector<number> domain_min{0., 0., 0.};
    std::vector<number> domain_max{1., 1., 1.};

    /// Number of cells per direction before the global refinements
    std::vector<unsigned int> cell_repetitions{1, 1};

    /// Particle list
    MeltPool::PowderBedData<number> powder_bed_data;

    /// Scaling factor of the initial signed distance function
    number distortion_factor = 1.0;

    /// Initial level set on the cut cells (union or cellwise)
    std::string initial_condition = "union";

    /// Signed distance function of the union of the particles
    std::shared_ptr<MeltPool::PowderBedLevelSet<dim, number>> signed_distance;
  };
} // namespace MeltPoolDG::Simulation::NarrowBandStaticParticles
