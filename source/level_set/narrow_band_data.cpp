#include <deal.II/base/exceptions.h>

#include <meltpooldg/level_set/narrow_band_data.hpp>
#include <meltpooldg/time_integration/time_integrator_util.hpp>

#include <algorithm>

namespace MeltPoolDG::LevelSet
{
  template <typename number>
  NarrowBandData<number>::NarrowBandData()
  {
    extension.linear_solver.solver_type         = LinearSolverType::CG;
    extension.linear_solver.preconditioner_type = PreconditionerType::ILU;
    extension.linear_solver.do_matrix_free      = false;
    extension.linear_solver.max_iterations      = 1000;
  }

  template <typename number>
  number
  NarrowBandData<number>::Extension::get_ghost_penalty_parameter(const unsigned int k) const
  {
    Assert(not ghost_penalty_parameters.empty(), dealii::ExcInternalError());

    // the last value is used for all remaining k
    return ghost_penalty_parameters[std::min<std::size_t>(k, ghost_penalty_parameters.size() - 1)];
  }

  template <typename number>
  void
  NarrowBandData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("narrow band");
    {
      prm.add_parameter(
        "type",
        type,
        "Type of the narrow band. extension: the band consists of the cut cells and their face "
        "neighbors; the level set of the neighbor cells is reconstructed by the extension from "
        "the cut cells in every time step. advected_layers: the band consists of the cut cells "
        "and several layers of vertex neighbors, which are advected together with the cut cells; "
        "the level set of the layers is only reconstructed by geometric redistancing (closest "
        "point on the discrete interface of the cut cells) for newly added cells and after the "
        "elliptic reinitialization of the cut cells.");
      prm.add_parameter("number of layers",
                        n_layers,
                        "Number of layers of vertex neighbors around the cut cells "
                        "(advected_layers).",
                        dealii::Patterns::Integer(2));

      prm.enter_subsection("extension");
      {
        prm.add_parameter(
          "type",
          extension.type,
          "Type of the extension of the level-set field from the cut cells to the neighbor cells: "
          "global face-based ghost-penalty extension (ghost_penalty) or closest-point extension "
          "(closest_point), where the level set at the nodes of the neighbor cells is given by "
          "the first-order Taylor expansion phi(y) + grad phi(y) . (x - y) from the closest point "
          "y on the discrete interface of the cut cells.");
        prm.add_parameter(
          "ghost penalty parameters",
          extension.ghost_penalty_parameters,
          "Dimensionless ghost-penalty parameters gamma_k > 0 for the jumps of the k-th normal "
          "derivative (k = 0, ..., p) on the ghost-penalty faces, which are scaled by h^(2k+1). "
          "Specify comma-separated values starting with k = 0. If less than p+1 values are "
          "given, the last value is used for all remaining k.");
        prm.add_parameter("ghost penalty faces",
                          extension.ghost_penalty_faces,
                          "Set of faces on which the ghost-penalty terms of the extension are "
                          "evaluated (ghost-penalty extension).");
        prm.add_parameter(
          "sign limiter tolerance",
          extension.sign_limiter_tolerance,
          "Tolerance of the sign limiter for the extended level-set field on the neighbor cells, "
          "relative to the cell size h. If the extended level set attains values of the opposite "
          "sign of its cell mean with a magnitude larger than tolerance * h in a neighbor cell, "
          "its deviation from the cell mean is scaled such that it does not change its sign. "
          "Neighbor cells do not contain the interface by definition; spurious zero crossings, "
          "e.g., due to the polynomial extrapolation near under-resolved interface tips, would "
          "otherwise create spurious cut cells. Set to a negative value to disable the limiter.");
        prm.add_parameter(
          "minimum source volume fraction",
          extension.minimum_source_volume_fraction,
          "Minimum volume fraction |K cap {phi > 0}| / |K| of the positive phase of a cut cell K, "
          "for which K serves as source of the extension. Ghost-penalty extension: the "
          "ghost-penalty faces between cut cells with a smaller volume fraction and neighbor "
          "cells are not evaluated, i.e., the neighbor cells are extended from the other cut cells "
          "via the faces to these cut cells and to other neighbor cells (if the ghost-penalty "
          "faces include all band faces). If a neighbor cell is not connected to a cut cell with a "
          "sufficient volume fraction, its faces to the small cut cells are kept. Closest-point "
          "extension: small cut cells are only considered in the closest-point search, if no "
          "other cut cells share a vertex with the neighbor cell. In contrast to 'neighbor "
          "activation volume fraction', all neighbor cells remain active. Set to 0 to use all cut "
          "cells as sources.",
          dealii::Patterns::Double(0.0, 1.0));
        extension.linear_solver.add_parameters(prm);
      }
      prm.leave_subsection();

      prm.enter_subsection("advection");
      {
        time_integrator.add_parameters(prm);
      }
      prm.leave_subsection();

      prm.add_parameter(
        "neighbor activation volume fraction",
        neighbor_activation_volume_fraction,
        "Minimum volume fraction |K cap {phi > 0}| / |K| of the positive phase of a cut cell K, "
        "for which its face neighbors are activated as neighbor cells of the narrow band (and "
        "filled by the extension). Face neighbors of cut cells with a smaller volume fraction "
        "remain inactive (FE_Nothing), unless they are face neighbors of another cut cell with a "
        "sufficient volume fraction. Set to 0 to activate all face neighbors of the cut cells.",
        dealii::Patterns::Double(0.0, 1.0));
      prm.add_parameter(
        "reinitialization faces",
        reinitialization_faces,
        "Set of faces on which the SIPG terms of the elliptic reinitialization on the cut cells "
        "are evaluated: all faces between two cut cells (cut_cell_faces) or only the faces "
        "between two cut cells, which are crossed by the interface (interface_faces). In the "
        "latter case, the cut cells are only coupled across the faces crossed by the interface.");
      prm.add_parameter("reinitialization frequency",
                        reinitialization_frequency,
                        "Reinitialize the level-set field on the cut cells every n-th time step. "
                        "Set to 0 to disable the frequency-based reinitialization.");
      prm.add_parameter("reinitialization eikonal threshold",
                        reinitialization_eikonal_threshold,
                        "Reinitialize the level-set field on the cut cells, if the eikonal error "
                        "sqrt(int (|grad phi| - 1)^2 dx / V) on the cut cells exceeds this value. "
                        "Set to 0 to disable the error-based reinitialization.",
                        dealii::Patterns::Double(0.0));
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  NarrowBandData<number>::check_input_parameters(const unsigned int fe_degree) const
  {
    AssertThrow(fe_degree >= 1, dealii::ExcMessage("The narrow-band level set requires p >= 1."));

    if (type == NarrowBandType::advected_layers)
      {
        AssertThrow(n_layers >= 2,
                    dealii::ExcMessage("The advected band requires at least two layers."));
        AssertThrow(neighbor_activation_volume_fraction == 0.,
                    dealii::ExcMessage("The neighbor activation volume fraction is not "
                                       "supported by the advected band."));
      }

    if (type == NarrowBandType::extension and
        extension.type == NarrowBandExtensionType::ghost_penalty)
      {
        AssertThrow(fe_degree <= 3,
                    dealii::ExcMessage(
                      "The ghost-penalty extension of the narrow-band level set is "
                      "implemented for polynomial degrees 1 <= p <= 3."));

        AssertThrow(not extension.ghost_penalty_parameters.empty() and
                      extension.ghost_penalty_parameters.size() <= fe_degree + 1,
                    dealii::ExcMessage(
                      "Specify between one and p+1 ghost-penalty parameters for the "
                      "narrow-band extension."));

        AssertThrow(std::all_of(extension.ghost_penalty_parameters.begin(),
                                extension.ghost_penalty_parameters.end(),
                                [](const number gamma) { return gamma > 0.0; }),
                    dealii::ExcMessage("The ghost-penalty parameters of the narrow-band extension "
                                       "must be positive."));

        AssertThrow(not extension.linear_solver.do_matrix_free,
                    dealii::ExcMessage(
                      "The ghost-penalty extension of the narrow-band level set is "
                      "implemented matrix-based. Set 'do matrix free' to false."));
      }

    AssertThrow(TimeIntegration::time_integrator_scheme_is_explicit(
                  time_integrator.integrator_type),
                dealii::ExcMessage("The advection of the narrow-band level set requires an "
                                   "explicit time integration scheme."));
  }

  template struct NarrowBandData<double>;
} // namespace MeltPoolDG::LevelSet
