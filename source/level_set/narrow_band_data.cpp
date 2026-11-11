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

    continuous_level_set.linear_solver.solver_type         = LinearSolverType::CG;
    continuous_level_set.linear_solver.preconditioner_type = PreconditionerType::ILU;
    continuous_level_set.linear_solver.do_matrix_free      = true;
    continuous_level_set.linear_solver.max_iterations      = 1000;
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
        "elliptic reinitialization of the cut cells. semi_lagrangian: the level set is only "
        "stored on the cut cells; in each time step, it is evaluated on the cut cells and their "
        "vertex neighbors (candidate cells) by nodal interpolation of the back-traced level set "
        "(foot point of the characteristic in a cut cell: polynomial of the cut cell; otherwise: "
        "distance to the discrete interface with the sign of the cell containing the foot point). "
        "The new cut cells are determined from the candidate cells; no extension is needed. "
        "Requires a displacement below one cell per time step.");
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
          "sign clamp epsilon",
          extension.sign_clamp_epsilon,
          "Sign clamp of the extrapolated level set (extension and geometric redistancing): each "
          "extended cell is assigned to the side of the interface, on which it is located (from "
          "the traces of the adjacent cut cells or from the closest point on the interface). "
          "Nodal values of the opposite sign (or zero) are set to -epsilon * h on the negative side "
          "and to +epsilon * h on the positive side (h: cell size), such that extrapolated values "
          "never introduce a new interface. The sign clamp is applied after the sign limiter. Set "
          "to 0 to disable the sign clamp.",
          dealii::Patterns::Double(0.0));
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

        prm.add_parameter(
          "velocity",
          advection_velocity,
          "Velocity of the DG advection on the narrow band: the given velocity u (bulk) or the "
          "extension velocity u(y(x)) (extension), i.e., the given velocity at the closest point "
          "y(x) = x - phi grad phi / |grad phi|^2 on the zero level set, evaluated with the current "
          "level set at each stage of the time integration (on faces with the averages of the "
          "traces; |grad phi| is bounded from below by 0.5). The extension velocity is constant "
          "along the normals of the interface and keeps the level set of the band a signed "
          "distance, while the interface moves with u. Not available for the semi-Lagrangian band.");

        prm.enter_subsection("artificial viscosity");
        {
          prm.add_parameter(
            "factor",
            artificial_viscosity.factor,
            "Artificial viscosity of the advection in the cells flagged by the modal smoothness "
            "indicator of Persson & Peraire (2006): the term div(epsilon grad phi) is added to the "
            "advection equation (SIPG discretization on the band, natural boundary condition at "
            "the outer boundary of the band). The viscosity is constant per cell and determined "
            "at the beginning of each time step from the indicator S = log10(E_p / E_var), where "
            "E_p is the energy of the highest Legendre modes (degree p in any direction) and E_var "
            "the energy of all modes except the cell mean: epsilon = epsilon_0 for S > s_0 + "
            "kappa, epsilon = epsilon_0 / 2 (1 + sin(pi (S - s_0) / (2 kappa))) for |S - s_0| <= "
            "kappa and zero otherwise, with epsilon_0 = factor h max |u| / p (h: cell size, max "
            "|u| over the cell) and s_0 = -4 log10(p) + offset. The viscous term is integrated "
            "explicitly, i.e., large factors violate the stability limit of the explicit time "
            "integration (the maximum diffusion number epsilon dt / h^2 is printed for verbosity "
            "level 1). A factor of zero disables the artificial viscosity. Requires p >= 2.",
            dealii::Patterns::Double(0.0));
          prm.add_parameter("indicator offset",
                            artificial_viscosity.indicator_offset,
                            "Offset of the threshold s_0 = -4 log10(p) + offset of the smoothness "
                            "indicator.");
          prm.add_parameter("indicator width",
                            artificial_viscosity.indicator_width,
                            "Half width kappa of the transition of the viscosity around the "
                            "threshold s_0.",
                            dealii::Patterns::Double(0.0));
          prm.add_parameter("cells",
                            artificial_viscosity.cells,
                            "Cells, in which the artificial viscosity is applied: all cells of the "
                            "band (all_band_cells) or all cells of the band, which are not cut at "
                            "the beginning of the time step (non_cut_cells).");
        }
        prm.leave_subsection();

        prm.enter_subsection("limiter");
        {
          prm.add_parameter(
            "cells",
            advection_limiter.cells,
            "Bound-preserving limiter of the advection: after each stage of the SSP Runge-Kutta "
            "scheme, the level set of the limited cells is scaled around its cell mean, such that "
            "its values at the Gauss-Lobatto points remain within the bounds given by the minimum "
            "m and maximum M of the level set at the beginning of the time step on the domain of "
            "dependence of the cell. none: no limiter; all_band_cells: all cells of the band; "
            "non_cut_cells: all cells of the band, which are not cut at the beginning of the time "
            "step.");
          prm.add_parameter(
            "type",
            advection_limiter.type,
            "Type of the bounds. maximum_principle: local maximum principle [m - r h, M + r h]; "
            "cells with an inflow face to an inactive cell or at the boundary of the domain are "
            "not limited (unknown domain of dependence). sign: cells, whose domain of dependence "
            "has a definite sign, keep this sign, i.e., phi >= min(m, epsilon h) if m > 0 and "
            "phi <= max(M, -epsilon h) if M < 0; cells with values of both signs in their domain "
            "of dependence are not limited; inactive cells and the boundary of the domain have "
            "the sign of the adjacent band cell. In contrast to the sign preservation, the local "
            "maximum principle also clips smooth extrema of the level set (e.g., the ridge of a "
            "thin filament).");
          prm.add_parameter(
            "bounds",
            advection_limiter.bounds,
            "Approximation of the domain of dependence of a cell: the cell and all its face "
            "neighbors in the band (face_neighbors), the cell and its face neighbors in the band "
            "across inflow faces (upwind_face_neighbors), or the cell and the strips of width "
            "|u.n| dt of the face neighbors in the band along the inflow faces, which are "
            "extended by |u| dt beyond the edges of the faces (upwind_strips).");
          prm.add_parameter("relaxation",
                            advection_limiter.relaxation,
                            "Relaxation r of the bounds of the local maximum principle relative to "
                            "the cell size h: [m - r h, M + r h].",
                            dealii::Patterns::Double(0.0));
          prm.add_parameter(
            "sign epsilon",
            advection_limiter.sign_epsilon,
            "Minimum distance epsilon h of the level set from zero (relative to the "
            "cell size h), which is enforced by the sign-preserving limiter, unless "
            "the domain of dependence attains values closer to zero.",
            dealii::Patterns::Double(0.0));
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      prm.add_parameter(
        "neighbor activation",
        neighbor_activation,
        "Face neighbors of the cut cells, which are activated as neighbor cells of the band "
        "(extension band). all_face_neighbors: all face neighbors; downstream_face_neighbors: "
        "only the face neighbors, into which the velocity points across a face shared with a cut "
        "cell (at any Gauss-Lobatto point of the face at the beginning or the end of the next "
        "time step), i.e., the cells into which the interface can move. The level set is only "
        "extended to these cells. At the faces of the cut cells to inactive upstream cells, the "
        "upwind flux uses the interior trace of the cut cell, i.e., the cut cell is not "
        "influenced by extrapolated values from upstream.");
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
      prm.add_parameter(
        "reinitialization interface",
        reinitialization_interface,
        "Interface, at which the reinitialization on the cut cells enforces phi = 0 weakly by the "
        "interface penalty term: the zero level set of the level set before the "
        "reinitialization (zero_level_set) or points of the interface (advected_points), which "
        "are seeded on the initial zero level set (Gauss points of the surface quadrature) and "
        "advected with the velocity (classical fourth-order Runge-Kutta method, weights and "
        "normal vectors by Nanson's formula) independently of the interpolation of the level set "
        "(front tracking). Points outside of the band or with |phi| > 0.2 h are removed, the "
        "quadrature points of the zero level set of the advected level set phi, whose distance to "
        "all points exceeds h / (p+1), are added, and in cells with more than 2 (p+1)^(dim-1) "
        "points, the closest points are merged. Points with a point of opposite orientation "
        "closer than 2 h (under-resolved interface) are not used. The cut cells are determined "
        "from phi; in cut cells without usable points, the zero level set of phi is used. The "
        "factors 0.2 and 2 are set in the subsection 'advected points'.");
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

      prm.enter_subsection("advected points");
      {
        prm.add_parameter(
          "unresolved distance",
          advected_points.unresolved_distance,
          "Distance relative to the cell size, below which advected interface points of opposite "
          "orientation are not used in the reinitialization (unresolved interface, e.g., both "
          "sides of a thin filament); the zero level set of the advected level set is used there. "
          "Only used for reinitialization interface = advected_points.",
          dealii::Patterns::Double(0.0));
        prm.add_parameter(
          "support tolerance",
          advected_points.support_tolerance,
          "Advected interface points, at which the magnitude of the advected level set exceeds "
          "this value times the cell size, are removed (parts of the interface, which the level "
          "set has lost). Only used for reinitialization interface = advected_points.",
          dealii::Patterns::Double(0.0));
      }
      prm.leave_subsection();

      prm.enter_subsection("continuous level set");
      {
        prm.add_parameter(
          "enable",
          continuous_level_set.enable,
          "Compute a continuous level-set field psi on the cut cells (FE_Q of the same degree, "
          "continuous across the faces between cut cells) after each time step by the "
          "constrained L2-projection of the DG level set phi: (psi, w) + sum_K gamma_K (psi, "
          "w)_{Gamma cap K} = (phi, w) on the cut cells, where Gamma is the zero level set of phi. "
          "The normal vector (projection of grad psi, normalized at the nodes) and the curvature "
          "(projection of -div n) are computed from psi with the filter parameters and linear "
          "solvers of the subsections 'normal vector' and 'curvature'. A filter parameter of zero "
          "is recommended: the damping biases the normal vector and the curvature on the thin band "
          "of cut cells independently of the mesh size. The fields are written to the output as "
          "level_set_cg, normal and curvature. The continuous level set does not feed back into "
          "the DG level set.");
        prm.add_parameter(
          "interface penalty",
          continuous_level_set.interface_penalty,
          "Dimensionless factor c of the interface penalty gamma_K = c h_K (h_K: cell size), which "
          "pins the zero level set of the continuous level set to the zero level set of the DG "
          "level set in a least-squares sense. Since the DG level set is discontinuous, the "
          "interface may still move; the maximum shift is printed for verbosity level 1. Set to "
          "0 for the plain L2-projection.",
          dealii::Patterns::Double(0.0));
        continuous_level_set.linear_solver.add_parameters(prm);
      }
      prm.leave_subsection();
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
        AssertThrow(neighbor_activation == NarrowBandNeighborActivation::all_face_neighbors,
                    dealii::ExcMessage("The downstream neighbor activation is not supported by "
                                       "the advected band."));
      }

    AssertThrow(advection_velocity == NarrowBandAdvectionVelocity::bulk or
                  type != NarrowBandType::semi_lagrangian,
                dealii::ExcMessage("The extension velocity is only available for the DG "
                                   "advection of the narrow band."));

    if (type == NarrowBandType::semi_lagrangian)
      {
        AssertThrow(neighbor_activation_volume_fraction == 0. and
                      neighbor_activation == NarrowBandNeighborActivation::all_face_neighbors,
                    dealii::ExcMessage("The neighbor activation options are not supported by the "
                                       "semi-Lagrangian narrow band."));
        AssertThrow(advection_limiter.cells == NarrowBandLimiterCells::none and
                      artificial_viscosity.factor == 0.,
                    dealii::ExcMessage("The limiter and the artificial viscosity of the DG "
                                       "advection are not supported by the semi-Lagrangian "
                                       "narrow band."));
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

    AssertThrow(artificial_viscosity.factor == 0. or fe_degree >= 2,
                dealii::ExcMessage("The artificial viscosity of the narrow-band advection "
                                   "requires p >= 2."));
    AssertThrow(artificial_viscosity.factor == 0. or artificial_viscosity.indicator_width > 0.,
                dealii::ExcMessage("The indicator width of the artificial viscosity must be "
                                   "positive."));

    AssertThrow(not continuous_level_set.enable or
                  continuous_level_set.linear_solver.do_matrix_free,
                dealii::ExcMessage("The continuous level set of the narrow band is only "
                                   "implemented matrix-free. Set 'do matrix free' to true."));

    AssertThrow(advection_limiter.cells == NarrowBandLimiterCells::none or
                  time_integrator.integrator_type ==
                    TimeIntegration::TimeIntegratorSchemes::SSP_RK_stage_1_order_1 or
                  time_integrator.integrator_type ==
                    TimeIntegration::TimeIntegratorSchemes::SSP_RK_stage_2_order_2 or
                  time_integrator.integrator_type ==
                    TimeIntegration::TimeIntegratorSchemes::SSP_RK_stage_3_order_3,
                dealii::ExcMessage("The bound-preserving limiter of the narrow-band advection "
                                   "requires an SSP Runge-Kutta scheme."));
  }

  template struct NarrowBandData<double>;
} // namespace MeltPoolDG::LevelSet
