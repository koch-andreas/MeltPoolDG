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
                        "(advected_layers). The interface must move less than one cell per time "
                        "step (warning with one layer: the interface reaches the outer boundary "
                        "of the band; with more layers: new cells in the first layer).",
                        dealii::Patterns::Integer(1));

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
        prm.add_parameter(
          "redistancing",
          extension.redistancing,
          "Geometric redistancing of the cells of the advected band (newly added cells and all "
          "cells of the layers after the reinitialization). taylor_expansion: first-order Taylor "
          "expansion phi(y) + grad phi(y) . (x - y) from the closest point y on the discrete "
          "interface. signed_distance: signed distance s |x - y| with the tracked sign s of the "
          "cell, which only changes if the interface passes through the cell; the redistancing "
          "never creates an interface, even if the cut cell at the closest point is under-resolved "
          "(e.g., at the tip of a thin filament): if the polynomial of a redistanced cell changes "
          "its sign, it is shifted such that the minimum of |phi| is 'sign clamp epsilon' * h "
          "(if 'sign clamp epsilon' is zero: 1/10 of the undershoot, such that the modification "
          "vanishes with the undershoot). scaled_signed_distance: as signed_distance, but scaled "
          "with the gradient magnitude at the closest point, s |grad phi(y)| |x - y| (limited to "
          "[1/2, 2]), i.e., the magnitude of the Taylor expansion, which is consistent with a "
          "level set of the cut cells, which is not a signed distance function (e.g., between "
          "two reinitializations).");
        prm.add_parameter(
          "layer redistancing",
          extension.layer_redistancing,
          "Redistancing of all cells of the layers of the advected band: after each "
          "reinitialization of the cut cells (after_reinitialization) or in every time step "
          "(every_time_step). The latter allows to reinitialize less often (e.g., 'reinitialization "
          "displacement'), while the layers, which provide the inflow data of the cut cells, do "
          "not degrade by the inflow condition at the outer boundary of the band.");
        prm.add_parameter(
          "cut clipped cells",
          extension.cut_clipped_cells,
          "Advected band with tracked signs (signed_distance, scaled_signed_distance): a cell, "
          "which is not cut, but at whose vertices or faces an adjacent cut cell has the other "
          "sign (the interface passes through a corner or along a face of the cell, which the "
          "discontinuous level set of the cell itself does not resolve), is redistanced by the "
          "Taylor expansion from the closest point on the interface of the cut cells, such that it "
          "becomes a cut cell. If the quadrature generator finds no interface in it afterwards "
          "(e.g., an interface, which is tangent to a face and dips slightly into the cell), it is "
          "excluded from the redistancing of the layers in this time step, which would otherwise "
          "remove this part of the interface with the tracked sign. Cells cleared by the thin "
          "feature removal are excluded, until they are cut again. Recommended for resolved "
          "interfaces and high orders; in strongly deformed, under-resolved regions, many cells "
          "may be clipped in every time step.");
        extension.linear_solver.add_parameters(prm);
      }
      prm.leave_subsection();

      prm.enter_subsection("thin feature removal");
      {
        prm.add_parameter(
          "thickness",
          thin_feature_removal.thickness,
          "Removal of thin features, which cannot be resolved by the mesh (DG advection of the "
          "band: types extension and advected_layers). After the advection, a cell K of the band "
          "is cleared from a phase, if the thickness 2 |K cap phase| / |interface in K| of the "
          "phase is below thickness * h (h: cell size) and if two boundaries of the phase in K "
          "(branches of the interface or faces, at which the face neighbor has the other sign) of "
          "opposite orientation enclose a region of this phase and are closer than thickness * h, "
          "or if the phase does not continue into any face neighbor of K (e.g., "
          "the remainder of a filament, whose neighboring part has been removed). The part of "
          "each face in the phase is determined by a face quadrature. The nodal "
          "values of the removed phase are set to the other phase. A cell is not cleared, if a "
          "face neighbor would have the other sign in the whole cell afterwards (two face "
          "neighbors, which would be cleared to different phases: only the one with the thinner "
          "removed phase), since the interface would lie on the shared face. A thin filament thus "
          "disappears cell by cell from its tip, once it is thinner than thickness * h, instead "
          "of creating spurious interfaces. Set to 0 to disable the removal.",
          dealii::Patterns::Double(0.0));
        prm.add_parameter("epsilon",
                          thin_feature_removal.epsilon,
                          "Minimum magnitude epsilon * h of the level set in a cleared cell.",
                          dealii::Patterns::Double(0.0));
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
        "level set space",
        level_set_space,
        "Finite element space of the level set on the cut cells of the semi-Lagrangian band: "
        "discontinuous (FE_DGQ) or continuous (FE_Q of the same degree, continuous across the "
        "faces between cut cells). For the continuous space, the reinitialization (fixed-point "
        "iteration without face terms, direct solver) and the subcell distance (L2-projection on "
        "the union of the unresolved cells with natural boundary conditions, or nodal "
        "interpolation) are computed in the continuous space; the semi-Lagrangian step maps a "
        "continuous level set to a continuous level set. The subcell coupling excluded is not "
        "available for the continuous space.");
      prm.add_parameter(
        "closed interface correction",
        closed_interface_correction,
        "Continuous level set space: correction of the level set after each time step (after the "
        "reinitialization and the subcell distance, and after the initial reinitialization), such "
        "that the zero level set is closed: on each face between a cut cell and a cell, which is "
        "not cut (excluding the boundary of the domain), the level set has the sign s of the "
        "neighbor cell. The minimum of s phi on a face is computed with the Bernstein coefficients "
        "of the restriction of the level set to the face and de Casteljau subdivision (any "
        "polynomial degree); faces, on which the interface only touches the face, are not "
        "changed. Where the interface crosses a face, the cell behind the face is added to the cut "
        "cells (closed interface activation; the nodes of the face keep their values, the other "
        "nodes of the cell take the values of the semi-Lagrangian step), since the "
        "reinitialization has moved the interface into this cell. On the remaining faces, all "
        "nodal values of the face are shifted by s (c h - m) (m: minimum of s phi on the face, c: "
        "closed interface offset), i.e., the interface is moved back into the cut cell. Thus, the "
        "interface only ends at the boundary of the domain. Not used for the discontinuous space.");
      prm.add_parameter(
        "closed interface activation",
        closed_interface_activation,
        "Closed-interface correction: add the cells, which are not cut, behind faces crossed by "
        "the interface to the cut cells (instead of moving the interface back into the cut cell). "
        "Without the activation, the correction pushes the interface back, when the "
        "reinitialization moves it towards the points into a cell, which the advected level set "
        "does not cut, which delays the interface and accumulates errors for small time steps.");
      prm.add_parameter("closed interface offset",
                        closed_interface_offset,
                        "Closed-interface correction: minimum c of s phi on a lifted face relative "
                        "to the cell size (distance of the corrected interface from the face).",
                        dealii::Patterns::Double(0.0));
      prm.add_parameter(
        "level set update cfl",
        level_set_update_cfl,
        "Semi-Lagrangian band: sub-cycling of the level-set update for small time steps. If "
        "positive, the level set of a time step is evaluated from the last reference level set at "
        "the foot points of the nodes at the time of the reference (backward integration of the "
        "characteristics over all time steps since the reference or characteristic map, see "
        "'level set update characteristics'), i.e., without accumulating the interpolation errors "
        "of the intermediate time steps; these level sets are only corrected by the "
        "closed-interface correction. The reinitialization, the insertion of new interface points and the update "
        "of the reference are performed, once the maximum displacement of the nodes since the "
        "reference exceeds this value times the cell size (the reinitialization frequency then "
        "counts these updates). Thus, the number of these operations, whose errors accumulate, "
        "does not grow with the number of time steps. Recommended: 0.5 (at most 0.9, since the "
        "interface must not leave the candidate cells of the reference). Zero: update in every "
        "time step.",
        dealii::Patterns::Double(0.0, 0.9));
      prm.add_parameter(
        "level set update characteristics",
        level_set_update_characteristics,
        "Sub-cycling of the level-set update (level set update cfl > 0): computation of the foot "
        "points of the nodes at the time of the reference. integration: backward integration of "
        "the characteristics from the current time to the time of the reference (one "
        "fourth-order Runge-Kutta step per time step since the reference), which requires the "
        "velocity at all these times. map: characteristic map, i.e., the displacement D(x) = x - "
        "X(t_ref; x, t) of the nodes since the reference is stored as a continuous (FE_Q) field "
        "on the candidate cells of the reference and composed in each time step with the "
        "backward characteristic of the time step, D^{n+1}(x) = x - y + D^n(y) with y = X(t^n; "
        "x, t^{n+1}) (one fourth-order Runge-Kutta step), which only requires the velocity of the "
        "current time step. The interpolation error of D^n at y is proportional to the "
        "displacement of the time step, i.e., it does not accumulate with the number of time "
        "steps.");
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
        "quadrature points of the zero level set of the level set at the end of the time step (or "
        "of the advected level set phi, see 'point insertion'), whose distance to all points "
        "exceeds h / n, are added, and in cells with more than 2 n^(dim-1) points, the "
        "closest points are merged (n Gauss points per direction, default p + 1). Points with a "
        "point of opposite orientation closer than 2 h (under-resolved interface) are not used. "
        "The cut cells are determined from phi; in cut cells without usable points, the zero "
        "level set of phi is used. The factors 0.2 and 2 and the number n are set in the "
        "subsection 'advected points'.");
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
      prm.add_parameter(
        "reinitialization displacement",
        reinitialization_displacement,
        "Reinitialize the level-set field on the cut cells, once the displacement of the "
        "interface since the last reinitialization, estimated by the sum of max |u| dt over the "
        "time steps (maximum of the velocity on the band at the beginning of the time step), "
        "reaches this length (absolute length, not relative to the cell size). The number of "
        "reinitializations then depends on the distance travelled by the interface, but neither "
        "on the time step size nor on the cell size. Combined with 'reinitialization frequency' "
        "and 'reinitialization eikonal threshold' (a reinitialization is performed, if any "
        "criterion holds); set 'reinitialization frequency' to 0 to use this criterion only. For "
        "the advected band, combine it with 'extension/layer redistancing = every_time_step'. "
        "Set to 0 to disable the displacement-based reinitialization.",
        dealii::Patterns::Double(0.0));

      prm.enter_subsection("advected points");
      {
        prm.add_parameter(
          "unresolved distance",
          advected_points.unresolved_distance,
          "Distance relative to the cell size, below which advected interface points of opposite "
          "orientation are not used in the reinitialization (unresolved interface, e.g., both "
          "sides of a thin filament); the treatment of the cells around them is set by "
          "'unresolved cells'. Only used for reinitialization interface = advected_points.",
          dealii::Patterns::Double(0.0));
        prm.add_parameter(
          "support tolerance",
          advected_points.support_tolerance,
          "Advected interface points, at which the magnitude of the advected level set exceeds "
          "this value times the cell size, are removed (parts of the interface, which the level "
          "set has lost). Only used for reinitialization interface = advected_points.",
          dealii::Patterns::Double(0.0));
        prm.add_parameter(
          "merge distance",
          advected_points.merge_distance,
          "Merge distance c_m relative to the cell size of the merge and break-up rules for "
          "topology changes. Merge rule (e.g., merging droplets): with the approach a = dt max(0, "
          "(u(x - h n) - u(x)) . n) of the flow on the outer side of an advected interface point "
          "within one time step, a point in a converging flow (the positive part of the normal "
          "velocity difference (u(x - h n) - u(x)) . n exceeds a quarter of the magnitude "
          "|u(x - h n) - u(x)| of the velocity difference, independent of a superposed uniform "
          "translation) is "
          "removed, if the advected level set at the distance c_m h + a on the outer side of the "
          "point (opposite to its normal vector) exceeds the value at the point by more than a "
          "quarter of this distance (the point lies inside of the phase or at a gap thinner than "
          "about c_m h + a), or if a point of a different orientation (n_i . n_j < 1/2) is closer "
          "than the sum of the approaches of the two points and each of them lies on the outer "
          "side of the other one (the gap has closed within the time step). Break-up rule (e.g., "
          "a neck, which pinches off): the same on the inner side of the point (in the direction "
          "of its normal vector; the point is removed, if the level set on the inner side is "
          "smaller than the value at the point by more than a quarter of the distance, or if a "
          "point of a different "
          "orientation within the sum of the approaches lies on the inner side, face to face); in "
          "addition, the points, which have crossed a line of converging characteristics within "
          "the time step (the velocity changes along the path by D = u(x^n, t^n) - u(x^{n+1}, "
          "t^{n+1}), and the velocity at x^{n+1} - dt D differs from the velocity at the new "
          "position by more than |D| / 2), are removed (e.g., the two sides of a tip, which are "
          "squeezed together). The "
          "points move forward in time and cannot represent the disappearance of colliding parts "
          "of the interface, while the advected level set (backward characteristics) already "
          "contains the merged or broken-up interface. Wider gaps, thicker parts of the phase and "
          "thin filaments in shear or rotating flows are kept. Zero disables both rules. Only "
          "used for reinitialization interface = advected_points.",
          dealii::Patterns::Double(0.0));
        prm.add_parameter(
          "points per direction",
          advected_points.points_per_direction,
          "Number n of Gauss points per direction of the surface quadrature, at which the advected "
          "interface points are seeded on the initial zero level set and inserted into gaps of the "
          "point set (0: n = p + 1, i.e., p + 1 points per cut cell in 2D and (p + 1)^2 in 3D). It "
          "also sets the minimum distance h / n of an inserted point to all points and the maximum "
          "number 2 n^(dim-1) of points per cell, above which the closest points are merged. More "
          "points cover small pieces of the interface in the cut cells (fewer cut cells, whose "
          "penalty term falls back to the zero level set of the advected level set), pin the zero "
          "level set more tightly between the points and shorten the patches of the subcell "
          "distance, at a cost proportional to n^(dim-1). Only used for reinitialization interface "
          "= advected_points.");
        prm.add_parameter(
          "point insertion",
          advected_points.point_insertion,
          "Level set, from whose zero level set new advected interface points are inserted into "
          "the gaps of the point set (Gauss points farther than h / n from all points): the "
          "advected level set before the reinitialization (advected_level_set) or the level set at "
          "the end of the time step, i.e., after the reinitialization, the subcell distance and "
          "the closed-interface correction (final_level_set). In the latter case, the new points "
          "lie on the corrected interface, and the reinitialization of the time step uses the zero "
          "level set of the advected level set in the cut cells without usable points. Only used "
          "for reinitialization interface = advected_points.");
        prm.add_parameter(
          "conservative insertion",
          advected_points.conservative_insertion,
          "If false, a new advected interface point is added with the weight of its quadrature "
          "point. Since the weights of the existing points grow with the stretching of the "
          "interface (Nanson's formula), they already contain the area of the gap, in which the "
          "new point is inserted, and this area is counted twice. If true, the weights of all "
          "points of a cell with new points are scaled by W / (W + W*) with the sums W and W* of "
          "the weights of the existing and the new points of the cell, i.e., the sum of the "
          "weights of the cell is conserved; in cells without existing points (e.g., cells "
          "emptied by the merge or break-up rules), the new points keep their weights. The sum of "
          "the weights then approximates the area of the interface. The weights enter the penalty "
          "term of the reinitialization and the size of the patches of the subcell distance. Only "
          "used for reinitialization interface = advected_points.");
        prm.add_parameter(
          "unresolved cells",
          advected_points.unresolved_cells,
          "Treatment of the cut cells, which contain unresolved points or are closer than "
          "'subcell buffer' times the cell size to an unresolved point. zero_level_set: the "
          "reinitialization uses the zero level set of the advected level set in the cut cells "
          "without usable points (and only the usable points in the other cut cells). "
          "subcell_distance: the level set of these cells is given by the L2-projection of the "
          "signed distance to the advected points (all points, "
          "evaluated at the points of an iterated Gauss quadrature with 'subcell subdivisions' "
          "subdivisions per direction; each point represents a flat patch of its weight with its "
          "normal vector), i.e., the reinitialization of these cells is replaced by an exact "
          "redistancing at the subcell scale, whose result is projected onto the polynomials of "
          "the cells. Beyond the edges of the patches (e.g., the open end of a chain of points), "
          "the signed distance takes the sign of the advected level set, and a cell keeps the "
          "reinitialized level set, if the signed distance has a different sign than the "
          "advected level set at a quadrature point farther than a quarter of the cell size from "
          "the points (the points do not create new parts of the interface); the coupling to the "
          "reinitialization of the other cut cells is set by 'subcell coupling' and the nodal "
          "interpolation instead of the projection by 'subcell transfer'. Only used for "
          "reinitialization interface = advected_points.");
        prm.add_parameter("subcell buffer",
                          advected_points.subcell_buffer,
                          "Distance relative to the cell size, within which the cut cells around "
                          "unresolved points are treated as unresolved cells.",
                          dealii::Patterns::Double(0.0));
        prm.add_parameter("subcell subdivisions",
                          advected_points.subcell_subdivisions,
                          "Number of subdivisions per direction of the iterated Gauss quadrature "
                          "of the L2-projection of the subcell distance and of the sign check.",
                          dealii::Patterns::Integer(1));
        prm.add_parameter(
          "subcell coupling",
          advected_points.subcell_coupling,
          "Coupling of the cells of the subcell distance to the reinitialization of the other cut "
          "cells. overwrite: the projection replaces the reinitialized level set in these cells "
          "after the reinitialization (the reinitialization is solved on all cut cells, and the "
          "level set may jump at the faces between projected and reinitialized cells). "
          "prescribed: the projection is computed before the reinitialization, which keeps it as "
          "prescribed values in these cells, i.e., only the other cut cells are reinitialized, "
          "and they are coupled to the projected cells by the SIPG face terms (weak Dirichlet "
          "data). excluded: as prescribed, but without the face terms between projected and "
          "reinitialized cells (natural boundary condition of the reinitialized cells at these "
          "faces). With prescribed and excluded, the reinitialization does not contain the "
          "conflicting terms of the unresolved cells and needs considerably fewer fixed-point "
          "iterations; with prescribed, errors of the projection (e.g., remnants at the tips of "
          "filaments) can spread into the adjacent cells.");
        prm.add_parameter(
          "subcell transfer",
          advected_points.subcell_transfer,
          "Representation of the signed distance to the advected points in the finite element "
          "space of the unresolved cells. projection: L2-projection of the signed distance "
          "evaluated at the points of the iterated Gauss quadrature with 'subcell subdivisions' "
          "subdivisions per direction, which resolves the kink of the distance on the medial axis "
          "of a thin filament inside the cells (best approximation by the polynomials of the "
          "cells). interpolation: nodal interpolation, i.e., the signed distance is only evaluated "
          "at the nodes of the cells (no linear system; the quadrature points are still used for "
          "the sign check). The interpolation samples the kink of the distance only at the nodes, "
          "which places the zeros of thin filaments less accurately and loses thin filaments "
          "already at larger widths than the projection, depending on their position relative "
          "to the nodes. Only available for the continuous level set space.");
        prm.add_parameter(
          "subcell orientation filter",
          advected_points.subcell_orientation_filter,
          "If true, the subcell distance of a cell, whose points all have the same orientation "
          "(pairwise n_i . n_j > 0), is only computed to the points of this orientation, i.e., to "
          "the branch of the interface in this cell, and not to a close branch of opposite "
          "orientation in a neighbor cell (e.g., the other side of a narrow gap or of a thin "
          "filament, whose sides lie in different cells). The kink of the distance on the medial "
          "axis then lies on the face between the cells (jump of the level set) instead of inside "
          "a cell, where the polynomials cannot represent it. Cells with points of both "
          "orientations (e.g., a thin filament inside the cell) or without points use all points. "
          "Only recommended for interfaces at rest (e.g., close particles): for advected "
          "interfaces, the semi-Lagrangian step evaluates the polynomials of the cells anywhere in "
          "the cells, where the distance to one branch overestimates the magnitude of the level "
          "set between the branches, so that thin filaments and gaps widen.");
        prm.add_parameter(
          "subcell pinning penalty",
          advected_points.subcell_pinning_penalty,
          "Dimensionless penalty parameter c of the pinned subcell projection (zero: plain "
          "L2-projection). In a projected cell, whose points all have the same orientation (one "
          "branch of the interface in the cell), the projection minimizes the L2-error to the "
          "signed distance to all points plus c h sum_k w_k phi(x_k)^2 over the points x_k of the "
          "cell with the weights w_k. The penalty keeps the zero level set of the cell on its "
          "points, while the distance to a close branch of opposite orientation in a neighbor cell "
          "(e.g., the other side of a narrow gap) still determines the level set away from the "
          "points: the kink of the distance on the medial axis only perturbs the level set away "
          "from the interface. Cells with points of both orientations are not pinned (the "
          "polynomials may not vanish at the points of both branches). Recommended for interfaces "
          "at rest or weakly deformed interfaces (e.g., close particles): for strongly stretched "
          "thin filaments, the penalty also keeps fragments of filaments, which the L2-projection "
          "would remove, so that small droplets and wiggles remain.");
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
        prm.add_parameter(
          "sign constraints",
          continuous_level_set.sign_constraints,
          "Enforce the signs of the cells, which are not cut, on the continuous level set, such "
          "that its zero level set does not leave the cut cells (it may otherwise cut the corner "
          "of a neighboring cell, where the continuous level set is not defined, if the DG level "
          "set jumps near a vertex of the band boundary). If the continuous level set has the "
          "wrong sign (or zero) at a node or between the nodes (checked at 4p+1 points per "
          "direction) of a face, edge or vertex shared with a cell, which is not cut, all nodal "
          "values of this face, edge or vertex are constrained to their values with the sign s of "
          "this cell and at least the magnitude c h (c: 'sign constraint offset'; the interior "
          "nodal values are raised further, if the sign still changes between the nodes), and the "
          "projection is repeated (at most five times). The numbers of constrained nodal values "
          "and of projections are printed for verbosity level 1. Requires narrow band/type = "
          "semi_lagrangian.");
        prm.add_parameter(
          "sign constraint offset",
          continuous_level_set.sign_constraint_offset,
          "Magnitude c of the constrained nodal values s c h of the sign constraints relative to "
          "the cell size h.",
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
        AssertThrow(n_layers >= 1,
                    dealii::ExcMessage("The advected band requires at least one layer."));
        AssertThrow(neighbor_activation_volume_fraction == 0.,
                    dealii::ExcMessage("The neighbor activation volume fraction is not "
                                       "supported by the advected band."));
        AssertThrow(neighbor_activation == NarrowBandNeighborActivation::all_face_neighbors,
                    dealii::ExcMessage("The downstream neighbor activation is not supported by "
                                       "the advected band."));
      }

    AssertThrow(extension.redistancing == NarrowBandRedistancing::taylor_expansion or
                  type == NarrowBandType::advected_layers,
                dealii::ExcMessage("The redistancing signed_distance (scaled_signed_distance) "
                                   "requires the advected band (narrow band/type = "
                                   "advected_layers)."));
    AssertThrow(extension.layer_redistancing ==
                    NarrowBandLayerRedistancing::after_reinitialization or
                  type == NarrowBandType::advected_layers,
                dealii::ExcMessage("The layer redistancing every_time_step requires the advected "
                                   "band (narrow band/type = advected_layers)."));
    AssertThrow(reinitialization_displacement == 0. or type != NarrowBandType::semi_lagrangian,
                dealii::ExcMessage("The displacement-based reinitialization requires the DG "
                                   "advection of the narrow band (narrow band/type = extension "
                                   "or advected_layers); for the semi-Lagrangian band, use the "
                                   "sub-cycling ('level set update cfl')."));
    AssertThrow(not extension.cut_clipped_cells or
                  (type == NarrowBandType::advected_layers and
                   extension.redistancing != NarrowBandRedistancing::taylor_expansion),
                dealii::ExcMessage("The option 'cut clipped cells' requires the advected band "
                                   "with tracked signs (narrow band/type = advected_layers, "
                                   "extension/redistancing = signed_distance or "
                                   "scaled_signed_distance)."));

    AssertThrow(thin_feature_removal.thickness == 0. or type != NarrowBandType::semi_lagrangian,
                dealii::ExcMessage("The thin feature removal requires the DG advection of the "
                                   "narrow band (narrow band/type = extension or "
                                   "advected_layers)."));
    AssertThrow(thin_feature_removal.thickness == 0. or thin_feature_removal.epsilon > 0.,
                dealii::ExcMessage("The epsilon of the thin feature removal must be positive."));

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

    AssertThrow(level_set_update_cfl == 0. or type == NarrowBandType::semi_lagrangian,
                dealii::ExcMessage("The sub-cycling of the level-set update requires the "
                                   "semi-Lagrangian narrow band."));

    if (level_set_space == NarrowBandLevelSetSpace::continuous)
      {
        AssertThrow(type == NarrowBandType::semi_lagrangian,
                    dealii::ExcMessage("The continuous level set space requires the "
                                       "semi-Lagrangian narrow band."));
        AssertThrow(advected_points.subcell_coupling != NarrowBandSubcellCoupling::excluded,
                    dealii::ExcMessage("The subcell coupling excluded is not available for the "
                                       "continuous level set space."));
        AssertThrow(not advected_points.subcell_orientation_filter and
                      advected_points.subcell_pinning_penalty == 0.,
                    dealii::ExcMessage("The orientation filter and the pinning penalty of the "
                                       "subcell distance are not available for the continuous "
                                       "level set space."));
      }
    else
      AssertThrow(advected_points.subcell_transfer == NarrowBandSubcellTransfer::projection,
                  dealii::ExcMessage("The subcell transfer interpolation is only available for "
                                     "the continuous level set space."));

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
    AssertThrow(not continuous_level_set.enable or not continuous_level_set.sign_constraints or
                  type == NarrowBandType::semi_lagrangian,
                dealii::ExcMessage("The sign constraints of the continuous level set require the "
                                   "signs of the cells, which are not cut, i.e., the "
                                   "semi-Lagrangian narrow band (narrow band/type = "
                                   "semi_lagrangian)."));
    AssertThrow(not continuous_level_set.sign_constraints or
                  continuous_level_set.sign_constraint_offset > 0.,
                dealii::ExcMessage("The sign constraint offset must be positive."));

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
