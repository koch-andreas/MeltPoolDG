#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/non_matching/mesh_classifier.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/level_set/narrow_band_level_set_operation.hpp>
#include <meltpooldg/level_set/narrow_band_utilities.hpp>
#include <meltpooldg/time_integration/time_integrator_util.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <string>

namespace MeltPoolDG::LevelSet
{
  template <int dim, typename number>
  NarrowBandLevelSetOperation<dim, number>::NarrowBandLevelSetOperation(
    ScratchData<dim, dim, number>                &scratch_data_in,
    const dealii::Triangulation<dim>             &triangulation,
    const NarrowBandData<number>                 &narrow_band_data_in,
    const ReinitializationData<number>           &reinit_data_in,
    const unsigned int                            fe_degree_in,
    const std::shared_ptr<dealii::Function<dim>> &velocity_function)
    : scratch_data(scratch_data_in)
    , narrow_band_data(narrow_band_data_in)
    , reinit_data(reinit_data_in)
    , fe_degree(fe_degree_in)
    , fe_collection_band(narrow_band_data_in.type == NarrowBandType::advected_layers ?
                           NarrowBand::create_layered_band_fe_collection<dim>(fe_degree_in) :
                           NarrowBand::create_band_fe_collection<dim>(fe_degree_in))
    , fe_collection_cut(narrow_band_data_in.type == NarrowBandType::advected_layers ?
                          NarrowBand::create_layered_cut_fe_collection<dim>(fe_degree_in) :
                          NarrowBand::create_cut_fe_collection<dim>(fe_degree_in))
    , dof_handler_band(triangulation)
    , dof_handler_cut(triangulation)
    , band_dof_idx(
        scratch_data_in.attach_dof_handler_and_constraint(dof_handler_band, constraints_band))
    , cut_dof_idx(
        scratch_data_in.attach_dof_handler_and_constraint(dof_handler_cut, constraints_cut))
    , quad_idx(scratch_data_in.attach_quadrature(dealii::QGauss<dim>(fe_degree_in + 1)))
    , ghost_penalty_extension(narrow_band_data_in.extension,
                              scratch_data_in.get_mapping(),
                              scratch_data_in.get_pcout(1))
    , closest_point_extension(narrow_band_data_in.extension,
                              scratch_data_in.get_mapping(),
                              scratch_data_in.get_pcout(1))
    , advection_operator(scratch_data_in, band_dof_idx, quad_idx)
  {
    narrow_band_data.check_input_parameters(fe_degree);

    AssertThrow(reinit_data.modeltype == ModelType::elliptic and
                  reinit_data.elliptic.intersected_cells_only,
                dealii::ExcMessage("The narrow-band level set requires the elliptic "
                                   "reinitialization on intersected cells only."));

    advection_operator.set_velocity_function(velocity_function);

    time_integrator.reset(TimeIntegration::explicit_time_integrator_factory<number>(
      [this](const number time,
             const number /*time_step*/,
             VectorType                             &dst,
             const VectorType                       &src,
             std::function<void(unsigned, unsigned)> post) {
        advection_operator.apply_operator(time, dst, src, post);
      },
      narrow_band_data.time_integrator));

    AssertThrow(time_integrator,
                dealii::ExcMessage("The time integrator of the narrow-band advection could not "
                                   "be created."));

    solution_history_band.resize(std::max(1U, time_integrator->required_solution_history_size()));

    reinit_operation = std::make_unique<ReinitializationEllipticOperation<dim, number>>(
      scratch_data, reinit_data, cut_dof_idx, quad_idx, cut_dof_idx);
    reinit_operation->set_face_terms_on_interface_faces_only(
      narrow_band_data.reinitialization_faces == NarrowBandReinitializationFaces::interface_faces);
  }

  template <int dim, typename number>
  void
  NarrowBandLevelSetOperation<dim, number>::setup_dof_system(
    const std::vector<bool> &is_cut_cell,
    const std::vector<bool> &is_small_cut_cell_in)
  {
    ScopedName                 sc("narrow_band::setup_dof_system");
    dealii::TimerOutput::Scope scope(scratch_data.get_timer(), sc);

    is_small_cut_cell = is_small_cut_cell_in;

    if (is_advected_band())
      NarrowBand::distribute_dofs_layered(is_cut_cell,
                                          narrow_band_data.n_layers,
                                          dof_handler_cut,
                                          fe_collection_cut,
                                          dof_handler_band,
                                          fe_collection_band);
    else
      NarrowBand::distribute_dofs(is_cut_cell,
                                  dof_handler_cut,
                                  fe_collection_cut,
                                  dof_handler_band,
                                  fe_collection_band,
                                  is_small_cut_cell);

    // DG: no constraints
    constraints_band.clear();
    constraints_band.close();
    constraints_cut.clear();
    constraints_cut.close();

    scratch_data.create_partitioning();
    scratch_data.build(false /*boundary face integrals*/,
                       true /*inner face integrals*/,
                       true /*normal vectors*/);

    // MatrixFree forms the cell batches according to the active FE indices of the first
    // hp-DoFHandler. Check that the cell batches are uniform with respect to both DoFHandlers.
    {
      const auto &matrix_free = scratch_data.get_matrix_free();
      for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
        {
          const unsigned int category_band = matrix_free.get_cell_category(cell, band_dof_idx);
          const unsigned int category_cut  = matrix_free.get_cell_category(cell, cut_dof_idx);
          for (unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(cell); ++v)
            AssertThrow(matrix_free.get_cell_iterator(cell, v, band_dof_idx)->active_fe_index() ==
                            category_band and
                          matrix_free.get_cell_iterator(cell, v, cut_dof_idx)->active_fe_index() ==
                            category_cut,
                        dealii::ExcMessage(
                          "The cell batches of MatrixFree are not uniform with respect "
                          "to the active FE indices of the narrow band. No other "
                          "hp-DoFHandler may be attached to the ScratchData object."));
        }
    }

    scratch_data.initialize_dof_vector(level_set, cut_dof_idx);
    solution_history_band.apply(
      [this](VectorType &v) { scratch_data.initialize_dof_vector(v, band_dof_idx); });

    time_integrator->reinit(solution_history_band);
    reinit_operation->reinit();
  }

  template <int dim, typename number>
  std::vector<bool>
  NarrowBandLevelSetOperation<dim, number>::compute_small_cut_cells(
    const std::vector<bool>   &is_cut_cell,
    const std::vector<number> &volume_fractions) const
  {
    if (narrow_band_data.neighbor_activation_volume_fraction <= 0.)
      return {};

    AssertDimension(is_cut_cell.size(), volume_fractions.size());

    std::vector<bool> is_small(is_cut_cell.size(), false);
    for (unsigned int i = 0; i < is_cut_cell.size(); ++i)
      is_small[i] = is_cut_cell[i] and
                    volume_fractions[i] < narrow_band_data.neighbor_activation_volume_fraction;

    return is_small;
  }

  template <int dim, typename number>
  void
  NarrowBandLevelSetOperation<dim, number>::set_initial_condition(
    const dealii::Function<dim> &level_set_function,
    const bool                   do_reinitialize)
  {
    ScopedName                 sc("narrow_band::set_initial_condition");
    dealii::TimerOutput::Scope scope(scratch_data.get_timer(), sc);

    // 1) Determine the candidates of the cut cells from the analytical function. Note: the
    // MeshClassifier keeps a pointer to the element; thus, it must outlive the MeshClassifier.
    const dealii::FE_Q<dim>                  fe_classification(fe_degree);
    dealii::NonMatching::MeshClassifier<dim> mesh_classifier(dof_handler_band.get_triangulation(),
                                                             level_set_function,
                                                             fe_classification);
    mesh_classifier.reclassify();

    std::vector<bool> is_candidate(dof_handler_band.get_triangulation().n_active_cells(), false);
    for (const auto &cell : dof_handler_band.get_triangulation().active_cell_iterators())
      if (cell->is_locally_owned() and mesh_classifier.location_to_level_set(cell) ==
                                         dealii::NonMatching::LocationToLevelSet::intersected)
        is_candidate[cell->active_cell_index()] = true;

    setup_dof_system(is_candidate);

    // 2) Determine the cut cells from the interpolated level-set field on the band of the
    // candidates.
    VectorType &level_set_band = solution_history_band.get_current_solution();
    dealii::VectorTools::interpolate(scratch_data.get_mapping(),
                                     dof_handler_band,
                                     level_set_function,
                                     level_set_band);

    std::vector<number>     volume_fractions;
    const std::vector<bool> is_cut_cell =
      NarrowBand::compute_cut_cells(dof_handler_band, level_set_band, fe_degree, &volume_fractions);

    setup_dof_system(is_cut_cell, compute_small_cut_cells(is_cut_cell, volume_fractions));

    // 3) level-set field on the cut cells
    dealii::VectorTools::interpolate(scratch_data.get_mapping(),
                                     dof_handler_cut,
                                     level_set_function,
                                     level_set);

    if (is_advected_band())
      {
        // level-set field on the band
        dealii::VectorTools::interpolate(scratch_data.get_mapping(),
                                         dof_handler_band,
                                         level_set_function,
                                         level_set_band);

        if (do_reinitialize)
          {
            reinitialize();
            redistance_layers();
          }

        print_band_statistics();
        return;
      }

    if (do_reinitialize)
      reinitialize();

    extend();
  }

  template <int dim, typename number>
  void
  NarrowBandLevelSetOperation<dim, number>::advance_time_step(const number       time,
                                                              const number       time_step,
                                                              const unsigned int time_step_number)
  {
    ScopedName                 sc("narrow_band::advance_time_step");
    dealii::TimerOutput::Scope scope(scratch_data.get_timer(), sc);

    VectorType &level_set_band = solution_history_band.get_current_solution();

    // S4: advection on the band
    {
      ScopedName                 sc_advection("narrow_band::advection");
      dealii::TimerOutput::Scope scope_advection(scratch_data.get_timer(), sc_advection);

      level_set_band.zero_out_ghost_values();
      time_integrator->perform_time_step(time, time_step, solution_history_band);
    }

    // S1: new cut cells and band, cell-wise transfer of the level-set values of the new cut cells
    {
      ScopedName                 sc_band("narrow_band::update_band");
      dealii::TimerOutput::Scope scope_band(scratch_data.get_timer(), sc_band);

      std::vector<number>     volume_fractions;
      const std::vector<bool> is_cut_cell = NarrowBand::compute_cut_cells(dof_handler_band,
                                                                          level_set_band,
                                                                          fe_degree,
                                                                          &volume_fractions);
      std::vector<bool>       is_small_cut_cell_new =
        compute_small_cut_cells(is_cut_cell, volume_fractions);

      // The DoF system is only rebuilt if the cut cells or the cut cells, which do not activate
      // their face neighbors, have changed on any process.
      bool cut_cells_changed = is_small_cut_cell_new.size() != is_small_cut_cell.size();
      for (const auto &cell : dof_handler_band.active_cell_iterators())
        {
          if (cut_cells_changed)
            break;

          if (not cell->is_locally_owned())
            continue;

          const unsigned int index = cell->active_cell_index();
          if (is_cut_cell[index] != (cell->active_fe_index() == NarrowBand::ActiveFEIndex::cut) or
              (not is_small_cut_cell_new.empty() and
               is_small_cut_cell_new[index] != is_small_cut_cell[index]))
            cut_cells_changed = true;
        }
      cut_cells_changed =
        dealii::Utilities::MPI::logical_or(cut_cells_changed, scratch_data.get_mpi_comm());

      level_set.zero_out_ghost_values();

      if (is_advected_band())
        {
          if (cut_cells_changed)
            {
              // Transfer the advected level set of all cells of the band. The cells of the new
              // band, which do not belong to the old band, are initialized by geometric
              // redistancing.
              std::vector<bool> is_band_cell(is_cut_cell.size(), false);
              for (const auto &cell : dof_handler_band.active_cell_iterators())
                if (cell->is_locally_owned() and cell->get_fe().n_dofs_per_cell() > 0)
                  is_band_cell[cell->active_cell_index()] = true;

              const auto band_cell_values =
                NarrowBand::extract_cell_values(dof_handler_band, level_set_band, is_band_cell);

              setup_dof_system(is_cut_cell);

              VectorType &level_set_band_new = solution_history_band.get_current_solution();
              NarrowBand::set_cell_values(dof_handler_band, band_cell_values, level_set_band_new);

              std::vector<bool> is_new_cell(is_cut_cell.size(), false);
              unsigned int      n_new_first_layer_cells = 0;
              for (const auto &cell : dof_handler_band.active_cell_iterators())
                if (cell->is_locally_owned() and cell->get_fe().n_dofs_per_cell() > 0 and
                    band_cell_values[cell->active_cell_index()].size() == 0)
                  {
                    is_new_cell[cell->active_cell_index()] = true;
                    if (cell->active_fe_index() == NarrowBand::ActiveFEIndex::neighbor)
                      ++n_new_first_layer_cells;
                  }

              // new cells next to the cut cells indicate that the interface has moved more
              // than one layer within the time step
              n_new_first_layer_cells =
                dealii::Utilities::MPI::sum(n_new_first_layer_cells, scratch_data.get_mpi_comm());
              if (n_new_first_layer_cells > 0)
                Journal::print_line(scratch_data.get_pcout(0),
                                    "Warning: " + std::to_string(n_new_first_layer_cells) +
                                      " new cells in the first layer of the band.",
                                    "narrow_band");

              closest_point_extension.redistance(dof_handler_band,
                                                 level_set_band_new,
                                                 is_new_cell,
                                                 "new cells");
            }

          VectorType &level_set_band_current = solution_history_band.get_current_solution();
          NarrowBand::copy_cut_cell_values(dof_handler_band,
                                           level_set_band_current,
                                           dof_handler_cut,
                                           level_set);
        }
      else if (cut_cells_changed)
        {
          const auto cut_cell_values =
            NarrowBand::extract_cell_values(dof_handler_band, level_set_band, is_cut_cell);

          setup_dof_system(is_cut_cell, is_small_cut_cell_new);

          NarrowBand::set_cell_values(dof_handler_cut, cut_cell_values, level_set);
        }
      else
        NarrowBand::copy_cut_cell_values(dof_handler_band,
                                         level_set_band,
                                         dof_handler_cut,
                                         level_set);
    }

    // S5: reinitialization on the cut cells
    if (reinit_data.enable)
      {
        const number eikonal_error =
          NarrowBand::compute_eikonal_error(scratch_data.get_mapping(), dof_handler_cut, level_set);

        const bool do_reinitialize =
          (narrow_band_data.reinitialization_frequency > 0 and
           time_step_number % narrow_band_data.reinitialization_frequency == 0) or
          (narrow_band_data.reinitialization_eikonal_threshold > 0. and
           eikonal_error > narrow_band_data.reinitialization_eikonal_threshold);

        std::ostringstream str;
        str << "eikonal error on cut cells: " << std::scientific << std::setprecision(4)
            << eikonal_error;

        if (do_reinitialize)
          {
            reinitialize();
            str << " -> "
                << NarrowBand::compute_eikonal_error(scratch_data.get_mapping(),
                                                     dof_handler_cut,
                                                     level_set)
                << " (reinitialized)";
          }

        Journal::print_line(scratch_data.get_pcout(1), str.str(), "narrow_band");

        // advected band: geometric redistancing of the layers after the reinitialization
        if (do_reinitialize and is_advected_band())
          redistance_layers();
      }

    if (is_advected_band())
      {
        print_band_statistics();
        return;
      }

    // S2: extension to the neighbor cells
    extend();
  }

  template <int dim, typename number>
  void
  NarrowBandLevelSetOperation<dim, number>::reinitialize()
  {
    ScopedName                 sc("narrow_band::reinitialize");
    dealii::TimerOutput::Scope scope(scratch_data.get_timer(), sc);

    reinit_operation->set_initial_condition(level_set);
    reinit_operation->solve();

    level_set.zero_out_ghost_values();
    level_set.copy_locally_owned_data_from(reinit_operation->get_level_set());
  }

  template <int dim, typename number>
  bool
  NarrowBandLevelSetOperation<dim, number>::is_advected_band() const
  {
    return narrow_band_data.type == NarrowBandType::advected_layers;
  }

  template <int dim, typename number>
  void
  NarrowBandLevelSetOperation<dim, number>::redistance_layers()
  {
    ScopedName                 sc("narrow_band::redistance");
    dealii::TimerOutput::Scope scope(scratch_data.get_timer(), sc);

    VectorType &level_set_band = solution_history_band.get_current_solution();
    level_set_band.zero_out_ghost_values();

    // reinitialized level set of the cut cells
    NarrowBand::copy_cut_cell_values(dof_handler_cut, level_set, dof_handler_band, level_set_band);

    // all cells of the layers
    std::vector<bool> is_layer_cell(dof_handler_band.get_triangulation().n_active_cells(), false);
    for (const auto &cell : dof_handler_band.active_cell_iterators())
      if (cell->is_locally_owned() and (cell->active_fe_index() == NarrowBand::neighbor or
                                        cell->active_fe_index() == NarrowBand::outer))
        is_layer_cell[cell->active_cell_index()] = true;

    closest_point_extension.redistance(dof_handler_band, level_set_band, is_layer_cell, "layers");
  }

  template <int dim, typename number>
  void
  NarrowBandLevelSetOperation<dim, number>::extend()
  {
    ScopedName                 sc("narrow_band::extend");
    dealii::TimerOutput::Scope scope(scratch_data.get_timer(), sc);

    VectorType &level_set_band = solution_history_band.get_current_solution();

    level_set_band.zero_out_ghost_values();
    level_set_band = 0.0;

    NarrowBand::copy_cut_cell_values(dof_handler_cut, level_set, dof_handler_band, level_set_band);

    if (narrow_band_data.extension.type == NarrowBandExtensionType::closest_point)
      closest_point_extension.extend(dof_handler_band, level_set_band);
    else
      ghost_penalty_extension.extend(dof_handler_band, level_set_band);

    print_band_statistics();
  }

  template <int dim, typename number>
  void
  NarrowBandLevelSetOperation<dim, number>::print_band_statistics() const
  {
    std::ostringstream str;
    str << "narrow band: " << std::setw(6)
        << NarrowBand::n_cells_with_active_fe_index(dof_handler_band, NarrowBand::cut)
        << " cut cells";

    // number of cut cells, which do not activate their face neighbors
    if (narrow_band_data.neighbor_activation_volume_fraction > 0.)
      {
        const unsigned int n_small_cut_cells =
          dealii::Utilities::MPI::sum(static_cast<unsigned int>(std::count(
                                        is_small_cut_cell.begin(), is_small_cut_cell.end(), true)),
                                      scratch_data.get_mpi_comm());
        str << " (" << n_small_cut_cells << " small)";
      }

    if (is_advected_band())
      str << ", " << std::setw(6)
          << NarrowBand::n_cells_with_active_fe_index(dof_handler_band, NarrowBand::neighbor) +
               NarrowBand::n_cells_with_active_fe_index(dof_handler_band, NarrowBand::outer)
          << " layer cells, " << std::setw(8) << dof_handler_band.n_dofs() << " DoFs";
    else
      str << ", " << std::setw(6)
          << NarrowBand::n_cells_with_active_fe_index(dof_handler_band, NarrowBand::neighbor)
          << " neighbor cells, " << std::setw(8) << dof_handler_band.n_dofs() << " DoFs";

    Journal::print_line(scratch_data.get_pcout(1), str.str(), "narrow_band");
  }

  template <int dim, typename number>
  const typename NarrowBandLevelSetOperation<dim, number>::VectorType &
  NarrowBandLevelSetOperation<dim, number>::get_level_set() const
  {
    return level_set;
  }

  template <int dim, typename number>
  const typename NarrowBandLevelSetOperation<dim, number>::VectorType &
  NarrowBandLevelSetOperation<dim, number>::get_level_set_band() const
  {
    return solution_history_band.get_current_solution();
  }

  template <int dim, typename number>
  const dealii::DoFHandler<dim> &
  NarrowBandLevelSetOperation<dim, number>::get_dof_handler_cut() const
  {
    return dof_handler_cut;
  }

  template <int dim, typename number>
  const dealii::DoFHandler<dim> &
  NarrowBandLevelSetOperation<dim, number>::get_dof_handler_band() const
  {
    return dof_handler_band;
  }

  template <int dim, typename number>
  void
  NarrowBandLevelSetOperation<dim, number>::attach_output_vectors(
    GenericDataOut<dim, number> &data_out) const
  {
    level_set.update_ghost_values();
    data_out.add_data_vector(dof_handler_cut, level_set, "level_set");

    const auto &level_set_band = solution_history_band.get_current_solution();
    level_set_band.update_ghost_values();
    data_out.add_data_vector(dof_handler_band, level_set_band, "level_set_band");
  }

  template <int dim, typename number>
  number
  NarrowBandLevelSetOperation<dim, number>::compute_cfl_number(const number time,
                                                               const number time_step) const
  {
    return advection_operator.compute_cfl_number(time, time_step);
  }

  template class NarrowBandLevelSetOperation<1, double>;
  template class NarrowBandLevelSetOperation<2, double>;
  template class NarrowBandLevelSetOperation<3, double>;
} // namespace MeltPoolDG::LevelSet
