#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/non_matching/mesh_classifier.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/core/exceptions.hpp>
#include <meltpooldg/level_set/narrow_band_level_set_operation.hpp>
#include <meltpooldg/level_set/narrow_band_utilities.hpp>
#include <meltpooldg/time_integration/time_integrator_util.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>

namespace MeltPoolDG::LevelSet
{
  namespace
  {
    /**
     * Throw an exception, if no process has a cut cell, i.e., if the interface has vanished
     * (e.g., a droplet smaller than a cell) or does not intersect the domain.
     */
    void
    check_cut_cells_exist(const std::vector<bool> &is_cut_cell, const MPI_Comm mpi_comm)
    {
      const unsigned int n_cut_cells = std::count(is_cut_cell.begin(), is_cut_cell.end(), true);
      AssertThrow(dealii::Utilities::MPI::sum(n_cut_cells, mpi_comm) > 0,
                  dealii::ExcMessage("The narrow band contains no cut cells: the interface has "
                                     "vanished or does not intersect the domain."));
    }

    /**
     * Restart: append the bytes of a value of a trivially copyable type to a buffer.
     */
    template <typename T>
    void
    append_bytes(std::vector<char> &buffer, const T &value)
    {
      const char *begin = reinterpret_cast<const char *>(&value);
      buffer.insert(buffer.end(), begin, begin + sizeof(T));
    }

    /**
     * Restart: append the number of values and the values to a buffer.
     */
    template <typename number>
    void
    append_values(std::vector<char> &buffer, const dealii::Vector<number> &values)
    {
      append_bytes(buffer, static_cast<std::uint32_t>(values.size()));
      for (const number value : values)
        append_bytes(buffer, value);
    }

    /**
     * Restart: read a value of a trivially copyable type from a buffer and advance the position.
     */
    template <typename T>
    T
    read_bytes(std::vector<char>::const_iterator      &position,
               const std::vector<char>::const_iterator end)
    {
      AssertThrow(end - position >= static_cast<std::ptrdiff_t>(sizeof(T)),
                  dealii::ExcMessage("The cell-wise restart data of the narrow band is invalid."));
      T value;
      std::memcpy(&value, &*position, sizeof(T));
      position += sizeof(T);
      return value;
    }

    /**
     * Restart: read the number of values and the values written by append_values().
     */
    template <typename number>
    std::vector<number>
    read_values(std::vector<char>::const_iterator      &position,
                const std::vector<char>::const_iterator end)
    {
      const auto          n_values = read_bytes<std::uint32_t>(position, end);
      std::vector<number> values(n_values);
      for (auto &value : values)
        value = read_bytes<number>(position, end);
      return values;
    }

    /**
     * Restart: set the DoF values of a cell from the restart data.
     */
    template <typename CellIteratorType, typename VectorType>
    void
    set_restart_values(const CellIteratorType                             &cell,
                       const std::vector<typename VectorType::value_type> &values,
                       VectorType                                         &vector)
    {
      AssertThrow(values.size() == cell->get_fe().n_dofs_per_cell(),
                  dealii::ExcMessage("The restart data does not match the DoFs of the narrow "
                                     "band."));
      if (values.empty())
        return;
      const dealii::Vector<typename VectorType::value_type> cell_values(values.begin(),
                                                                        values.end());
      cell->set_dof_values(cell_values, vector);
    }
  } // namespace

  template <int dim, typename number>
  NarrowBandLevelSetOperation<dim, number>::NarrowBandLevelSetOperation(
    ScratchData<dim, dim, number>                &scratch_data_in,
    const dealii::Triangulation<dim>             &triangulation,
    const NarrowBandData<number>                 &narrow_band_data_in,
    const ReinitializationData<number>           &reinit_data_in,
    const unsigned int                            fe_degree_in,
    const std::shared_ptr<dealii::Function<dim>> &velocity_function,
    const NormalVectorData<number>               *normal_vector_data,
    const CurvatureData<number>                  *curvature_data)
    : scratch_data(scratch_data_in)
    , narrow_band_data(narrow_band_data_in)
    , reinit_data(reinit_data_in)
    , fe_degree(fe_degree_in)
    , fe_collection_band(narrow_band_data_in.type != NarrowBandType::extension ?
                           NarrowBand::create_layered_band_fe_collection<dim>(fe_degree_in) :
                           NarrowBand::create_band_fe_collection<dim>(fe_degree_in))
    , fe_collection_cut(narrow_band_data_in.type != NarrowBandType::extension ?
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
    , semi_lagrangian_advection(scratch_data_in.get_pcout(1))
    , dof_handler_cut_reference(triangulation)
    , dof_handler_band_reference(triangulation)
    , advection_operator(scratch_data_in, band_dof_idx, quad_idx)
    , velocity_function(velocity_function)
    , advection_limiter(narrow_band_data_in.advection_limiter, scratch_data_in.get_mapping())
  {
    narrow_band_data.check_input_parameters(fe_degree);

    AssertThrow(reinit_data.modeltype == ModelType::elliptic and
                  reinit_data.elliptic.intersected_cells_only,
                dealii::ExcMessage("The narrow-band level set requires the elliptic "
                                   "reinitialization on intersected cells only."));

    advection_operator.set_velocity_function(velocity_function);
    advection_operator.set_extension_velocity(narrow_band_data.advection_velocity ==
                                              NarrowBandAdvectionVelocity::extension);

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

    // advected interface points for the interface penalty term of the reinitialization
    if (narrow_band_data.reinitialization_interface ==
        NarrowBandReinitializationInterface::advected_points)
      {
        AssertThrow(reinit_data.enable,
                    dealii::ExcMessage("The reinitialization interface advected_points requires "
                                       "the reinitialization."));
        interface_points = std::make_unique<NarrowBandInterfacePoints<dim, number>>(
          scratch_data.get_pcout(1),
          narrow_band_data.advected_points.unresolved_distance,
          narrow_band_data.advected_points.support_tolerance,
          narrow_band_data.advected_points.merge_distance);
      }

    // reinitialization in the continuous level set space
    if (narrow_band_data.level_set_space == NarrowBandLevelSetSpace::continuous)
      continuous_reinitialization =
        std::make_unique<NarrowBandContinuousReinitialization<dim, number>>(
          scratch_data.get_mapping(),
          triangulation,
          reinit_data,
          fe_degree,
          fe_collection_cut,
          scratch_data.get_pcout(1));

    // continuous level set on the cut cells: its DoFHandler is attached to the scratch data after
    // the band and the cut DoFHandler
    if (narrow_band_data.continuous_level_set.enable)
      {
        // default parameters: no damping (see NarrowBandContinuousLevelSet)
        NormalVectorData<number> default_normal_vector_data;
        CurvatureData<number>    default_curvature_data;
        default_normal_vector_data.filter_parameter = 0.;
        default_curvature_data.filter_parameter     = 0.;

        continuous_level_set = std::make_unique<NarrowBandContinuousLevelSet<dim, number>>(
          scratch_data,
          narrow_band_data.continuous_level_set,
          normal_vector_data != nullptr ? *normal_vector_data : default_normal_vector_data,
          curvature_data != nullptr ? *curvature_data : default_curvature_data,
          fe_collection_cut,
          fe_degree,
          cut_dof_idx,
          quad_idx);
      }
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

    if (is_layered_band())
      NarrowBand::distribute_dofs_layered(is_cut_cell,
                                          is_semi_lagrangian() ? 1 : narrow_band_data.n_layers,
                                          dof_handler_cut,
                                          fe_collection_cut,
                                          dof_handler_band,
                                          fe_collection_band);
    else if (narrow_band_data.neighbor_activation ==
             NarrowBandNeighborActivation::downstream_face_neighbors)
      {
        AssertThrow(not neighbor_activation_times.empty(), dealii::ExcInternalError());

        // the face neighbors are only activated across the faces, across which the velocity
        // points from the cut cell into the neighbor, i.e., the interface can only move into these
        // cells within the next time step
        NarrowBand::distribute_dofs<dim>(
          is_cut_cell,
          dof_handler_cut,
          fe_collection_cut,
          dof_handler_band,
          fe_collection_band,
          is_small_cut_cell,
          [&](const typename dealii::Triangulation<dim>::active_cell_iterator &cell,
              const unsigned int                                               face_no) {
            return NarrowBand::is_downstream_face<dim>(scratch_data.get_mapping(),
                                                       *velocity_function,
                                                       neighbor_activation_times,
                                                       cell,
                                                       face_no,
                                                       fe_degree + 2);
          });
      }
    else
      NarrowBand::distribute_dofs(is_cut_cell,
                                  dof_handler_cut,
                                  fe_collection_cut,
                                  dof_handler_band,
                                  fe_collection_band,
                                  is_small_cut_cell);

    // continuous level set: same active FE indices as the cut DoFHandler
    if (continuous_level_set)
      continuous_level_set->distribute_dofs(dof_handler_cut);

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
    if (narrow_band_data.artificial_viscosity.factor > 0.)
      scratch_data.initialize_dof_vector(artificial_viscosity, band_dof_idx);
    solution_history_band.apply(
      [this](VectorType &v) { scratch_data.initialize_dof_vector(v, band_dof_idx); });

    time_integrator->reinit(solution_history_band);
    reinit_operation->reinit();

    if (continuous_level_set)
      continuous_level_set->reinit();
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
    const bool                   do_reinitialize,
    const CellwiseFunction      &cellwise_level_set_function)
  {
    ScopedName                 sc("narrow_band::set_initial_condition");
    dealii::TimerOutput::Scope scope(scratch_data.get_timer(), sc);

    // the band of the first time step: the velocity is evaluated at its current time
    neighbor_activation_times = {velocity_function->get_time()};

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
    check_cut_cells_exist(is_cut_cell, scratch_data.get_mpi_comm());

    setup_dof_system(is_cut_cell, compute_small_cut_cells(is_cut_cell, volume_fractions));

    // semi-Lagrangian band: signs of all cells from the initial level set
    if (is_semi_lagrangian())
      update_cell_signs(is_cut_cell, &level_set_function);

    // 3) level-set field on the cut cells
    if (cellwise_level_set_function)
      {
        // interpolation cell by cell at the support points
        dealii::Vector<number> values;
        for (const auto &cell : dof_handler_cut.active_cell_iterators())
          {
            if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
              continue;

            const auto &unit_support_points = cell->get_fe().get_unit_support_points();
            values.reinit(unit_support_points.size());
            for (unsigned int i = 0; i < unit_support_points.size(); ++i)
              {
                const dealii::Point<dim> point =
                  scratch_data.get_mapping().transform_unit_to_real_cell(cell,
                                                                         unit_support_points[i]);
                values[i] = cellwise_level_set_function(*cell, point);
              }
            cell->set_dof_values(values, level_set);
          }
      }
    else
      dealii::VectorTools::interpolate(scratch_data.get_mapping(),
                                       dof_handler_cut,
                                       level_set_function,
                                       level_set);

    if (is_layered_band())
      {
        // level-set field on the band
        dealii::VectorTools::interpolate(scratch_data.get_mapping(),
                                         dof_handler_band,
                                         level_set_function,
                                         level_set_band);

        // closed-interface correction: level set on the candidate cells, which are not cut
        if (is_semi_lagrangian())
          store_candidate_cell_values(dof_handler_band, level_set_band, is_cut_cell);

        if (do_reinitialize)
          {
            reinitialize();
            if (is_advected_band())
              redistance_layers();
          }

        // continuous level set space: closed interface
        close_interface();

        // advected interface points: seeding on the initial zero level set
        if (interface_points)
          interface_points->seed(dof_handler_cut,
                                 level_set,
                                 narrow_band_data.advected_points.get_points_per_direction(
                                   fe_degree));

        // sub-cycling: the initial level set is the reference (its time is the time of the first
        // time step)
        if (is_subcycling())
          {
            n_reference_updates = 0;
            set_reference(std::numeric_limits<number>::quiet_NaN());
          }

        print_band_statistics();
        update_continuous_level_set();
        return;
      }

    if (do_reinitialize)
      reinitialize();

    // advected interface points: seeding on the initial zero level set
    if (interface_points)
      interface_points->seed(dof_handler_cut,
                             level_set,
                             narrow_band_data.advected_points.get_points_per_direction(fe_degree));

    extend();
    update_continuous_level_set();
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

    // sub-cycling: the time of the initial reference is the time of the first time step
    if (is_subcycling() and std::isnan(time_reference))
      time_reference = time;

    // semi-Lagrangian band: statistics of the evaluation (maximum displacement of the nodes)
    typename NarrowBandSemiLagrangianAdvection<dim, number>::Statistics sl_statistics;

    // S4: advection on the band
    {
      ScopedName                 sc_advection("narrow_band::advection");
      dealii::TimerOutput::Scope scope_advection(scratch_data.get_timer(), sc_advection);

      level_set_band.zero_out_ghost_values();

      // stage output: level set at the beginning of the time step (extended band)
      output_stage(0, time, time_step, level_set_band);

      // semi-Lagrangian band: evaluation of the back-traced level set of the cut cells on the
      // candidate cells; with sub-cycling, of the reference level set on the candidate cells of
      // the reference (backward integration over all time steps since the reference)
      if (is_subcycling())
        sl_statistics = semi_lagrangian_advection.advect(dof_handler_cut_reference,
                                                         level_set_reference,
                                                         dof_handler_band_reference,
                                                         level_set_band_reference,
                                                         cell_signs_reference,
                                                         *velocity_function,
                                                         time_reference,
                                                         time + time_step - time_reference,
                                                         n_steps_since_reference + 1);
      else if (is_semi_lagrangian())
        sl_statistics = semi_lagrangian_advection.advect(dof_handler_cut,
                                                         level_set,
                                                         dof_handler_band,
                                                         level_set_band,
                                                         cell_signs,
                                                         *velocity_function,
                                                         time,
                                                         time_step);

      // artificial viscosity from the level set at the beginning of the time step
      else if (const auto &data = narrow_band_data.artificial_viscosity; data.factor > 0.)
        {
          const auto statistics = advection_operator.compute_artificial_viscosity(
            level_set_band,
            artificial_viscosity,
            time,
            time_step,
            data.factor,
            -4. * std::log10(static_cast<number>(fe_degree)) + data.indicator_offset,
            data.indicator_width,
            data.cells == NarrowBandViscosityCells::non_cut_cells);
          advection_operator.set_artificial_viscosity(&artificial_viscosity);

          if (statistics.n_cells > 0)
            {
              std::ostringstream str;
              str << "artificial viscosity: " << statistics.n_cells
                  << " cells, max diffusion number eps dt / h^2: " << std::scientific
                  << std::setprecision(2) << statistics.max_diffusion_number;
              Journal::print_line(scratch_data.get_pcout(1), str.str(), "narrow_band");
            }
        }

      if (is_semi_lagrangian())
        {
          // no DG advection; stage output: semi-Lagrangian advection (not available with
          // sub-cycling, where the level set is evaluated on the candidate cells of the reference)
          if (not is_subcycling())
            output_stage(1, time, time_step, level_set_band);
        }
      else
        {
          // bound-preserving limiter: bounds from the level set at the beginning of the time
          // step; limiting after each stage of the SSP Runge-Kutta scheme
          const bool use_limiter =
            narrow_band_data.advection_limiter.cells != NarrowBandLimiterCells::none;
          const unsigned int n_bounded_cells =
            use_limiter ? advection_limiter.compute_bounds(
                            dof_handler_band, level_set_band, *velocity_function, time, time_step) :
                          0;

          // stage post-processing: limiter and stage output
          typename NarrowBandBoundPreservingLimiter<dim, number>::Statistics statistics;
          unsigned int                                                       stage = 0;
          std::function<void(number, number, VectorType &, const VectorType &)>
            stage_post_processing;
          if (use_limiter or stage_output_function)
            stage_post_processing =
              [&](const number, const number, VectorType &stage_solution, const VectorType &) {
                if (use_limiter)
                  statistics += advection_limiter.limit(dof_handler_band, stage_solution);
                output_stage(++stage, time, time_step, stage_solution);
              };

          time_integrator->perform_time_step(
            time, time_step, solution_history_band, {}, stage_post_processing);

          if (statistics.n_limited_cells + statistics.n_mean_violations > 0)
            Journal::print_line(scratch_data.get_pcout(1),
                                "advection limiter: " + std::to_string(n_bounded_cells) +
                                  " bounded cells, " + std::to_string(statistics.n_limited_cells) +
                                  " limited cells, " +
                                  std::to_string(statistics.n_mean_violations) +
                                  " mean violations (sum over stages)",
                                "narrow_band");
        }
    }

    // advected interface points: Lagrangian tracking from t^n to t^{n+1}
    if (interface_points)
      interface_points->advect(*velocity_function, time, time_step);

    // S1: new cut cells and band, cell-wise transfer of the level-set values of the new cut cells
    {
      ScopedName                 sc_band("narrow_band::update_band");
      dealii::TimerOutput::Scope scope_band(scratch_data.get_timer(), sc_band);

      // level set on the candidate cells (with sub-cycling: candidate cells of the reference)
      const dealii::DoFHandler<dim> &dof_handler_evaluation =
        is_subcycling() ? dof_handler_band_reference : dof_handler_band;
      const VectorType &level_set_evaluation =
        is_subcycling() ? level_set_band_reference : level_set_band;

      std::vector<number>     volume_fractions;
      const std::vector<bool> is_cut_cell = NarrowBand::compute_cut_cells(dof_handler_evaluation,
                                                                          level_set_evaluation,
                                                                          fe_degree,
                                                                          &volume_fractions);
      check_cut_cells_exist(is_cut_cell, scratch_data.get_mpi_comm());

      // closed-interface correction: level set on the candidate cells, which are not cut (cells,
      // into which the reinitialization may move the interface)
      if (is_semi_lagrangian())
        store_candidate_cell_values(dof_handler_evaluation, level_set_evaluation, is_cut_cell);

      // advected interface points: assignment to the cells, repair of the point set
      if (interface_points)
        interface_points->locate(dof_handler_evaluation,
                                 level_set_evaluation,
                                 narrow_band_data.advected_points.get_points_per_direction(
                                   fe_degree),
                                 is_cut_cell,
                                 narrow_band_data.advected_points.point_insertion ==
                                   NarrowBandPointInsertion::advected_level_set);
      std::vector<bool> is_small_cut_cell_new =
        compute_small_cut_cells(is_cut_cell, volume_fractions);

      // semi-Lagrangian band: signs of the candidate cells, which are not cut at t^{n+1} (with
      // sub-cycling: of the candidate cells of the reference, all other cells keep the signs of
      // the reference)
      if (is_subcycling())
        {
          cell_signs = cell_signs_reference;
          update_cell_signs(is_cut_cell, nullptr, dof_handler_evaluation, level_set_evaluation);
        }
      else if (is_semi_lagrangian())
        update_cell_signs(is_cut_cell, nullptr);

      // diagnostics: new cut cells, whose interface is not connected to the previous interface
      // Note: the output stream is only active on the root process; the decision has to be
      // taken by all processes.
      if (dealii::Utilities::MPI::logical_or(scratch_data.get_pcout(1).is_active(),
                                             scratch_data.get_mpi_comm()))
        {
          const auto [n_new_cut_cells, n_unconnected_cut_cells] = NarrowBand::count_new_cut_cells(
            scratch_data.get_mapping(), dof_handler_evaluation, level_set_evaluation, is_cut_cell);
          if (n_new_cut_cells > 0)
            Journal::print_line(scratch_data.get_pcout(1),
                                "new cut cells: " + std::to_string(n_new_cut_cells) + " (" +
                                  std::to_string(n_unconnected_cut_cells) +
                                  " not connected to the previous interface)",
                                "narrow_band");
        }

      // The DoF system is only rebuilt if the cut cells or the cut cells, which do not activate
      // their face neighbors, have changed on any process. For the downstream neighbor
      // activation, the neighbor cells depend on the velocity of the next time step; thus, the
      // DoF system is rebuilt in every time step.
      neighbor_activation_times = {time + time_step, time + 2. * time_step};
      bool cut_cells_changed    = is_small_cut_cell_new.size() != is_small_cut_cell.size() or
                               narrow_band_data.neighbor_activation ==
                                 NarrowBandNeighborActivation::downstream_face_neighbors;
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
      else if (cut_cells_changed or is_subcycling())
        {
          const auto cut_cell_values = NarrowBand::extract_cell_values(dof_handler_evaluation,
                                                                       level_set_evaluation,
                                                                       is_cut_cell);

          if (cut_cells_changed)
            setup_dof_system(is_cut_cell, is_small_cut_cell_new);

          NarrowBand::set_cell_values(dof_handler_cut, cut_cell_values, level_set);

          // sub-cycling: level set on the band (output) from the candidate cells of the reference
          if (is_subcycling())
            {
              VectorType &level_set_band_new = solution_history_band.get_current_solution();
              level_set_band_new.zero_out_ghost_values();
              level_set_band_new = 0.;
              NarrowBand::copy_common_cell_values(dof_handler_band_reference,
                                                  level_set_band_reference,
                                                  dof_handler_band,
                                                  level_set_band_new);
            }
        }
      else
        NarrowBand::copy_cut_cell_values(dof_handler_band,
                                         level_set_band,
                                         dof_handler_cut,
                                         level_set);
    }

    // sub-cycling: update of the reference (reinitialization, insertion of new points), once the
    // displacement since the reference exceeds the threshold; otherwise, the level set of this
    // time step is only evaluated from the reference
    const bool update_reference = not is_subcycling() or sl_statistics.max_displacement >=
                                                           narrow_band_data.level_set_update_cfl;
    const unsigned int update_number = is_subcycling() ? n_reference_updates + 1 : time_step_number;

    // S5: reinitialization on the cut cells
    if (reinit_data.enable and update_reference)
      {
        const number eikonal_error =
          NarrowBand::compute_eikonal_error(scratch_data.get_mapping(), dof_handler_cut, level_set);

        const bool do_reinitialize =
          (narrow_band_data.reinitialization_frequency > 0 and
           update_number % narrow_band_data.reinitialization_frequency == 0) or
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

    // continuous level set space: closed interface
    close_interface();

    // advected interface points: new points in the gaps of the point set from the zero level set
    // of the final level set of the cut cells
    if (interface_points and update_reference and
        narrow_band_data.advected_points.point_insertion ==
          NarrowBandPointInsertion::final_level_set)
      {
        level_set.zero_out_ghost_values();
        interface_points->insert_points(dof_handler_cut,
                                        level_set,
                                        dof_handler_band,
                                        narrow_band_data.advected_points.get_points_per_direction(
                                          fe_degree));
      }

    // sub-cycling: new reference or evaluation from the reference
    if (is_subcycling())
      {
        std::ostringstream str;
        str << "sub-cycling: displacement since the reference " << std::scientific
            << std::setprecision(2) << sl_statistics.max_displacement << " h ("
            << n_steps_since_reference + 1 << " time steps)";
        if (update_reference)
          {
            str << ", update of the reference";
            ++n_reference_updates;
            set_reference(time + time_step);
          }
        else
          ++n_steps_since_reference;
        Journal::print_line(scratch_data.get_pcout(1), str.str(), "narrow_band");
      }

    if (is_layered_band())
      {
        print_band_statistics();
        update_continuous_level_set();
        return;
      }

    // S2: extension to the neighbor cells
    extend();

    // S6, S7: continuous level set, normal vector and curvature
    update_continuous_level_set();
  }

  template <int dim, typename number>
  void
  NarrowBandLevelSetOperation<dim, number>::reinitialize()
  {
    ScopedName                 sc("narrow_band::reinitialize");
    dealii::TimerOutput::Scope scope(scratch_data.get_timer(), sc);

    if (continuous_reinitialization)
      {
        reinitialize_continuous();
        return;
      }

    // interface penalty term: advected interface points (if they are assigned to the cells) or
    // the zero level set of the level set before the reinitialization
    if (interface_points and interface_points->is_located())
      reinit_operation->set_surface_quadrature_function(
        [this](const typename dealii::Triangulation<dim>::cell_iterator &cell) {
          return interface_points->get_surface_quadrature(cell);
        });
    else
      reinit_operation->set_surface_quadrature_function({});

    // subcell distance in the unresolved cells: advected level set for the sign check
    const bool use_subcell_distance = interface_points and interface_points->is_located() and
                                      narrow_band_data.advected_points.unresolved_cells ==
                                        NarrowBandUnresolvedCells::subcell_distance;
    const bool prescribe_subcell_distance =
      use_subcell_distance and
      narrow_band_data.advected_points.subcell_coupling != NarrowBandSubcellCoupling::overwrite;
    std::optional<VectorType> level_set_advected;
    if (use_subcell_distance)
      {
        level_set_advected.emplace(level_set);
        level_set_advected->zero_out_ghost_values();
      }

    // unresolved cells: L2-projection of the signed distance to the points at the subcell scale,
    // either before the reinitialization (prescribed values of the reinitialization) or after it
    // (replacing the reinitialized level set)
    const auto project_subcell_distance = [&](VectorType *projected_dofs) {
      const auto statistics = interface_points->project_signed_distance(
        scratch_data.get_mapping(),
        dof_handler_cut,
        *level_set_advected,
        level_set,
        narrow_band_data.advected_points.subcell_buffer,
        narrow_band_data.advected_points.subcell_subdivisions,
        projected_dofs,
        narrow_band_data.advected_points.subcell_orientation_filter,
        narrow_band_data.advected_points.subcell_pinning_penalty);
      if (statistics.n_cells + statistics.n_rejected_cells > 0)
        Journal::print_line(scratch_data.get_pcout(1),
                            "subcell distance: " + std::to_string(statistics.n_cells) +
                              " unresolved cells projected, " +
                              std::to_string(statistics.n_rejected_cells) +
                              " rejected (sign check)" +
                              (narrow_band_data.advected_points.subcell_orientation_filter ?
                                 ", " + std::to_string(statistics.n_filtered_cells) +
                                   " with orientation filter" :
                                 std::string()) +
                              (narrow_band_data.advected_points.subcell_pinning_penalty > 0. ?
                                 ", " + std::to_string(statistics.n_pinned_cells) + " pinned" :
                                 std::string()),
                            "narrow_band");
      return statistics.n_cells;
    };

    VectorType subcell_dofs;
    if (prescribe_subcell_distance)
      {
        scratch_data.initialize_dof_vector(subcell_dofs, cut_dof_idx);
        level_set.zero_out_ghost_values();
        if (project_subcell_distance(&subcell_dofs) > 0)
          reinit_operation->set_prescribed_dofs(&subcell_dofs,
                                                &level_set,
                                                narrow_band_data.advected_points.subcell_coupling ==
                                                  NarrowBandSubcellCoupling::prescribed);
      }

    reinit_operation->set_initial_condition(level_set);

    // Subcell distance: if most of the interface is unresolved, the penalty and the gradient terms
    // of the reinitialization may conflict in many cells, and the fixed-point iteration may not
    // converge. Outside of the cells of the subcell distance, the advected level set is kept in
    // this case.
    bool converged = true;
    if (use_subcell_distance)
      {
        try
          {
            reinit_operation->solve();
          }
        catch (const ExcFixedPointDidNotConverge &)
          {
            converged = false;
          }
      }
    else
      reinit_operation->solve();

    if (prescribe_subcell_distance)
      reinit_operation->set_prescribed_dofs(nullptr, nullptr);

    level_set.zero_out_ghost_values();
    if (converged)
      level_set.copy_locally_owned_data_from(reinit_operation->get_level_set());
    else
      Journal::print_line(scratch_data.get_pcout(0),
                          "Warning: the reinitialization did not converge; the advected level set "
                          "is kept outside of the cells of the subcell distance.",
                          "narrow_band");

    if (use_subcell_distance and not prescribe_subcell_distance)
      project_subcell_distance(nullptr);
  }

  template <int dim, typename number>
  void
  NarrowBandLevelSetOperation<dim, number>::reinitialize_continuous()
  {
    auto &cg = *continuous_reinitialization;

    level_set.zero_out_ghost_values();
    cg.distribute_dofs(dof_handler_cut);
    cg.interpolate_from_discontinuous(dof_handler_cut, level_set);

    // interface penalty term: advected interface points or the zero level set
    typename NarrowBandContinuousReinitialization<dim, number>::SurfaceQuadratureFunction
      surface_quadrature_function;
    if (interface_points and interface_points->is_located())
      surface_quadrature_function =
        [this](const typename dealii::Triangulation<dim>::cell_iterator &cell) {
          return interface_points->get_surface_quadrature(cell);
        };

    const bool use_subcell_distance = interface_points and interface_points->is_located() and
                                      narrow_band_data.advected_points.unresolved_cells ==
                                        NarrowBandUnresolvedCells::subcell_distance;
    const bool prescribe_subcell_distance =
      use_subcell_distance and
      narrow_band_data.advected_points.subcell_coupling == NarrowBandSubcellCoupling::prescribed;

    const auto project = [&]() {
      const auto statistics =
        cg.project_subcell_distance(*interface_points,
                                    dof_handler_cut,
                                    level_set,
                                    narrow_band_data.advected_points.subcell_buffer,
                                    narrow_band_data.advected_points.subcell_subdivisions);
      if (statistics.n_cells + statistics.n_rejected_cells > 0)
        Journal::print_line(scratch_data.get_pcout(1),
                            "subcell distance (continuous): " + std::to_string(statistics.n_cells) +
                              " unresolved cells projected, " +
                              std::to_string(statistics.n_rejected_cells) +
                              " rejected (sign check)",
                            "narrow_band");
      return statistics.n_cells;
    };

    // Note: level_set holds the advected level set (sign check of the subcell distance) until it
    // is overwritten at the end.
    bool keep_prescribed_dofs = false;
    if (prescribe_subcell_distance)
      keep_prescribed_dofs = project() > 0;

    const auto statistics = cg.reinitialize(surface_quadrature_function, keep_prescribed_dofs);

    std::ostringstream str;
    str << "continuous reinitialization: " << statistics.n_dofs << " DoFs, "
        << statistics.n_iterations << " fixed-point iterations";
    if (keep_prescribed_dofs)
      str << ", " << statistics.n_prescribed_dofs << " prescribed DoFs";
    Journal::print_line(scratch_data.get_pcout(1), str.str(), "narrow_band");
    if (not statistics.converged)
      Journal::print_line(scratch_data.get_pcout(0),
                          "Warning: the continuous reinitialization did not converge; the level "
                          "set before the reinitialization is kept.",
                          "narrow_band");

    if (use_subcell_distance and not prescribe_subcell_distance)
      project();

    cg.interpolate_to_discontinuous(dof_handler_cut, level_set);
  }

  template <int dim, typename number>
  void
  NarrowBandLevelSetOperation<dim, number>::close_interface()
  {
    if (not continuous_reinitialization or not narrow_band_data.closed_interface_correction)
      return;

    ScopedName                 sc("narrow_band::close_interface");
    dealii::TimerOutput::Scope scope(scratch_data.get_timer(), sc);

    auto          &cg            = *continuous_reinitialization;
    const auto    &triangulation = dof_handler_cut.get_triangulation();
    const MPI_Comm mpi_comm      = scratch_data.get_mpi_comm();

    level_set.zero_out_ghost_values();
    cg.distribute_dofs(dof_handler_cut);
    cg.interpolate_from_discontinuous(dof_handler_cut, level_set);

    // activation: the cells, which are not cut, behind faces crossed by the interface are added
    // to the cut cells (the reinitialization has moved the interface into them); the nodes shared
    // with the cut cells keep their values, the other nodes take the level set of the
    // semi-Lagrangian step (at most two layers of cells)
    unsigned int n_activated_cells = 0;
    if (narrow_band_data.closed_interface_activation and
        candidate_cell_values.size() == triangulation.n_active_cells())
      for (unsigned int round = 0; round < 2; ++round)
        {
          // cells behind open faces of all processes (CellIds in binary form)
          std::vector<unsigned int> local_ids;
          for (const auto &id : cg.find_cells_behind_open_faces(cell_signs))
            for (const auto value : id.template to_binary<dim>())
              local_ids.push_back(value);

          std::vector<bool> is_cut_cell(triangulation.n_active_cells(), false);
          for (const auto &cell : dof_handler_cut.active_cell_iterators())
            if (cell->is_locally_owned() and cell->get_fe().n_dofs_per_cell() > 0)
              is_cut_cell[cell->active_cell_index()] = true;
          const std::vector<bool> is_old_cut_cell = is_cut_cell;

          // locally owned candidate cells, which are not cut (only these cells can be added)
          std::map<dealii::CellId, unsigned int> candidates;
          for (const auto &cell : dof_handler_cut.active_cell_iterators())
            if (cell->is_locally_owned() and not is_cut_cell[cell->active_cell_index()] and
                candidate_cell_values[cell->active_cell_index()].size() > 0)
              candidates.emplace(cell->id(), cell->active_cell_index());

          unsigned int n_cells = 0;
          for (const auto &ids : dealii::Utilities::MPI::all_gather(mpi_comm, local_ids))
            for (std::size_t k = 0; k < ids.size();
                 k += std::tuple_size_v<typename dealii::CellId::binary_type>)
              {
                typename dealii::CellId::binary_type binary;
                std::copy(ids.begin() + k, ids.begin() + k + binary.size(), binary.begin());
                const auto candidate = candidates.find(dealii::CellId(binary));
                if (candidate != candidates.end() and not is_cut_cell[candidate->second])
                  {
                    is_cut_cell[candidate->second] = true;
                    ++n_cells;
                  }
              }
          n_cells = dealii::Utilities::MPI::sum(n_cells, mpi_comm);
          if (n_cells == 0)
            break;
          n_activated_cells += n_cells;

          // level set of the enlarged set of cut cells
          auto cell_values =
            NarrowBand::extract_cell_values(dof_handler_cut, level_set, is_cut_cell);
          for (unsigned int index = 0; index < is_cut_cell.size(); ++index)
            if (is_cut_cell[index] and not is_old_cut_cell[index])
              {
                cell_values[index] = candidate_cell_values[index];
                candidate_cell_values[index].reinit(0);
              }

          setup_dof_system(is_cut_cell);
          NarrowBand::set_cell_values(dof_handler_cut, cell_values, level_set);

          cg.distribute_dofs(dof_handler_cut);
          cg.interpolate_from_discontinuous(dof_handler_cut, level_set, &is_old_cut_cell);
        }

    // remaining open faces: lift
    const auto statistics =
      cg.close_interface(cell_signs, narrow_band_data.closed_interface_offset);

    cg.interpolate_to_discontinuous(dof_handler_cut, level_set);

    std::ostringstream str;
    str << "closed interface: " << statistics.n_faces << " faces to cells, which are not cut, "
        << n_activated_cells << " cells added, " << statistics.n_lifted_faces
        << " lifted faces, max correction " << std::scientific << std::setprecision(2)
        << statistics.max_correction << " h";
    if (statistics.n_conflicting_dofs + statistics.n_open_faces > 0)
      str << ", " << statistics.n_conflicting_dofs << " nodes between faces of opposite signs, "
          << statistics.n_open_faces << " open faces";
    Journal::print_line(scratch_data.get_pcout(1), str.str(), "narrow_band");
  }

  template <int dim, typename number>
  bool
  NarrowBandLevelSetOperation<dim, number>::is_advected_band() const
  {
    return narrow_band_data.type == NarrowBandType::advected_layers;
  }

  template <int dim, typename number>
  void
  NarrowBandLevelSetOperation<dim, number>::update_cell_signs(const std::vector<bool> &is_cut_cell,
                                                              const dealii::Function<dim> *function)
  {
    update_cell_signs(is_cut_cell,
                      function,
                      dof_handler_band,
                      solution_history_band.get_current_solution());
  }

  template <int dim, typename number>
  void
  NarrowBandLevelSetOperation<dim, number>::update_cell_signs(
    const std::vector<bool>       &is_cut_cell,
    const dealii::Function<dim>   *function,
    const dealii::DoFHandler<dim> &dof_handler_band,
    const VectorType              &level_set_band)
  {
    const auto &triangulation = dof_handler_band.get_triangulation();
    AssertDimension(is_cut_cell.size(), triangulation.n_active_cells());

    if (function != nullptr or cell_signs.size() != triangulation.n_active_cells())
      cell_signs.assign(triangulation.n_active_cells(), 0);

    const auto sign = [](const number value) { return value > 0. ? 1 : (value < 0. ? -1 : 0); };

    if (function != nullptr)
      {
        // initialization of all locally owned cells
        for (const auto &cell : triangulation.active_cell_iterators())
          if (cell->is_locally_owned())
            cell_signs[cell->active_cell_index()] = sign(function->value(cell->center()));
      }
    else
      {
        // candidate cells (band), which are not cut: sign of the mean of the nodal values
        dealii::Vector<number> values;
        for (const auto &cell : dof_handler_band.active_cell_iterators())
          {
            if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0 or
                is_cut_cell[cell->active_cell_index()])
              continue;

            values.reinit(cell->get_fe().n_dofs_per_cell());
            cell->get_dof_values(level_set_band, values);
            number mean = 0.;
            for (const auto value : values)
              mean += value;
            if (const int s = sign(mean); s != 0)
              cell_signs[cell->active_cell_index()] = s;
          }
      }

    // communicate the signs of the locally owned cells to the ghost cells
    exchange_cell_signs(cell_signs);
  }

  template <int dim, typename number>
  void
  NarrowBandLevelSetOperation<dim, number>::exchange_cell_signs(std::vector<int> &signs) const
  {
    const auto &triangulation = dof_handler_cut.get_triangulation();
    AssertDimension(signs.size(), triangulation.n_active_cells());

    if (dynamic_cast<const dealii::parallel::TriangulationBase<dim> *>(&triangulation) != nullptr)
      dealii::GridTools::exchange_cell_data_to_ghosts<int, dealii::Triangulation<dim>>(
        triangulation,
        [&](const typename dealii::Triangulation<dim>::active_cell_iterator &cell)
          -> std::optional<int> { return signs[cell->active_cell_index()]; },
        [&](const typename dealii::Triangulation<dim>::active_cell_iterator &cell, const int &s) {
          signs[cell->active_cell_index()] = s;
        });
  }

  template <int dim, typename number>
  bool
  NarrowBandLevelSetOperation<dim, number>::is_semi_lagrangian() const
  {
    return narrow_band_data.type == NarrowBandType::semi_lagrangian;
  }

  template <int dim, typename number>
  void
  NarrowBandLevelSetOperation<dim, number>::store_candidate_cell_values(
    const dealii::DoFHandler<dim> &dof_handler_candidates,
    const VectorType              &level_set_candidates,
    const std::vector<bool>       &is_cut_cell)
  {
    candidate_cell_values.clear();
    if (not continuous_reinitialization or not narrow_band_data.closed_interface_correction or
        not narrow_band_data.closed_interface_activation)
      return;

    std::vector<bool> is_candidate(is_cut_cell.size(), false);
    for (const auto &cell : dof_handler_candidates.active_cell_iterators())
      if (cell->is_locally_owned() and cell->get_fe().n_dofs_per_cell() > 0 and
          not is_cut_cell[cell->active_cell_index()])
        is_candidate[cell->active_cell_index()] = true;

    candidate_cell_values =
      NarrowBand::extract_cell_values(dof_handler_candidates, level_set_candidates, is_candidate);
  }

  template <int dim, typename number>
  bool
  NarrowBandLevelSetOperation<dim, number>::is_subcycling() const
  {
    return is_semi_lagrangian() and narrow_band_data.level_set_update_cfl > 0.;
  }

  template <int dim, typename number>
  void
  NarrowBandLevelSetOperation<dim, number>::set_reference(const number time)
  {
    ScopedName                 sc("narrow_band::set_reference");
    dealii::TimerOutput::Scope scope(scratch_data.get_timer(), sc);

    // cut cells of the current level set and their candidate cells
    std::vector<bool> is_cut_cell(dof_handler_cut.get_triangulation().n_active_cells(), false);
    for (const auto &cell : dof_handler_cut.active_cell_iterators())
      if (cell->is_locally_owned() and cell->get_fe().n_dofs_per_cell() > 0)
        is_cut_cell[cell->active_cell_index()] = true;

    NarrowBand::distribute_dofs_layered(is_cut_cell,
                                        1,
                                        dof_handler_cut_reference,
                                        fe_collection_cut,
                                        dof_handler_band_reference,
                                        fe_collection_band);

    const MPI_Comm mpi_comm = scratch_data.get_mpi_comm();
    level_set_reference.reinit(dof_handler_cut_reference.locally_owned_dofs(),
                               dealii::DoFTools::extract_locally_relevant_dofs(
                                 dof_handler_cut_reference),
                               mpi_comm);
    level_set_band_reference.reinit(dof_handler_band_reference.locally_owned_dofs(),
                                    dealii::DoFTools::extract_locally_relevant_dofs(
                                      dof_handler_band_reference),
                                    mpi_comm);

    NarrowBand::copy_common_cell_values(dof_handler_cut,
                                        level_set,
                                        dof_handler_cut_reference,
                                        level_set_reference);

    cell_signs_reference    = cell_signs;
    time_reference          = time;
    n_steps_since_reference = 0;
  }

  template <int dim, typename number>
  bool
  NarrowBandLevelSetOperation<dim, number>::is_layered_band() const
  {
    return is_advected_band() or is_semi_lagrangian();
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

    if (is_semi_lagrangian())
      str << ", " << std::setw(6)
          << NarrowBand::n_cells_with_active_fe_index(dof_handler_band, NarrowBand::neighbor)
          << " candidate cells, " << std::setw(8) << dof_handler_band.n_dofs() << " DoFs";
    else if (is_advected_band())
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
  void
  NarrowBandLevelSetOperation<dim, number>::update_continuous_level_set()
  {
    if (continuous_level_set)
      continuous_level_set->compute(level_set, is_semi_lagrangian() ? &cell_signs : nullptr);
  }

  template <int dim, typename number>
  void
  NarrowBandLevelSetOperation<dim, number>::set_stage_output_function(
    const StageOutputFunction &function)
  {
    stage_output_function = function;
  }

  template <int dim, typename number>
  void
  NarrowBandLevelSetOperation<dim, number>::output_stage(const unsigned int stage,
                                                         const number       time,
                                                         const number       time_step,
                                                         const VectorType  &level_set_band_stage)
  {
    if (not stage_output_function)
      return;

    // level set of the cut cells: at the beginning of the time step the level set of the cut
    // cells, otherwise the restriction of the stage to the cut cells
    if (stage > 0)
      {
        level_set_stage.reinit(level_set);
        NarrowBand::copy_cut_cell_values(dof_handler_band,
                                         level_set_band_stage,
                                         dof_handler_cut,
                                         level_set_stage);
      }
    const VectorType &level_set_cut = stage > 0 ? level_set_stage : level_set;

    const bool level_set_cut_has_ghosts  = level_set_cut.has_ghost_elements();
    const bool level_set_band_has_ghosts = level_set_band_stage.has_ghost_elements();
    level_set_cut.update_ghost_values();
    level_set_band_stage.update_ghost_values();

    stage_output_function(stage, time, time_step, level_set_cut, level_set_band_stage);

    if (not level_set_cut_has_ghosts)
      level_set_cut.zero_out_ghost_values();
    if (not level_set_band_has_ghosts)
      level_set_band_stage.zero_out_ghost_values();
  }

  template <int dim, typename number>
  const NarrowBandContinuousLevelSet<dim, number> *
  NarrowBandLevelSetOperation<dim, number>::get_continuous_level_set() const
  {
    return continuous_level_set.get();
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

    if (continuous_level_set)
      continuous_level_set->attach_output_vectors(data_out);
  }

  template <int dim, typename number>
  number
  NarrowBandLevelSetOperation<dim, number>::compute_cfl_number(const number time,
                                                               const number time_step) const
  {
    return advection_operator.compute_cfl_number(time, time_step);
  }

  template <int dim, typename number>
  void
  NarrowBandLevelSetOperation<dim, number>::update_restart_flags()
  {
    has_small_cut_cells_on_restart = not is_small_cut_cell.empty();
    has_cell_signs_on_restart      = not cell_signs.empty();
    has_reference_on_restart = is_subcycling() and dof_handler_cut_reference.has_active_dofs();
  }

  template <int dim, typename number>
  void
  NarrowBandLevelSetOperation<dim, number>::prepare_for_serialization()
  {
    update_restart_flags();

    std::vector<const VectorType *> band_vectors;
    solution_history_band.apply([&](VectorType &vector) { band_vectors.push_back(&vector); });

    // Note: the callback is called by Triangulation::save() for each locally owned cell.
    auto &triangulation =
      const_cast<dealii::Triangulation<dim> &>(dof_handler_cut.get_triangulation());
    triangulation.register_data_attach(
      [this, band_vectors](const typename dealii::Triangulation<dim>::cell_iterator &cell,
                           const dealii::CellStatus                                  status) {
        AssertThrow(status == dealii::CellStatus::cell_will_persist, dealii::ExcNotImplemented());

        const unsigned int index     = cell->active_cell_index();
        const auto         cell_cut  = cell->as_dof_handler_iterator(dof_handler_cut);
        const auto         cell_band = cell->as_dof_handler_iterator(dof_handler_band);
        const bool         is_cut    = cell_cut->get_fe().n_dofs_per_cell() > 0;
        const bool         is_reference =
          has_reference_on_restart and
          cell->as_dof_handler_iterator(dof_handler_cut_reference)->get_fe().n_dofs_per_cell() > 0;

        std::uint8_t flags = 0;
        if (is_cut)
          flags |= 1;
        if (has_small_cut_cells_on_restart and is_small_cut_cell[index])
          flags |= 2;
        if (is_reference)
          flags |= 4;

        std::vector<char> buffer;
        append_bytes(buffer, flags);
        append_bytes(buffer,
                     static_cast<std::int8_t>(has_cell_signs_on_restart ? cell_signs[index] : 0));
        append_bytes(buffer,
                     static_cast<std::int8_t>(
                       has_reference_on_restart ? cell_signs_reference[index] : 0));
        append_bytes(buffer, static_cast<std::uint8_t>(cell_band->active_fe_index()));

        dealii::Vector<number> values;
        const auto             append_cell_values = [&](const auto &dof_cell, const VectorType &v) {
          values.reinit(dof_cell->get_fe().n_dofs_per_cell());
          if (values.size() > 0)
            dof_cell->get_dof_values(v, values);
          append_values(buffer, values);
        };

        append_cell_values(cell_cut, level_set);
        append_bytes(buffer, static_cast<std::uint8_t>(band_vectors.size()));
        for (const VectorType *vector : band_vectors)
          append_cell_values(cell_band, *vector);
        if (has_reference_on_restart)
          append_cell_values(cell->as_dof_handler_iterator(dof_handler_cut_reference),
                             level_set_reference);
        else
          append_values(buffer, dealii::Vector<number>());

        return buffer;
      },
      /*returns_variable_size_data=*/true);
  }

  template <int dim, typename number>
  void
  NarrowBandLevelSetOperation<dim, number>::deserialize()
  {
    ScopedName                 sc("narrow_band::deserialize");
    dealii::TimerOutput::Scope scope(scratch_data.get_timer(), sc);

    auto &triangulation =
      const_cast<dealii::Triangulation<dim> &>(dof_handler_cut.get_triangulation());
    const unsigned int n_cells = triangulation.n_active_cells();

    // cell-wise data of the locally owned cells
    std::vector<bool>                             is_cut_cell(n_cells, false);
    std::vector<bool>                             is_small(n_cells, false);
    std::vector<bool>                             is_reference(n_cells, false);
    std::vector<int>                              signs(n_cells, 0);
    std::vector<int>                              signs_reference(n_cells, 0);
    std::vector<unsigned int>                     band_fe_indices(n_cells, 0);
    std::vector<std::vector<number>>              cut_values(n_cells);
    std::vector<std::vector<number>>              reference_values(n_cells);
    std::vector<std::vector<std::vector<number>>> band_values(n_cells);

    // Note: the handle is obtained by registering the data as in prepare_for_serialization().
    const unsigned int handle = triangulation.register_data_attach(
      [](const typename dealii::Triangulation<dim>::cell_iterator &, const dealii::CellStatus) {
        return std::vector<char>();
      },
      /*returns_variable_size_data=*/true);
    triangulation.notify_ready_to_unpack(
      handle,
      [&](const typename dealii::Triangulation<dim>::cell_iterator       &cell,
          const dealii::CellStatus                                        status,
          const boost::iterator_range<std::vector<char>::const_iterator> &data) {
        AssertThrow(status == dealii::CellStatus::cell_will_persist, dealii::ExcNotImplemented());

        const unsigned int index    = cell->active_cell_index();
        auto               position = data.begin();
        const auto         end      = data.end();

        const auto flags       = read_bytes<std::uint8_t>(position, end);
        is_cut_cell[index]     = (flags & 1) != 0;
        is_small[index]        = (flags & 2) != 0;
        is_reference[index]    = (flags & 4) != 0;
        signs[index]           = read_bytes<std::int8_t>(position, end);
        signs_reference[index] = read_bytes<std::int8_t>(position, end);
        band_fe_indices[index] = read_bytes<std::uint8_t>(position, end);

        cut_values[index]         = read_values<number>(position, end);
        const auto n_band_vectors = read_bytes<std::uint8_t>(position, end);
        band_values[index].resize(n_band_vectors);
        for (auto &values : band_values[index])
          values = read_values<number>(position, end);
        reference_values[index] = read_values<number>(position, end);
      });

    // DoF system of the restored cut cells and band (for the downstream neighbor activation with
    // the restored times of the activation)
    setup_dof_system(is_cut_cell, has_small_cut_cells_on_restart ? is_small : std::vector<bool>());

    unsigned int n_band_mismatches = 0;
    for (const auto &cell : dof_handler_band.active_cell_iterators())
      if (cell->is_locally_owned() and
          cell->active_fe_index() != band_fe_indices[cell->active_cell_index()])
        ++n_band_mismatches;
    AssertThrow(dealii::Utilities::MPI::sum(n_band_mismatches, scratch_data.get_mpi_comm()) == 0,
                dealii::ExcMessage("The band of the restart data could not be restored (the "
                                   "parameters of the neighbor activation have changed)."));

    // DoF values of the cut cells and the band
    std::vector<VectorType *> band_vectors;
    solution_history_band.apply([&](VectorType &vector) { band_vectors.push_back(&vector); });
    for (const auto &cell : dof_handler_cut.active_cell_iterators())
      {
        if (not cell->is_locally_owned())
          continue;
        const unsigned int index = cell->active_cell_index();
        set_restart_values(cell, cut_values[index], level_set);

        const auto cell_band = cell->as_dof_handler_iterator(dof_handler_band);
        for (unsigned int i = 0; i < std::min(band_vectors.size(), band_values[index].size()); ++i)
          set_restart_values(cell_band, band_values[index][i], *band_vectors[i]);
      }

    // signs of the cells, which are not cut (semi-Lagrangian band)
    if (has_cell_signs_on_restart)
      {
        cell_signs = signs;
        exchange_cell_signs(cell_signs);
      }

    // sub-cycling: reference (or a new reference, if the restart data has none)
    if (is_subcycling())
      {
        if (has_reference_on_restart)
          {
            NarrowBand::distribute_dofs_layered(is_reference,
                                                1,
                                                dof_handler_cut_reference,
                                                fe_collection_cut,
                                                dof_handler_band_reference,
                                                fe_collection_band);

            const MPI_Comm mpi_comm = scratch_data.get_mpi_comm();
            level_set_reference.reinit(dof_handler_cut_reference.locally_owned_dofs(),
                                       dealii::DoFTools::extract_locally_relevant_dofs(
                                         dof_handler_cut_reference),
                                       mpi_comm);
            level_set_band_reference.reinit(dof_handler_band_reference.locally_owned_dofs(),
                                            dealii::DoFTools::extract_locally_relevant_dofs(
                                              dof_handler_band_reference),
                                            mpi_comm);
            for (const auto &cell : dof_handler_cut_reference.active_cell_iterators())
              if (cell->is_locally_owned())
                set_restart_values(cell,
                                   reference_values[cell->active_cell_index()],
                                   level_set_reference);

            cell_signs_reference = signs_reference;
            exchange_cell_signs(cell_signs_reference);
          }
        else
          set_reference(std::numeric_limits<number>::quiet_NaN());
      }

    candidate_cell_values.clear();

    Journal::print_line(scratch_data.get_pcout(1),
                        "restart: state of the narrow band restored",
                        "narrow_band");
    print_band_statistics();
    update_continuous_level_set();
  }

  template class NarrowBandLevelSetOperation<1, double>;
  template class NarrowBandLevelSetOperation<2, double>;
  template class NarrowBandLevelSetOperation<3, double>;
} // namespace MeltPoolDG::LevelSet
