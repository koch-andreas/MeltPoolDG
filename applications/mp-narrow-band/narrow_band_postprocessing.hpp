#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>

#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/non_matching/fe_immersed_values.h>
#include <deal.II/non_matching/quadrature_generator.h>

#include <meltpooldg/level_set/narrow_band_utilities.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/utilities/journal.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace MeltPoolDG::LevelSet
{
  /**
   * @brief Print the number of cut cells and of the other cells of the narrow band (neighbor cells
   * or cells of the layers of the advected band).
   *
   * @param generic_data_out Output data containing the level-set vector of the band.
   * @param vector_name Name of the level-set vector of the band in @p generic_data_out.
   * @param mpi_communicator MPI communicator.
   * @param label Label of the printed line.
   */
  template <int dim, typename number>
  void
  print_narrow_band_statistics(const GenericDataOut<dim, number> &generic_data_out,
                               const std::string                 &vector_name,
                               const MPI_Comm                     mpi_communicator,
                               const std::string                 &label)
  {
    const auto &dof_handler = generic_data_out.get_dof_handler(vector_name);

    dealii::ConditionalOStream pcout(std::cout,
                                     dealii::Utilities::MPI::this_mpi_process(mpi_communicator) ==
                                       0);

    std::ostringstream str;
    str << "t = " << std::scientific << std::setprecision(4) << generic_data_out.get_time() << ": "
        << NarrowBand::n_cells_with_active_fe_index(dof_handler, NarrowBand::cut) << " cut cells, "
        << NarrowBand::n_cells_with_active_fe_index(dof_handler, NarrowBand::neighbor) +
             NarrowBand::n_cells_with_active_fe_index(dof_handler, NarrowBand::outer)
        << " neighbor cells";
    Journal::print_line(pcout, str.str(), label);
  }

  /**
   * @brief Compute and print geometric measures of the zero level-set iso-surface $\Gamma_h$ of a
   * narrow-band level-set field: the area (volume in 3D) of the region $\Omega^+ = \{\phi_h >
   * 0\}$, the length (area in 3D) of $\Gamma_h$ and the circularity (sphericity in 3D).
   *
   * Since the level-set field is only defined in the narrow band, the volume of $\Omega^+$ is
   * computed with the divergence theorem from the surface integral over the discrete interface,
   * $|\Omega^+| = \frac{1}{d} \int_{\Gamma_h} \boldsymbol{x} \cdot \boldsymbol{n}^+ \,
   * \mathrm{d}s$, where $\boldsymbol{n}^+$ is the outer normal of $\Omega^+$. This assumes that
   * $\Omega^+$ does not touch the domain boundary. The circularity is the ratio of the surface
   * measure of a hypersphere with the volume $|\Omega^+|$ and $|\Gamma_h|$, i.e., one for a
   * circle (sphere).
   *
   * @note Since a DG level-set field may be discontinuous across faces, $\Gamma_h$ is not
   * necessarily closed. Gaps at the faces enter the measures.
   *
   * @param generic_data_out Output data containing the level-set vector and its DoFHandler.
   * @param vector_name Name of the level-set vector in @p generic_data_out.
   * @param mpi_communicator MPI communicator.
   * @param label Label of the printed line.
   */
  template <int dim, typename number>
  void
  print_interface_measures(const GenericDataOut<dim, number> &generic_data_out,
                           const std::string                 &vector_name,
                           const MPI_Comm                     mpi_communicator,
                           const std::string                 &label)
  {
    const auto &dof_handler = generic_data_out.get_dof_handler(vector_name);
    const auto &level_set   = generic_data_out.get_vector(vector_name);
    const auto &mapping     = generic_data_out.get_mapping();

    const bool has_ghost_elements = level_set.has_ghost_elements();
    if (not has_ghost_elements)
      level_set.update_ghost_values();

    unsigned int max_degree = 1;
    for (const auto &fe : dof_handler.get_fe_collection())
      max_degree = std::max(max_degree, fe.degree);

    const dealii::hp::QCollection<1> q_collection_1d(dealii::QGauss<1>(max_degree + 3));
    dealii::NonMatching::DiscreteQuadratureGenerator<dim> quadrature_generator(q_collection_1d,
                                                                               dof_handler,
                                                                               level_set);

    number volume = 0., interface_measure = 0.;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
          continue;

        quadrature_generator.generate(cell);
        const auto &surface_quadrature = quadrature_generator.get_surface_quadrature();
        if (surface_quadrature.size() == 0)
          continue;

        dealii::NonMatching::FEImmersedSurfaceValues<dim> fe_surface_values(
          mapping,
          cell->get_fe(),
          surface_quadrature,
          dealii::update_JxW_values | dealii::update_quadrature_points |
            dealii::update_normal_vectors);
        fe_surface_values.reinit(cell);

        for (const unsigned int q : fe_surface_values.quadrature_point_indices())
          {
            // The normal of the immersed surface points from {phi < 0} to {phi > 0}, i.e., it is
            // the inner normal of the region {phi > 0}.
            volume -= fe_surface_values.quadrature_point(q) * fe_surface_values.normal_vector(q) *
                      fe_surface_values.JxW(q) / dim;
            interface_measure += fe_surface_values.JxW(q);
          }
      }

    if (not has_ghost_elements)
      level_set.zero_out_ghost_values();

    volume            = dealii::Utilities::MPI::sum(volume, mpi_communicator);
    interface_measure = dealii::Utilities::MPI::sum(interface_measure, mpi_communicator);

    // surface measure of a hypersphere with the same volume
    number sphere_surface_measure = 0.;
    if constexpr (dim == 2)
      sphere_surface_measure = 2. * std::sqrt(dealii::numbers::PI * volume);
    else if constexpr (dim == 3)
      sphere_surface_measure = std::cbrt(36. * dealii::numbers::PI * volume * volume);

    dealii::ConditionalOStream pcout(std::cout,
                                     dealii::Utilities::MPI::this_mpi_process(mpi_communicator) ==
                                       0);

    std::ostringstream str;
    str << std::scientific << std::setprecision(4) << "area (phi>0): " << volume
        << "  interface length: " << interface_measure;
    if (dim > 1 and interface_measure > 0.)
      str << "  circularity: " << sphere_surface_measure / interface_measure;
    Journal::print_line(pcout, str.str(), label);
  }

  /**
   * @brief Compute and print the errors of a narrow-band level-set field with respect to an exact
   * signed distance function $d$.
   *
   * The following errors are evaluated on all cells, on which the level-set field $\phi_h$ is
   * defined (cells with FE_Nothing elements are skipped):
   * - the relative L2-error $\|\phi_h - d\| / \|d\|$,
   * - the eikonal error $\sqrt{\int (|\nabla \phi_h| - 1)^2 \, \mathrm{d}x / V}$,
   * - the interface position errors $\sqrt{\int_{\Gamma_h} d^2 \, \mathrm{d}s / |\Gamma_h|}$ and
   *   $\max_{\Gamma_h} |d|$ at the quadrature points of the discrete interface
   *   $\Gamma_h = \{\phi_h = 0\}$,
   * - the relative error of the interface measure $|\Gamma_h|$, if the exact value is given.
   *
   * @param generic_data_out Output data containing the level-set vector and its DoFHandler.
   * @param vector_name Name of the level-set vector in @p generic_data_out.
   * @param exact_signed_distance Exact signed distance function (evaluated at its current time).
   * @param exact_interface_measure Exact measure of the interface. Set to zero to skip the
   * corresponding error.
   * @param mpi_communicator MPI communicator.
   * @param label Label of the printed lines.
   */
  template <int dim, typename number>
  void
  print_narrow_band_errors(const GenericDataOut<dim, number>   &generic_data_out,
                           const std::string                   &vector_name,
                           const dealii::Function<dim, number> &exact_signed_distance,
                           const number                         exact_interface_measure,
                           const MPI_Comm                       mpi_communicator,
                           const std::string                   &label)
  {
    const auto &dof_handler = generic_data_out.get_dof_handler(vector_name);
    const auto &level_set   = generic_data_out.get_vector(vector_name);
    const auto &mapping     = generic_data_out.get_mapping();

    const bool has_ghost_elements = level_set.has_ghost_elements();
    if (not has_ghost_elements)
      level_set.update_ghost_values();

    const auto                  &fe_collection = dof_handler.get_fe_collection();
    dealii::hp::QCollection<dim> q_collection;
    for (unsigned int i = 0; i < fe_collection.size(); ++i)
      q_collection.push_back(dealii::QGauss<dim>(fe_collection[i].degree + 3));

    // Note: hp::FEValues keeps pointers to the collections; thus, they must outlive it.
    const dealii::hp::MappingCollection<dim> mapping_collection(mapping);
    dealii::hp::FEValues<dim>                hp_fe_values(mapping_collection,
                                           fe_collection,
                                           q_collection,
                                           dealii::update_values | dealii::update_gradients |
                                             dealii::update_JxW_values |
                                             dealii::update_quadrature_points);

    unsigned int max_degree = 1;
    for (unsigned int i = 0; i < fe_collection.size(); ++i)
      max_degree = std::max(max_degree, fe_collection[i].degree);

    const dealii::hp::QCollection<1> q_collection_1d(dealii::QGauss<1>(max_degree + 3));
    dealii::NonMatching::DiscreteQuadratureGenerator<dim> quadrature_generator(q_collection_1d,
                                                                               dof_handler,
                                                                               level_set);

    std::vector<number>                         values;
    std::vector<dealii::Tensor<1, dim, number>> gradients;

    number error_value = 0, norm_exact = 0, error_eikonal = 0, volume = 0;
    number error_interface = 0, max_error_interface = 0, interface_measure = 0;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
          continue;

        // volume errors
        hp_fe_values.reinit(cell);
        const auto &fe_values = hp_fe_values.get_present_fe_values();

        values.resize(fe_values.n_quadrature_points);
        gradients.resize(fe_values.n_quadrature_points);
        fe_values.get_function_values(level_set, values);
        fe_values.get_function_gradients(level_set, gradients);

        for (const unsigned int q : fe_values.quadrature_point_indices())
          {
            const number exact = exact_signed_distance.value(fe_values.quadrature_point(q), 0);
            error_value += dealii::Utilities::fixed_power<2>(values[q] - exact) * fe_values.JxW(q);
            norm_exact += exact * exact * fe_values.JxW(q);
            error_eikonal +=
              dealii::Utilities::fixed_power<2>(gradients[q].norm() - 1.0) * fe_values.JxW(q);
            volume += fe_values.JxW(q);
          }

        // interface errors
        quadrature_generator.generate(cell);
        const auto &surface_quadrature = quadrature_generator.get_surface_quadrature();
        if (surface_quadrature.size() == 0)
          continue;

        dealii::NonMatching::FEImmersedSurfaceValues<dim> fe_surface_values(
          mapping,
          cell->get_fe(),
          surface_quadrature,
          dealii::update_JxW_values | dealii::update_quadrature_points);
        fe_surface_values.reinit(cell);

        for (const unsigned int q : fe_surface_values.quadrature_point_indices())
          {
            const number distance =
              exact_signed_distance.value(fe_surface_values.quadrature_point(q), 0);
            error_interface += distance * distance * fe_surface_values.JxW(q);
            max_error_interface = std::max(max_error_interface, std::abs(distance));
            interface_measure += fe_surface_values.JxW(q);
          }
      }

    if (not has_ghost_elements)
      level_set.zero_out_ghost_values();

    error_value         = dealii::Utilities::MPI::sum(error_value, mpi_communicator);
    norm_exact          = dealii::Utilities::MPI::sum(norm_exact, mpi_communicator);
    error_eikonal       = dealii::Utilities::MPI::sum(error_eikonal, mpi_communicator);
    volume              = dealii::Utilities::MPI::sum(volume, mpi_communicator);
    error_interface     = dealii::Utilities::MPI::sum(error_interface, mpi_communicator);
    interface_measure   = dealii::Utilities::MPI::sum(interface_measure, mpi_communicator);
    max_error_interface = dealii::Utilities::MPI::max(max_error_interface, mpi_communicator);

    dealii::ConditionalOStream pcout(std::cout,
                                     dealii::Utilities::MPI::this_mpi_process(mpi_communicator) ==
                                       0);

    {
      std::ostringstream str;
      str << std::scientific << std::setprecision(4)
          << "rel. L2 error signed distance: " << std::sqrt(error_value / norm_exact)
          << "  L2 error |grad phi|-1: " << std::sqrt(error_eikonal / volume);
      Journal::print_line(pcout, str.str(), label);
    }
    {
      std::ostringstream str;
      str << std::scientific << std::setprecision(4)
          << "interface error L2: " << std::sqrt(error_interface / interface_measure)
          << "  max: " << max_error_interface;
      if (exact_interface_measure > 0.)
        str << "  rel. error measure: "
            << (interface_measure - exact_interface_measure) / exact_interface_measure;
      Journal::print_line(pcout, str.str(), label);
    }
  }

  /**
   * @brief Compute and print the errors of the curvature and the normal vector of the continuous
   * level set on the cut cells (see NarrowBandContinuousLevelSet) at the quadrature points of the
   * zero level set of the continuous level set.
   *
   * The output vectors "level_set_cg", "curvature" and "normal" have to be attached to
   * @p generic_data_out. The following errors are printed:
   * - the L2-error $\sqrt{\int_{\Gamma_h} (\kappa_h - \kappa)^2 \, \mathrm{d}s / |\Gamma_h|}$
   *   of the curvature relative to $|\kappa|$ and the maximum relative error,
   * - the L2-error and the maximum error of the normal vector $|\boldsymbol{n}_h -
   *   \boldsymbol{n}|$.
   *
   * @param generic_data_out Output data containing the continuous fields.
   * @param exact_curvature Exact curvature (constant along the interface).
   * @param exact_normal Exact unit normal vector at a point on the interface.
   * @param mpi_communicator MPI communicator.
   * @param label Label of the printed lines.
   */
  template <int dim, typename number>
  void
  print_interface_geometry_errors(
    const GenericDataOut<dim, number> &generic_data_out,
    const number                       exact_curvature,
    const std::function<dealii::Tensor<1, dim, number>(const dealii::Point<dim> &)> &exact_normal,
    const MPI_Comm     mpi_communicator,
    const std::string &label)
  {
    const auto &dof_handler        = generic_data_out.get_dof_handler("level_set_cg");
    const auto &level_set          = generic_data_out.get_vector("level_set_cg");
    const auto &curvature          = generic_data_out.get_vector("curvature");
    const auto &dof_handler_normal = generic_data_out.get_dof_handler("normal");
    const auto &normal             = generic_data_out.get_vector("normal");
    const auto &mapping            = generic_data_out.get_mapping();

    for (const auto *vector : {&level_set, &curvature, &normal})
      AssertThrow(vector->has_ghost_elements(),
                  dealii::ExcMessage("The output vectors require ghost values."));

    unsigned int max_degree = 1;
    for (const auto &fe : dof_handler.get_fe_collection())
      max_degree = std::max(max_degree, fe.degree);

    const dealii::hp::QCollection<1> q_collection_1d(dealii::QGauss<1>(max_degree + 1));
    dealii::NonMatching::DiscreteQuadratureGenerator<dim> quadrature_generator(q_collection_1d,
                                                                               dof_handler,
                                                                               level_set);

    const dealii::FEValuesExtractors::Vector normal_extractor(0);

    std::vector<number>                         curvature_values;
    std::vector<dealii::Tensor<1, dim, number>> normal_values;

    number error_curvature = 0., max_error_curvature = 0., error_normal = 0., max_error_normal = 0.,
           interface_measure = 0.;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() or cell->get_fe().n_dofs_per_cell() == 0)
          continue;

        quadrature_generator.generate(cell);
        const auto &surface_quadrature = quadrature_generator.get_surface_quadrature();
        if (surface_quadrature.size() == 0)
          continue;

        dealii::NonMatching::FEImmersedSurfaceValues<dim> fe_values(
          mapping,
          cell->get_fe(),
          surface_quadrature,
          dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
        fe_values.reinit(cell);
        curvature_values.resize(surface_quadrature.size());
        fe_values.get_function_values(curvature, curvature_values);

        const auto cell_normal = cell->as_dof_handler_iterator(dof_handler_normal);
        dealii::NonMatching::FEImmersedSurfaceValues<dim> fe_values_normal(mapping,
                                                                           cell_normal->get_fe(),
                                                                           surface_quadrature,
                                                                           dealii::update_values);
        fe_values_normal.reinit(cell_normal);
        normal_values.resize(surface_quadrature.size());
        fe_values_normal[normal_extractor].get_function_values(normal, normal_values);

        for (const unsigned int q : fe_values.quadrature_point_indices())
          {
            const number e_curvature = std::abs(curvature_values[q] - exact_curvature);
            const number e_normal =
              (normal_values[q] - exact_normal(fe_values.quadrature_point(q))).norm();

            error_curvature += e_curvature * e_curvature * fe_values.JxW(q);
            error_normal += e_normal * e_normal * fe_values.JxW(q);
            max_error_curvature = std::max(max_error_curvature, e_curvature);
            max_error_normal    = std::max(max_error_normal, e_normal);
            interface_measure += fe_values.JxW(q);
          }
      }

    error_curvature     = dealii::Utilities::MPI::sum(error_curvature, mpi_communicator);
    error_normal        = dealii::Utilities::MPI::sum(error_normal, mpi_communicator);
    interface_measure   = dealii::Utilities::MPI::sum(interface_measure, mpi_communicator);
    max_error_curvature = dealii::Utilities::MPI::max(max_error_curvature, mpi_communicator);
    max_error_normal    = dealii::Utilities::MPI::max(max_error_normal, mpi_communicator);

    dealii::ConditionalOStream pcout(std::cout,
                                     dealii::Utilities::MPI::this_mpi_process(mpi_communicator) ==
                                       0);

    const number scale = std::abs(exact_curvature) > 0. ? std::abs(exact_curvature) : 1.;

    {
      std::ostringstream str;
      str << std::scientific << std::setprecision(4)
          << "rel. curvature error L2: " << std::sqrt(error_curvature / interface_measure) / scale
          << "  max: " << max_error_curvature / scale;
      Journal::print_line(pcout, str.str(), label);
    }
    {
      std::ostringstream str;
      str << std::scientific << std::setprecision(4)
          << "normal error L2: " << std::sqrt(error_normal / interface_measure)
          << "  max: " << max_error_normal;
      Journal::print_line(pcout, str.str(), label);
    }
  }
} // namespace MeltPoolDG::LevelSet
