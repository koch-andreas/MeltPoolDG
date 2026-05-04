set(TEST_LIBRARIES ${meltpooldg_lib})
set(TEST_TARGET ${meltpooldg_lib})
deal_ii_pickup_tests()

file(GLOB unit_test_cmake_commands "cmake/*.cmake")
foreach(command ${unit_test_cmake_commands})
    include(${command})
endforeach()

# In order to run tests with MPI, we need to know the MPI executable and its semantics.
find_package(MPI REQUIRED)

mpdg_setup_mpi_unit_tests(
   N_MPI_PROCESSES
   1
   TEST_SOURCE_FILES
   material_data.cpp
)

add_subdirectory(flow)
add_subdirectory(heat)
add_subdirectory(particles)
add_subdirectory(utilities)
add_subdirectory(cut)
