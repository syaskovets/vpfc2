#ifndef VPFC2_H
#define VPFC2_H
#include <limits>
#include <random>
#include <cmath>
#include <iostream>
#include <vector>
#include <utility>
#include <fstream>

#include <amdis/AMDiS.hpp>
#include <amdis/AdaptInstationary.hpp>
#include <amdis/LocalOperators.hpp>
#include <amdis/ProblemInstat.hpp>
#include <amdis/ProblemStat.hpp>
#include <amdis/GridFunctions.hpp>
#include <amdis/Marker.hpp>
#include <dune/grid/albertagrid.hh>
#include <dune/alugrid/grid.hh>
#include <dune/grid/uggrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <amdis/io/FileWriterCreator.hpp>

#include <typeinfo>
#include <mpi.h>

#define YASPGRID 1
// #define _DEBUG 1

using namespace AMDiS;
using namespace Dune::Functions::BasisFactory;

// using Grid = Dune::UGGrid<2>;
// using Grid = Dune::ALUGrid<2,2,Dune::simplex,Dune::conforming>;

#ifdef YASPGRID
using Grid = Dune::YaspGrid<GRIDDIM, Dune::EquidistantOffsetCoordinates<double,2>>;
#else
using Grid = Dune::AlbertaGrid<GRIDDIM, WORLDDIM>;
#endif

using Param = LagrangeBasis<Grid, 2, 2, 2>;

// using Grid = Dune::YaspGrid<2>;
// using Param = LagrangeBasis<Grid, 1, 1, 1>;
// using VpfcParam   = YaspGridBasis<GRIDDIM, 2>;
// using VpfcProblem = ProblemStat<VpfcParam>;
// using VpfcProblemInstat = ProblemInstat<VpfcParam>;

double q, r, H, M, v0;
int N, cooldown, domain, bc;
bool addVacancy = true;
bool addNoise = false;
bool vicsek = false; double vicsekR = 0;
bool runAndTumble = false; double RTRate = 0;
bool logParticles = false;
bool periodicBC_ = false;
std::string logParticlesFname;
double noiseSigma, compression;
double cooldownTimestep, timestep_;

FieldVector<double,2> scale;

// id is double as it is later used in FieldVector<double,3>
// should be size_t ideally
struct CellPeakProp {
  double id;
  double x; double y;
  double v_x; double v_y;
  // rv stands for real velocity
  // as opposed to unit velocity field
  double rv_x; double rv_y;
  double phi; double dd_phi;

  CellPeakProp()
  : id(-1), x(0), y(0), v_x(0), v_y(0), rv_x(0), rv_y(0), phi(0), dd_phi(0) {}

  CellPeakProp(
    double id,
    double x, double y,
    double v_x, double v_y,
    double rv_x, double rv_y,
    double phi, double dd_phi)
  : id(id), x(x), y(y), v_x(v_x), v_y(v_y), rv_x(rv_x), rv_y(rv_y), phi(phi), dd_phi(dd_phi) {}
};

double v0XInit[1000];
double v0YInit[1000];

std::vector<CellPeakProp> peaks;
std::vector<CellPeakProp> particles;
std::map<unsigned int,CellPeakProp> particlesOld;

template <typename T, typename T1>
double dist(const T& p1, const T1& p2) {
  double dist_x = p1[0]-p2[0]; double dist_y = p1[1]-p2[1];

  if (periodicBC_) {
    dist_x = std::abs(dist_x) < scale[0] ? dist_x : scale[0]*2-std::abs(dist_x);
    dist_y = std::abs(dist_y) < scale[1] ? dist_y : scale[1]*2-std::abs(dist_y);
  }

  return std::sqrt(std::pow(dist_x, 2)+std::pow(dist_y, 2));
}

std::pair<double, double> generateGaussianNoise(double mu, double sigma)
{
    constexpr double epsilon = std::numeric_limits<double>::epsilon();
    constexpr double two_pi = 2.0 * M_PI;

    static std::mt19937 rng(std::random_device{}()); // Standard mersenne_twister_engine seeded with rd()
    static std::uniform_real_distribution<> runif(0.0, 1.0);

    double u1, u2;
    do
    {
        u1 = runif(rng);
        u2 = runif(rng);
    }
    while (u1 <= epsilon);

    auto mag = sigma * sqrt(-2.0 * log(u1));
    auto z0  = mag * cos(two_pi * u2) + mu;
    auto z1  = mag * sin(two_pi * u2) + mu;

    return std::make_pair(z0, z1);
}

#include "MyProblemInstat.h"
#endif