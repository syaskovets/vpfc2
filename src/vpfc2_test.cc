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
#include <dune/grid/utility/structuredgridfactory.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <typeinfo>
#include <mpi.h>


using namespace AMDiS;
using namespace Dune::Functions::BasisFactory;

using Grid = Dune::ALUGrid<2,2,Dune::simplex,Dune::conforming>;
using Param = LagrangeBasis<Grid, 1, 1, 1>;

double q, r, H, M;
double c1, c2, beta;
int N, domain, bc;
double compression;
FieldVector<double,2> scale;

/// inital value function
class NgridOrientedDots {
public:

  NgridOrientedDots()  
  {
    N = Parameters::get<int>("vpfc->N").value();
    // number of particls

    int Nx = floor(sqrt(N)+0.5);
    int Ny = floor(sqrt(N)) + ((N%(Nx*Nx)>0) ? 1 : 0) ;

    double dx = 2*scale[0]/(Nx)*compression;
    double dy = 2*scale[1]/(Ny)*compression;
    int cntr = 0;

    // initalizing a regular grid 
    // adding a noise on each grid point 
    // using the first N grid points 

    if (domain == 0)
    {
      for (int i=0; i<Nx ; i++)
        for (int j=0; j<Ny ; j++){
          y[cntr][0] = -scale[0]*compression + 0.5*dx + i*dx;
          y[cntr][1] = -scale[1]*compression + 0.5*dx + j*dy;

          cntr++;
        }
    }

    else if (domain == 1) {
      int i = 0; int j = 0;
      double _x = 0; double _y = 0;

      while (cntr < N) {
        _x = -scale[0]*compression + 0.5*dx + i*dx;
        _y = -scale[1]*compression + 0.5*dx + j*dy;

        if ((_x/scale[0])*(_x/scale[0])+(_y/scale[1])*(_y/scale[1]) < 1) {
          y[cntr][0] = _x;
          y[cntr][1] = _y;

          cntr++;
        }

        ++j; if (j == Ny) {j = 0; ++i;}
      }
    }
  }

  template <typename T>
  double operator()(const T& x) const
  {
    double sum = 0.0;

    for (int i=0; i < N ;i++ )
    {
      if ( sqrt(pow(x[0] - y[i][0] ,2) + pow(x[1] - y[i][1],2)) < 2*M_PI/sqrt(3)) {
        sum += (cos(sqrt(3)/2*sqrt(pow(x[0]-y[i][0],2) + pow(x[1]-y[i][1],2)) )+1.0) + 0.0002*(rand()%1000-500) ;
      }
    }

    return sum;
  }

private:
  double y [5000][2];
};

template <class Traits>
class MyProblemInstat : public ProblemInstat<Traits> {
  public:
    MyProblemInstat(std::string const& name, ProblemStat<Traits>& prob)
      : ProblemInstat<Traits>(name, prob) {}

    ~MyProblemInstat() {}

    void closeTimestep(AdaptInfo& adaptInfo) {
      auto phi = this->problemStat_->solution(0);
      auto mu = this->problemStat_->solution(2);

      for (int k = 0; k < 5; ++k) {
        this->problemStat_-> markElements(adaptInfo);
        this->problemStat_-> adaptGrid(adaptInfo);
      }

      ProblemInstat<Traits>::closeTimestep(adaptInfo);
    }
};

void setInitValues(ProblemStat<Param>& prob, AdaptInfo& adaptInfo) {
  auto phi = prob.solution(0);
  auto psi = prob.solution(1);

  NgridOrientedDots fct;
  // G2fix fct;

  double density;
  double B0 = integrate(constant(1.0), prob.gridView(), 6);
  double psibar = N*0.9*16*M_PI*M_PI/3/B0* sqrt( (-48.0 - 56.0*r)/133.0  );

  int interface_ref = Parameters::get<int>("vpfc->interface refinements").value_or(15);
  int bulk_ref = Parameters::get<int>("vpfc->bulk refinements").value_or(10);
  int outer_ref = Parameters::get<int>("vpfc->outer refinements").value_or(5);

  double threshold = 0.90;

  static GridFunctionMarker marker("interface", prob.grid(),
    invokeAtQP([interface_ref,bulk_ref,outer_ref,threshold](double const& phi) -> int {
      return ( phi < -threshold ) ? outer_ref : (phi > threshold) ? bulk_ref : interface_ref;
    }, 10.0*valueOf(prob.solution(0))-1));
  prob.addMarker(marker);

  phi.interpolate(fct);
  for (int i=0; i<7;i++){
    phi.interpolate(fct);
    B0 = integrate(constant(1.0), prob.gridView(), 6);
    density = integrate(valueOf(phi), prob.gridView(), 6)/B0; // density of the initial value
    phi << 1.2*psibar/(density+0.0000001)*valueOf(phi);

    prob.markElements(adaptInfo);
    prob.adaptGrid(adaptInfo);
  }
  // prob.removeMarker("interface");

  psi.interpolate(constant(0.0));
}

void setDensityOperators(ProblemStat<Param>& prob, MyProblemInstat<Param>& probInstat) {
  q = Parameters::get<double>("vpfc->q").value_or(10);
  r = Parameters::get<double>("vpfc->r").value_or(0.5);
  H = Parameters::get<double>("vpfc->H").value_or(1500);
  M = Parameters::get<double>("vpfc->mobility").value();

  auto phi = prob.solution(0);
  auto phiOld = probInstat.oldSolution(0);
  auto invTau = std::ref(probInstat.invTau());

  prob.addMatrixOperator(sot(M), 0, 1);
  prob.addMatrixOperator(sot(1), 1, 2);
  prob.addMatrixOperator(sot(1), 2, 0);
  prob.addMatrixOperator(zot(invTau), 0, 0);
  prob.addVectorOperator(zot(phiOld * invTau), 0);
  prob.addMatrixOperator(zot(1.0), 1, 1);

  prob.addMatrixOperator(zot(-2*q*q), 1, 2);
  prob.addMatrixOperator(zot(1.0), 2, 2);

// lhs of the density
  auto op1Impl = zot(-3*pow<2>(phi) - std::pow(q, 4) - r - 6*H*(abs(phi)-phi), 6);
  prob.addMatrixOperator(op1Impl, 1, 0);

// rhs of the density
  auto op1Fexpl = zot(-2*pow<3>(phi) + 3*H*(pow<2>(phi) - phi*abs(phi)), 6);
  prob.addVectorOperator(op1Fexpl, 1);

  std::cout << " params " << q << " " << r << " " << H << std::endl;
  std::cout << "setBounderyConditions " << scale[0] << " " << scale[1] << std::endl;

  // rectangular or circular domain
  if (domain == 0)
  {
    auto op1BC = zot(-H*(tanh(100*(pow<20>(X(0)/scale[0])+pow<20>(X(1)/scale[1])-0.99))+1), 6);
    prob.addMatrixOperator(op1BC,1,0);
  }
  else if (domain == 1)
  {
    auto op1BC = zot(-H*(tanh(100*(pow<20>(pow<2>(X(0)/scale[0])+pow<2>(X(1)/scale[1]))-0.99))+1), 6);
    prob.addMatrixOperator(op1BC,1,0);
  }
  else
    std::cout << "ERROR: unrecognized domain!\n";
}


int main(int argc, char** argv)

{  
  Environment env(argc, argv);
  const Dune::MPIHelper& mpiHelper = Dune::MPIHelper::instance(argc,argv);

  double Nl = Parameters::get<int>("Nl").value_or(4);
  double lattice = Parameters::get<int>("lattice").value();

  lattice = lattice*M_PI/sqrt(3);
  double _scale = Nl*lattice;
  scale = FieldVector<double,2>({_scale, _scale});

  N = Parameters::get<int>("vpfc->N").value_or(1);
  domain = Parameters::get<int>("vpfc->domain").value();
  compression = Parameters::get<double>("vpfc->c").value();
  bc = Parameters::get<double>("vpfc->bc").value();

  // // Start with a structured grid
  const std::array < unsigned, 2 > n = { 16, 16 };
  const FieldVector < double, 2 > lower = {-scale[0], -scale[1]};
  const FieldVector < double, 2 > upper = {scale[0], scale[1]};

  std::shared_ptr < Grid > grid = Dune::StructuredGridFactory < Grid > ::createSimplexGrid(lower, upper, n);
  grid->loadBalance();

  ProblemStat<Param> prob("vpfc", grid);
  prob.initialize(INIT_ALL);

  MyProblemInstat<Param> probInstat("vpfc", prob);
  probInstat.initialize(INIT_UH_OLD);

  AdaptInfo adaptInfo("adapt");

  setDensityOperators(prob, probInstat);
  setInitValues(prob, adaptInfo);

  AdaptInstationary adapt("adapt", prob, adaptInfo, probInstat, adaptInfo);
  adapt.adapt();
  
  return 0;
}
