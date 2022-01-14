#include <limits>
#include <random>
#include <cmath>
#include <iostream>
#include <vector>
#include <utility>

#include "boost/variant.hpp"
#include <boost/mpl/int.hpp>
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/at_c.hpp>

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

using namespace AMDiS;
using namespace Dune::Functions::BasisFactory;

// using Grid = Dune::YaspGrid<2, Dune::EquidistantOffsetCoordinates<double,2>>;
using Grid = Dune::UGGrid<2>;
// using Grid = Dune::ALUGrid<2,2,Dune::simplex,Dune::conforming>;
// using Param = LagrangeBasis<Grid, 1, 1, 1, 1, 1>;
using Param = LagrangeBasis<Grid, 1, 1, 1, 1, 1, 1, 1>;

// using Grid = Dune::YaspGrid<2>;
// using Grid = Dune::AlbertaGrid<GRIDDIM, WORLDDIM>;
// using Param = LagrangeBasis<Grid, 1, 1, 1>;
// using VpfcParam   = YaspGridBasis<GRIDDIM, 2>;
// using VpfcProblem = ProblemStat<VpfcParam>;
// using VpfcProblemInstat = ProblemInstat<VpfcParam>;

double K;
double q, r, H, M;
double c1, c2, beta, v0;
int N, cooldown;
bool addVacancy = true;
bool addNoise = false;

FieldVector<double,2> scale;

using GridView = Grid::LeafGridView;

typedef std::pair<double,double> coord;
// <[x,y], phi, v_x, v_y>
typedef boost::fusion::vector<coord, double, double, double> peakProp;

double v0XInit[] = {1.0, -1.0};
double v0YInit[] = {1.0, -1.0};

std::vector<peakProp> peaks;
std::vector<peakProp> particles;

/// inital value function
class G2fix
{
public:

  G2fix()  
  {
    posX1 = Parameters::get<double>("vpfc->posX1").value();
    posY1 = Parameters::get<double>("vpfc->posY1").value();
    posX2 = Parameters::get<double>("vpfc->posX2").value();
    posY2 = Parameters::get<double>("vpfc->posY2").value();

    std::cout << "TEST " << posX1 << " " << posX2 << " " << posY1 << " " << posY2 << std::endl;

    y[0][0] =  posX1;
    y[0][1] =  posY1;
    y[1][0] =  posX2;
    y[1][1] =  posY2;
    // y[0][0] =  posX1 + 0.00001*(rand()%100-50);
    // y[0][1] =  posY1 + 0.00001*(rand()%100-50);
    // y[1][0] =  posX2 + 0.00001*(rand()%100-50);
    // y[1][1] =  posY2 + 0.00001*(rand()%100-50);

  }

  template <typename T>
  double operator()(const T& x) const
  {
    double sum = 0.0;
  
    for (int i = 0; i < 2; ++i)
    {
      if ( sqrt(pow(x[0]-y[i][0] ,2) + pow(x[1]-y[i][1],2)) < 2*M_PI/sqrt(3)) {
        sum += (cos(sqrt(3)/2*sqrt(pow(x[0]-y[i][0],2) + pow(x[1]-y[i][1],2)) )+1.0);//*(1.0+0.0001*(rand()%1000-500)) ;
      }
    }

    return sum;
  }

private:

  double y[2][2];
  double posX1,posY1,posX2,posY2;
};


/// inital value function
class NgridOrientedDots {
public:

  NgridOrientedDots()  
  {
    N = Parameters::get<int>("vpfc->N").value();
    // number of particls

    compression = Parameters::get<double>("vpfc->c").value();    
    bc = Parameters::get<double>("vpfc->bc").value();

    int Nx = floor(sqrt(N)+0.5);
    int Ny = floor(sqrt(N)) + ((N%(Nx*Nx)>0) ? 1 : 0) ;

    double dx = 2*scale[0]/(Nx)*compression;
    double dy = 2*scale[1]/(Ny)*compression;
    int cntr = 0;

    std::cout << "N " << N << " scale[0] " << scale[0] << " compression " << compression << " Nx " << Nx << " dx " << dx << " dy " << dy << std::endl;

    // initalizing a regular grid 
    // adding a noise on each grid point 
    // using the first N grid points 

    for (int i=0; i<Nx ; i++)
      for (int j=0; j<Ny ; j++){
        y[cntr][0] = -scale[0]*compression + 0.5*dx + i*dx;// + dx*0.002*dis*(rand()%1000-500) ;
        // y[cntr][0] = -scale[0]*compression + 0.5*15.4778 + i*15.4778;// + dx*0.002*dis*(rand()%1000-500) ;
        y[cntr][1] = -scale[1]*compression + 0.5*dx + j*dy;// + dy*0.002*dis*(rand()%1000-500) ;

        std::cout << "y[cntr][0] " << y[cntr][0] << " y[cntr][1] " << y[cntr][1] << std::endl;
        cntr++;
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
  double compression;
  // WorldVector<double> scale;
  int bc;
};

coord x_to_coord(const auto& x) {
  return std::make_pair(x[0], x[1]);
} 

double dist(const coord& x, const coord& y) {
  return std::sqrt(std::pow(x.first - y.first, 2)+std::pow(x.second - y.second, 2));
}



template <class Traits>
class MyProblemInstat : public ProblemInstat<Traits> {
    int iter;
    bool vInit;
    std::shared_ptr < Grid > grid;
    // GridView& gridView;

  public:
    MyProblemInstat(std::string const& name, ProblemStat<Traits>& prob)
      : iter(0), vInit(false), ProblemInstat<Traits>(name, prob) {}
    // MyProblemInstat(std::string const& name, ProblemStat<Traits>& prob, std::shared_ptr < Grid > grid, GridView& gridView)
    //   : iter(0), vInit(false), grid(grid), gridView(gridView), ProblemInstat<Traits>(name, prob) {}

    void closeTimestep(AdaptInfo& adaptInfo) {
      auto phi = this->problemStat_->solution(0);
      auto temp1 = this->problemStat_->solution(3);
      auto temp2 = this->problemStat_->solution(4);
      auto v1 = this->problemStat_->solution(5);
      auto v2 = this->problemStat_->solution(6);

      tag::partial der0, der1;
      der0.comp = 0;
      der1.comp = 1;

      if (++iter > cooldown) {
        peaks.clear();
        particles.clear();

        temp1 << derivativeOf(valueOf(phi), der0) + derivativeOf(valueOf(phi), der1);
        temp2 << derivativeOf(valueOf(temp1), der0) + derivativeOf(valueOf(temp1), der1);

        double min = 0.0;

        temp1 << invokeAtQP([&](auto dd_p_x, auto const& x) {
          if (min > dd_p_x)
            min = dd_p_x;

          return dd_p_x;
        }, temp2, X());

        temp1 << invokeAtQP([&](auto dd_p_x, auto v1_x, auto v2_x, auto const& x) {
          // consider only ampls within 20% of the max (min)
          if ((min-dd_p_x)/min < 0.2) {
            coord c_x = x_to_coord(x);
            peakProp p(c_x, dd_p_x[0], v1_x[0], v2_x[0]);
            peaks.push_back(p);
            return 1;
          }

          return 0;
        }, temp2, v1, v2, X());

        // sort peaks by second derivative ampl
        std::sort(peaks.begin(), peaks.end(), [](auto &left, auto &right) {
            return boost::fusion::at<boost::mpl::int_<1> >(left) < boost::fusion::at<boost::mpl::int_<1> >(right);
        });

        for(auto i = 0; i < peaks.size() || particles.size() < N; ++i) {
          bool insert = true;

          for(auto it2 = particles.begin(); it2 != particles.end(); ++it2)
            // 2 is an empirical min dist to filter out the same cells
            if (dist(boost::fusion::at<boost::mpl::int_<0> >(peaks[i]), boost::fusion::at<boost::mpl::int_<0> >(*it2)) < 2)
              insert = false;

          if (insert)
            particles.push_back(peaks[i]);
        }

        std::cout <<" particles.size() "<< particles.size() << " " << vInit << " " << addNoise << std::endl;

        if (vInit && addNoise) {
          for(auto i = 0; i < particles.size(); ++i) {
            double x = boost::fusion::at<boost::mpl::int_<2> >(particles[i]);
            double y = boost::fusion::at<boost::mpl::int_<3> >(particles[i]);
            double alpha = atan2 (y,x) * 180.0 / M_PI;

            alpha += generateGaussianNoise(0,20).first;

            x = cos(alpha * M_PI / 180.0); boost::fusion::at<boost::mpl::int_<2> >(particles[i]) = x;
            y = sin(alpha * M_PI / 180.0); boost::fusion::at<boost::mpl::int_<3> >(particles[i]) = y;
          }
        }

        if (particles.size() < N) {
          std::cout << "Error!!! " << std::endl;
        }

        v1 << invokeAtQP([&](auto phi_x, auto const& x) {
          if (phi_x > (0.001))
          {
            double min_dist = dist(boost::fusion::at<boost::mpl::int_<0> >(particles[0]), x_to_coord(x));
            int min_dist_i = 0;

            for (int i = 0; i < particles.size(); ++i) {
              double d = dist(boost::fusion::at<boost::mpl::int_<0> >(particles[i]), x_to_coord(x));
              if (d < min_dist) {
                min_dist_i = i;
                min_dist = d;
              }
            }

            if (vInit)
              return boost::fusion::at<boost::mpl::int_<2> >(particles[min_dist_i]);
            else
              return v0XInit[min_dist_i];
          }
          return 0.0;
        }, phi, X());

        v2 << invokeAtQP([&](auto phi_x, auto const& x) {
          if (phi_x > (0.001))
          {
            double min_dist = dist(boost::fusion::at<boost::mpl::int_<0> >(particles[0]), x_to_coord(x));
            int min_dist_i = 0;

            for (int i = 0; i < particles.size(); ++i) {
              double d = dist(boost::fusion::at<boost::mpl::int_<0> >(particles[i]), x_to_coord(x));
              if (d < min_dist) {
                min_dist_i = i;
                min_dist = d;
              }
            }

            if (vInit)
              return boost::fusion::at<boost::mpl::int_<3> >(particles[min_dist_i]);
            else
              return v0YInit[min_dist_i];
          }
          return 0.0;
        }, phi, X());

        if (!vInit)
          vInit = true;
      } else {
        v1.interpolate(constant(0.0));
        v2.interpolate(constant(0.0));
      }

      // for (int k = 0; k < 3; ++k) {
      //   for (const auto & element: elements(gridView))
      //     grid -> mark(1, element);
      //   grid -> preAdapt();
      //   grid -> adapt();
      //   grid -> postAdapt();
      // }
      // for (int i=0 ; i<2; i++){
      //   refinement->refine(1,   function_(indicator3("vpfc", 0.95 ), 5.0*valueOf(psi) , pow<2>(valueOf(q1)) + pow<2>(valueOf(q2)), abs_(valueOf(problemStat->getSolution()->getDOFVector(2)))     ));    
      // }

      ProblemInstat<Traits>::closeTimestep(adaptInfo);
    }
};

void setInitValues(ProblemStat<Param>& prob, AdaptInfo& adaptInfo) {  
  auto phi = prob.solution(0);
  auto temp1 = prob.solution(3);
  auto temp2 = prob.solution(4);
  auto v1 = prob.solution(5);
  auto v2 = prob.solution(6);

  // NgridOrientedDots fct;
  G2fix fct;

  double density;
  double B0 = integrate(constant(1.0), prob.gridView(), 6);
  // double B0 = integrate( constant(1.0) , prob.getMesh() ); // size of the domain
  double psibar = N*0.9*16*M_PI*M_PI/3/B0* sqrt( (-48.0 - 56.0*r)/133.0  );

  std::cout << "B0 " << B0 << " psibar " << psibar << " r " << r << " N " << N << std::endl; 

  int interface_ref = Parameters::get<int>("vpfc->interface refinements").value_or(15);
  int bulk_ref = Parameters::get<int>("vpfc->bulk refinements").value_or(10);
  int outer_ref = Parameters::get<int>("vpfc->outer refinements").value_or(5);

  double threshold = 0.90;

  GridFunctionMarker marker("interface", prob.grid(),
    invokeAtQP([interface_ref,bulk_ref,outer_ref,threshold](double const& phi) -> int {
      return ( phi < -threshold ) ? outer_ref : (phi > threshold) ? bulk_ref : interface_ref;
    }, 10.0*valueOf(prob.solution(0))-1));
  prob.addMarker(marker);  

  phi.interpolate(fct);
  for (int i=0; i<4;i++){
    phi.interpolate(fct);
    B0 = integrate(constant(1.0), prob.gridView(), 6);
    density = integrate(valueOf(phi), prob.gridView(), 6)/B0; // density of the initial value
    std::cout << "density " << density << " B0 " << B0 << " psibar " << psibar << std::endl;
    phi << 1.2*psibar/(density+0.0000001)*valueOf(phi);

    prob.markElements(adaptInfo);
    prob.adaptGrid(adaptInfo);
  }

  // prob.removeMarker("interface");

  v1.interpolate(constant(0.0));
  v2.interpolate(constant(0.0));
  temp1.interpolate(constant(0.0));
  temp2.interpolate(constant(0.0));
}

void setDensityOperators(ProblemStat<Param>& prob, MyProblemInstat<Param>& probInstat) {
  q = Parameters::get<double>("vpfc->q").value_or(10);
  r = Parameters::get<double>("vpfc->r").value_or(0.5);
  H = Parameters::get<double>("vpfc->H").value_or(1500);
  M = Parameters::get<double>("vpfc->mobility").value_or(1.0);

  auto phi = prob.solution(0);
  auto phiOld = probInstat.oldSolution(0);
  auto invTau = std::ref(probInstat.invTau());
  auto v1 = prob.solution(5);
  auto v2 = prob.solution(6);
  auto temp = prob.solution(7);

  prob.addMatrixOperator(sot(M), 0, 1);
  prob.addMatrixOperator(sot(1), 1, 2);
  prob.addMatrixOperator(sot(1), 2, 0);
  prob.addMatrixOperator(zot(invTau), 0, 0);
  prob.addVectorOperator(zot(phiOld * invTau), 0);
  tag::partial der0, der1;
  der0.comp = 0;
  der1.comp = 1;
  prob.addVectorOperator(zot(-v0*v1* derivativeOf(valueOf(phi), der0)), 0);
  prob.addVectorOperator(zot(-v0*v2* derivativeOf(valueOf(phi), der1)), 0);
  prob.addMatrixOperator(zot(1.0), 1, 1);

  // !sot?
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
  auto op1BC = zot(-H*(tanh(100*(pow<20>(X(0)/scale[0])+pow<20>(X(1)/scale[1])-0.99))+1), 6);
  prob.addMatrixOperator(op1BC,1,0);  
}

void setTempOperators(ProblemStat<Param>& prob, MyProblemInstat<Param>& probInstat) {
  prob.addMatrixOperator(zot(1), 3, 3);
  prob.addVectorOperator(zot(0), 3);

  prob.addMatrixOperator(zot(1), 4, 4);
  prob.addVectorOperator(zot(0), 4);
}

void setVelocityOperators(ProblemStat<Param>& prob, MyProblemInstat<Param>& probInstat) {
  auto v1 = prob.solution(5);
  auto v2 = prob.solution(6);

  prob.addMatrixOperator(zot(1), 5, 5);
  prob.addVectorOperator(zot(v1), 5);

  prob.addMatrixOperator(zot(1), 6, 6);
  prob.addVectorOperator(zot(v2), 6);
}

int main(int argc, char** argv)
{
  Environment env(argc, argv);
  
  double Nl = 4;
  double lattice = 4*M_PI/sqrt(3);
  double _scale = Nl*lattice;
  scale = FieldVector<double,2>({_scale, _scale});

  K = Parameters::get<double>("vpfc->K").value();
  // Dune::YaspGrid<2> grid(Dune::FieldVector<double, 2>({2*scale[0], 2*scale[1]}), {2u, 2u});
  // using Factory2 = Dune::StructuredGridFactory<Grid>;
  // auto grid = Factory2::createSimplexGrid({0.0,0.0}, {1.0,1.0},
  //                                           std::array<unsigned int,2>{2u,2u});

  // using Factory3 = Dune::StructuredGridFactory<Grid>;
  // auto grid = Factory3::createSimplexGrid(Dune::FieldVector<double, 2>({-scale[0], -scale[1]}), Dune::FieldVector<double, 2>({scale[0], scale[1]}),
  //                                           std::array<unsigned int,2>{2u,2u});

  // using Grid = UGGrid < dim > ;

  // Start with a structured grid
  const std::array < unsigned, 2 > n = { 8, 8 };
  const FieldVector < double, 2 > lower = {-scale[0], -scale[1]};
  const FieldVector < double, 2 > upper = {scale[0], scale[1]};


  std::shared_ptr < Grid > grid = Dune::StructuredGridFactory < Grid > ::createSimplexGrid(lower, upper, n);

  using GridView = Grid::LeafGridView;
  GridView gridView = grid -> leafGridView();

  // Grid grid(Dune::FieldVector<double, 2>(), Dune::FieldVector<double, 2>(), {2u, 2u});

  // ProblemStat<Param> prob("vpfc", *grid);
  ProblemStat<Param> prob("vpfc", grid);
  prob.initialize(INIT_ALL);

  MyProblemInstat<Param> probInstat("vpfc", prob);
  probInstat.initialize(INIT_UH_OLD);

  AdaptInfo adaptInfo("adapt");

  N = Parameters::get<int>("vpfc->N").value_or(1);
  v0 = Parameters::get<double>("vpfc->v0").value_or(200);
  cooldown = Parameters::get<int>("vpfc->cooldown").value_or(5);
  addVacancy = Parameters::get<bool>("vpfc->add vacancy").value_or(true);
  addNoise = Parameters::get<bool>("vpfc->add noise").value_or(true);
  double scale_ = Parameters::get<double>("scale").value_or(1.0);

  setDensityOperators(prob, probInstat);
  setVelocityOperators(prob, probInstat);
  setTempOperators(prob, probInstat);
  setInitValues(prob, adaptInfo);

  AdaptInstationary adapt("adapt", prob, adaptInfo, probInstat, adaptInfo);
  adapt.adapt();

  return 0;
}
