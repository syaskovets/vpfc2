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
#include <boost/core/demangle.hpp>

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
#include <typeinfo>

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
// using Grid = Dune::UGGrid<2>;
using Grid = Dune::ALUGrid<2,2,Dune::simplex,Dune::conforming>;
using Param = LagrangeBasis<Grid, 1, 1, 1>;

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
double noiseSigma;

FieldVector<double,2> scale;

using GridView = Grid::LeafGridView;

typedef std::pair<double,double> coord;
// <[x,y], phi, v_x, v_y>
typedef boost::fusion::vector<coord, double, double, double> peakProp;

double v0XInit[] = {1.0, -1.0};
double v0YInit[] = {0.0, 0.0};

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

using btype = AMDiS::GlobalBasis<Dune::Functions::PowerPreBasis<Dune::Functions::BasisFactory::FlatLexicographic, 
Dune::Functions::LagrangePreBasis<Dune::GridView<AMDiS::DefaultLeafGridViewTraits<
const AMDiS::AdaptiveGrid<Dune::ALUGrid<2, 2, Dune::simplex, Dune::conforming> > > >, 1, double>, 8> >;

// using btype = AMDiS::GlobalBasis<Dune::Functions::PowerPreBasis<Dune::Functions::BasisFactory::FlatLexicographic, 
// Dune::Functions::LagrangePreBasis<Dune::GridView<AMDiS::DefaultLeafGridViewTraits<const AMDiS::AdaptiveGrid<Dune::ALUGrid<2, 2, 
// Dune::simplex, Dune::conforming> > > >, 1, double>, 8> >;

// using btype = AMDiS::GlobalBasis<Dune::Functions::PowerPreBasis<Dune::Functions::BasisFactory::FlatLexicographic, 
// Dune::Functions::LagrangePreBasis<Dune::GridView<AMDiS::DefaultLeafGridViewTraits<
// const AMDiS::AdaptiveGrid<Dune::ALUGrid<2, 2, Dune::simplex, Dune::conforming> > > >, 1, double>, 8> >;


template <class B, class GF, class TP, class NTRE>
void iterateTreeSubset(B const& basis, GF const& gf, TP const& treePath, NTRE const& nodeToRangeEntry)
{
  auto lf = localFunction(gf);
  auto localView = basis.localView();

  for (const auto& e : elements(basis.gridView(), typename BackendTraits<B>::PartitionSet{}))
  {
    localView.bind(e);
    lf.bind(e);

    auto&& subTree = Dune::TypeTree::child(localView.tree(),treePath);
    Traversal::forEachLeafNode(subTree, [&](auto const& node, auto const& tp)
    {
      using Traits = typename TYPEOF(node)::FiniteElement::Traits::LocalBasisType::Traits;
      using RangeField = typename Traits::RangeFieldType;

      auto&& fe = node.finiteElement();

      // extract component of local function result corresponding to node in tree
      auto localFj = [&](auto const& local)
      {
        const auto& tmp = lf(local);
        return nodeToRangeEntry(node, tp, Dune::MatVec::as_vector(tmp));
      };

      thread_local std::vector<RangeField> interpolationCoeff;
      fe.localInterpolation().interpolate(localFj, interpolationCoeff);
    });
  }
}


template < template <class, class, class, class> typename DiscreteFunction, 
          class Coeff, class GB, class TreePath, class R, typename Expr>
void applyComposerGridFunction(DiscreteFunction<Coeff, GB, TreePath, R>& df, Expr&& expr) {
  const auto & basis = df.basis();
  auto const& treePath = df.treePath();
  auto&& gf = makeGridFunction(FWD(expr), basis.gridView());
  auto ntrm = AMDiS::HierarchicNodeToRangeMap();

  iterateTreeSubset(basis, gf, treePath, ntrm);
}

template <class... Args>
static auto create(const Args&... args) {
  return AMDiS::GlobalBasis{args..., power<2>(lagrange<1>())};
}

template <class Traits>
class MyProblemInstat : public ProblemInstat<Traits> {
  int iter;
  bool vInit;

  typedef decltype(create(std::declval<typename ProblemStat<Traits>::GridView>())) u_type;
  DOFVector<u_type>& u;

  public:
    MyProblemInstat(std::string const& name, ProblemStat<Traits>& prob, DOFVector<u_type>& u)
      : iter(0), vInit(false), ProblemInstat<Traits>(name, prob), u(u) {}

    void closeTimestep(AdaptInfo& adaptInfo) {
      auto phi = this->problemStat_->solution(0);
      auto mu = this->problemStat_->solution(2);
      auto v1 = valueOf(u,0);
      auto v2 = valueOf(u,1);

      // double B0 = integrate(constant(1.0), this->problemStat_->gridView(), 6);
      // double density = integrate(valueOf(phi), this->problemStat_->gridView(), 6)/B0; // density of the initial value
      // std::cout << "B0 " << B0 << "density " << density << std::endl;

      if (++iter > cooldown) {
        peaks.clear();
        particles.clear();

        double min_mu = 0.0;
        double max_phi_x = 0.0;

        auto f = invokeAtQP([&](auto dd_p_x, auto phi_x, auto const& x) {
          if (min_mu > dd_p_x[0])
            min_mu = dd_p_x[0];

          if (max_phi_x < phi_x[0])
            max_phi_x = phi_x[0];

          return dd_p_x;
        }, mu, phi, X());

        applyComposerGridFunction(phi, f);

        auto f1 = invokeAtQP([&](auto dd_p_x, auto v1_x, auto v2_x, auto const& x) {
          // consider only ampls within 20% of the max (min_mu)
          if ((min_mu-dd_p_x[0])/min_mu < 0.2) {
            coord c_x = x_to_coord(x);
            peakProp p(c_x, dd_p_x[0], v1_x[0], v2_x[0]);
            peaks.push_back(p);
            return 1;
          }

          return 0;
        }, mu, v1, v2, X());

        applyComposerGridFunction(phi, f1);

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

            alpha += generateGaussianNoise(0,noiseSigma).first;

            x = cos(alpha * M_PI / 180.0); boost::fusion::at<boost::mpl::int_<2> >(particles[i]) = x;
            y = sin(alpha * M_PI / 180.0); boost::fusion::at<boost::mpl::int_<3> >(particles[i]) = y;
          }
        }

        if (particles.size() < N) {
          std::cout << "Error!!! " << std::endl;
        }

        v1 << invokeAtQP([&](auto phi_x, auto const& x) {
          if (phi_x > (0.00001))
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

            double phi_x_norm = phi_x / max_phi_x;

            if (vInit)
              return phi_x_norm * boost::fusion::at<boost::mpl::int_<2> >(particles[min_dist_i]);
            else
              return phi_x_norm * v0XInit[min_dist_i];
          }
          return 0.0;
        }, phi, X());

        v2 << invokeAtQP([&](auto phi_x, auto const& x) {
          if (phi_x > (0.00001))
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

            double phi_x_norm = phi_x / max_phi_x;

            if (vInit)
              return phi_x_norm * boost::fusion::at<boost::mpl::int_<3> >(particles[min_dist_i]);
            else
              return phi_x_norm * v0YInit[min_dist_i];
          }
          return 0.0;
        }, phi, X());

        if (!vInit)
          vInit = true;
      } else {
        v1.interpolate(constant(0.0));
        v2.interpolate(constant(0.0));
      }

      for (int k = 0; k < 5; ++k) {
        this->problemStat_-> markElements(adaptInfo);
        this->problemStat_-> adaptGrid(adaptInfo);
      }

      ProblemInstat<Traits>::closeTimestep(adaptInfo);
    }
};

template <typename DOFVectorType>
void setInitValues(ProblemStat<Param>& prob, AdaptInfo& adaptInfo, DOFVectorType& u) {
  auto phi = prob.solution(0);
  auto psi = prob.solution(1);

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

  static GridFunctionMarker marker("interface", prob.grid(),
    invokeAtQP([interface_ref,bulk_ref,outer_ref,threshold](double const& phi) -> int {
    // std::cout << " marker f " << ((phi > -threshold) && (phi < threshold)) << " " << phi << std::endl;
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

  valueOf(u).interpolate(constant(0.0));
  psi.interpolate(constant(0.0));
}

template <typename DOFVectorType>
void setDensityOperators(ProblemStat<Param>& prob, MyProblemInstat<Param>& probInstat, DOFVectorType& u) {
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
  prob.addVectorOperator(zot(-v0*valueOf(u)* derivativeOf(valueOf(phi), tag::gradient{})), 0);
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
  // Grid grid(Dune::FieldVector<double, 2>({-scale[0], -scale[1]}), Dune::FieldVector<double, 2>({scale[0], scale[1]}), {2u, 2u});

  // using Grid = UGGrid < dim > ;

  // Start with a structured grid
  const std::array < unsigned, 2 > n = { 16, 16 };
  const FieldVector < double, 2 > lower = {-scale[0], -scale[1]};
  const FieldVector < double, 2 > upper = {scale[0], scale[1]};

  std::shared_ptr < Grid > grid = Dune::StructuredGridFactory < Grid > ::createSimplexGrid(lower, upper, n);


  // using GridView = Grid::LeafGridView;
  // GridView gridView = grid -> leafGridView();

  // Grid grid(Dune::FieldVector<double, 2>(), Dune::FieldVector<double, 2>(), {2u, 2u});

  // ProblemStat<Param> prob("vpfc", *grid);
  ProblemStat<Param> prob("vpfc", grid);
  prob.initialize(INIT_ALL);

  DOFVector u(prob.gridView(), power<2>(lagrange<1>()));

  MyProblemInstat<Param> probInstat("vpfc", prob, u);
  probInstat.initialize(INIT_UH_OLD);

  AdaptInfo adaptInfo("adapt");

  N = Parameters::get<int>("vpfc->N").value_or(1);
  v0 = Parameters::get<double>("vpfc->v0").value_or(200);
  cooldown = Parameters::get<int>("vpfc->cooldown").value_or(5);
  addVacancy = Parameters::get<bool>("vpfc->add vacancy").value_or(true);
  addNoise = Parameters::get<bool>("vpfc->add noise").value_or(true);
  noiseSigma = Parameters::get<double>("vpfc->noise sigma").value();
  double scale_ = Parameters::get<double>("scale").value_or(1.0);

  setDensityOperators(prob, probInstat, u);
  setInitValues(prob, adaptInfo, u);

  AdaptInstationary adapt("adapt", prob, adaptInfo, probInstat, adaptInfo);
  adapt.adapt();

  return 0;
}
