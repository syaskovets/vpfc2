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
#include <typeinfo>
#include <mpi.h>

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

double q, r, H, M, v0;
int N, cooldown, domain, bc;
bool addVacancy = true;
bool addNoise = false;
bool logParticles = false;
std::string logParticlesFname;
double noiseSigma, compression;

FieldVector<double,2> scale;

using GridView = Grid::LeafGridView;

// id is double as it is later used in FieldVector<double,3>
// should be size_t ideally
struct CellPeakProp {
  double id;
  double x; double y;
  double v_x; double v_y;
  double phi; double dd_phi;
};

double v0XInit[100];
double v0YInit[100];

std::vector<CellPeakProp> peaks;
std::vector<CellPeakProp> particles;

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

    v0XInit[0] = 1.0; v0XInit[0] = -1.0;
    v0YInit[0] = 0.0; v0YInit[0] = 0.0;
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

    for (int i = 0; i < N; ++i)
    {
      double alpha = std::rand()%360;

      double v_x = cos(alpha * M_PI / 180.0); v0XInit[i] = v_x;
      double v_y = sin(alpha * M_PI / 180.0); v0YInit[i] = v_y;
    }

    int Nx = floor(sqrt(N)+0.5);
    int Ny = floor(sqrt(N)) + ((N%(Nx*Nx)>0) ? 1 : 0) ;

    double dx = 2*scale[0]/(Nx)*compression;
    double dy = 2*scale[1]/(Ny)*compression;
    int cntr = 0;

    std::cout << "N " << N << " scale[0] " << scale[0] << " compression " << compression << " Nx " << Nx << " dx " << dx << " dy " << dy << std::endl;

    // initalizing a regular grid 
    // adding a noise on each grid point 
    // using the first N grid points 


    if (domain == 0)
    {
      for (int i=0; i<Nx ; i++)
        for (int j=0; j<Ny ; j++){
          y[cntr][0] = -scale[0]*compression + 0.5*dx + i*dx;// + dx*0.002*dis*(rand()%1000-500) ;
          // y[cntr][0] = -scale[0]*compression + 0.5*15.4778 + i*15.4778;// + dx*0.002*dis*(rand()%1000-500) ;
          y[cntr][1] = -scale[1]*compression + 0.5*dx + j*dy;// + dy*0.002*dis*(rand()%1000-500) ;

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

template <typename T, typename T1>
double dist(const T& x, const T1& y) {
  return std::sqrt(std::pow(x[0] - y[0], 2)+std::pow(x[1] - y[1], 2));
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


template <class B, class GF, class TP>
void iterateTreeSubset(B const& basis, GF const& gf, TP const& treePath)
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
      thread_local std::vector<RangeField> interpolationCoeff;

      node.finiteElement().localInterpolation().interpolate(AMDiS::HierarchicNodeWrapper{tp,lf}, interpolationCoeff);
    });
  }
}


template < template <class, class, class, class> typename DiscreteFunction, 
          class Coeff, class GB, class TreePath, class R, typename Expr>
void applyComposerGridFunction(DiscreteFunction<Coeff, GB, TreePath, R>& df, Expr&& expr) {
  const auto & basis = df.basis();
  auto const& treePath = df.treePath();
  auto&& gf = makeGridFunction(FWD(expr), basis.gridView());

  iterateTreeSubset(basis, gf, treePath);
}

template <class... Args>
static auto create(const Args&... args) {
  return AMDiS::GlobalBasis{args..., power<3>(lagrange<1>())};
}

template <class Traits>
class MyProblemInstat : public ProblemInstat<Traits> {
  int iter;
  bool vInit;
  std::ofstream logFile;
  const Dune::MPIHelper& mpiHelper;

  // particle id is incorporated into the velocity field
  // to save additional iteration loop
  typedef decltype(create(std::declval<typename ProblemStat<Traits>::GridView>())) u_type;
  DOFVector<u_type>& u;

  public:
    MyProblemInstat(std::string const& name, ProblemStat<Traits>& prob, DOFVector<u_type>& u, const Dune::MPIHelper& mpiHelper)
      : iter(0), vInit(false), ProblemInstat<Traits>(name, prob), u(u), mpiHelper(mpiHelper) {
        if (logParticles) logFile.open(logParticlesFname);
    }

    ~MyProblemInstat() {
        if (logParticles) logFile.close();
    }

    void closeTimestep(AdaptInfo& adaptInfo) {
      auto phi = this->problemStat_->solution(0);
      auto mu = this->problemStat_->solution(2);
      auto u_df = valueOf(u);

      // double B0 = integrate(constant(1.0), this->problemStat_->gridView(), 6);
      // double density = integrate(valueOf(phi), this->problemStat_->gridView(), 6)/B0; // density of the initial value
      // std::cout << "B0 " << B0 << "density " << density << std::endl;

      if (++iter > cooldown) {
        peaks.clear();
        particles.clear();

        double min_mu = 0.0;

        auto f = invokeAtQP([&](auto dd_p_x, auto phi_x, auto const& x) {
          if (min_mu > dd_p_x[0])
            min_mu = dd_p_x[0];

          return dd_p_x;
        }, mu, phi, X());

        applyComposerGridFunction(phi, f);

        auto f1 = invokeAtQP([&](auto phi_x, auto dd_p_x, auto v, auto const& x) {
          // consider only ampls within 20% of the max (min_mu)
          if ((min_mu-dd_p_x[0])/min_mu < 0.2) {
            peaks.push_back(CellPeakProp{v[2], x[0], x[1], v[0], v[1], phi_x[0], dd_p_x[0]});
            return 1;
          }

          return 0;
        }, phi, mu, u_df, X());

        applyComposerGridFunction(phi, f1);

        if (mpiHelper.size() > 1)
          exchgPeaksMPI();

        // sort peaks by second derivative ampl
        std::sort(peaks.begin(), peaks.end(), [](auto &left, auto &right) {
            return left.dd_phi < right.dd_phi;
        });

        for(auto i = 0; i < peaks.size() || particles.size() < N; ++i) {
          bool insert = true;

          for(auto it2 = particles.begin(); it2 != particles.end(); ++it2)
            // 2 is an empirical min dist to filter out the same cells
            if (dist(FieldVector<double,2>{peaks[i].x, peaks[i].y}, FieldVector<double,2>{it2->x,it2->y}) < 2)
              insert = false;

          if (insert)
            particles.push_back(peaks[i]);
        }

        std::cout <<" particles.size() "<< particles.size() << " " << vInit << " " << addNoise << std::endl;

        if (vInit && addNoise) {
          for(auto i = 0; i < particles.size(); ++i) {
            double x = particles[i].v_x;
            double y = particles[i].v_y;
            double alpha = atan2 (y,x) * 180.0 / M_PI;

            alpha += generateGaussianNoise(0,noiseSigma).first;

            x = cos(alpha * M_PI / 180.0); particles[i].v_x = x;
            y = sin(alpha * M_PI / 180.0); particles[i].v_y = y;
          }
        }

        if (particles.size() < N) {
          std::cout << "Error!!! " << std::endl;
        }

        u_df << invokeAtQP([&](auto phi_x, auto const& x) {
          if (phi_x > (0.00001))
          {
            double min_dist = dist(FieldVector<double,2>{particles[0].x, particles[0].y}, x);
            int min_dist_i = 0;

            for (int i = 0; i < particles.size(); ++i) {
              double d = dist(FieldVector<double,2>{particles[i].x, particles[i].y}, x);
              if (d < min_dist) {
                min_dist_i = i;
                min_dist = d;
              }
            }

            double phi_x_norm = phi_x / particles[min_dist_i].phi;

            // particle id is incorporated into the velocity field
            // to save additional iteration loop
            if (vInit)
              return FieldVector<double,3>{phi_x_norm*particles[min_dist_i].v_x, phi_x_norm*particles[min_dist_i].v_y, particles[min_dist_i].id};
            else
              return FieldVector<double,3>{phi_x_norm*v0XInit[min_dist_i], phi_x_norm*v0YInit[min_dist_i], min_dist_i};
          }
          return FieldVector<double,3>{0, 0, 0};
        }, phi, X());

        if (!vInit)
          vInit = true;
      } else {
        u_df.interpolate(constant(0.0));
      }

      for (int k = 0; k < 5; ++k) {
        this->problemStat_-> markElements(adaptInfo);
        this->problemStat_-> adaptGrid(adaptInfo);
      }

      if (logParticles && mpiHelper.rank() == 0) {
        logFile << "Time: " << this->time_ << "\n";
        for (size_t i = 0; i < particles.size(); ++i)
        {
          logFile << "\t" << particles[i].id << " " << particles[i].x << " " << particles[i].y \
                  << " " << particles[i].v_x << " " << particles[i].v_y << "\n";
        }
        logFile.flush();
      }

      ProblemInstat<Traits>::closeTimestep(adaptInfo);
    }

    void exchgPeaksMPI() {
      // MPI_Comm comm;
      int gsize;
      int *recvcounts, *displs;

      MPI_Comm_size(mpiHelper.getCommunicator(), &gsize);

      int locPeaksN, *globPeaksN;
      globPeaksN = (int *)malloc(gsize*1*sizeof(CellPeakProp));

      locPeaksN = peaks.size();
      // exchange peaks local array size
      MPI_Allgather(&locPeaksN, 1, MPI_INT, globPeaksN, 1, MPI_INT, mpiHelper.getCommunicator());

      recvcounts = (int *)malloc(gsize*sizeof(CellPeakProp));
      displs = (int *)malloc(gsize*sizeof(CellPeakProp));

      int TotalPeaksN = 0;

      for (int i=0; i<gsize; ++i) {
        int _N = globPeaksN[i];

        displs[i] = TotalPeaksN;
        recvcounts[i] = _N;
        TotalPeaksN += _N;
      }

      // rbuf = (CellPeakProp *)malloc(TotalPeaksN*sizeof(CellPeakProp));
      std::vector<CellPeakProp> rbuf(TotalPeaksN);

      // MPI type for a CellPeakProp struct
      const int nitems = 7;
      int blocklengths[7] = {1,1,1,1,1,1,1};
      MPI_Datatype types[7] = {MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE};
      MPI_Datatype CellPeakProp_type;
      MPI_Aint offsets[7];

      offsets[0] = offsetof(CellPeakProp, id);
      offsets[1] = offsetof(CellPeakProp, x);
      offsets[2] = offsetof(CellPeakProp, y);
      offsets[3] = offsetof(CellPeakProp, v_x);
      offsets[4] = offsetof(CellPeakProp, v_y);
      offsets[5] = offsetof(CellPeakProp, phi);
      offsets[6] = offsetof(CellPeakProp, dd_phi);

      MPI_Type_create_struct(nitems, blocklengths, offsets, types, &CellPeakProp_type);
      MPI_Type_commit(&CellPeakProp_type);

      MPI_Allgatherv(&peaks[0], peaks.size(), CellPeakProp_type, &rbuf[0], recvcounts, displs, CellPeakProp_type, mpiHelper.getCommunicator());
      peaks = rbuf;

      MPI_Type_free(&CellPeakProp_type);

      if (mpiHelper.rank() == 0) {
        for (int j = 0; j < TotalPeaksN; ++j)
        {
          CellPeakProp temp = peaks[j];
          std::cout << j << " ! " << temp.id << " " << temp.x << " " << temp.y << " " << temp.v_x << " " << temp.v_y << " " << temp.phi << " " << temp.dd_phi << "\n";
        }
      }
    }
};

template <typename DOFVectorType>
void setInitValues(ProblemStat<Param>& prob, AdaptInfo& adaptInfo, DOFVectorType& u) {
  auto phi = prob.solution(0);
  auto psi = prob.solution(1);

  NgridOrientedDots fct;
  // G2fix fct;

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

  tag::partial der0, der1;
  der0.comp = 0;
  der1.comp = 1;

  prob.addMatrixOperator(sot(M), 0, 1);
  prob.addMatrixOperator(sot(1), 1, 2);
  prob.addMatrixOperator(sot(1), 2, 0);
  prob.addMatrixOperator(zot(invTau), 0, 0);
  prob.addVectorOperator(zot(phiOld * invTau), 0);
  prob.addVectorOperator(zot(-v0*valueOf(u,0)* derivativeOf(valueOf(phi), der0)), 0);
  prob.addVectorOperator(zot(-v0*valueOf(u,1)* derivativeOf(valueOf(phi), der1)), 0);
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


int main(int argc, char** argv) {
  Environment env(argc, argv);
  const Dune::MPIHelper& mpiHelper = Dune::MPIHelper::instance(argc,argv);

  double Nl = Parameters::get<int>("Nl").value_or(4);
  double lattice = Parameters::get<int>("lattice").value();

  lattice = lattice*M_PI/sqrt(3);
  double _scale = Nl*lattice;
  scale = FieldVector<double,2>({_scale, _scale});

  N = Parameters::get<int>("vpfc->N").value_or(1);
  v0 = Parameters::get<double>("vpfc->v0").value_or(200);
  cooldown = Parameters::get<int>("vpfc->cooldown").value_or(5);
  domain = Parameters::get<int>("vpfc->domain").value();
  compression = Parameters::get<double>("vpfc->c").value();
  bc = Parameters::get<double>("vpfc->bc").value();
  addVacancy = Parameters::get<bool>("vpfc->add vacancy").value_or(true);
  addNoise = Parameters::get<bool>("vpfc->add noise").value_or(true);
  noiseSigma = Parameters::get<double>("vpfc->noise sigma").value();
  logParticles = Parameters::get<bool>("vpfc->log particles").value_or(false);
  if (logParticles) logParticlesFname = Parameters::get<std::string>("vpfc->log particles fname").value();
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
  grid->loadBalance();

  // using GridView = Grid::LeafGridView;
  // GridView gridView = grid -> leafGridView();

  // Grid grid(Dune::FieldVector<double, 2>(), Dune::FieldVector<double, 2>(), {2u, 2u});

  // ProblemStat<Param> prob("vpfc", *grid);
  ProblemStat<Param> prob("vpfc", grid);
  prob.initialize(INIT_ALL);

  DOFVector u(prob.gridView(), power<3>(lagrange<1>()));

  MyProblemInstat<Param> probInstat("vpfc", prob, u, mpiHelper);
  probInstat.initialize(INIT_UH_OLD);

  AdaptInfo adaptInfo("adapt");

  setDensityOperators(prob, probInstat, u);
  setInitValues(prob, adaptInfo, u);

  AdaptInstationary adapt("adapt", prob, adaptInfo, probInstat, adaptInfo);
  adapt.adapt();
  return 0;
}
