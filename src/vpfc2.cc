#include "vpfc2.h"

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

    v0XInit[0] = -1.0; v0XInit[1] = 1.0;
    v0YInit[0] = 0.0; v0YInit[1] = 0.0;
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

    for (int i = 0; i < N; ++i)
    {
      double alpha = std::rand()%360;

      double v_x = cos(alpha * M_PI / 180.0); v0XInit[i] = v_x;
      double v_y = sin(alpha * M_PI / 180.0); v0YInit[i] = v_y;
    }

    // initalizing a regular grid 
    // adding a noise on each grid point 
    // using the first N grid points 

    int cntr = 0;

    if (domain == 0)
    {
      int Nx = floor(sqrt(N)+0.5);
      int Ny = floor(sqrt(N)) + ((N%(Nx*Nx)>0) ? 1 : 0) ;

      double dx = 2*scale[0]/(Nx)*compression;
      double dy = 2*scale[1]/(Ny)*compression;

      for (int i=0; i<Nx ; i++)
        for (int j=0; j<Ny ; j++){
          y[cntr][0] = -scale[0]*compression + 0.5*dx + i*dx;// + dx*0.002*dis*(rand()%1000-500) ;
          // y[cntr][0] = -scale[0]*compression + 0.5*15.4778 + i*15.4778;// + dx*0.002*dis*(rand()%1000-500) ;
          y[cntr][1] = -scale[1]*compression + 0.5*dx + j*dy;// + dy*0.002*dis*(rand()%1000-500) ;

          cntr++;
        }
    }
    else if (domain == 1)
    {
      int Nx = floor(sqrt(N)+0.5);
      int Ny = floor(sqrt(N)) ;

      double dx = 2*scale[0]/(Nx)*compression;
      double dy = 2*scale[1]/(Ny)*compression;

      int i = 0; int j = 0;
      double _x = 0; double _y = 0;

      while (cntr < N) {
        _x = -scale[0]*compression - 0.5*dx + i*dx;
        _y = -scale[1]*compression - 0.5*dy + j*dy;

        if (pow(-1+(2.0/Nx)*(-0.5+i),2) + pow(-1+(2.0/Ny)*(-0.5+j),2) < 1.4) {
          y[cntr][0] = _x;
          y[cntr][1] = _y;

          cntr++;
        }

        ++j; if (_y > scale[1]) {j = 0; ++i;}
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

template <typename DOFVectorType, typename FuncType>
void setInitValues(ProblemStat<Param>& prob, AdaptInfo& adaptInfo, DOFVectorType& u, FuncType& fct) {
  auto phi = prob.solution(0);
  auto psi = prob.solution(1);

  double density;
  double B0 = integrate(constant(1.0), prob.gridView(), 6);
  // double B0 = integrate( constant(1.0) , prob.getMesh() ); // size of the domain
  double psibar = N*0.9*16*M_PI*M_PI/3/B0* sqrt( (-48.0 - 56.0*r)/133.0  );

  std::cout << "B0 " << B0 << " psibar " << psibar << " r " << r << " N " << N << std::endl;

  int interface_ref = Parameters::get<int>("vpfc->interface refinements").value_or(15);
  int bulk_ref = Parameters::get<int>("vpfc->bulk refinements").value_or(10);
  int outer_ref = Parameters::get<int>("vpfc->outer refinements").value_or(5);

  double threshold = 0.90;

#ifdef YASPGRID
  phi.interpolate(fct);
  B0 = integrate(constant(1.0), prob.gridView(), 6);
  density = integrate(valueOf(phi), prob.gridView(), 6)/B0; // density of the initial value
  phi << 1.2*psibar/(density+0.0000001)*valueOf(phi);
#else
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
#endif

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
  if (domain == 0 && !periodicBC_)
  {
    auto op1BC = zot(-H*(tanh(100*(pow<20>(X(0)/scale[0])+pow<20>(X(1)/scale[1])-0.99))+1), 6);
    prob.addMatrixOperator(op1BC,1,0);
  }
  else if (domain == 1)
  {
    auto op1BC = zot(-H*(tanh(100*(pow<10>(pow<2>(X(0)/scale[0])+pow<2>(X(1)/scale[1]))-0.99))+1), 6);
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
  timestep_ = Parameters::get<double>("adapt->timestep").value();
  // cooldownTimestep = Parameters::get<double>("adapt->cooldown timestep").value_or(timestep_);
  cooldownTimestep = Parameters::get<double>("adapt->cooldown timestep").value_or(0.01);
  domain = Parameters::get<int>("vpfc->domain").value();
  compression = Parameters::get<double>("vpfc->c").value();
  bc = Parameters::get<double>("vpfc->bc").value();
  addVacancy = Parameters::get<bool>("vpfc->add vacancy").value_or(true);
  addNoise = Parameters::get<bool>("vpfc->add noise").value_or(true);
  runAndTumble = Parameters::get<bool>("vpfc->run and tumble").value_or(false);
  vicsek = Parameters::get<bool>("vpfc->vicsek").value_or(false);
  vicsekR = Parameters::get<bool>("vpfc->vicsekR").value_or(10.0);
  noiseSigma = Parameters::get<double>("vpfc->noise sigma").value();
  logParticles = Parameters::get<bool>("vpfc->log particles").value_or(false);
  periodicBC_ = Parameters::get<bool>("vpfc->periodic").value_or(false);
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

#ifdef YASPGRID
  // Grid grid(lower, upper, {Nl*20, Nl*20}, std::bitset<2>("10"));
  Grid grid(lower, upper, {Nl*20, Nl*20});
#else
  std::shared_ptr < Grid > grid = Dune::StructuredGridFactory < Grid > ::createSimplexGrid(lower, upper, n);
#endif

  // grid->loadBalance();

  // using GridView = Grid::LeafGridView;
  // GridView gridView = grid -> leafGridView();

  // Grid grid(Dune::FieldVector<double, 2>(), Dune::FieldVector<double, 2>(), {2u, 2u});

  // ProblemStat<Param> prob("vpfc", *grid);
  ProblemStat<Param> prob("vpfc", grid);
  prob.initialize(INIT_ALL);

  DOFVector u(prob.gridView(), power<3>(lagrange<2>()));

  MyProblemInstat<Param> probInstat("vpfc", prob, u, mpiHelper);
  probInstat.initialize(INIT_UH_OLD);

  AdaptInfo adaptInfo("adapt");
  adaptInfo.setTimestep(cooldownTimestep);

  setDensityOperators(prob, probInstat, u);

#ifdef YASPGRID
  if (periodicBC_) {
    prob.boundaryManager()->setBoxBoundary({-1,-1,1,1});
    prob.addPeriodicBC(-1,{{1.0,0.0}, {0.0,1.0}}, {2*scale[0], 0.0});
    prob.addPeriodicBC(1,{{1.0,0.0}, {0.0,1.0}}, {0.0, 2*scale[1]});
  }
#endif

  if (N == 2) {
    G2fix fct;
    setInitValues(prob, adaptInfo, u, fct);
  } else {
    NgridOrientedDots fct;
    setInitValues(prob, adaptInfo, u, fct);
  }

  AdaptInstationary adapt("adapt", prob, adaptInfo, probInstat, adaptInfo);
  adapt.adapt();
  return 0;
}
