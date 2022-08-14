#ifndef MyProblemInstat_H
#define MyProblemInstat_H

#include "vpfc2.h"

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

  for (const auto& e : elements(basis.gridView()))
  {
    localView.bind(e);
    lf.bind(e);

    auto&& subTree = Dune::TypeTree::child(localView.tree(),treePath);
    Traversal::forEachLeafNode(subTree, [&](auto const& node, auto const& tp)
    {
      using Traits = typename TYPEOF(node)::FiniteElement::Traits::LocalBasisType::Traits;
      using Range = typename Traits::RangeType;
      // using RangeField = typename Traits::RangeFieldType;
      thread_local std::vector<Range> interpolationCoeff;

      node.finiteElement().localInterpolation().interpolate(lf, interpolationCoeff);
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
  return AMDiS::GlobalBasis{args..., power<3>(lagrange<2>())};
}

template <class Traits>
class MyProblemInstat : public ProblemInstat<Traits> {
  int iter;
  bool vInit;
  std::ofstream logFile;
  const Dune::MPIHelper& mpiHelper;
  // time of last run and tumble for every particle
  std::map<unsigned, double> rntTime;
  std::random_device rd;
  std::default_random_engine randGen;
  std::mt19937 randGen2;
  std::poisson_distribution<> poissonDistr;
  std::uniform_real_distribution<double> distribution;

  // particle id is incorporated into the velocity field
  // to save additional iteration loop
  typedef decltype(create(std::declval<typename ProblemStat<Traits>::GridView>())) u_type;


  DOFVector<u_type>& u;
  std::list<std::shared_ptr<FileWriterInterface>> filewriter__;
  unsigned int newPartId;

  public:
    MyProblemInstat(std::string const& name, ProblemStat<Traits>& prob, DOFVector<u_type>& u, const Dune::MPIHelper& mpiHelper)
      : iter(0), vInit(false), ProblemInstat<Traits>(name, prob), u(u), mpiHelper(mpiHelper), distribution(0.0,1.0), newPartId(N+100), randGen2(rd()), poissonDistr(RTRate/timestep_) {
      if (logParticles) logFile.open(logParticlesFname);

#ifdef _DEBUG
      FileWriterCreator<DOFVector<u_type>> creator(std::shared_ptr<DOFVector<u_type>>(&u, [](DOFVector<u_type> *) {}), this->problemStat_->boundaryManager());
      filewriter__.clear();

      auto localView = u.basis().localView();
      Traversal::forEachNode(localView.tree(), [&](auto const& /*node*/, auto treePath) -> void
      {
        std::string componentName = "u->output[" + to_string(treePath) + "]";
        std::string format = "vtk";
        auto writer = creator.create(format, componentName, treePath);
        if (writer)
          filewriter__.push_back(std::move(writer));
      });
#endif

      if (runAndTumble) {
        // determine after how many time steps to tumble
        // average \nu is \lamdba/dt
        // \lamdba is in tumbles/s
        for (size_t i = 0; i < N; ++i)
        {
          rntTime[i] = poissonDistr(randGen2);
        }
      }
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

        double min_mu = 0.0;

        auto f = invokeAtQP([&](auto dd_p_x, auto phi_x, auto const& x) {
          if (min_mu > dd_p_x[0])
            min_mu = dd_p_x[0];

          return dd_p_x;
        }, mu, phi, X());

        applyComposerGridFunction(phi, f);


        auto f1 = invokeAtQP([&](auto phi_x, auto dd_p_x, auto v, auto const& x) {
          // consider only ampls within 25% of the max (min_mu)
          if ((min_mu-dd_p_x[0])/min_mu < 0.25) {
            peaks.push_back(CellPeakProp{v[2], x[0], x[1], v[0], v[1], 0.0, 0.0, phi_x[0], dd_p_x[0]});
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

        particles.clear();

        for(int i = 0; i < peaks.size() && particles.size() < N; ++i) {
          bool insert = true;
#ifdef _DEBUG
          std::cout << min_mu << " min_mu " <<  peaks[i].id << " being inserted " << peaks[i].x << " " << peaks[i].y << " " << peaks[i].phi << " " << peaks[i].dd_phi <<  std::endl;
#endif
          for(auto it2 = particles.begin(); it2 != particles.end(); ++it2) {
#ifdef _DEBUG
            // bool flag = peaks[i].id == it2->id;
            // std::cout << "distance to " << it2->id << " " << dist(FieldVector<double,2>{peaks[i].x, peaks[i].y}, FieldVector<double,2>{it2->x,it2->y}) << " " << flag << std::endl;
#endif
            // 2.5 is an empirical min dist to filter out the same cells
            if (dist(FieldVector<double,2>{peaks[i].x, peaks[i].y}, FieldVector<double,2>{it2->x,it2->y}) < 2.5 || (vInit && peaks[i].id == it2->id))
              insert = false;
          }

          if (insert)
            particles.push_back(peaks[i]);
        }

        // if new particles emerge
        for(int i = 0; i < peaks.size() && particles.size() < N; ++i) {
          bool insert = true;
#ifdef _DEBUG
          std::cout << min_mu << " new_part " <<  peaks[i].id << " " << newPartId << " being inserted " << peaks[i].x << " " << peaks[i].y << " " << peaks[i].phi << " " << peaks[i].dd_phi <<  std::endl;
#endif
          for(auto it2 = particles.begin(); it2 != particles.end(); ++it2) {
#ifdef _DEBUG
            // std::cout << "distance to " << it2->id << " " << dist(FieldVector<double,2>{peaks[i].x, peaks[i].y}, FieldVector<double,2>{it2->x,it2->y}) << std::endl;
#endif
            // 2.5 is an empirical min dist to filter out the same cells
            if (dist(FieldVector<double,2>{peaks[i].x, peaks[i].y}, FieldVector<double,2>{it2->x,it2->y}) < 2.5)
              insert = false;
          }

          if (insert) {
            peaks[i].id = ++newPartId;
            double rn = distribution(randGen);
            double x = cos(2*rn*M_PI); peaks[i].v_x = x;
            double y = sin(2*rn*M_PI); peaks[i].v_y = y;
            peaks[i].rv_x = 0;
            peaks[i].rv_y = 0;

            particles.push_back(peaks[i]);
          }
        }

        // +3 is when particlesOld[i].rv_x/y start to have correct values
        if ((iter > cooldown+3) && runAndTumble) {
          for(auto i = 0; i < particlesOld.size(); ++i) {
            // if ((std::abs(particlesOld[i].rv_x) + std::abs(particlesOld[i].rv_y))/(this->tau_*v0) < 0.05) {
            if (--rntTime[i] < 1) {
              // std::cout << "particle " << i << " " << particlesOld[i].x << " " << particlesOld[i].y << " tumbles at " << this->time_ << " " << std::abs(particlesOld[i].rv_x)+std::abs(particlesOld[i].rv_y) << " " << this->tau_*v0 << " " << particlesOld[i].v_x << " "
              // << particlesOld[i].v_y << " " << this->time_-rntTime[i] << std::endl;
              double rn = distribution(randGen);

              for (size_t j = 0; j < particles.size(); ++j)
                if (particles[j].id == i) {
                  double x = cos(2*rn*M_PI); particles[j].v_x = x;
                  double y = sin(2*rn*M_PI); particles[j].v_y = y;
                  break;
                }


              rntTime[i] = poissonDistr(randGen2);
            }
          }
        }

        if (vInit && vicsek) {
          std::vector<CellPeakProp> theta;
          for(auto i = 0; i < particles.size(); ++i) {
            theta.clear();

            for (int j = 0; j < particles.size(); ++j)
            {
              if (dist(FieldVector<double,2>{particles[i].x, particles[i].y}, FieldVector<double,2>{particles[j].x, particles[j].y}) < vicsekR) {
                theta.push_back(particles[j]);
              }
            }

            double u1 = 0, u2 = 0;
            for (int j = 0; j < theta.size(); ++j)
            {
              u1 += theta[j].v_x;
              u2 += theta[j].v_y;
            }

            particles[i].v_x = u1/particles.size();
            particles[i].v_y = u2/particles.size();
          }
        }

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

#ifdef _DEBUG
          std::cout << particles.size() << " particles left out of " << N << std::endl;
#endif

        // arrays used to store weighted sum particle locations
        std::vector<FieldVector<double,2>> particleWSPositions;
        std::vector<double> particleWSPositionsTotal;
        particleWSPositions.resize(particles.size());
        particleWSPositionsTotal.resize(particles.size());

        u_df << invokeAtQP([&](auto phi_x, auto const& x) {
          if (phi_x > (0.000001))
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

            if (periodicBC_)
            {
              // account for periodicity when determining weighted sum location of a particle
              FieldVector<double,2> pos = FieldVector<double,2>(
                {particleWSPositions[min_dist_i][0] / particleWSPositionsTotal[min_dist_i],
                particleWSPositions[min_dist_i][1] / particleWSPositionsTotal[min_dist_i]});

              if (std::abs(pos[0]-x[0]) > scale[0]) {
                int sign_x = (pos[0] > 0) ? 1 : ((pos[0] < 0) ? -1 : 0);
                particleWSPositions[min_dist_i][0] += sign_x*(2*scale[0]-std::abs(x[0])) * phi_x;
              } else {
                particleWSPositions[min_dist_i][0] += x[0] * phi_x;
              }

              if (std::abs(pos[1]-x[1]) > scale[1]) {
                int sign_y = (pos[1] > 0) ? 1 : ((pos[1] < 0) ? -1 : 0);
                particleWSPositions[min_dist_i][1] += sign_y*(2*scale[1]-std::abs(x[1])) * phi_x;
              } else {
                particleWSPositions[min_dist_i][1] += x[1] * phi_x;
              }
            }
            else
            {
              particleWSPositions[min_dist_i][0] += x[0] * phi_x;
              particleWSPositions[min_dist_i][1] += x[1] * phi_x;
            }

            particleWSPositionsTotal[min_dist_i] += phi_x;

            // particle id is incorporated into the velocity field
            // to save additional iteration loop
            if (vInit){
              // normalize the velocity field
              double v_norm = sqrt(pow(particles[min_dist_i].v_x,2)+pow(particles[min_dist_i].v_y, 2));
              return FieldVector<double,3>{phi_x_norm*particles[min_dist_i].v_x/v_norm, phi_x_norm*particles[min_dist_i].v_y/v_norm, particles[min_dist_i].id};
            }
            else
              return FieldVector<double,3>{phi_x_norm*v0XInit[min_dist_i], phi_x_norm*v0YInit[min_dist_i], min_dist_i};
          }
          return FieldVector<double,3>{0, 0, 0};
        }, phi, X());

        // first iteration: velocity and id fields are no correct
        if (!vInit) {
          particles.clear();
        }

        // compute particle locations as weighted sums of all non-zero phi elements
        for (int i = 0; i < particles.size(); ++i)
        {
          particles[i].x = particleWSPositions[i][0] / particleWSPositionsTotal[i];
          particles[i].y = particleWSPositions[i][1] / particleWSPositionsTotal[i];

          if (periodicBC_) {
            // account for periodicity when determining weighted sum location of a particle
            if (std::abs(particles[i].x) > scale[0]) {
              int sign_x = (particles[i].x > 0) ? 1 : ((particles[i].x < 0) ? -1 : 0);
              particles[i].x = -1*sign_x*(2*scale[0]-std::abs(particles[i].x));
            }
            if (std::abs(particles[i].y) > scale[1]) {
              int sign_y = (particles[i].y > 0) ? 1 : ((particles[i].y < 0) ? -1 : 0);
              particles[i].y = -1*sign_y*(2*scale[1]-std::abs(particles[i].y));
            }
          }

          if (particlesOld.size())
          {
            particles[i].rv_x = particles[i].x - particlesOld[particles[i].id].x;
            particles[i].rv_y = particles[i].y - particlesOld[particles[i].id].y;

            if (periodicBC_) {
              // account for periodicity when determining real velocity of the particles
              if (std::abs(particles[i].rv_x) > scale[0]) {
                int sign_x = (particles[i].rv_x > 0) ? 1 : ((particles[i].rv_x < 0) ? -1 : 0);
                particles[i].rv_x = -1*sign_x*(2*scale[0]-std::abs(particles[i].rv_x));
              }
              if (std::abs(particles[i].rv_y) > scale[1]) {
                int sign_y = (particles[i].rv_y > 0) ? 1 : ((particles[i].rv_y < 0) ? -1 : 0);
                particles[i].rv_y = -1*sign_y*(2*scale[1]-std::abs(particles[i].rv_y));
              }
            }
          }
        }

        if (vInit)
        {
          particlesOld.clear();
          for (int i = 0; i < particles.size(); ++i)
            particlesOld[particles[i].id] = particles[i];
        }

        if (!vInit) {
          vInit = true;
          adaptInfo.setTimestep(timestep_);
        }
      } else {
        u_df.interpolate(constant(0.0));
      }

#ifndef YASPGRID
      for (int k = 0; k < 5; ++k) {
        this->problemStat_-> markElements(adaptInfo);
        this->problemStat_-> adaptGrid(adaptInfo);
      }
#endif

      double B0 = integrate(constant(1.0), this->problemStat_->gridView(), 6);
      double density = integrate(valueOf(phi), this->problemStat_->gridView(), 6)/B0; // density of the phi

      if (logParticles && mpiHelper.rank() == 0) {
        logFile << "Time: " << this->time_ << " density " << density << "\n";
        for (size_t i = 0; i < particles.size(); ++i)
        {
          logFile << "\t" << particles[i].id << " " << particles[i].x << " " << particles[i].y \
                  << " " << particles[i].v_x << " " << particles[i].v_y << " " << particles[i].rv_x << " " << particles[i].rv_y << "\n";
        }
        logFile.flush();
      }

#ifdef _DEBUG
      for (auto writer : filewriter__)
        writer->write(adaptInfo, true);
#endif

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
      const int nitems = 9;
      int blocklengths[9] = {1,1,1,1,1,1,1,1,1};
      MPI_Datatype types[9] = {MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE};
      MPI_Datatype CellPeakProp_type;
      MPI_Aint offsets[9];

      offsets[0] = offsetof(CellPeakProp, id);
      offsets[1] = offsetof(CellPeakProp, x);
      offsets[2] = offsetof(CellPeakProp, y);
      offsets[3] = offsetof(CellPeakProp, v_x);
      offsets[4] = offsetof(CellPeakProp, v_y);
      offsets[5] = offsetof(CellPeakProp, rv_x);
      offsets[6] = offsetof(CellPeakProp, rv_y);
      offsets[7] = offsetof(CellPeakProp, phi);
      offsets[8] = offsetof(CellPeakProp, dd_phi);

      MPI_Type_create_struct(nitems, blocklengths, offsets, types, &CellPeakProp_type);
      MPI_Type_commit(&CellPeakProp_type);

      MPI_Allgatherv(&peaks[0], peaks.size(), CellPeakProp_type, &rbuf[0], recvcounts, displs, CellPeakProp_type, mpiHelper.getCommunicator());
      peaks = rbuf;

      MPI_Type_free(&CellPeakProp_type);

      if (mpiHelper.rank() == 0) {
        for (int j = 0; j < TotalPeaksN; ++j)
        {
          CellPeakProp temp = peaks[j];
          std::cout << j << " ! " << temp.id << " " << temp.x << " " << temp.y << " " << temp.v_x << " " << temp.v_y << " "
                    << temp.rv_x << " " << temp.rv_y << " " << temp.phi << " " << temp.dd_phi << "\n";
        }
      }
    }
};
#endif