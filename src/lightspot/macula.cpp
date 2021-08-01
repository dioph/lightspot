#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;

template<typename T>
using npy = py::array_t<T, py::array::c_style | py::array::forcecast>;

template<typename T>
T zeta(T x)
{
  const T halfpi = T(1.5707963267948966);
  if (x < 0.0) return T(1.0);
  else if (x > halfpi) return T(0.0);
  else return std::cos(x);
}

template<typename T>
T safe_sqrt(T x)
{
  if (x <= T(0.0)) return T(0.0);
  else return std::sqrt(x);
}

template<typename T>
T safe_acos(T x)
{
  const T pi = T(3.141592653589793);
  if (x <= T(-1.0)) return pi;
  else if (x >= T(1.0)) return T(0.0);
  else return std::acos(x);
}

template<typename T>
T macula_singlespot(T t, T* theta_star, T* c, T* d, T* theta_spot)
{
  const int pLD = 5;
  const T pi = T(3.141592653589793);
  const T halfpi = T(1.5707963267948966);
  const T inv_pi = T(0.3183098861837907);
  const T tol = T(0.0001);
  const T mingress = T(0.0416667);
  T SinInc = std::sin(theta_star[0]);
  T CosInc = std::cos(theta_star[0]);
  T Phi0 = theta_spot[1];
  T SinPhi0 = std::sin(Phi0);
  T CosPhi0 = std::cos(Phi0);
  T Prot = theta_star[1] / (
    T(1.0)
    - theta_star[2] * std::pow(SinPhi0, T(2.0))
    - theta_star[3] * std::pow(SinPhi0, T(4.0))
  );
  T alphamax = theta_spot[2];
  T fspot = theta_spot[3];
  T tmax = theta_spot[4];
  T life = theta_spot[5];
  T ingress = std::fmax(mingress, theta_spot[6]);
  T egress = std::fmax(mingress, theta_spot[7]);
  T tcrit1 = tmax - T(0.5) * life - ingress;
  T tcrit2 = tcrit1 + ingress;
  T tcrit3 = tcrit2 + life;
  T tcrit4 = tcrit3 + egress;
  T alpha, A, Psi, Xi, Upsilon, w;
  if ((t < tcrit1) || (t > tcrit4))
    alpha = T(0.0);
  else if ((tcrit2 < t) && (t < tcrit3))
    alpha = alphamax;
  else  if ((tcrit1 <= t) && (t <= tcrit2))
    alpha = alphamax * ((t - tcrit1) / ingress);
  else
    alpha = alphamax * ((tcrit4 - t) / egress);
  T sinalpha = std::sin(alpha);
  T cosalpha = std::cos(alpha);
  T Lambda = theta_spot[0] + T(2.0) * pi * (t - tmax) / Prot;
  T cosLambda = std::cos(Lambda);
  T cosbeta = CosInc * SinPhi0 + SinInc * CosPhi0 * cosLambda;
  T beta = safe_acos<T>(cosbeta);
  T sinbeta = std::sin(beta);
  T zetapos = zeta<T>(beta + alpha);
  T zetaneg = zeta<T>(beta - alpha);
  if (alpha > tol)
  {
    if (beta > (halfpi + alpha))
      A = T(0.0);
    else if (beta < (halfpi - alpha))
      A = pi * cosbeta * sinalpha * sinalpha;
    else
    {
      Psi = safe_sqrt<T>(T(1.0) - (cosalpha / sinbeta) * (cosalpha / sinbeta));
      Xi = sinalpha * safe_acos<T>(-(cosalpha * cosbeta) / (sinalpha * sinbeta));
      A = safe_acos<T>(cosalpha / sinbeta) + Xi * cosbeta * sinalpha - Psi * sinbeta * cosalpha;
    }
  }
  else
    A = T(0.0);
  T q = T(0.0);
  T Upsilon_den = zetaneg * zetaneg - zetapos * zetapos;
  if (zetapos == zetaneg)
    Upsilon_den++;
  for (int n = 0; n < pLD; n++)
  {
    Upsilon = (
      std::sqrt(std::pow(zetaneg, T(n + 4))) - std::sqrt(std::pow(zetapos, T(n + 4)))
    ) / Upsilon_den;
    w = Upsilon * (T(4.0) * (c[n] - d[n] * fspot)) / (n + T(4.0));
    q += (A * inv_pi) * w;
  }
  return q;
}

template<typename T>
npy<T> macula(npy<T> t, npy<T> theta, npy<T> tstart, npy<T> tend)
{
  const int pstar = 12;
  const int pspot = 8;
  const int pinst = 2;
  const int pLD = 5;
  auto t_ = t.request();
  auto theta_ = theta.request();
  auto tstart_ = tstart.request();
  auto tend_ = tend.request();
  if (t_.ndim != 1) {
    throw std::runtime_error("t should be 1-D, shape (ndata,)");
  }
  if (theta_.ndim != 2) {
    throw std::runtime_error("theta should be 2-D, shape (ntrials, jmax)");
  }
  if (tstart_.ndim != 1) {
    throw std::runtime_error("tstart should be 1-D, shape (mmax,)");
  }
  if (tend_.ndim != 1) {
    throw std::runtime_error("tend should be 1-D, shape (mmax,)");
  }
  if (tstart_.shape[0] != tend_.shape[0]) {
    throw std::runtime_error("tstart and tend should both have shape (mmax,)");
  }
  const int ndata = t_.shape[0];
  const int ntrials = theta_.shape[0];
  const int jmax = theta_.shape[1];
  const int mmax = tstart_.shape[0];
  if ((jmax - mmax * pinst - pstar) % pspot != 0) {
    throw std::runtime_error("Invalid number of parameters!");
  }
  const int nspots = (jmax - mmax * pinst - pstar) / pspot;
  T* t_ptr = (T*) t_.ptr;
  T* theta_ptr = (T*) theta_.ptr;
  T* tstart_ptr = (T*) tstart_.ptr;
  T* tend_ptr = (T*) tend_.ptr;
  npy<T> fmod(ntrials * ndata);
  T* fmod_ptr = (T*) fmod.request().ptr;
  T theta_star[4];
  T c[5];
  T d[5];
  T theta_spot[8];
  T Fab0, Fab, U, B;
  for (int trial = 0; trial < ntrials; trial++)
  {
    for (int j = 0; j < 4; j++)
      theta_star[j] = theta_ptr[trial * jmax + j];
    c[0] = T(1.0);
    d[0] = T(1.0);
    for (int n = 1; n < pLD; n++)
    {
      c[n] = theta_ptr[trial * jmax + n + 3];
      d[n] = theta_ptr[trial * jmax + n + 7];
      c[0] -= c[n];
      d[0] -= d[n];
    }
    Fab0 = T(1.0);
    for (int n = 0; n < pLD; n++)
      Fab0 -= (n * c[n]) / (n + T(4.0));
    for (int i = 0; i < ndata; i++)
    {
      for (int m = 0; m < mmax; m++)
      {
        if ((tstart_ptr[m] < t_ptr[i]) && (t_ptr[i] < tend_ptr[m]))
        {
          U = theta_ptr[trial * jmax + pstar + nspots * pspot + m];
          B = theta_ptr[trial * jmax + pstar + nspots * pspot + mmax + m];
          break;
        }
        else
        {
          U = T(0.0);
          B = T(1.0);
        }
      }
      Fab = Fab0;
      for (int k = 0; k < nspots; k++)
      {
        for (int j = 0; j < pspot; j++)
          theta_spot[j] = theta_ptr[trial * jmax + pstar + j * nspots + k];
        Fab -= macula_singlespot<T>(t_ptr[i], theta_star, c, d, theta_spot);
      }
      fmod_ptr[trial * ndata + i] = U * (Fab / (Fab0 * B) + (B - T(1.0)) / B);
    }
  }
  fmod.resize({ntrials, ndata});
  return fmod;
}

PYBIND11_MODULE(macula, m)
{
    m.doc() = "";
    m.def("macula", &macula<float>, "", "t"_a, "theta"_a, "tstart"_a, "tend"_a);
    m.def("macula", &macula<double>, "", "t"_a, "theta"_a, "tstart"_a, "tend"_a);
}
