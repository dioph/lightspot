#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;
using npy = py::array_t<double, py::array::c_style | py::array::forcecast>;
using tuple = std::tuple<npy,npy,npy,npy,npy,npy,npy>;
using vec = std::vector<double>;
using mat = std::vector<vec>;

using std::cos;
using std::sin;
using std::acos;
using std::pow;
using std::sqrt;

double zeta (double x)
{
  const double halfpi = 1.5707963267948966;
  if (x < 0.0) return 1.0;
  else if (x > halfpi) return 0.0;
  else return cos(x);
}

double kronecker (double x, double y)
{
  if (x == y) return 1.0;
  else return 0.0;
}

tuple macula (npy T, npy Theta_star, npy Theta_spot, npy Theta_inst, npy Tstart, npy Tend,
              bool derivatives = false, bool temporal = false, bool TdeltaV = false)
{
  // PRE-AMBLE ////////////////////////////////////////////////////////////////
  // Note - There should be no need to ever change these four parameters.
  const int pstar = 12;
  const int pspot = 8;
  const int pinst = 2;
  const int pLD = 5;
  // INPUTS ///////////////////////////////////////////////////////////////////
  auto t_ = T.request();
  auto theta_star_ = Theta_star.request();
  auto theta_spot_ = Theta_spot.request();
  auto theta_inst_ = Theta_inst.request();
  auto tstart_ = Tstart.request();
  auto tend_ = Tend.request();
  if (t_.ndim != 1) {
    throw std::runtime_error("t should be 1-D, shape (ndata,)");
  }
  if (theta_star_.ndim != 1) {
    throw std::runtime_error("theta_star should be 1-D, shape (12,)");
  }
  if (theta_star_.shape[0] != pstar) {
    throw std::runtime_error("Wrong number of star params (there should be 12)");
  }
  if (theta_spot_.ndim != 2) {
    throw std::runtime_error("theta_spot should be 2-D, shape (8, Nspot)");
  }
  if (theta_spot_.shape[0] != pspot) {
    throw std::runtime_error("Wrong number of spot params (there should be 8)");
  }
  if (theta_inst_.ndim != 2) {
    throw std::runtime_error("theta_inst should be 2-D, shape (2, mmax)");
  }
  if (theta_inst_.shape[0] != pinst) {
    throw std::runtime_error("Wrong number of inst params (there should be 2)");
  }
  if (tstart_.shape[0] != theta_inst_.shape[1]) {
    throw std::runtime_error("tstart should have shape (mmax,)");
  }
  if (tend_.shape[0] != theta_inst_.shape[1]) {
    throw std::runtime_error("tend should have shape (mmax,)");
  }
  double *t = (double *) t_.ptr,
    *theta_star = (double *) theta_star_.ptr,
    *theta_spot = (double *) theta_spot_.ptr,
    *theta_inst = (double *) theta_inst_.ptr,
    *tstart = (double *) tstart_.ptr,
    *tend = (double *) tend_.ptr;
  // VARIABLES ////////////////////////////////////////////////////////////////
  const int ndata = t_.shape[0];
  const int Nspot = theta_spot_.shape[1];
  const int mmax = theta_inst_.shape[1];
  const int jmax = pstar + pspot * Nspot + pinst * mmax;
  vec tref(Nspot); // By default, macula will set tref[k]=tmax[k]
  double SinInc, CosInc;
  const double pi = 3.141592653589793;
  const double halfpi = 1.5707963267948966;
  const double piI = 0.3183098861837907;
  const double tol = 0.0001; // alpha values below this will be ignored
  const double mingress = 0.0416667; // minimum ingress/egress time allowed
  // OUTPUTS //////////////////////////////////////////////////////////////////
  npy Fmod(ndata), Deltaratio(ndata);
  double *fmod = (double *) Fmod.request().ptr;
  double *deltaratio = (double *) Deltaratio.request().ptr;
  npy dFmoddt(ndata);
  double *dfmoddt = (double *) dFmoddt.request().ptr;
  npy dFmod_star(ndata * pstar);
  npy dFmod_spot(ndata * pspot * Nspot);
  npy dFmod_inst(ndata * pinst * mmax);
  double *dfmod_star = (double *) dFmod_star.request().ptr;
  double *dfmod_spot = (double *) dFmod_spot.request().ptr;
  double *dfmod_inst = (double *) dFmod_inst.request().ptr;
  
  /////////////////////////////////////////////////////////////////////////////
  //                       SECTION 1: THETA ASSIGNMENT                       //
  /////////////////////////////////////////////////////////////////////////////
  
  vec Theta(jmax);

  int l = 0;
  for (int j = 0; j < pstar; ++j) {
    Theta[l++] = theta_star[j];
  }
  for (int k = 0; k < Nspot; ++k) {
    for (int j = 0; j < pspot; ++j) {
      Theta[l++] = theta_spot[j * Nspot + k];
    }
  }
  for (int j = 0; j < pinst; ++j) {
    for (int m = 0; m < mmax; ++m) {
      Theta[l++] = theta_inst[j * mmax + m];
    }
  }
  
  // Thus we have...
  // Theta_star[j] = Theta[j], 0 <= j < pstar
  // Theta_spot[j][k] = Theta[pstar + pspot * k + j], 0 <= j < pspot, 0 <= k < Nspot
  // Theta_inst[j][m] = Theta[pstar + pspot * Nspot + mmax * j + m], 0 <= j < pinst, 0 <= m < mmax

  /////////////////////////////////////////////////////////////////////////////
  //                       SECTION 2: BASIC PARAMETERS                       //
  /////////////////////////////////////////////////////////////////////////////
  
  vec c(pLD), d(pLD);
  vec U(mmax), B(mmax);

  vec Box(mmax);
  vec Phi0(Nspot), SinPhi0(Nspot), CosPhi0(Nspot), Prot(Nspot);
  vec beta(Nspot), sinbeta(Nspot), cosbeta(Nspot);
  vec alpha(Nspot), sinalpha(Nspot), cosalpha(Nspot);
  vec Lambda(Nspot), sinLambda(Nspot), cosLambda(Nspot);
  vec tcrit1(Nspot), tcrit2(Nspot), tcrit3(Nspot), tcrit4(Nspot);
  vec alphamax(Nspot), fspot(Nspot), tmax(Nspot), life(Nspot), ingress(Nspot), egress(Nspot);

  // section 3
  vec zetaneg(Nspot), zetapos(Nspot);
  mat Upsilon(pLD, vec(Nspot)), w(pLD, vec(Nspot));
  vec Psi(Nspot), Xi(Nspot);
  double q;
  vec A(Nspot);
  double Fab0, Fab;

  // section 4
  mat dc(pLD, vec(jmax, 0)), dd(pLD, vec(jmax, 0));
  mat dU(mmax, vec(jmax, 0)), dB(mmax, vec(jmax, 0));
  mat dfspot(Nspot, vec(jmax, 0));

  // section 5
  mat dalpha(Nspot, vec(jmax, 0));
  mat dbeta(Nspot, vec(jmax, 0));

  // section 6
  double epsil;
  vec dAda(Nspot), dAdb(Nspot);

  // section 7
  double dA, dUpsilon, dw, dzetaneg, dzetapos, dq, dFtilde, dFab;
  vec dFmod(jmax);
  vec dzetanegda(Nspot), dzetaposda(Nspot);
  vec dFab0(jmax, 0);

  // section 8
  double dalphadt, dbetadt, dzetanegdt, dzetaposdt;
  double dqdt, dAdt, dwdt, dUpsilondt;
  double dFtildedt, dFabdt;

  // c and d assignment
  for (int n = 1; n < pLD; ++n) {
    c[n] = Theta[n+3];
    d[n] = Theta[n+7];
  }
  c[0] = 1.0 - c[1] - c[2] - c[3] - c[4]; // c0
  d[0] = 1.0 - d[1] - d[2] - d[3] - d[4]; // d0
  
  // inclination substitutions
  SinInc = sin(Theta[0]);
  CosInc = cos(Theta[0]);

  // U and B assignment
  for (int m = 0; m < mmax; ++m) {
    U[m] = Theta[pstar + pspot * Nspot + m];
    B[m] = Theta[pstar + pspot * Nspot + mmax + m];
  }
  
  // Spot params
  for (int k = 0; k < Nspot; ++k) {
    // Phi0 & Prot calculation
        Phi0[k] = Theta[pstar + pspot * k + 1];
    SinPhi0[k] = sin(Phi0[k]);
    CosPhi0[k] = cos(Phi0[k]);
    Prot[k] = Theta[1] / (1.0 - Theta[2] * pow(SinPhi0[k], 2) - Theta[3] * pow(SinPhi0[k], 4));
    // alpha calculation
    alphamax[k] = Theta[pstar + pspot * k + 2];
       fspot[k] = Theta[pstar + pspot * k + 3];
        tmax[k] = Theta[pstar + pspot * k + 4];
        life[k] = Theta[pstar + pspot * k + 5];
     ingress[k] = Theta[pstar + pspot * k + 6];
      egress[k] = Theta[pstar + pspot * k + 7];
    if (ingress[k] < mingress) { // minimum ingress time
      ingress[k] = mingress;
    }
    if (egress[k] < mingress) { // minimum egress time
      egress[k] = mingress;
    }
    // macula defines the reference time = maximum spot-size time
    // However, one can change the line below to whatever they wish.
    tref[k] = tmax[k];
    // tcrit points = critical instances in the evolution of the spot
    tcrit1[k] = tmax[k] - 0.5 * life[k] - ingress[k];
    tcrit2[k] = tmax[k] - 0.5 * life[k];
    tcrit3[k] = tmax[k] + 0.5 * life[k];
    tcrit4[k] = tmax[k] + 0.5 * life[k] + egress[k];
  }

  // Fab0
  Fab0 = 0.0;
  for (int n = 0; n < pLD; ++n) {
    Fab0 += (n * c[n]) / (n + 4.0);
  }
  Fab0 = 1.0 - Fab0;
  
  /////////////////////////////////////////////////////////////////////////////
  //                       SECTION 4: BASIS DERIVATIVES                      //
  /////////////////////////////////////////////////////////////////////////////

  if (derivatives) {
    // derivatives of c & d.
    dc[1][4] = 1.0;
    dc[2][5] = 1.0;
    dc[3][6] = 1.0;
    dc[4][7] = 1.0;
    dc[0][4] = -1.0;
    dc[0][5] = -1.0;
    dc[0][6] = -1.0;
    dc[0][7] = -1.0;
    dd[1][8] = 1.0;
    dd[2][9] = 1.0;
    dd[3][10] = 1.0;
    dd[4][11] = 1.0;
    dd[0][8] = -1.0;
    dd[0][9] = -1.0;
    dd[0][10] = -1.0;
    dd[0][11] = -1.0;

    // derivatives of U
    for (int m = 0; m < mmax; ++m) {
      dU[m][pstar + pspot * Nspot + m] = 1.0;
    }

    // derivatives of B
    for (int m = 0; m < mmax; ++m) {
      dB[m][pstar + pspot * Nspot + mmax + m] = 1.0;
    }

    // Derivatives of fspot (4th spot parameter)
    for (int k = 0; k < Nspot; ++k) {
      dfspot[k][pstar + pspot * k + 3] = 1.0;
    }

    // Derivatives of Fab0
    dFab0[4] = -0.2;
    dFab0[5] = -0.33333333;
    dFab0[6] = -0.42857143;
    dFab0[7] = -0.5;
  }

  // MASTER LOOP //////////////////////////////////////////////////////////////
  for (int i = 0; i < ndata; ++i) {
    // Box-car function (labelled as Pi_m in the paper)
    for (int m = 0; m < mmax; ++m) {
      if (t[i] > tstart[m] && t[i] < tend[m])
        Box[m] = 1.0;
      else
        Box[m] = 0.0;
    }
    // alpha, lambda & beta
    for (int k = 0; k < Nspot; ++k) {
      // temporal evolution of alpha
      if (t[i] < tcrit1[k] || t[i] > tcrit4[k])
        alpha[k] = 0.0;
      else if (t[i] < tcrit3[k] && t[i] > tcrit2[k])
        alpha[k] = alphamax[k];
      else if (t[i] <= tcrit2[k] && t[i] >= tcrit1[k])
        alpha[k] = alphamax[k] * ((t[i] - tcrit1[k]) / ingress[k]);
      else
        alpha[k] = alphamax[k] * ((tcrit4[k] - t[i]) / egress[k]);
      sinalpha[k] = sin(alpha[k]);
      cosalpha[k] = cos(alpha[k]);
      // Lambda & beta calculation
      Lambda[k] = Theta[pstar + pspot * k] + 2.0 * pi * (t[i] - tref[k]) / Prot[k];
      sinLambda[k] = sin(Lambda[k]);
      cosLambda[k] = cos(Lambda[k]);
      cosbeta[k] = CosInc * SinPhi0[k] + SinInc * CosPhi0[k] * cosLambda[k];
      beta[k] = acos(cosbeta[k]);
      sinbeta[k] = sin(beta[k]);
    }

    ///////////////////////////////////////////////////////////////////////////
    //                       SECTION 3: COMPUTING FMOD                       //
    ///////////////////////////////////////////////////////////////////////////

    Fab = Fab0;
    fmod[i] = 0.0;
    for (int k = 0; k < Nspot; ++k) {
      // zetapos and zetaneg
      zetapos[k] = zeta(beta[k] + alpha[k]);
      zetaneg[k] = zeta(beta[k] - alpha[k]);
      // Area A
      if (alpha[k] > tol) {
        if (beta[k] > (halfpi + alpha[k])) {
          // Case IV
          A[k] = 0.0;
        }
        else if (beta[k] < (halfpi - alpha[k])) {
          // Case I
          A[k] = pi * cosbeta[k] * pow(sinalpha[k], 2);
        }
        else {
          // Case II & III
          Psi[k] = sqrt(1.0 - pow((cosalpha[k] / sinbeta[k]), 2));
          Xi[k] = sinalpha[k] * acos(-(cosalpha[k] * cosbeta[k])
                                      / (sinalpha[k] * sinbeta[k]));
          A[k] = acos(cosalpha[k] / sinbeta[k])
            + Xi[k] * cosbeta[k] * sinalpha[k]
            - Psi[k] * sinbeta[k] * cosalpha[k];
        }
      }
      else {
        A[k] = 0.0;
      }
      q = 0.0;
      // Upsilon & w
      for (int n = 0; n < pLD; ++n) {
        Upsilon[n][k] = pow(zetaneg[k], 2) - pow(zetapos[k], 2) + kronecker(zetapos[k], zetaneg[k]);
        Upsilon[n][k] = (sqrt(pow(zetaneg[k], n+4)) - sqrt(pow(zetapos[k], n+4))) / Upsilon[n][k];
        w[n][k] = (4.0 * (c[n] - d[n] * fspot[k])) / (n + 4.0);
        w[n][k] *= Upsilon[n][k];
        // q
        q += (A[k] * piI) * w[n][k];
      }
      // Fab
      Fab -= q;
    }
    for (int m = 0; m < mmax; ++m) {
      fmod[i] += U[m] * Box[m] * (Fab / (Fab0 * B[m]) + (B[m] - 1.0) / B[m]);
    }
    // delta {obs}/delta
    if (TdeltaV) {
      deltaratio[i] = 0.0;
      for (int m = 0; m < mmax; ++m) {
        deltaratio[i] += B[m] * Box[m];
      }
      deltaratio[i] = (Fab0 / Fab) / deltaratio[i];
    }
    else {
      deltaratio[i] = 1.0;
    }

    // Master if-loop
    if (derivatives) {

      /////////////////////////////////////////////////////////////////////////
      //                 SECTION 5: ALPHA & BETA DERIVATIVES                 //
      /////////////////////////////////////////////////////////////////////////

      for (int k = 0; k < Nspot; ++k) {
        // Derivatives of alpha(alphamax,tmax,life,ingress,egress)
        // [function of 5*Nspot parameters]
        // wrt alphamax (3rd spot parameter)
        dalpha[k][pstar + pspot * k + 2] = alpha[k] / alphamax[k];
        // wrt tmax (5th spot parameter)
        if (t[i] < tcrit2[k] && t[i] > tcrit1[k])
          dalpha[k][pstar + pspot * k + 4] = -alphamax[k] / ingress[k];
        else if (t[i] < tcrit4[k] && t[i] > tcrit3[k])
          dalpha[k][pstar + pspot * k + 4] = alphamax[k] / egress[k];
        // wrt life (6th spot parameter)
        if (t[i] < tcrit2[k] && t[i] > tcrit1[k])
          dalpha[k][pstar + pspot * k + 5] = 0.5 * alphamax[k] / ingress[k];
        else if (t[i] < tcrit4[k] && t[i] > tcrit3[k])
          dalpha[k][pstar + pspot * k + 5] = 0.5 * alphamax[k] / egress[k];
        // wrt ingress (7th spot parameter)
        if (t[i] < tcrit2[k] && t[i] > tcrit1[k]) {
          dalpha[k][pstar + pspot * k + 6] = -(alphamax[k] / pow(ingress[k], 2)) *
            (t[i] - 0.50 * (tcrit1[k] + tcrit2[k]));
        }
        // wrt egress (8th spot parameter)
        if (t[i] < tcrit4[k] && t[i] > tcrit3[k]) {
          dalpha[k][pstar + pspot * k + 7] = (alphamax[k] / pow(egress[k], 2)) *
            (t[i] - 0.50 * (tcrit3[k] + tcrit4[k]));
        }
        // Stellar derivatives of beta(Istar,Phi0,Lambda0,Peq,kappa2,kappa4)
        // [Function of 4+2*Nspot parameters]
        // wrt Istar (1st star parameter)
        dbeta[k][0] = SinPhi0[k] * SinInc - cosLambda[k] * CosPhi0[k] * CosInc;
        dbeta[k][0] /= sinbeta[k];
        // wrt Peq (2nd star parameter)
        dbeta[k][1] = CosPhi0[k] * sinLambda[k] * SinInc / sinbeta[k];
        dbeta[k][1] *= 2.0 * pi * (t[i] - tref[k]) / Theta[1]; // Temporary
        // wrt kappa2 (3rd star parameter)
        dbeta[k][2] = -dbeta[k][1] * pow(SinPhi0[k], 2);
        // wrt kappa4 (4th star parameter)
        dbeta[k][3] = -dbeta[k][1] * pow(SinPhi0[k], 4);
        // wrt Peq continued
        dbeta[k][1] = -dbeta[k][1] / Prot[k];
        // Spot-derivatives of beta
        // wrt Lambda [1st spot parameter]
        dbeta[k][pstar + pspot * k] = SinInc * CosPhi0[k] * sinLambda[k] / sinbeta[k];
        // wrt Phi0 [2nd spot parameter]
        dbeta[k][pstar + pspot * k + 1] = 2.0 * Theta[2] * pow(CosPhi0[k], 2)
          + Theta[3] * pow(2.0 * SinPhi0[k] * CosPhi0[k], 2);
        dbeta[k][pstar + pspot * k + 1] *= 2.0 * pi * (t[i] - tref[k]) / Theta[1];
        dbeta[k][pstar + pspot * k + 1] = cosLambda[k] - dbeta[k][pstar + pspot * k + 1];
        dbeta[k][pstar + pspot * k + 1] *= SinInc * SinPhi0[k] / sinbeta[k];
      }

      /////////////////////////////////////////////////////////////////////////
      //                       SECTION 6: A DERIVATIVES                      //
      /////////////////////////////////////////////////////////////////////////

      for (int k = 0; k < Nspot; ++k) {
        if (alpha[k] > tol) {
          if (beta[k] > (halfpi + alpha[k])) {
            // Case IV
            dAda[k] = 0.0;
            dAdb[k] = 0.0;
          }
          else if (beta[k] < (halfpi - alpha[k])) {
            // Case I
            dAda[k] = 2.0 * pi * cosbeta[k] * sinalpha[k] * cosalpha[k];
            dAdb[k] = -pi * pow(sinalpha[k], 2) * sinbeta[k];
          }
          else {
            // Case II & III
            epsil = 2.0 * (pow(cosalpha[k], 2) + pow(cosbeta[k], 2) - 1.0) 
                            / (pow(sinbeta[k], 2) * Psi[k]);
            dAda[k] = -sinalpha[k] * sinbeta[k] * epsil + 2.0 * cosalpha[k] * cosbeta[k] * Xi[k];
            dAdb[k] = 0.5 * cosalpha[k] * cosbeta[k] * epsil - sinalpha[k] * sinbeta[k] * Xi[k];
          }
        }
        else {
          dAda[k] = 0.0;
          dAdb[k] = 0.0;
        }
      }

      /////////////////////////////////////////////////////////////////////////
      //                     SECTION 7: FINAL DERIVATIVES                    //
      /////////////////////////////////////////////////////////////////////////

      for (int j = 0; j < jmax; ++j) {
        dFab = dFab0[j];
        dFmod[j] = 0.0;
        for (int k = 0; k < Nspot; ++k) {
          // Derivatives of A
          dA = dAda[k] * dalpha[k][j] + dAdb[k] * dbeta[k][j];
          // Derivatives of zeta wrt alpha (and implicitly beta)
          // dzetanegda
          if ((beta[k] - alpha[k]) < halfpi && (beta[k] - alpha[k]) > 0.0)
            dzetanegda[k] = cosalpha[k] * sinbeta[k] - cosbeta[k] * sinalpha[k];
          else
            dzetanegda[k] = 0.0;
          // dzetaposda
          if ((beta[k] + alpha[k]) < halfpi && (beta[k] + alpha[k]) > 0.0)
            dzetaposda[k] = -cosalpha[k] * sinbeta[k] - cosbeta[k] * sinalpha[k];
          else
            dzetaposda[k] = 0.0;
          // Derivatives of zeta
          dzetaneg = dzetanegda[k] * (dalpha[k][j] - dbeta[k][j]);
          dzetapos = dzetaposda[k] * (dalpha[k][j] + dbeta[k][j]);
          dq = 0.0;
          // Derivatives of Upsilon
          for (int n = 0; n < pLD; ++n) {
            dUpsilon = sqrt(pow(zetaneg[k], n+2)) * dzetaneg
              - sqrt(pow(zetapos[k], n+2)) * dzetapos;
            dUpsilon = 0.5 * (n+4.0) * dUpsilon - 2.0 * Upsilon[n][k]
                        * (dzetaneg - dzetapos);
            dUpsilon /= (pow(zetaneg[k], 2) - pow(zetapos[k], 2) + kronecker(zetapos[k], zetaneg[k]));
            // Derivatives of w
            dw = Upsilon[n][k] * dc[n][j] + (c[n] - d[n] * fspot[k]) * dUpsilon -
                  d[n] * Upsilon[n][k] * dfspot[k][j] - 
                  fspot[k] * Upsilon[n][k] * dd[n][j];
            dw *= 4.0 / (n + 4.0);
            // Derivatives of q
            dq += (A[k] * dw + dA * w[n][k]) * piI;
          }
          dFab -= dq;
        }
        for (int m = 0; m < mmax; ++m) {
          dFtilde = Fab0 * B[m] * (Fab + Fab0 * (B[m] - 1.0)) * dU[m][j]
                    + U[m] * (B[m] * Fab0 * dFab - B[m] * Fab * dFab0[j]
                    + Fab0 * (Fab0 - Fab) * dB[m][j]);
          dFtilde *= Box[m] / pow(Fab0*B[m], 2);
          dFmod[j] += dFtilde;
        }
      }

      /////////////////////////////////////////////////////////////////////////
      //                   SECTION 8: TEMPORAL DERIVATIVES                   //
      /////////////////////////////////////////////////////////////////////////

      if (temporal) {
        // Temporal derivatives of alpha and beta
        dFabdt = 0.0;
        dfmoddt[i] = 0.0;
        for (int k = 0; k < Nspot; ++k) {
          if (t[i] < tcrit2[k] && t[i] > tcrit1[k])
            dalphadt = alphamax[k] / ingress[k];
          else if (t[i] < tcrit4[k] && t[i] > tcrit3[k])
            dalphadt = -alphamax[k] / egress[k];
          else
            dalphadt = 0.0;
          dbetadt = 1.0 - pow(SinInc * CosPhi0[k] * cosLambda[k] + CosInc * SinPhi0[k], 2);
          dbetadt = (2.0 * pi * SinInc * CosPhi0[k] * sinLambda[k]) / (Prot[k] * sqrt(dbetadt));
          // Temporal derivatives of zeta
          dzetanegdt = dzetanegda[k] * (dalphadt - dbetadt);
          dzetaposdt = dzetaposda[k] * (dalphadt + dbetadt);
          // Temporal derivatives of A
          dAdt = dAda[k] * dalphadt + dAdb[k] * dbetadt;
          dqdt = 0.0;
          // Temporal derivatives of Upsilon
          for (int n = 0; n < pLD; ++n) {
            dUpsilondt = sqrt(pow(zetaneg[k], n+2)) * dzetanegdt 
                        - sqrt(pow(zetapos[k], n+2)) * dzetaposdt;
            dUpsilondt = 0.5 * (n + 4) * dUpsilondt - 2.0 * Upsilon[n][k] * (dzetanegdt - dzetaposdt);
            dUpsilondt /= (pow(zetaneg[k], 2) - pow(zetapos[k], 2)
                           + kronecker(zetapos[k], zetaneg[k]));
            // Temporal derivatives of w
            dwdt = dUpsilondt * ((4.0 * (c[n] - d[n] * fspot[k])) / (n + 4.0));
            // Temporal derivatives of q
            dqdt += (A[k] * dwdt + dAdt * w[n][k]) * piI;
          }
          // Temporal derivatives of Fab
          dFabdt -= dqdt;
        }
        // Temporal derivatives of Ftilde
        for (int m = 0; m < mmax; ++m) {
          dFtildedt = ((U[m] * Box[m]) / (B[m] * Fab0)) * dFabdt;
          // Temporal derivatives of Fmod
          dfmoddt[i] += dFtildedt;
        }
      }
      else {
        dfmoddt[i] = 0.0;
      }

      /////////////////////////////////////////////////////////////////////////
      //                   SECTION 9: RE-SPLIT DERIVATIVES                   //
      /////////////////////////////////////////////////////////////////////////

      l = 0;
      // Derivatives provided for Theta_star, Theta_inst, Theta_spot discretely
      for (int j = 0; j < pstar; ++j) {
        dfmod_star[i * pstar + j] = dFmod[l];
        l++;
      }
      for (int k = 0; k < Nspot; ++k) {
        for (int j = 0; j < pspot; ++j) {
          dfmod_spot[i * pspot * Nspot + j * Nspot + k] = dFmod[l];
          l++;
        }
      }
      for (int m = 0; m < mmax; ++m) {
        for (int j = 0; j < pinst; j++) {
          dfmod_inst[i * pinst * mmax + j * mmax + m] = dFmod[l];
          l++;
        }
      }
    }
  }

  if (!derivatives) {
    std::fill(dfmod_star, dfmod_star+pstar*ndata, 0.0);
    std::fill(dfmod_spot, dfmod_spot+pspot*Nspot*ndata, 0.0);
    std::fill(dfmod_inst, dfmod_inst+pinst*mmax*ndata, 0.0);
    std::fill(dfmoddt, dfmoddt+ndata, 0.0);
  }

  // reshape output arrays
  dFmod_star.resize({ndata, pstar});
  dFmod_spot.resize({ndata, pspot, Nspot});
  dFmod_inst.resize({ndata, pinst, mmax});

  return std::make_tuple(Fmod, dFmoddt, dFmod_star, dFmod_spot, 
                         dFmod_spot, dFmod_inst, Deltaratio);
}

PYBIND11_MODULE(macula, m)
{
    m.doc() = "";
    m.def("macula", &macula, "",
          "t"_a, "theta_star"_a, "theta_spot"_a, "theta_inst"_a, "tstart"_a, "tend"_a,
          "derivatives"_a=false, "temporal"_a=false, "tdeltav"_a=false);
}
